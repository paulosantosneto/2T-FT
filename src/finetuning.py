import torch
import os
import json
import re
import transformers
import torch.nn as nn

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Union, Literal
from torch.utils.data import DataLoader, Dataset

@dataclass
class LORAConfig:
    rank: int = 8
    lora_alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'dense', 'fc1', 'fc2'])
    lora_dropout: float = 0.1
    bias: str = 'none'
    task_type: str = 'CAUSAL_LM'
    use_rslora: bool = True

@dataclass
class TrainerConfig:
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: int = 1,
    warmup_ratio: float = 0.1,
    learning_rate: float = 1e-4,
    fp16: bool = False,
    group_by_length: bool = True,
    lr_scheduler_type: str = 'cosine',
    optim: str = 'paged_adamw_8bit',
    report_to: str = 'none'

class TextDataset(Dataset):
    def __init__(self, encodings, question_lengths, answer_lengths, tokenizer):
        self.encodings = encodings
        self.question_lengths = question_lengths
        self.answer_lengths = answer_lengths
        self.tokenizer = tokenizer
        #print(self.question_lengths)
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        
        item['labels'] = item['input_ids'].clone()

        item['labels'][:-1] = item['labels'].clone()[1:]
        item['labels'][-1] = self.tokenizer.eos_token_id
        
        item['loss_mask'] = torch.zeros_like(item['input_ids'])
        item['loss_mask'][self.question_lengths[idx] - 1] = 1
        item['loss_mask'][self.question_lengths[idx] + self.answer_lengths[idx]] = 1
        
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

def prepare_dataset(dataset, tokenizer):
    
    formatted_dataset = dataset.map(
        lambda x: {
            'text': ''.join([x['question'], x['reasoning']]),
            'reasoning_text': ''.join(x['reasoning'])
        }
    )
    
    encodings = tokenizer(
        [data['text'] for data in formatted_dataset], 
        truncation=True, 
        padding='max_length', 
        max_length=512, 
        return_tensors='pt'
    ) 
    question_lengths = [len(tokenizer.encode(data['question'], truncation=True, padding=False, max_length=512)) for data in formatted_dataset]
    answer_lengths = [len(tokenizer.encode(data['reasoning_text'], truncation=True, padding=False, max_length=512)) for data in formatted_dataset]

    text_dataset = TextDataset(encodings, question_lengths, answer_lengths, tokenizer)
    
    return text_dataset

class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        labels = inputs.pop('labels')
        loss_mask = inputs.pop('loss_mask')
        
        # forward
        
        outputs = model(**inputs)
        
        logits = outputs.logits
        
        if torch.isnan(logits).any():
            print('NaN detected in logits')
            print(logits)
        
        probs = nn.functional.softmax(logits, dim=-1)
        
        predicted_token_ids = torch.argmax(probs, dim=-1)
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
        
        losses = losses.view(-1, inputs['input_ids'].size(1))
        
        masked_loss = losses * loss_mask
        
        loss = masked_loss.sum() / (loss_mask.sum() + 1e-9)
        
        batch_size, seq_length = inputs['input_ids'].size()
        
        return (loss, outputs) if return_outputs else loss
    
    def get_train_dataloader(self):
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        dataloader_params = {
            'batch_size': self.args.train_batch_size,
            'collate_fn': data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory
        }
        
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params['shuffle'] = True
            dataloader_params['drop_last'] = self.args.dataloader_drop_last
        
        return DataLoader(train_dataset, **dataloader_params)
    
    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        data_collator = self.data_collator
        
        dataloader_params = {
            'batch_size': self.args.eval_batch_size,
            'collate_fn': data_collator,
            'num_workers': self.args.dataloader_num_workers,
            'pin_memory': self.args.dataloader_pin_memory,
            'shuffle': False,
            'drop_last': self.args.dataloader_drop_last,
        }
        
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params.pop('shuffle', None)
            dataloader_params.pop('drop_last', None)
        
        return DataLoader(eval_dataset, **dataloader_params)

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        loss_mask = torch.stack([item['loss_mask'] for item in batch])
        
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_mask': loss_mask
        }

def run_trainer(train_dataset, model, data_collator, args):

    trainer_configs = TrainerConfig()

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        args=TrainingArguments(
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_grad_norm=1,
            warmup_ratio=0.1,
            learning_rate=1e-4,
            fp16=trainer_configs.fp16,
            logging_steps=args.logging_steps,
            output_dir=args.output_ft,
            optim='paged_adamw_8bit',
            group_by_length=trainer_configs.group_by_length,
            lr_scheduler_type='cosine',
            report_to=trainer_configs.report_to,
        ),
        data_collator=data_collator,
    )

    model.config.use_cache = False

    trainer.train()

    return model, trainer.state.log_history