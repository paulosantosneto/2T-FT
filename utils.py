import os
import argparse
import json
import datasets
from datasets import load_dataset, DatasetDict, Dataset
from typing import List, Dict, Optional, Union, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
import matplotlib.pyplot as plt
from src import *
import re
from tqdm import tqdm
from dataclasses import dataclass, asdict
from peft import LoraConfig, get_peft_model

def get_args():
    
    parser = argparse.ArgumentParser()
    
    # Global configurations
    parser.add_argument('--choice', default='demo', choices=['demo', 'evaluating', 'finetuning'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default='123', type=int)
    parser.add_argument('--model', default='microsoft/phi-2', type=str)
    parser.add_argument('--parallel_size', default=1, type=int)
    
    # [DEMO]

    parser.add_argument('--demo_prompt', default="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?")
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--max_new_tokens', default=300, type=int)
    parser.add_argument('--stop_criteria', default=['Q:', '\n\nQ:'], type=list)
    parser.add_argument('--bert', action='store_true')
    parser.add_argument('--add_shot', default='', type=str)
    parser.add_argument('--pattern', default=r'-?\d+\.?\d*', type=str)
    parser.add_argument('--dtype', default='float16', type=str)
    parser.add_argument('--max_model_len', default=2048, type=int)
    
    # [GENERATE DATASET]

    parser.add_argument('--dataset', nargs='+', choices=['gsm8k', 'multiarith', 'svamp', 'last', 'singleq', 'addsub', 'coin'])
    parser.add_argument('--dataset_path', default='./dataset', type=str)
    parser.add_argument('--log_path', default='/content/log', type=str)
    parser.add_argument('--plot_path', default='/content/plot', type=str)

    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--save_plot', action='store_true')
    parser.add_argument('--dataset_type', default='aritmetic', type=str)
    parser.add_argument('--log_outputs_path', type=str)
    parser.add_argument('--leco_name', type=str)
    parser.add_argument('--cot_name', type=str)

    # [FINE TUNING]

    parser.add_argument('--output_ft', default='/content/ft', type=str)
    parser.add_argument('--logging_steps', default=25, type=int)

    args = parser.parse_args()
    
    return args

def print_output(prompt, cot_outputs, leco_outputs):
    print('-'*100)
    print(f'Your prompt: {prompt}\n')

    print('CoT-Decoding Paths:\n')

    for path in cot_outputs.paths:
 
        print(f"\t(k={path.num_path}) Reasoning Text: {path.reasoning_text} (Score: {path.score:.4f}) (Span: {path.answer_span})")
        
    print('LeCo* Paths:\n')

    for path in leco_outputs.paths:
 
        print(f"\t(k={path.num_path}) Reasoning Text: {path.reasoning_text} (Score: {path.score:.4f})")

def convert_to_string(example):
    example['answer'] = str(example['answer'])
    return example

def concatenate_fields(example):
    example['question'] = example['Body'] + '. ' + example['Question']
    return example

def remove_characters(example):
    example['question'] = example['question'][2:]  # Remove o primeiro caractere
    return example

def load_datasets(datasets: List[str]):

    loaded_datasets = {}

    # math tasks
    for dataset_name in datasets:
        if dataset_name == 'gsm8k':
            loaded_datasets['gsm8k'] = load_dataset('gsm8k', 'main')
        elif dataset_name == 'multiarith':
            dataset = load_dataset('ChilleD/MultiArith')
            dataset = dataset.rename_column('final_ans', 'answer')
            dataset = dataset.map(convert_to_string)
            loaded_datasets['multiarith'] = dataset
        elif dataset_name == 'svamp':
            dataset = load_dataset('ChilleD/SVAMP')
            dataset = dataset.rename_column('Answer', 'answer')
            dataset = dataset.map(concatenate_fields)
            loaded_datasets['svamp'] = dataset
        elif dataset_name == 'singleq':
            dataset = load_dataset("allenai/lila", "singleq")
            dataset = dataset.rename_column('input', 'question')
            dataset = dataset.rename_column('output_answer', 'answer')
            loaded_datasets['singleq'] = dataset
        elif dataset_name == 'addsub':
            dataset = load_dataset("allenai/lila", "addsub")
            dataset = dataset.rename_column('input', 'question')
            dataset = dataset.rename_column('output_answer', 'answer')
            loaded_datasets['addsub'] = dataset
        elif dataset_name == 'coin':
            dataset = load_dataset("skrishna/coin_flip")
            dataset = dataset.rename_column('inputs', 'question')
            dataset = dataset.rename_column('targets', 'answer')
            dataset = dataset.map(remove_characters)
            loaded_datasets['coin'] = {split: dataset.select(range(100)) for split, dataset in dataset.items()}

        elif dataset_name == 'last':
            loaded_datasets['last'] = load_dataset("ChilleD/LastLetterConcat")


    
    return loaded_datasets

def save_logs(log_outputs: Dict, dataset_name: str, log_path: str):

    with open(f'{dataset_name}_cot_decoding.json', 'w') as f:
        json.dump(log_outputs['cot_decoding'], f, indent=4)
    
    with open(f'{dataset_name}_leco_decoding.json', 'w') as f:
        json.dump(log_outputs['leco*'], f, indent=4)

def extract_cot_paths_from_dataset(dataset: DatasetDict,
                                      dataset_name: str,
                                      generator: GeneratePaths,
                                      cot_decoding: CoTDecoding,
                                      leco_decoding: LeCoDecoding,
                                      max_samples: int,
                                      prompt_key: str,
                                      field: Optional[Literal['train', 'val', 'test']]):

    if field is not None:
        dataset = dataset[field]
    else:
        raise ValueError("Field must be one of 'train', 'val', or 'test'.")

    
    prompts = dataset[prompt_key][:max_samples]
    
    log_outputs = {'cot_decoding': {}, 'leco*': {}}
    
    for i, prompt in enumerate(tqdm(prompts, desc=dataset_name, total=len(prompts))):
        
        topk_tokens, outputs = generator.search_cots(prompt)
        cot_paths = cot_decoding.calculate_score(prompt, topk_tokens, outputs)
        diver_paths = leco_decoding.calculate_score(prompt, topk_tokens, outputs)
        
        log_outputs['cot_decoding'][f'idx={i}:{prompt}'] = asdict(cot_paths)['paths']
        log_outputs['leco*'][f'idx={i}:{prompt}'] = asdict(diver_paths)['paths']

    return log_outputs

def greedy_decoding(log_outputs: Dict) -> List:
    
    return [paths[0]['answer_span'] for key, paths in log_outputs.items()]

def choose_specific_depth(log_outputs: Dict, k: int) -> List:
    
    return [paths[k]['answer_span'] for key, paths in log_outputs.items()]

def best_score(log_outputs: Dict, k: int) -> List:
    
    return [max(paths[:k], key=lambda x: x['score'])['answer_span'] for key, paths in log_outputs.items()]

def self_consistency(log_outputs: Dict, k: int) -> List:
    
    consistency_answer_span = []
    for key, paths in log_outputs.items():
        consistency = {}
        for path in paths[:k]:
            if len(path['answer_span']) > 0:
                if path['answer_span'] not in consistency:
                    consistency[path['answer_span']] = 0
                consistency[path['answer_span']] += path['score']
            
        if len(consistency) > 0:
            major_answer_span = max(consistency, key=consistency.get)
            consistency_answer_span.append(max([item for item in paths if item['answer_span'] == major_answer_span], key=lambda x: x['score'])['answer_span'])
        else:
            consistency_answer_span.append(paths[0]['answer_span'])
            
    return consistency_answer_span

def exact_match(predicted, ground_truth):
    
    return sum([pred == gt for pred, gt in zip(predicted, ground_truth)]) / len(ground_truth)

def extract_exact_match(log_outputs, pattern):
    for method in log_outputs.keys():
        for key, paths in log_outputs[method].items():
            for path in paths:
                reasoning_text = str(path.get('reasoning_text', ''))
                answer_span = re.findall(pattern, reasoning_text)
                if answer_span:
                    path['answer_span'] = answer_span[-1].replace(',', '')
                else:
                    path['answer_span'] = ''
    return log_outputs

                
def evaluating(log_outputs, ground_truth, pattern: str, num_paths: int, tokenizer: AutoTokenizer, max_new_tokens: int):
    
    evaluations = []
    
    for k in range(1, num_paths):
        
        log_outputs = extract_exact_match(log_outputs, pattern)
        
        greedy_decoding_acc = exact_match(greedy_decoding(log_outputs['cot_decoding']), ground_truth)                       
        cot_decoding_max_acc = exact_match(best_score(log_outputs['cot_decoding'], k=k), ground_truth)
        cot_decoding_agg_acc = exact_match(self_consistency(log_outputs['cot_decoding'], k=k), ground_truth)
        leco_acc = exact_match(best_score(log_outputs['leco*'], k=k), ground_truth)
        
        evaluations.append({'Greedy Decoding': greedy_decoding_acc,
                            'CoT-Decoding (max)': cot_decoding_max_acc,
                            'LeCo*': leco_acc,
                            'CoT-Decoding (agg)': cot_decoding_agg_acc})

    return evaluations

def print_evaluations(evaluations: List, dataset_name: str):
    
    print(f'--- Evaluation using Exact Match for {dataset_name} ---')
    print(f'\tGreedy Decoding: {evaluations[-1]["Greedy Decoding"]:.4f}')
    print(f'\tCoT-Decoding (max): {evaluations[-1]["CoT-Decoding (max)"]:.4f}')
    print(f'\tLeCO*: {evaluations[-1]["LeCo*"]:.4f}')
    print(f'\tCoT-Decoding (agg): {evaluations[-1]["CoT-Decoding (agg)"]:.4f}')

def load_cot_dataset(args):

    log_outputs = {'cot_decoding': {}, 'leco*': {}}

    with open(f'{args.log_path}/{args.dataset[0]}_cot_decoding.json', 'r') as f:
        log_outputs['cot_decoding'] = json.load(f)
    
    with open(f'{args.log_path}/{args.dataset[0]}_leco_decoding.json', 'r') as f:
        log_outputs['leco_decoding'] = json.load(f)

    return log_outputs

def prepare_finetuning(args, log_outputs):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.unk_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", quantization_config=quantization_config)

    loraconfig = LORAConfig()

    '''
    lora_config = LORAConfig(
        rank=loraconfig.rank,
        lora_alpha=loraconfig.lora_alpha,
        target_modules=loraconfig.target_modules,
        lora_dropout=loraconfig.lora_dropout,
        bias=loraconfig.bias,
        task_type=loraconfig.task_type,
        use_rslora=loraconfig.use_rslora,
    )
    '''
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'dense', 'fc1', 'fc2'],
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM',
        #modules_to_save=['lm_head'],
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)

    ft_dataset = {'question': [], 'reasoning': []}

    pattern = r"idx=\d+:"

    for key in log_outputs['cot_decoding'].keys():
        match = re.search(pattern, key)
        start_pos = match.end()
        ft_dataset['question'].append(key[start_pos:])

    ft_dataset['reasoning'] = best_score(log_outputs['cot_decoding'], args.topk)

    dataset = datasets.Dataset.from_dict(ft_dataset)

    return model, tokenizer, dataset

def save_plot(evaluations: Dict, title: str, plot_path: str):
    
    components = ['Greedy Decoding', 'CoT-Decoding (max)', 'LeCo*', 'CoT-Decoding (agg)']
    symbols = ['o', 's', '^', 'd']
    colors = [(0.4, 0.8, 0.4, 0.8), (0.4, 0.4, 0.8, 0.8), (0.8, 0.4, 0.4, 0.8), (0.8, 0.8, 0.4, 0.8)]  # RGBA format

    plt.figure(figsize=(10, 6))

    for idx, comp in enumerate(components):
        exact_match_values = [evaluation[comp] for evaluation in evaluations]
        plt.plot(range(1, len(evaluations) + 1), exact_match_values, marker=symbols[idx], color=colors[idx], label=comp)

    plt.title(title)
    plt.xlabel('Paths (K)')
    plt.ylabel('Exact Match Acc')
    plt.xticks(range(1, len(evaluations) + 1))
    plt.legend(loc='upper left')
    
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_path + '.png')

def plot_loss(log_history):
    
    losses = []
    steps = []
    
    for log in log_history:
        if 'loss' in log:
            losses.append(log['loss'])
            steps.append(log['step'])

    # Plota a perda ao longo dos passos
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Steps')
    plt.grid(True)
    plt.savefig(plot_path + '_loss.png')
    