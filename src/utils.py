import os
import argparse
from datasets import load_dataset, DatasetDict
from typing import List, Dict, Optional, Union, Literal
import matplotlib.pyplot as plt
from .decoding import *
from dataclasses import dataclass

def get_args():
    
    parser = argparse.ArgumentParser()
    
    # Global configurations
    parser.add_argument('--choice', default='demo', choices=['demo', 'generate_dataset', 'finetuning'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default='123', type=int)
    parser.add_argument('--model', default='microsoft/phi-2', type=str)
    parser.add_argument('--parallel_size', default=1, type=int)
    
    # [DEMO]

    parser.add_argument('--demo_prompt', default="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?")
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--max_new_tokens', default=300, type=int)
    parser.add_argument('--stop_criteria', default=['Q:', '\n\nQ:'], type=list)
    parser.add_argument('--answer_span_model', default='huggingface-course/bert-finetuned-squad', type=str)
    parser.add_argument('--add_shot', default='', type=str)
    parser.add_argument('--pattern', default=r'-?\b\d+(?:[,.]\d{1,10})*\b', type=str)

    # [GENERATE DATASET]

    parser.add_argument('--dataset', nargs='+', choices=['gsm8k', 'multiarith', 'svamp', 'last_letters', 'single_eq', 'addsub'])

    # [FINE TUNING]


    args = parser.parse_args()
    
    return args

def print_output(prompt, cot_outputs, leco_outputs):
    print('-'*100)
    print(f'Your prompt: {prompt}\n')

    print('\tCoT-Decoding Paths:\n')

    for path in cot_outputs.paths:
 
        print(f"\t\t(k={path.num_path}) Reasoning Text: {path.reasoning_text} (Score: {path.score:.4f}) (Span: {path.answer_span})\n")
        print('- '*80)
        
    print('\tLeCo* Paths:\n')

    for path in leco_outputs.paths:
 
        print(f"\t\t(k={path.num_path}) Reasoning Text: {path.reasoning_text} (Score: {path.score:.4f})\n")
        print('- '*80)

    print(leco_outputs)

def load_datasets(datasets: List[str]):

    loaded_datasets = {}

    # math tasks
    for dataset_name in datasets:
        if dataset_name == 'gsm8k':
            loaded_datasets['gsm8k'] = load_dataset('gsm8k', 'main')
        elif dataset_name == 'multiarith':
            loaded_datasets['multiarith'] = load_dataset('ChilleD/MultiArith')
        elif dataset_name == 'svamp':
            loaded_datasets['svamp'] = load_dataset('ChilleD/SVAMP')
    
    return loaded_datasets

def extract_cot_decoding_from_dataset(dataset: DatasetDict,
                                      dataset_name: str,
                                      generator: GeneratePaths,
                                      cot_decoding: CoTDecoding,
                                      divergence_decoding: LeCoDecoding,
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
        
        topk_tokens, outputs = generator.search_cots(prompt, verbose=True)
        cot_paths = cot_decoding.calculate_score(prompt, topk_tokens, outputs)
        diver_paths = divergence_decoding.calculate_score(prompt, topk_tokens, outputs)
        
        log_outputs['cot_decoding'][i] = asdict(cot_paths)['paths']
        log_outputs['leco*'][i] = asdict(diver_paths)['paths']

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
        cot_decoding_acc = exact_match(best_score(log_outputs['cot_decoding'], k=k), ground_truth)
        self_consistency_acc = exact_match(self_consistency(log_outputs['cot_decoding'], k=k), ground_truth)
        leco_acc = exact_match(best_score(log_outputs['leco*'], k=k), ground_truth)
        
        evaluations.append({'Greedy Decoding': greedy_decoding_acc,
                            'CoT-Decoding': cot_decoding_acc,
                            'LeCO*': leco_acc,
                            'CoT-Decoding + Self-Consistency': self_consistency_acc})

    return evaluations

def print_evaluations(evaluations: List):
    
    print('--- Evaluation using Exact Match ---')
    print(f'\tGreedy Decoding: {evaluations[-1]["Greedy Decoding"]:.4f}')
    print(f'\tCoT-Decoding: {evaluations[-1]["CoT-Decoding"]:.4f}')
    print(f'\tLeCO*: {evaluations[-1]["LeCO*"]:.4f}')
    print(f'\tCoT-Decoding + Self-Consistency: {evaluations[-1]["CoT-Decoding + Self-Consistency"]:.4f}')

def plot_k_paths(evaluations, title: str):
    
    components = ['Greedy Decoding', 'CoT-Decoding', 'LeCO*', 'CoT-Decoding + Self-Consistency']
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
    plt.show()
    