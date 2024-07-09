import torch
import re
from src import *
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
from utils import *

# Suppress all warnings globally
warnings.simplefilter("ignore")

def configurate_method(args):

    torch.manual_seed(args.seed)

    model = LLM(model=args.model, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
        
    generator = GeneratePaths(model=model, 
                              topk=args.topk, 
                              max_new_tokens=args.max_new_tokens, 
                              stop=args.stop_criteria,
                              prompt=args.add_shot)
    
    answer_span_model = None
    if args.answer_span_model:
        answer_span_model = pipeline('question-answering', model=args.answer_span_model)

    cot_decoding = CoTDecoding(answer_span_model=answer_span_model,
                               pattern=args.pattern,
                               tokenizer=tokenizer,
                               prompt=args.add_shot)
    
    leco_decoding = LeCoDecoding()

    return generator, cot_decoding, leco_decoding, tokenizer

def simple_demo(generator, cot_decoding, leco_decoding, args):

    topk_tokens, outputs = generator.search_cots(args.demo_prompt)
    cot_paths = cot_decoding.calculate_score(args.demo_prompt, topk_tokens, outputs)
    leco_paths = leco_decoding.calculate_score(args.demo_prompt, topk_tokens, outputs)

    return cot_paths, leco_paths

def evaluating_decoding_methods(args):

    generator, cot_decoding, leco_decoding, tokenizer = configurate_method(args)
    datasets = load_datasets(args.datasets)

    print(datasets.keys())
    
    for dataset in datasets:
        log_outputs = extract_cot_paths_from_dataset(dataset=datasets[dataset],
                                           dataset_name=dataset,
                                           max_samples=datasets[dataset]['test'].num_rows,
                                           #max_samples=10,
                                           field='test',
                                           prompt_key='question',
                                           generator=generator,
                                           cot_decoding=cot_decoding,
                                           leco_decoding=leco_decoding        
        )

        if args.save_log:
            save_logs(log_outputs, dataset, args.log_path)
        
        ground_truth = [re.findall(args.pattern, string)[-1] for string in datasets[dataset]['test']['answer']]
        
        evaluations = evaluating(log_outputs=log_outputs,
                                 ground_truth=ground_truth,
                                 pattern=args.pattern,
                                 num_paths=args.topk, 
                                 tokenizer=tokenizer,
                                 max_new_tokens=args.max_new_tokens)
        
        if args.save_plot:
            save_plot(evaluations, f'Exatch Match Evaluation - {dataset}', dataset)
        
        print_evaluations(evaluations, dataset)
        
if __name__ == "__main__":

    args = get_args()

    if args.choice == 'demo':
        generator, cot_decoding, leco_decoding, tokenizer = configurate_method(args)
        cot_paths, leco_paths = simple_demo(generator, cot_decoding, leco_decoding, args)
    elif args.choice == 'evaluating':
        evaluating_decoding_methods(args)
