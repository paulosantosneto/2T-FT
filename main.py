import torch

from src import *
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings

# Suppress all warnings globally
warnings.simplefilter("ignore")

def generate_paths(args):

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

    topk_tokens, outputs = generator.search_cots(args.demo_prompt)
    cot_paths = cot_decoding.calculate_score(args.demo_prompt, topk_tokens, outputs)
    leco_paths = leco_decoding.calculate_score(args.demo_prompt, topk_tokens, outputs)


    return cot_paths, leco_paths

if __name__ == "__main__":

    args = get_args()

    if args.choice == 'demo':
        cot_paths, leco_paths = generate_paths(args)
        print_output(args.demo_prompt, cot_paths, leco_paths)
