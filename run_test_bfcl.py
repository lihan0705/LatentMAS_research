import argparse
import json
import os
import time
from typing import List, Dict
from tqdm import tqdm

from models import ModelWrapper
from models_tool import ModelWrapperWithTools
from methods.baseline import BaselineMethod
from methods.latent_mas_tool import LatentMASToolMethod
from utils import set_seed, auto_device, compare_tool_calls

def load_bfcl_dataset(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            dataset = json.loads(content)
        else:
            dataset = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    # Standardize BFCL v3 fields: 'function' -> 'functions', 'ground_truth' -> 'gold'
    for item in dataset:
        if 'function' in item and 'functions' not in item:
            item['functions'] = item['function']
        if 'ground_truth' in item and 'gold' not in item:
            # BFCL ground_truth is often a list, we take the first one or the whole string
            gt = item['ground_truth']
            item['gold'] = gt[0] if isinstance(gt, list) and len(gt) > 0 else str(gt)
            
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Test LatentMAS on BFCL dataset')
    parser.add_argument('--method', choices=['baseline', 'latent_mas_tool'], required=True)
    parser.add_argument('--model_name', default='Qwen/Qwen3-8B')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--max_samples', type=int, default=20)
    parser.add_argument('--latent_steps', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--think', action='store_true')
    parser.add_argument('--latent_space_realign', action='store_true', default=True)
    
    # Missing args required by latent_mas_tool.py:
    parser.add_argument('--prompt', choices=['sequential', 'hierarchical'], default='sequential')
    parser.add_argument('--task', default='gsm8k')
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--latent_only', action='store_true')
    parser.add_argument('--sequential_info_only', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--generate_bs', type=int, default=1)
    
    args = parser.parse_args()
    args.task = "toolcalling"

    set_seed(args.seed)
    device = auto_device(args.device)

    print(f"Initializing {args.method} with {args.model_name}...")
    
    if args.method == 'baseline':
        model = ModelWrapper(args.model_name, device, args=args)
        method = BaselineMethod(model, args=args)
    else:
        model = ModelWrapperWithTools(args.model_name, device, args=args, enable_tools=True)
        # DummyExecutor will be set inside the loop for each sample
        
        method = LatentMASToolMethod(
            model,
            latent_steps=args.latent_steps,
            args=args,
            enable_tools=True
        )

    dataset = load_bfcl_dataset(args.dataset)
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]

    print(f"Testing {len(dataset)} samples...")
    
    results = []
    start_time = time.time()
    
    for i, item in enumerate(tqdm(dataset)):
        print(f"\n--- Problem #{i+1} ---")
        # Register tools for LTE-MAS retrieval
        if hasattr(model, 'tool_vectorizer'):
            # Clear anchors from previous sample to avoid contamination
            model.tool_vectorizer.clear_anchors()
            for func in item.get('functions', []):
                model.tool_vectorizer.add_tool(func['name'], str(func))
        
        # Prepare function schema for Phase 2 decoding
        functions_str = json.dumps(item.get('functions', []), indent=2)
        
        # Inject schema into the model context for the generator
        model.current_functions_schema = functions_str
        
        # Improve DummyExecutor to be more robust
        class DummyExecutor:
            def execute(self, code):
                if not code or code == "None" or code == "func_name(param=value)":
                    return False, "No valid tool call generated"
                
                # Strip [CALL] tags if they still exist for some reason
                import re
                code = re.sub(r'\[/?CALL\]', '', code, flags=re.IGNORECASE).strip()
                
                # If it contains balanced parens or looks like a function call, assume success
                if "(" in code and ")" in code:
                    return True, code.strip()
                
                # Truncated call handling: if it ends with ( or , it's definitely truncated but we might still accept it for "detected"
                if "(" in code and (code.endswith(",") or code.endswith("(") or len(code) > 20):
                    return True, code.strip() + ")" # Try to close it
                
                return False, f"Invalid function call format: {code[:50]}"
        model._tool_executor = DummyExecutor()
        
        res = method.run_item(item)
        
        # For BFCL, we care about what Tool was called
        # The method (Baseline or LatentMAS) already handles extraction into 'prediction'
        generated_tool_call = res.get('prediction', 'None')
        
        # If the method didn't extract anything, we can try to look into agents as a secondary fallback
        if generated_tool_call == "None" or generated_tool_call == "func_name(param=value)":
            agents = res.get('agents', [])
            for agent in agents:
                if agent.get('tool_calls'):
                    t_info = agent['tool_calls'][0]
                    if t_info.get('success'):
                        generated_tool_call = t_info.get('tool_call', 'None')
                        break
                
        print(f"Gold API: {item['gold']}")
        print(f"Pred API: {generated_tool_call}")
        
        # Simple Exact Match for now (BFCL official uses AST matching)
        # This will be low, but it's a safe starting metric
        is_correct = compare_tool_calls(item["gold"], generated_tool_call)
        print(f"Match: {is_correct}")
        
        res['bfcl_generated_api'] = generated_tool_call
        res['bfcl_correct'] = is_correct
        results.append(res)

    total_time = time.time() - start_time
    correct_count = sum(1 for r in results if r['bfcl_correct'])
    
    summary = {
        'method': args.method,
        'accuracy': correct_count / len(results) if results else 0,
        'correct': correct_count,
        'total_time_sec': total_time,
        'time_per_sample': total_time / len(results) if results else 0
    }
    
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    
    os.makedirs('results', exist_ok=True)
    out_file = f"results/bfcl_{args.method}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump({'summary': summary, 'predictions': results}, f, indent=2)

if __name__ == '__main__':
    main()
