import argparse
import json
import os
import time
from typing import List, Dict
from tqdm import tqdm

from models_tool import ModelWrapperWithTools
from methods.latent_mas_tool import LatentMASToolMethod
from utils import normalize_answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--dataset", type=str, default="data/test_dataset_hotpot_latent.json")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--latent_steps", type=int, default=20)
    parser.add_argument("--tool_threshold", type=float, default=0.5)
    parser.add_argument("--method", type=str, default="latent_mas_tool")
    parser.add_argument("--prompt", type=str, default="sequential")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    # Define hotpot tool schema
    hotpot_tools = [
        {
            "name": "search",
            "description": "Search for information about a specific entity or film title to get background details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title of the entity or film to search for."}
                },
                "required": ["title"]
            }
        }
    ]

    print(f"Initializing {args.method} with {args.model_name}...")
    model = ModelWrapperWithTools(args.model_name, device=args.device, enable_tools=True, tool_threshold=args.tool_threshold, args=args)
    
    # Custom executor that looks up context from the JSON
    class HotpotExecutor:
        def __init__(self, current_item):
            self.item = current_item
            
        def execute(self, code):
            import re
            # Extract title from search(title="...") or search("...")
            match = re.search(r'search\((?:title\s*=\s*)?["\'](.*?)["\']\)', code)
            if not match:
                return False, "Invalid search format. Use search(title='Name')"
            
            search_title = match.group(1).lower()
            for title, lines in self.item.get('context', []):
                if search_title in title.lower() or title.lower() in search_title:
                    result = " ".join(lines)
                    print(f"[HotpotExecutor] Found info for '{search_title}': {result[:50]}...")
                    return True, result
            
            return False, f"No information found for '{search_title}'"

    method = LatentMASToolMethod(model, args=args)
    
    with open(args.dataset, 'r') as f:
        data = json.load(f)

    results = []
    correct_count = 0
    
    print(f"Testing {len(data)} samples on HotpotQA reasoning...")
    for i, item in enumerate(tqdm(data)):
        print(f"\n--- Problem #{i+1} ---")
        # Update current context for the executor
        model._tool_executor = HotpotExecutor(item)
        
        # Injected functions schema
        item['functions'] = hotpot_tools
        
        # We need to tell the method NOT to short-circuit if we want to see final reasoning
        # Our modified LatentMASToolMethod will use the tool result and then Judger will decide
        res = method.run_item(item)
        
        prediction = res.get('prediction', '')
        gold = item['gold']
        
        is_correct = normalize_answer(gold) in normalize_answer(prediction)
        if is_correct:
            correct_count += 1
            
        print(f"Question: {item['question']}")
        print(f"Gold: {gold}")
        print(f"Pred: {prediction}")
        print(f"Correct: {is_correct}")
        
        # Log tool usage
        for agent in res.get('agents', []):
            if agent.get('tool_calls'):
                t = agent['tool_calls'][0]
                print(f"  - {agent['name']} used tool: {t.get('tool_call')} (Success: {t.get('success')})")

        results.append({
            "question": item['question'],
            "prediction": prediction,
            "gold": gold,
            "correct": is_correct,
            "trace": res.get('agents', [])
        })

    accuracy = correct_count / len(data)
    print(f"\nFinal Accuracy: {accuracy:.2f} ({correct_count}/{len(data)})")

    out_file = f"results/hotpot_latent_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump({'accuracy': accuracy, 'results': results}, f, indent=2)

if __name__ == '__main__':
    main()
