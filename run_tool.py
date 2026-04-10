"""
Run script with Tool Calling Support

This script extends the original run.py with tool calling capabilities
while maintaining full backward compatibility.

Usage:
    # Original usage (no tools)
    python run_tool.py --method latent_mas --model_name Qwen/Qwen3-4B --task gsm8k
    
    # With tool calling (new)
    python run_tool.py --method latent_mas_tool --model_name Qwen/Qwen3-4B --task toolcalling --enable_tools
    
    # Baseline for comparison
    python run_tool.py --method baseline --model_name Qwen/Qwen3-4B --task toolcalling
"""

import argparse
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import time
import os
import sys
from datetime import datetime


class TeeLogger:
    """
    A logger that writes to both terminal and a log file.
    Usage:
        tee = TeeLogger(log_file_path)
        tee.start()
        # ... all print statements go to both terminal and file ...
        tee.stop()
    """
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.terminal = sys.stdout
        self.log_file = None
    
    def start(self):
        """Start capturing output to both terminal and file."""
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        sys.stdout = self
    
    def stop(self):
        """Stop capturing and restore normal output."""
        if self.log_file:
            sys.stdout = self.terminal
            self.log_file.close()
            self.log_file = None
    
    def write(self, message):
        """Write to both terminal and file."""
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()

# Import data loaders
from data import (
    load_gsm8k,
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)

# Import methods
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from methods.latent_mas_tool import LatentMASToolMethod

# Import models
from models import ModelWrapper
from models_tool import ModelWrapperWithTools

from utils import auto_device, set_seed


def load_toolcalling_dataset(split: str = "test") -> List[Dict]:
    """Load the tool calling dataset."""
    data_path = "data/test_dataset_toolcalling.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to standard format
    items = []
    for item in data:
        items.append({
            "question": item["question"],
            "gold": str(item["final_answer"]),
            "solution": item.get("thought", ""),
            "tool_call": item.get("tool_call", ""),
            "tool_output": item.get("tool_output", ""),
        })
    return items


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    
    # Handle vLLM for latent_mas
    if args.method in ["latent_mas", "latent_mas_tool"] and args.use_vllm:
        if hasattr(method, 'run_batch_vllm'):
            results = method.run_batch_vllm(current_batch)
        else:
            results = method.run_batch(current_batch)
    else:
        results = method.run_batch(current_batch)
    
    if len(results) > remaining:
        results = results[:remaining]
    
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        
        # Print agents
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            print(f"----- Agent: {name} ({role}) -----")
            
            # Print tool calls if any
            tool_calls = a.get("tool_calls")
            if tool_calls:
                print("[Tool Calls]")
                for tc in tool_calls:
                    print(f"  Agent: {tc.get('agent')}")
                    print(f"  Call: {tc.get('tool_call', 'N/A')[:100]}...")
                    print(f"  Result: {tc.get('tool_result', 'N/A')}")
                    print(f"  Success: {tc.get('success')}")
            
            latent_steps = a.get("latent_steps", None)
            if latent_steps is not None:
                print(f"[Latent Steps: {latent_steps}]")
            
            output = a.get("output", "").rstrip()
            if output:
                print(f"[Output]\n{output}")
            print("----------------------------------------------")
        
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")
    
    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    
    return processed, preds


def main():
    parser = argparse.ArgumentParser()
    
    # Core args
    parser.add_argument("--method", 
                        choices=["baseline", "text_mas", "latent_mas", "latent_mas_tool"],
                        required=True,
                        help="Method to run. 'latent_mas_tool' enables tool calling.")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--task", 
                        choices=["gsm8k", "aime2024", "aime2025", "gpqa", 
                                 "arc_easy", "arc_challenge", "mbppplus", 
                                 "humanevalplus", "medqa", "toolcalling"],
                        default="gsm8k")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], 
                        default="sequential")
    
    # Model args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=1)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    # vLLM args
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--use_second_HF_model", action="store_true")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    
    # Tool calling args (NEW)
    parser.add_argument("--enable_tools", action="store_true",
                        help="Enable tool calling for latent_mas_tool method")
    parser.add_argument("--tool_threshold", type=float, default=0.3,
                        help="Threshold for tool detection (default 0.3, lowered from 0.7)")
    parser.add_argument("--tool_detection_step", type=int, default=-1,
                        help="Step to check for tool need (-1 = middle step)")
    
    # Logging arg
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save log files")
    
    args = parser.parse_args()
    
    # === Setup logging to both terminal and file ===
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{args.log_dir}/{args.method}_{args.model_name.replace('/', '_')}_{args.task}_{timestamp}.log"
    tee = TeeLogger(log_file)
    tee.start()
    print(f"[Log] Logging to: {log_file}")
    print(f"[Log] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Auto-enable tool settings for latent_mas_tool
    if args.method == "latent_mas_tool":
        args.enable_tools = True
    
    # vLLM auto-configuration for latent_mas methods
    if args.method in ["latent_mas", "latent_mas_tool"] and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    
    # Create model with or without tool support
    if args.enable_tools:
        print(f"[Info] Creating model with tool support")
        model = ModelWrapperWithTools(
            args.model_name, 
            device, 
            use_vllm=args.use_vllm, 
            args=args,
            enable_tools=True,
            tool_threshold=args.tool_threshold
        )
    else:
        model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()
    
    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Method selection
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_mas":
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "latent_mas_tool":
        method = LatentMASToolMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
            enable_tools=args.enable_tools,
            tool_threshold=args.tool_threshold,
            tool_detection_step=args.tool_detection_step,
        )
    
    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # Dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    elif args.task == "toolcalling":
        dataset_iter = load_toolcalling_dataset(split='test')
    else:
        raise ValueError(f'no {args.task} support')
    
    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)
        args.max_samples = len(dataset_iter)
    
    progress = tqdm(total=args.max_samples)
    
    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method, batch, processed, preds, progress, args.max_samples, args
            )
            batch = []
            if processed >= args.max_samples:
                break
    
    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method, batch, processed, preds, progress, max_samples=args.max_samples, args=args
        )
    
    progress.close()
    
    total_time = time.time() - start_time
    acc, correct = evaluate(preds)
    
    # Save results
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{args.method}_{args.model_name.replace('/', '_')}_{args.task}_{timestamp}.json"
    
    result_data = {
        "summary": {
            "method": args.method,
            "model": args.model_name,
            "dataset": f"data/test_dataset_toolcalling.json" if args.task == "toolcalling" else args.task,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "latent_steps": args.latent_steps,
            "enable_tools": args.enable_tools,
            "accuracy": acc,
            "correct": correct,
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / args.max_samples, 4),
        },
        "predictions": preds
    }
    
    # Add tool stats if available
    if args.enable_tools and hasattr(method, 'get_tool_stats'):
        result_data["tool_stats"] = method.get_tool_stats()
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Accuracy: {acc:.4f} ({correct}/{args.max_samples})")
    if args.enable_tools and hasattr(method, 'get_tool_stats'):
        print(f"Tool Stats: {method.get_tool_stats()}")
    print(f"{'='*60}")
    
    # Stop logging
    print(f"\n[Log] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee.stop()
    print(f"[Log] Log saved to: {log_file}")


if __name__ == "__main__":
    main()
