"""
Test script: Evaluate LatentMAS on custom math tool-calling dataset
Compare pure reasoning (without tool calling) capability
"""
import os

# HuggingFace mirror for China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Note: Do NOT set CUDA_VISIBLE_DEVICES here.
# GPU 0 is broken, but the working GPU (physical GPU 1) becomes cuda:0
# when CUDA_VISIBLE_DEVICES is not set or when filtering out the broken GPU.

import argparse
import json
from typing import Dict, List, Tuple

from tqdm import tqdm

from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.latent_mas_tool import LatentMASToolMethod
from models import ModelWrapper
from utils import auto_device, set_seed, normalize_answer, extract_gsm8k_answer
import time


def load_test_dataset(file_path: str = "data/test_dataset_toolcalling.json"):
    """Load custom test dataset"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        # Direct use of final_answer field (now contains just numbers/values)
        final_answer = item.get("final_answer", "")
        gold = final_answer  # Use the final_answer directly as gold standard
        
        yield {
            "question": item.get("question", ""),
            "solution": str(final_answer),  # Convert to string for consistency
            "gold": gold,
            # Keep original data for analysis
            "original_id": item.get("id", ""),
            "thought": item.get("thought", ""),
            "tool_call": item.get("tool_call", ""),
            "tool_output": item.get("tool_output", ""),
        }


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

    if args.method == "latent_mas" and args.use_vllm:
        results = method.run_batch_vllm(current_batch)
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
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser(description="Test LatentMAS on tool-calling dataset")

    # Core args
    parser.add_argument("--method", choices=["baseline", "latent_mas", "latent_mas_tool"], default="latent_mas",
                        help="Test method: baseline(pure reasoning) or latent_mas")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-14B", "Qwen/Qwen3-8B"],
                        help="Model choice")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of samples to test, -1 for all")
    parser.add_argument("--dataset", type=str, default="data/test_dataset_toolcalling.json",
                        help="Custom dataset path")

    # Other args
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=5, help="LatentMAS reasoning steps")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=10, help="Batch size")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend")
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--use_second_HF_model", action="store_true")
    parser.add_argument("--device2", type=str, default="cuda:0")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)

    # Prompt type for LatentMAS
    parser.add_argument("--prompt", type=str, default="sequential", choices=["sequential", "hierarchical"],
                        help="Prompt type for LatentMAS: sequential or hierarchical")

    # Task type (required by LatentMAS)
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "medqa"],
                        help="Task type for answer extraction")

    # Additional args required by LatentMAS
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true", help="Enable latent space realignment")

    args = parser.parse_args()

    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True

    set_seed(args.seed)
    device = auto_device(args.device)
    

    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Method selection
    # Optimized Model & Method Initialization (Ensures single model load)
    if args.method == "baseline":
        model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == 'latent_mas':
        model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas_tool':
        from models_tool import ModelWrapperWithTools
        model = ModelWrapperWithTools(args.model_name, device, use_vllm=args.use_vllm, args=args, enable_tools=True)
        method = LatentMASToolMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
            enable_tools=True
        )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset_iter = list(load_test_dataset(args.dataset))
    print(f"Total {len(dataset_iter)} samples")

    if args.max_samples == -1:
        args.max_samples = len(dataset_iter)

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()

    total_time = time.time() - start_time

    acc, correct = evaluate(preds)

    # Save results
    import os
    from datetime import datetime

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/{args.method}_{args.model_name.replace('/', '_')}_toolcalling_{timestamp}.json"

    result_data = {
        "summary": {
            "method": args.method,
            "model": args.model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "latent_steps": args.latent_steps,
            "accuracy": acc,
            "correct": correct,
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / args.max_samples, 4) if args.max_samples > 0 else 0,
        },
        "predictions": preds
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(json.dumps(result_data["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
