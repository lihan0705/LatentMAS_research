"""
LatentMAS with Tool Calling Support

This module extends LatentMASMethod with tool calling capabilities.
The implementation is backward compatible - if enable_tools=False,
it behaves exactly like the original LatentMASMethod.
"""

from typing import Dict, List, Optional, Tuple
import torch
import argparse
import json

from models import _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer
from methods import default_agents


class LatentMASToolMethod:
    """
    LatentMAS with tool calling support.
    
    Key Features:
    1. Tool detection during latent reasoning
    2. Seamless tool execution and result injection
    3. Backward compatible with original LatentMAS
    
    Usage:
        # With tool calling (new)
        method = LatentMASToolMethod(model, enable_tools=True, args=args)
        
        # Without tool calling (original behavior)
        method = LatentMASToolMethod(model, enable_tools=False, args=args)
    """
    
    def __init__(
        self,
        model,  # ModelWrapper or ModelWrapperWithTools
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
        enable_tools: bool = False,
        tool_threshold: float = 0.7,
        tool_detection_step: int = -1,  # -1 means middle step
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        
        # Tool calling settings
        self.enable_tools = enable_tools
        self.tool_threshold = tool_threshold
        self.tool_detection_step = tool_detection_step

        # Streamline agents for tool calling to reduce redundancy
        from methods import Agent
        if self.enable_tools:
            self.agents = [
                Agent(name="Planner", role="planner"),
                Agent(name="Judger", role="judger")
            ]
            print(f"[LatentMASToolMethod] Streamlined MAS for tool calling: Planner -> Judger")
        else:
            self.agents = default_agents()
        
        self.method_name = 'latent_mas_tool'
        
        # Additional settings from original
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False
        self.task = getattr(args, "task", "gsm8k") if args else "gsm8k"
        
        if self.latent_only:
            self.sequential_info_only = True
        
        # Tool statistics
        self._tool_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0
        }
    
    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()
    
    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        try:
            from transformers.cache_utils import Cache
            if isinstance(past_kv, Cache):
                legacy = past_kv.to_legacy_cache()
                trimmed_legacy = tuple(
                    tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                    for layer in legacy
                )
                return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        except ImportError:
            pass
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)
    
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        """
        Run batch processing with optional tool calling.
        
        This method is backward compatible:
        - If enable_tools=False, behaves exactly like original LatentMAS
        - If enable_tools=True, adds tool detection and calling
        """
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        
        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        
        # Track tool calls per sample
        tool_calls_per_sample: List[List[Dict]] = [[] for _ in range(batch_size)]
        # Track successful tool results for each sample (to inject into judger prompt)
        tool_results_for_judger: List[Optional[str]] = [None for _ in range(batch_size)]
        
                # Track if tool was successful for each sample
        tool_success_for_sample: List[bool] = [False for _ in range(batch_size)]
        
        # NEW: Global tool result cache for the current batch
        tool_result_cache: Dict[int, Dict] = {}
        
        for agent in self.agents:
            # Build prompts
            if self.args.prompt == "sequential":
                batch_messages = []
                for item in items:
                    context = ""
                    if self.task == "toolcalling" and "functions" in item:
                        context = json.dumps(item["functions"], indent=2)
                    
                    batch_messages.append(
                        build_agent_message_sequential_latent_mas(
                            role=agent.role, 
                            question=item["question"], 
                            context=context, 
                            method=self.method_name, 
                            args=self.args
                        )
                    )
            elif self.args.prompt == "hierarchical":
                batch_messages = []
                for item in items:
                    context = ""
                    if self.task == "toolcalling" and "functions" in item:
                        context = json.dumps(item["functions"], indent=2)
                        
                    batch_messages.append(
                        build_agent_message_hierarchical_latent_mas(
                            role=agent.role, 
                            question=item["question"], 
                            context=context, 
                            method=self.method_name, 
                            args=self.args
                        )
                    )
            
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )
            
            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)
                
                # Handle think token
                if self.args.think:
                    wrapped_prompts = [f"{prompt}<tool_call>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts
                
                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                
                # === Tool-enabled latent generation ===
                if self.enable_tools and hasattr(self.model, 'generate_latent_batch_with_tools'):
                    # Extract questions for tool detection
                    questions = [item["question"] for item in items]
                    past_kv, tool_info = self.model.generate_latent_batch_with_tools(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                        tool_detection_step=self.tool_detection_step,
                        questions=questions,  # Pass questions for keyword matching
                        tool_result_cache=tool_result_cache,  # NEW: shared cache
                    )
                    
                    # Record tool calls and save successful results for judger
                    if tool_info.get('detected', False):
                        for idx in range(batch_size):
                            tool_calls_per_sample[idx].append({
                                'agent': agent.name,
                                'step': self.tool_detection_step,
                                'tool_call': tool_info.get('tool_call'),
                                'tool_result': tool_info.get('tool_result'),
                                'success': tool_info.get('success', False),
                                'confidence': tool_info.get('confidence', 0.0)
                            })
                            self._tool_stats['total_calls'] += 1
                            if tool_info.get('success'):
                                self._tool_stats['successful_calls'] += 1
                                                                # Save tool result for judger prompt
                                tool_results_for_judger[idx] = tool_info.get('tool_result')
                                
                                # NEW: Update global cache for subsequent agents
                                if idx not in tool_result_cache:
                                    tool_result_cache[idx] = tool_info.copy()
                            else:
                                self._tool_stats['failed_calls'] += 1
                else:
                    # Fallback to standard latent generation
                    past_kv = self.model.generate_latent_batch(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )
                
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": wrapped_tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                        "tool_calls": tool_calls_per_sample[idx] if tool_calls_per_sample[idx] else None,
                    })
            else:
                # Judger: decode final answer
                # NOTE: We do NOT pass past_kv to the judger here because the judger prompt 
                # is not a continuation of the previous agents' prompts, which causes 
                # KV cache dimension mismatches in some transformers versions.
                # The judger prompt is self-contained.
                past_for_decoding = None
                
                # === Check if tool was successful - if so, use tool result directly ===
                for idx in range(batch_size):
                    if tool_results_for_judger[idx] is not None:
                        # short-circuit ONLY if task is toolcalling (e.g. BFCL)
                        # For reasoning tasks (GSM8K, HotpotQA), we want the Judger to see the result
                        tool_info = tool_result_cache.get(idx)
                        if self.task == "toolcalling" and tool_info and tool_info.get('success'):
                            tool_success_for_sample[idx] = True
                            final_texts[idx] = tool_info['tool_call']
                            print(f"[ShortCircuit] Using tool call from cache: {final_texts[idx]}")
                
                # === Inject tool results into judger prompt ===
                judger_prompts_raw = prompts.copy()
                for idx in range(batch_size):
                    # Inject result if it exists (either from latent run or cache)
                    result = tool_results_for_judger[idx]
                    if result is not None:
                        tool_info_text = f"\n\n[Tool Result]\nThe following information was obtained using tools: {result}\n\nPlease use this information in your final answer.\n"
                        judger_prompts_raw[idx] = prompts[idx] + tool_info_text
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<tool_call>" for prompt in judger_prompts_raw]
                else:
                    judger_prompts = judger_prompts_raw
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                
                # Only skip Judger if toolcalling task already short-circuited
                samples_need_judger = [idx for idx in range(batch_size) if not tool_success_for_sample[idx]]
                
                if samples_need_judger:
                    generated_batch, _ = self.model.generate_text_batch(
                        judger_ids,
                        judger_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=past_for_decoding,
                    )
                    
                    for idx in range(batch_size):
                        if not tool_success_for_sample[idx]:
                            final_text = generated_batch[idx].strip()
                            
                            # NEW: Clean tool call for BFCL
                            if self.task == "toolcalling" and hasattr(self.model, "_extract_code"):
                                final_text = self.model._extract_code(final_text)
                                
                            final_texts[idx] = final_text
                            mask = judger_mask[idx].bool()
                            trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                            agent_traces[idx].append({
                                "name": agent.name,
                                "role": agent.role,
                                "input": judger_prompts[idx],
                                "input_ids": trimmed_ids,
                                "input_tokens": judger_tokens_batch[idx],
                                "output": final_text,
                            })
                else:
                    # All samples have successful tool results - just record the input
                    for idx in range(batch_size):
                        mask = judger_mask[idx].bool()
                        trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append({
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": f"[Tool result used directly: {final_texts[idx]}]",
                        })
        
        # Build results
        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            gold = item.get("gold", "")
            
            if self.task == "toolcalling":
                # BFCL prediction should be the tool call itself
                pred = final_text
                ok = False # Evaluated by run_test_bfcl.py later via compare_tool_calls
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                ok = (pred == gold) if (pred and gold) else False
            
            result = {
                "question": item["question"],
                "gold": gold,
                "solution": item.get("solution", ""),
                "prediction": pred,
                "raw_prediction": final_text,
                "agents": agent_traces[idx],
                "correct": ok,
            }
            
            # Add tool call info if available
            if tool_calls_per_sample[idx]:
                result["tool_calls"] = tool_calls_per_sample[idx]
            
            results.append(result)
        
        return results
    
    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
    
    def get_tool_stats(self) -> Dict:
        """Get tool calling statistics."""
        return self._tool_stats.copy()
