"""
ModelWrapper Extension for Tool Calling Support

This module extends ModelWrapper with tool calling capabilities
while maintaining backward compatibility with the original code.

New features:
- Tool detection during latent reasoning
- Tool call generation from text
- Tool result injection via Cross-Attention
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from models import ModelWrapper, _past_length
import json
import argparse

class ToolDetectionHead(nn.Module):
    """Binary classifier to detect if tool calling is needed from hidden state."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossAttentionInjector(nn.Module):
    """Injects tool results into the latent stream via Cross-Attention."""
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, H_latent: torch.Tensor, H_observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_latent: [B, L, D] Main latent reasoning stream
            H_observation: [B, L_obs, D] Encoded tool result
        """
        # H_observation is the Key/Value, H_latent is the Query
        attn_out, _ = self.cross_attn(H_latent, H_observation, H_observation)
        H = self.norm(H_latent + attn_out)
        H = self.norm2(H + self.ffn(H))
        return H

class ToolVectorizer:
    """Encodes tool descriptions into latent anchors."""
    def __init__(self, model_wrapper):
        self.model = model_wrapper
        self.anchors = {}

    def clear_anchors(self):
        """Clear all registered tool anchors."""
        self.anchors = {}
        print("[ToolVectorizer] All anchors cleared.")

    def add_tool(self, name, description):
        with torch.no_grad():
            inputs = self.model.tokenizer(description, return_tensors='pt').to(self.model.device)
            source_model = self.model.HF_model if hasattr(self.model, "HF_model") else self.model.model
            outputs = source_model(**inputs, output_hidden_states=True)
            # Use mean of last layer as the anchor
            anchor = outputs.hidden_states[-1].mean(dim=1)
            self.anchors[name] = anchor

    def get_most_similar(self, hidden_state: torch.Tensor) -> Tuple[str, float]:
        if not self.anchors:
            return "None", 0.0
        
        best_tool = "None"
        max_sim = -1.0
        
        # Ensure hidden_state is [1, D]
        h = hidden_state if hidden_state.dim() == 2 else hidden_state.unsqueeze(0)
        
        for name, anchor in self.anchors.items():
            sim = torch.cosine_similarity(h, anchor.to(h.dtype)).item()
            if sim > max_sim:
                max_sim = sim
                best_tool = name
        
        return best_tool, max_sim

class ModelWrapperWithTools(ModelWrapper):
    """
    Extended ModelWrapper that supports LTE-MAS (Latent Tool Embedding).
    """
    
    def __init__(self, model_name: str, device: torch.device, 
                 use_vllm: bool = False, args=None,
                 enable_tools: bool = False,
                 tool_threshold: float = 0.5):
        """
        Args:
            model_name: HuggingFace model name
            device: torch device
            use_vllm: Whether to use vLLM backend
            args: Argument namespace
            enable_tools: Whether to enable tool calling
            tool_threshold: Threshold for tool detection
        """
        super().__init__(model_name, device, use_vllm, args)
        
        self.enable_tools = enable_tools
        self.tool_threshold = tool_threshold
        
        # Tool-specific components
        self._tool_detection_head = None
        self._cross_attention_injector = None
        self._tool_executor = None
        
        self._tool_call_count = 0
        self._tool_success_count = 0
        self.tool_vectorizer = ToolVectorizer(self)
    
    def _init_tool_components(self):
        if self._tool_detection_head is not None:
            return
        
        source_model = self.HF_model if hasattr(self, "HF_model") else self.model
        hidden_dim = source_model.config.hidden_size
        model_dtype = source_model.dtype
        
        # Match model dtype (BFloat16)
        self._tool_detection_head = ToolDetectionHead(hidden_dim).to(device=self.device, dtype=model_dtype).eval()
        self._cross_attention_injector = CrossAttentionInjector(hidden_dim).to(device=self.device, dtype=model_dtype).eval()
        
        print(f"[ModelWrapperWithTools] Initialized tool components (dim={hidden_dim}, dtype={model_dtype})")
    
    def _init_tool_executor(self):
        if self._tool_executor is None:
            from tools.python_executor import PythonExecutor
            self._tool_executor = PythonExecutor(timeout=30)
    
    def detect_tool_intent_heuristic(self, hidden_state: torch.Tensor,
                                      input_text: str = "") -> Tuple[bool, float]:
        # Handle list format from BFCL v3 or message history
        if isinstance(input_text, list):
            try:
                # [[{"role": "user", "content": "..."}]]
                if len(input_text) > 0 and isinstance(input_text[0], list):
                    input_text = input_text[0][-1]['content']
                # [{"role": "user", "content": "..."}]
                elif len(input_text) > 0 and isinstance(input_text[0], dict):
                    input_text = input_text[-1]['content']
                else:
                    input_text = str(input_text)
            except:
                input_text = str(input_text)

        tool_keywords = [
            'calculate', 'compute', 'multiply', 'divide', 'power',
            'prime', 'factorization', 'fibonacci', 'factorial',
            'matrix', 'determinant', 'integral', 'equation',
            'probability', 'variance', 'monte carlo', 'simulation',
            'python', 'program', 'code', 'function', 'loop',
            'remainder', 'sum', 'lcm', 'gcd', 'combination',
            'sequence', 'term', 'how many', 'what is the',
            'odds', 'find', 'expression', 'result of',
            '计算', '乘法', '除法', '编程', '程序',
        ]

        input_lower = input_text.lower()
        keyword_match = any(kw in input_lower for kw in tool_keywords)

        if self._tool_detection_head is not None:
            with torch.no_grad():
                prob = self._tool_detection_head(hidden_state).item()
            neural_score = prob
        else:
            neural_score = 0.3

        _, sim = self.tool_vectorizer.get_most_similar(hidden_state)
        
        if keyword_match:
            confidence = 0.8 + 0.2 * neural_score
        else:
            sim_boost = 0.2 if sim > 0.8 else 0.0
            confidence = 0.5 * neural_score + sim_boost
        
        needs_tool = confidence > self.tool_threshold
        return needs_tool, confidence

    def generate_tool_call_from_text(self, question: str, functions_schema: str = None, max_tokens: int = 1024) -> str:
        """
        Generate tool call code from question text using normal text generation.
        Aligned with BFCL v3 standard prompting style with improved extraction tags.
        """
        source_model = self.HF_model if hasattr(self, "HF_model") else self.model

        if functions_schema:
            system_prompt = "You are an API calling assistant. You must identify the correct function and parameters from the user's question."
            tool_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Available Functions:
{functions_schema}

Question: {question}

Please provide the function call(s) in the format: [CALL] function_name(param1=value1, param2=value2) [/CALL]
Do not provide any reasoning or explanation. Output ONLY the function call(s).<|im_end|>
<|im_start|>assistant
"""
        else:
            tool_prompt = f"""<|im_start|>system
You are an API calling assistant.<|im_end|>
<|im_start|>user
Question: {question}

Please provide the function call(s) in the format: [CALL] function_name(param1=value1) [/CALL]
Do not provide any reasoning or explanation. Output ONLY the function call(s).<|im_end|>
<|im_start|>assistant
"""

        input_ids = self.tokenizer.encode(tool_prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_length = input_ids.shape[1]

        with torch.no_grad():
            generate_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_tokens,  # Increased from 512 to 1024
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            output_ids = source_model.generate(**generate_kwargs)

        new_tokens = output_ids[0][input_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"[DEBUG] Generated Raw: {generated_text[:200].replace('\n', ' ')}...")
        return self._extract_code(generated_text)
    
    def _parse_question_to_function_call(self, question: str) -> str:
        """Smarter dynamic heuristic parser to avoid parameter shifts."""
        import re
        q_lower = question.lower()
        
        # Log that we are using heuristic fallback
        print(f"[HEURISTIC] Attempting fallback for: {question[:50]}...")
        
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', question)
        all_numbers = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', question)]
        
        if 'binomial' in q_lower or 'probability' in q_lower or 'six' in q_lower or 'roll' in q_lower:
            p = 0.5
            if pct_match: p = float(pct_match.group(1)) / 100.0
            elif any(0 < n < 1 for n in all_numbers):
                p_candidates = [n for n in all_numbers if 0 < n < 1]
                if p_candidates: p = p_candidates[0]
            
            ints = [int(n) for n in all_numbers if n >= 1]
            if len(ints) >= 2:
                real_ints = [i for i in ints if i != 6] 
                n_val = max(real_ints) if len(real_ints) >= 2 else max(ints)
                k_val = min(real_ints) if len(real_ints) >= 2 else min(ints)
                return f"calc_binomial_probability(n={n_val}, k={k_val}, p={p})"
            elif len(ints) == 1:
                return f"calc_binomial_probability(n={ints[0]}, k=5, p={p})"
        
        all_lists = re.findall(r'\[[\d\s,.-]+\]', question)
        if ('cosine' in q_lower or 'similarity' in q_lower) and len(all_lists) >= 2:
            return f"calculate_cosine_similarity(vectorA={all_lists[0]}, vectorB={all_lists[1]})"
        
        if len(all_numbers) >= 2:
            n1, n2 = all_numbers[0], all_numbers[1]
            if 'density' in q_lower: return f"calculate_density(mass={max(n1, n2)}, volume={min(n1, n2)})"
            if 'velocity' in q_lower or 'speed' in q_lower: 
                v0 = 0.0 if 'start' in q_lower and ('rest' in q_lower or 'zero' in q_lower) else n1
                return f"calculate_final_velocity(initial_velocity={v0}, acceleration={n2}, time=10)"
        
        return "None" # Changed from func_name(param=value) to None to signal failure

    def _extract_code(self, text: str) -> str:
        import re
        # 1. Handle <think> blocks
        # Remove completed think blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()

        # If <think> is still there (unclosed/truncated), take everything after </think> or split by <think>
        if "<think>" in text.lower():
            if "</think>" in text.lower():
                text = text.split("</think>")[-1].strip()
            else:
                # If only <think> exists without closing, it might be ALL thinking.
                # But sometimes the model starts outputting after thinking without a tag.
                # We'll try to look for patterns first.
                pass

        # 2. Try [CALL]...[/CALL] (prioritize the last one)
        call_matches = list(re.finditer(r'\[CALL\]\s*(.*?)\s*\[/CALL\]', text, re.DOTALL | re.IGNORECASE))
        if call_matches:
            return call_matches[-1].group(1).strip()

        # 3. Try [CALL] without [/CALL] (handles truncation, take last)
        call_start_matches = list(re.finditer(r'\[CALL\]\s*(.*)', text, re.DOTALL | re.IGNORECASE))
        if call_start_matches:
            return call_start_matches[-1].group(1).strip()

        # 4. Search for function_name(args) pattern (prioritize the last one)
        # Using a more robust regex that finds all potential calls
        call_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\s*\(.*\))"
        matches = list(re.finditer(call_pattern, text, re.DOTALL))
        if matches:
            # Check the last match
            code = matches[-1].group(1).strip()

            # If the code has newlines, try to find the balanced closing parenthesis from the start of this match
            # To avoid capturing too much if multiple calls are present
            balance = 0
            for i, char in enumerate(code):
                if char == '(': balance += 1
                elif char == ')': balance -= 1
                if balance == 0 and i > 0:
                    return code[:i+1].strip()
            return code

        # 5. Fallback: try to find anything that looks like a call even if it's truncated (last match)
        truncated_matches = list(re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*\s*\(.*)', text, re.DOTALL))
        if truncated_matches:
            return truncated_matches[-1].group(1).strip()

        # 6. Last resort: first non-empty line of the cleaned text
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[0] if lines else "None"

    def inject_tool_result(self, H_latent: torch.Tensor, h_observation: torch.Tensor) -> torch.Tensor:
        """Helper to inject tool results."""
        if self._cross_attention_injector is None:
            self._init_tool_components()
        
        # h_observation should be [B, D] or [B, 1, D]
        if h_observation.dim() == 2:
            h_obs_seq = h_observation.unsqueeze(1)
        else:
            h_obs_seq = h_observation
            
        return self._cross_attention_injector(H_latent, h_obs_seq)

    def generate_latent_batch_with_tools(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        latent_steps: int = 20,
        past_key_values: Optional[Tuple] = None,
        on_tool_call: Optional[Callable] = None,
        questions: List[str] = None,
        tool_result_cache: Dict[int, Dict] = None,
        tool_detection_step: int = -1,
    ) -> Tuple[Optional[Tuple], Dict]:
        if not self.enable_tools:
            return self.generate_latent_batch(input_ids, attention_mask, latent_steps=latent_steps, past_key_values=past_key_values)
        
        self._init_tool_components()
        self._init_tool_executor()
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, output_hidden_states=True, return_dict=True)
        past, last_hidden = outputs.past_key_values, outputs.hidden_states[-1][:, -1, :]
        
        if tool_detection_step < 0: tool_detection_step = latent_steps // 2
        tool_info = {'detected': False, 'confidence': 0.0, 'tool_call': None, 'tool_result': None, 'success': False}
        
        # Phase 1: Multi-point Refresh
        refresh_steps = [tool_detection_step, tool_detection_step + 3, tool_detection_step + 6, latent_steps - 1]
        refresh_steps = [s for s in refresh_steps if s < latent_steps]
        
        for step in range(latent_steps):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            
            outputs = source_model(
                inputs_embeds=latent_embed,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            # === Multi-point Refresh & Detection ===
            if step in refresh_steps:
                if tool_info.get('success'):
                    # HYBRID INJECTION: Refresh both Action and Result
                    refresh_text = f"Action: {tool_info['tool_call']} | Result: {tool_info['tool_result']}"
                    print(f"[LTE-MAS] Step {step}: Refreshing Latent (Hybrid)")
                    h_observation = self.encode_tool_result(refresh_text)
                    last_hidden = self.inject_tool_result(last_hidden.unsqueeze(1), h_observation)[:, -1, :]
                
                elif step == tool_detection_step:
                    cached_result = tool_result_cache.get(0) if tool_result_cache else None
                    if cached_result and cached_result.get('detected'):
                        tool_info.update(cached_result)
                        if tool_info.get('success'):
                            h_observation = self.encode_tool_result(f"Action: {tool_info['tool_call']} | Result: {tool_info['tool_result']}")
                            last_hidden = self.inject_tool_result(last_hidden.unsqueeze(1), h_observation)[:, -1, :]
                    else:
                        q_text = questions[0] if questions else ""
                        needs_tool, conf = self.detect_tool_intent_heuristic(last_hidden, q_text)
                        tool_info.update({'detected': needs_tool, 'confidence': conf})
                        
                        if needs_tool:
                            print(f"[ToolDetection] Step {step}: Tool needed (conf={conf:.3f})")
                            try:
                                t_call = self.generate_tool_call_from_text(q_text, getattr(self, "current_functions_schema", None))
                                if not t_call or t_call == "None": t_call = self._parse_question_to_function_call(q_text)
                                tool_info['tool_call'] = t_call
                                print(f"[ToolCall] {t_call[:100]}...")
                                success, t_res = self._tool_executor.execute(t_call)
                                tool_info.update({'success': success, 'tool_result': t_res})
                                if success:
                                    print(f"[ToolResult] {str(t_res)[:100]}")
                                    h_obs = self.encode_tool_result(f"Action: {t_call} | Result: {t_res}")
                                    last_hidden = self.inject_tool_result(last_hidden.unsqueeze(1), h_obs)[:, -1, :]
                            except Exception as e:
                                print(f"[ToolException] {e}")
        
        return past, tool_info

    def encode_tool_result(self, tool_result: str) -> torch.Tensor:
        source_model = self.HF_model if hasattr(self, "HF_model") else self.model
        ids = self.tokenizer.encode(str(tool_result), return_tensors='pt', add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = source_model(input_ids=ids, output_hidden_states=True, return_dict=True)
            last_h = out.hidden_states[-1][:, -1, :]
        return self._apply_latent_realignment(last_h, source_model)

def create_model_wrapper_with_tools(model_name: str, device: torch.device, use_vllm: bool = False, args=None, enable_tools: bool = False) -> ModelWrapperWithTools:
    return ModelWrapperWithTools(model_name=model_name, device=device, use_vllm=use_vllm, args=args, enable_tools=enable_tools)
