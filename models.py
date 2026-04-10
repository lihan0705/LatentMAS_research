"""
LatentMAS ModelWrapper - 潜在空间多智能体协作的核心实现

=== 核心理念 ===
传统方法：Agent 间通过文本通信 (token space)
LatentMAS：Agent 间通过 KV Cache 通信 (latent space)

=== 效率革命 ===
- 传统 CoT: 2000+ tokens → 复杂推理完成
- LatentMAS: 40-80 latent steps → 同等推理完成  
- 效率提升: O(d_h / log|V|) 倍 (Qwen3-4B: 235.7×)

=== 关键数据流 ===
1. 初始: input_ids [B, seq_len] → KV Cache + 隐藏状态
2. 潜在推理: 40-80 次自回归隐藏状态更新 (无 token 生成!)
3. Agent 通信: 完整 KV Cache 传递 (无损记忆转移)
4. 最终输出: 仅最后 Agent 解码生成答案

=== 两种通信模式 ===
- HF Backend: 直接传递 past_key_values 对象
- vLLM Backend: 传递嵌入序列作为 prompt_embeds

理论保证: Theorem 3.3 - KV Cache 传递 ≡ 显式输入传递 (信息无损)
"""

import os
import csv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

# Lazy import vllm to avoid initialization errors with broken GPUs
_HAS_VLLM = None
_LLM = None
_SamplingParams = None

def _ensure_vllm():
    """Lazy import vllm, returns (has_vllm, LLM, SamplingParams)"""
    global _HAS_VLLM, _LLM, _SamplingParams
    if _HAS_VLLM is None:
        try:
            from vllm import LLM, SamplingParams
            _HAS_VLLM = True
            _LLM = LLM
            _SamplingParams = SamplingParams
        except Exception:
            _HAS_VLLM = False
            _LLM = None
            _SamplingParams = None
    return _HAS_VLLM, _LLM, _SamplingParams


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


class ModelWrapper:
    """
    LatentMAS 的核心模型封装类
    
    支持双后端：
        - HuggingFace: 标准 Transformer，直接操作 KV Cache 对象
        - vLLM: 高性能推理，通过嵌入序列传递潜在状态
    
    核心功能：
        1. 潜在思考生成 (generate_latent_batch)
           - 在隐藏空间自回归推理，无需解码 token
           - 实现高效的多步思考过程
           
        2. 输入-输出对齐 (_apply_latent_realignment)
           - 解决隐藏状态与输入嵌入的分布不一致
           - 通过对齐矩阵 W_a = (W_out^T@W_out+λI)^(-1)@W_out^T@W_in
           
        3. 无损记忆传递
           - KV Cache 完整保存历史推理过程
           - 支持跨 Agent 的连续协作
           
    性能提升：
        - Token 使用: 减少 70-80%
        - 推理速度: 4-6× 加速  
        - 准确率: 提升 13.3% (平均)
    """
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        has_vllm, _, _ = _ensure_vllm()
        self.use_vllm = use_vllm and has_vllm
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            has_vllm, LLM, SamplingParams = _ensure_vllm()
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left')
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side='left')
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        """
        准备聊天输入的标准化流程，将消息列表转换为模型可处理的格式
        
        【输入】:
            messages: [{"role": "user", "content": "..."}, ...]
            add_generation_prompt: 是否添加生成提示符
            
        【输出】:
            prompt_text: str - 完整的提示文本
            input_ids: torch.Tensor [1, seq_len] - Token IDs
            attention_mask: torch.Tensor [1, seq_len] - 注意力掩码  
            tokens: List[str] - 去除 padding 的 token 列表
        
        【Shape 变化详解】:
            1. messages → prompt_text (字符串拼接)
            2. prompt_text → encoded (tokenizer 处理)
            3. input_ids [1, seq_len] + attention_mask [1, seq_len] 
            4. 过滤 padding → active_ids [actual_len]
            5. active_ids → tokens [token1, token2, ...]
        """
        # 步骤 1: 将聊天消息渲染成完整文本
        # 输入: [{"role": "user", "content": "你好"}]
        # 输出: "<|user|>\n你好\n</|user|><|assistant|>\n"
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        
        # 步骤 2: 文本编码为 token IDs
        # prompt_text: str → input_ids: [1, seq_len], attention_mask: [1, seq_len]
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        
        # 步骤 3: 转换到目标设备
        # Shape 保持不变: input_ids [1, seq_len], attention_mask [1, seq_len]
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # 步骤 4: 提取非 padding 的有效 tokens (调试和显示用)
        # attention_mask[0].bool(): [seq_len] → [True, True, False, ...]
        # input_ids[0][mask]: [seq_len] → [valid_id1, valid_id2, ...]  
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        
        # 步骤 5: 将 IDs 转换为可读的 tokens (用于显示和分析)
        # [id1, id2, ...] → ["<s>", "你", "好", ...]
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        """
        批量处理聊天输入，支持多个对话的并行处理
        
        【输入】:
            batch_messages: [
                [{"role": "user", "content": "你好"}],  # 对话 1
                [{"role": "user", "content": "世界"}],  # 对话 2
                ...
            ]
            
        【输出】:
            prompts: List[str] - 每个对话的文本
            input_ids: torch.Tensor [batch_size, max_seq_len] - 批量 Token IDs
            attention_mask: torch.Tensor [batch_size, max_seq_len] - 批量注意力掩码
            tokens_batch: List[List[str]] - 每个对话的 token 列表
        
        【关键区别 vs 单例】:
            - padding=True 自动补齐到最大长度
            - 所有 tensor 形状包含 batch 维度
        """
        # 步骤 1: 批量渲染聊天消息为文本
        prompts: List[str] = []
        for messages in batch_messages:
            # 每个 messages 列表转换为一个 prompt 文本
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        
        # 步骤 2: 批量编码，自动 padding 到最长序列
        # 输入: ["你好", "更长的句子..."] 
        # 输出: 
        #   input_ids: [batch_size=2, max_seq_len] 
        #   attention_mask: [batch_size=2, max_seq_len]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,  # 关键：自动 padding
            add_special_tokens=False,
        )
        
        # 步骤 3: 转换到目标设备
        input_ids = encoded["input_ids"].to(self.device)      # [B, max_len]
        attention_mask = encoded["attention_mask"].to(self.device)  # [B, max_len]
        
        # 步骤 4: 为每个对话提取非 padding tokens
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            # ids_row: [max_len], mask_row: [max_len]
            # 过滤掉 padding tokens (通常 mask_row=0 表示 padding)
            active_ids = ids_row[mask_row.bool()].tolist()
            # 转换为可读 tokens
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        generate_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'return_dict_in_generate': True,
            'output_scores': False,
            'past_key_values': past_key_values,
        }
        if top_p > 0 and top_p < 1.0:
            generate_kwargs['top_p'] = top_p
        outputs = self.model.generate(**generate_kwargs)
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        # === 步骤 2：潜在空间自回归循环 ===
        # 
        # 【关键洞察】这里的效率革命：
        # - 传统 CoT：需要生成 2000+ tokens 才能完成复杂推理
        # - LatentMAS：仅用 40-80 个 latent steps 即可实现同等推理深度
        # - 效率提升：每个 latent step ≈ 235-471 个 token 的表达能力
        #
        # 【Shape 变化链】：
        #   last_hidden [B,D] → 对齐 → latent_vec [B,D] → 增维 → latent_embed [B,1,D]
        #   → 模型推理 → 新的 last_hidden [B,D] + 累积的 KV Cache
        #
        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

