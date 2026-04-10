# LatentMAS 代码库架构分析

## 核心文件解析

### 1. `models.py` - 模型封装层

**ModelWrapper 类** (`models.py:69-567`)

```
┌─────────────────────────────────────────────────────────┐
│                    ModelWrapper                         │
├─────────────────────────────────────────────────────────┤
│  双后端支持:                                            │
│  - HuggingFace: 直接操作 KV Cache 对象                  │
│  - vLLM: 通过嵌入序列传递潜在状态                        │
├─────────────────────────────────────────────────────────┤
│  核心方法:                                              │
│  - generate_latent_batch()     # 潜在思考生成           │
│  - generate_text_batch()       # 文本生成               │
│  - _apply_latent_realignment() # 输入-输出对齐          │
└─────────────────────────────────────────────────────────┘
```

**关键数据流** (`models.py:416-500`):

```
input_ids [B, seq_len]
    ↓
outputs = model(input_ids, ...)  # 首次前向传播
    ↓
last_hidden [B, D]  # 最后一层隐藏状态
    ↓
┌── latent_vec = _apply_latent_realignment(last_hidden) ──┐
│                                                         │
│   latent_embed = latent_vec.unsqueeze(1)  [B, 1, D]     │
│       ↓                                                 │
│   outputs = model(inputs_embeds=latent_embed, ...)      │
│       ↓                                                 │
│   last_hidden = outputs.hidden_states[-1][:, -1, :]     │
│       ↓                                                 │
└── 循环 latent_steps 次 ─────────────────────────────────┘
    ↓
return past_key_values  # 累积的 KV Cache
```

**Latent Realignment 公式** (`models.py:297-352`):

```python
# 对齐矩阵计算
gram = W_out.T @ W_out + λI  # Gram 矩阵 + 正则化
rhs = W_out.T @ W_in         # 右侧项
W_align = solve(gram, rhs)   # 对齐矩阵

# 应用对齐
aligned = hidden @ W_align
aligned = aligned * (target_norm / aligned_norm)  # 归一化
```

---

### 2. `methods/latent_mas.py` - LatentMAS 核心

**LatentMASMethod 类** (`latent_mas.py:16-443`)

```
┌─────────────────────────────────────────────────────────┐
│                  LatentMASMethod                        │
├─────────────────────────────────────────────────────────┤
│  参数:                                                  │
│  - latent_steps: 潜在推理步数 (通常 10-80)              │
│  - judger_max_new_tokens: Judger 输出长度               │
│  - temperature, top_p: 采样参数                         │
├─────────────────────────────────────────────────────────┤
│  核心流程:                                              │
│  for agent in [Planner, Critic, Refiner, Judger]:       │
│      if agent != Judger:                                │
│          past_kv = generate_latent_batch(past_kv)       │
│      else:                                              │
│          output = generate_text_batch(past_kv)          │
└─────────────────────────────────────────────────────────┘
```

**两种运行模式**:

| 模式 | 方法 | 后端 | 适用场景 |
|------|------|------|----------|
| HF | `run_batch()` | HuggingFace | 复现论文结果 |
| vLLM | `run_batch_vllm()` | HF + vLLM 混合 | 高性能推理 |

**vLLM 模式的关键创新** (`latent_mas.py:252-439`):

```python
# 1. HF 模型执行 latent 推理
past_kv, hidden_emb = generate_latent_batch_hidden_state(...)

# 2. 将 latent embeddings 插入到 prompt 中
whole_prompt_emb = torch.cat([left_emb, past_embedding, right_emb], dim=0)

# 3. vLLM 从 embeddings 生成文本
outputs = vllm_engine.generate(prompt_embeds_list, sampling_params)
```

---

### 3. `methods/text_mas.py` - 文本 MAS 基线

**TextMASMethod 类** (`text_mas.py:11-181`)

```
对比 LatentMAS:
┌─────────────────────────────────────────────────────────┐
│  TextMAS:                                               │
│  Agent 1 → 生成文本 → 传递文本给 Agent 2 → ...          │
│  (累积 context 字符串，每个 Agent 重新编码)              │
├─────────────────────────────────────────────────────────┤
│  LatentMAS:                                             │
│  Agent 1 → latent 推理 → KV Cache → Agent 2 → ...       │
│  (直接传递 KV Cache，无需重新编码)                       │
└─────────────────────────────────────────────────────────┘
```

---

### 4. Agent 定义 (`methods/__init__.py`)

```python
@dataclass
class Agent:
    name: str   # Agent 名称
    role: str   # 角色: planner, critic, refiner, judger

def default_agents() -> List[Agent]:
    return [
        Agent(name="Planner", role="planner"),   # 规划
        Agent(name="Critic", role="critic"),     # 批评
        Agent(name="Refiner", role="refiner"),   # 精炼
        Agent(name="Judger", role="judger"),     # 最终输出
    ]
```

---

## 关键变量与 Shape

| 变量 | Shape | 说明 |
|------|-------|------|
| `input_ids` | `[B, seq_len]` | 输入 token IDs |
| `attention_mask` | `[B, seq_len]` | 注意力掩码 |
| `hidden_states` | `[num_layers, B, seq_len, D]` | 所有层隐藏状态 |
| `last_hidden` | `[B, D]` | 最后一层最后一个位置的隐藏状态 |
| `latent_vec` | `[B, D]` | 对齐后的潜在向量 |
| `latent_embed` | `[B, 1, D]` | 增维后的潜在嵌入 |
| `past_key_values` | `Tuple[Tuple[Tensor, Tensor], ...]` | KV Cache |
| `W_align` | `[D, D]` | 对齐矩阵 |

---

## 扩展点：如何添加 Tool Calling 支持

### 方案 A: 修改 `generate_latent_batch()`

```python
def generate_latent_batch_with_tools(self, input_ids, ...):
    for step in range(latent_steps):
        latent_vec = self._apply_latent_realignment(last_hidden)

        # 新增: 检测是否需要工具
        tool_prob = self.tool_head(latent_vec)
        if tool_prob > threshold:
            # 解码工具调用
            tool_call = self.decode_tool_call(latent_vec)
            # 执行工具
            observation = execute_tool(tool_call)
            # 编码结果回潜空间
            latent_vec = self.obs_encoder(observation)

        latent_embed = latent_vec.unsqueeze(1)
        outputs = self.model(inputs_embeds=latent_embed, ...)
```

### 方案 B: 修改 Agent 循环 (`latent_mas.py`)

```python
for agent in self.agents:
    if agent.role != "judger":
        past_kv = self.model.generate_latent_batch(...)

        # 新增: 检查是否需要工具
        if self._should_call_tool(past_kv):
            tool_result = self._execute_tool_injection(past_kv)
            past_kv = self._inject_observation(past_kv, tool_result)
```

### 方案 C: 添加 Gatekeeper Agent

```python
# 在 methods/__init__.py 中添加
Agent(name="Gatekeeper", role="gatekeeper"),

# 在 latent_mas.py 中处理
if agent.role == "gatekeeper":
    past_kv = self._handle_tool_calls(past_kv)
```

---

## 实验配置参考

### 基础配置
```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --task gsm8k \
    --latent_steps 40 \
    --latent_space_realign \
    --prompt sequential
```

### vLLM 高性能配置
```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --use_vllm \
    --use_second_HF_model \
    --device cuda:0 \
    --device2 cuda:1 \
    --enable_prefix_caching
```

---

*Last Updated: 2026-03-12*
