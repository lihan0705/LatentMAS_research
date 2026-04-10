# LatentMAS 项目 Memory

## 项目概述

**LatentMAS** 是一个在潜在空间进行多智能体协作的推理框架，将 Agent 协作从 token 空间转移到模型的 latent space。

### 核心优势
- **Token 减少**: 70-80%
- **推理加速**: 4-7x
- **Training-free**: 无需训练即可部署

### 项目结构
```
LatentMAS/
├── models.py              # ModelWrapper - HF + vLLM + latent realignment
├── methods/
│   ├── latent_mas.py      # 核心：潜在空间多智能体协作
│   ├── text_mas.py        # 基线：文本空间多智能体
│   └── baseline.py        # 基线：单智能体
├── prompts.py             # Prompt 构造器
├── run.py                 # 主实验入口
└── research_notes/        # 研究笔记
```

---

## 核心技术实现

### 1. Latent Space Communication (`models.py:416-500`)

```python
# 关键函数: generate_latent_batch()
# 数据流:
# input_ids [B, seq_len] → KV Cache + hidden states
# → latent_steps 次自回归更新（无 token 生成）
# → 累积 KV Cache 传递给下一个 Agent
```

**核心洞察**:
- 传统 CoT: 2000+ tokens 完成复杂推理
- LatentMAS: 40-80 latent steps 实现同等推理深度
- 效率提升: O(d_h / log|V|) 倍 (Qwen3-4B: 235.7x)

### 2. Input-Output Alignment (`models.py:297-352`)

解决 hidden states 与 input embeddings 的分布不一致问题：
- 对齐矩阵: `W_a = (W_out^T @ W_out + λI)^(-1) @ W_out^T @ W_in`
- 保持 latent space 的语义稳定性

### 3. Agent 协作模式 (`methods/__init__.py`)

```python
default_agents() → [
    Agent(name="Planner", role="planner"),
    Agent(name="Critic", role="critic"),
    Agent(name="Refiner", role="refiner"),
    Agent(name="Judger", role="judger"),  # 唯一解码输出的 Agent
]
```

---

## 核心挑战：Latent 空间无法支持 Tool Use

### 根本问题：张量错位 (Tensor Mismatch)

| 维度 | Latent Space | Tool Calling |
|------|--------------|--------------|
| 数据类型 | 连续高维向量 | 离散符号 (JSON) |
| 梯度流 | 端到端可微 | 不可导中断 |
| 精度要求 | 概率相似性 | Token 级零容忍 |

### 三大技术难点

1. **离散性断裂 (The Discreteness Break)**
   - API 需要精确字符: `{"tool": "calc", "input": "2+2"}`
   - Latent → Discrete 转换不可导
   - 破坏端到端梯度流

2. **观察值注入污染 (Observation Injection Problem)**
   - 工具返回离散结果
   - 编码回 Latent 空间引发语义偏移
   - 难以保持推理连续性

3. **因果链严苛性**
   - 离散推理: If-Then 硬逻辑
   - Latent 推理: 概率相似性软逻辑
   - 工具调用对 Token 精度零容忍

---

## 四个解决方案

### A. 混合表征：Observation-to-Latent 投影器

**核心思想**: 引入专门的 Encoder，将工具返回的离散结果投影到 Agent 隐空间

```python
class LatentAgent_with_Tool:
    def step(self, latent_msg_in):
        h_internal = self.transformer_block(latent_msg_in)
        prob_tool = sigmoid(self.tool_head(h_internal))

        if prob_tool > threshold:
            query = self.decoder(h_internal)
            observation = external_tool.execute(query)
            z_obs = self.obs_encoder(observation)  # 关键：离散→连续
            h_combined = self.fusion_layer(h_internal, z_obs)
            return h_combined
        return h_internal
```

**优势**: 工具结果看起来像"来自外部 Agent 的消息"
**挑战**: 需要训练轻量级 Encoder

---

### B. 潜空间工具索引 (Latent Tool Indexing) ⭐

**核心思想**: 将工具向量化，通过向量相似度匹配触发

```python
class LatentMAS_Indexing:
    def __init__(self):
        # 工具向量库，通过 Contrastive Learning 训练
        self.tool_bank = {"Search": v1, "Calc": v2, "DB": v3}

    def forward(self, h_i):
        query_vec = self.intent_projection(h_i)
        # 近似可导的 soft selection
        weights = softmax([dot(query_vec, v_tool) for v_tool in self.tool_bank.values()])
        selected_tool = self.tool_bank.keys()[argmax(weights)]
        return selected_tool.execute(...)
```

**优势**: 推理-动作同构，无需显式文本接口
**研究价值**: 挑战"文本即接口"范式

---

### C. 异步 Cross-Attention 注入 ⭐ 推荐

**核心思想**: Latent Backbone 不动，工具结果作为 Side-input 通过 Cross-Attention 注入

```python
def latent_reasoning_step(H_latent, tool_results_queue):
    # 1. 标准 LatentMAS 通信 (Backbone)
    H_communicated = multi_agent_attention(H_latent)

    # 2. 异步注入工具结果
    if not tool_results_queue.empty():
        K_tool, V_tool = tool_encoder(tool_results_queue.pop())
        # Cross-Attention 让 Latent Stream 按需读取工具结果
        H_refined = cross_attention(query=H_communicated, key=K_tool, value=V_tool)
        return H_refined
    return H_communicated
```

**架构类比**: Perceiver / Flamingo
**优势**:
- 不打断 Latent 连续性
- Cross-Attention 起到缓冲和过滤作用
- 学术接受度高

---

### D. 离散-连续混合架构 (Gatekeeper 模式)

**核心思想**: 设立专门的"对外联络官" Agent 负责翻译

```python
class GatekeeperAgent(BaseAgent):
    def process(self, group_latent_state):
        # 1. 监听潜空间中的"求助信号"
        instruction_vector = self.listen(group_latent_state)

        # 2. Latent → Symbolic 翻译
        api_call = self.latent_to_symbolic_decoder(instruction_vector)
        raw_result = call_external_api(api_call)

        # 3. Symbolic → Latent 广播
        latent_feedback = self.symbolic_to_latent_encoder(raw_result)
        return latent_feedback
```

**角色分工**:
- 纯潜空间 Agent: 思考者 (Intuition)
- Gatekeeper Agent: 翻译官 (Tool Interface)

---

## 研究路线图

### Phase 1: 现象观察 (当前)
- 数据集: GSM8K-Python, MATH, AIME
- 目标: 观察模型在复杂计算时，潜空间向量是否偏离"推理流形"
- 已创建: `data/test_dataset_toolcalling.json`

### Phase 2: 动作触发器开发
- 在 `latent_steps` 循环中增加 Action Head 或 VQ 层
- 验证：模型能否通过潜空间变化识别"该用工具了"

### Phase 3: 闭环验证
- ToolBench / GAIA 数据集
- 端到端潜空间推理 + 工具调用

---

## 未来研究方向

### 方向 1: 潜空间动作量化 (Latent Action Quantization)
- 引入 VQ-VAE Codebook
- 动作意图映射为潜空间量化向量
- 理论贡献: **Reasoning-Action Isomorphism**

### 方向 2: 递归世界模型 (O(1) Memory)
- 结合 Thinking States (2026.02)
- 固定大小 World State Vector
- 工具结果作为状态更新函数输入

### 方向 3: 异步双流协作
- Latent Stream (Slow): 长程规划
- Token Stream (Fast): 环境交互
- 模仿人类 System 1 / System 2

---

## 论文建议

**标题**: "Bridging the Gap: Enabling Discrete Tool Usage in Continuous Latent Multi-Agent Systems"

**核心 Contribution**: Encoder-Decoder Interaction Layer 解决离散信息注入导致的潜空间塌陷

---

*Last Updated: 2026-03-12*
