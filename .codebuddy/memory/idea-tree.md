# Idea Tree: LatentMAS Tool Calling

## Seed (Level 0)
**问题**: LatentMAS 实现了高效的潜在空间多智能体协作，但无法支持离散工具调用。需要桥接连续潜在空间与离散工具接口。

---

## Level 1: Technique Variants

### A. Observation-to-Latent 投影器
**描述**: 引入专门的 Encoder，将工具返回的离散结果投影到 Agent 隐空间。工具结果"看起来像"来自外部 Agent 的消息。

**核心技术**: 训练一个轻量级 Encoder (如 MLP 或小型 Transformer)，将离散文本编码为 latent vector。

**数据流**: 
```
tool_result (text) → obs_encoder → h_observation → 融入 KV Cache
```

---

### B. 潜空间工具索引 (Latent Tool Indexing)
**描述**: 将工具向量化，通过向量相似度匹配触发。推理-动作同构，无需显式文本接口。

**核心技术**: Contrastive Learning 训练工具向量库，Soft Selection 实现可微的工具选择。

**数据流**:
```
h_latent → intent_projection → cosine_similarity(tool_bank) → soft_select → execute_tool
```

**理论贡献**: 挑战"文本即接口"范式，建立 Reasoning-Action Isomorphism。

---

### C. 异步 Cross-Attention 注入
**描述**: Latent Backbone 不动，工具结果作为 Side-input 通过 Cross-Attention 注入。不打断 Latent 连续性。

**核心技术**: 单层 Cross-Attention，类比 Perceiver / Flamingo 架构。

**数据流**:
```
H_latent (backbone) ←── CrossAttention ──→ tool_encoder(tool_result)
```

**优势**: Cross-Attention 起到缓冲和过滤作用，学术接受度高。

---

### D. Gatekeeper 模式
**描述**: 设立专门的"对外联络官" Agent 负责翻译。角色分工：纯潜空间 Agent 做思考者，Gatekeeper Agent 做翻译官。

**核心技术**: 双 Agent 架构，Latent ↔ Symbolic 双向翻译。

**数据流**:
```
[Planner, Critic, Refiner] (latent) → Gatekeeper (translate) → Tool Call → Gatekeeper (encode) → [Judger] (latent)
```

---

## Level 2: Domain Adaptations

### A1. Observation-to-Latent + 轻量级 MLP Encoder
- **Domain**: 数学推理 + 计算器工具
- **特点**: Encoder 可以很简单，因为工具输出格式相对固定（数字、列表）
- **风险**: 泛化到其他工具类型可能需要重新训练

### A2. Observation-to-Latent + 预训练 Encoder
- **Domain**: 通用工具调用（搜索、数据库、API）
- **特点**: 使用预训练语言模型作为 Encoder 基础
- **风险**: 计算开销增加，可能破坏 latent space 的紧凑性

### B1. 潜空间工具索引 + 固定工具集
- **Domain**: 数学推理（Python 计算器）
- **特点**: 工具集有限且固定，索引效率高
- **风险**: 扩展新工具需要重新训练

### B2. 潜空间工具索引 + 动态工具注册
- **Domain**: 通用 Agent 系统（ToolBench）
- **特点**: 新工具通过文本描述自动注册到向量库
- **风险**: 工具描述可能不完整，影响匹配精度

### C1. Cross-Attention + 共享编码器
- **Domain**: 数学推理
- **特点**: 工具输出用 Input Embedding 层编码（利用对齐矩阵）
- **风险**: 数学输出格式相对固定，容易处理

### C2. Cross-Attention + 独立编码器
- **Domain**: 通用工具调用
- **特点**: 工具输出用独立的小型 Transformer 编码
- **风险**: 需要额外训练，但更灵活

### D1. Gatekeeper + 轮询模式
- **Domain**: 多 Agent 协作
- **特点**: Gatekeeper 定期检查是否需要工具调用
- **风险**: 增加通信开销

### D2. Gatekeeper + 中断模式
- **Domain**: 事件驱动
- **特点**: Agent 主动向 Gatekeeper 发送请求
- **风险**: 需要修改 Agent 通信协议

---

## Level 3: Formulation Variants (Leaf Nodes)

### A1.1 数学推理 + MLP Encoder + 单步工具调用
- **输入**: 数学问题文本
- **输出**: 最终答案
- **约束**: 每道题最多调用一次工具
- **评估**: 准确率、Token 效率

### A1.2 数学推理 + MLP Encoder + 多步工具调用
- **输入**: 复杂数学问题（如蒙特卡洛模拟）
- **输出**: 最终答案
- **约束**: 可多次调用工具，需要决策"何时调用"
- **评估**: 准确率、工具调用成功率

### B1.1 数学推理 + 固定工具索引 + 单工具
- **输入**: 数学问题
- **输出**: 最终答案
- **约束**: 仅支持 Python 计算器
- **评估**: 准确率、索引匹配准确率

### C1.1 数学推理 + Cross-Attention + 共享编码器 + 单步
- **输入**: 数学问题
- **输出**: 最终答案
- **约束**: 利用现有对齐矩阵，无需额外训练
- **评估**: 准确率、训练开销

### C1.2 数学推理 + Cross-Attention + 共享编码器 + 多步
- **输入**: 复杂数学问题
- **输出**: 最终答案
- **约束**: 支持多次工具调用
- **评估**: 准确率、推理稳定性

### C2.1 通用工具 + Cross-Attention + 独立编码器
- **输入**: 自然语言请求
- **输出**: 执行结果
- **约束**: 支持搜索、数据库等多种工具
- **评估**: 工具调用准确率、泛化能力

### D1.1 多 Agent + Gatekeeper 轮询 + 单工具
- **输入**: 数学问题
- **输出**: 最终答案
- **约束**: Gatekeeper 作为独立 Agent 加入协作链
- **评估**: 准确率、通信开销

---

## Summary

| ID | 方案 | Domain | Formulation | 关键特点 |
|----|------|--------|-------------|---------|
| A1.1 | Obs-to-Latent | 数学推理 | 单步调用 | MLP Encoder，训练开销小 |
| A1.2 | Obs-to-Latent | 数学推理 | 多步调用 | 需要决策模块 |
| B1.1 | 工具索引 | 数学推理 | 单工具 | 向量匹配，无需解码 |
| C1.1 | Cross-Attention | 数学推理 | 单步+共享编码 | 无需训练，利用对齐矩阵 |
| C1.2 | Cross-Attention | 数学推理 | 多步+共享编码 | 推荐候选 |
| C2.1 | Cross-Attention | 通用工具 | 多步+独立编码 | 扩展性强 |
| D1.1 | Gatekeeper | 多Agent | 轮询模式 | 架构清晰，开销大 |

**进入 Tournament 的候选 (N_I=7)**:
- A1.1, A1.2, B1.1, C1.1, C1.2, C2.1, D1.1
