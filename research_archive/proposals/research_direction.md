# LatentMAS 项目概述

## 项目目标

解决 Agent 系统的 **"Token 税"** 和 **KV-Cache 膨胀** 问题，实现：
- **O(1) 空间复杂度** 的恒定内存占用
- 具备 **深度推理能力** 的智能体架构
- 多智能体系统在潜在空间的高效协作

**预期效果**: 节省 90% Token 前提下，长程规划成功率不降反升

---

## 核心技术路线

### 1. CoConut (Chain of Continuous Thought)
- **来源**: Google, 2024.12, arxiv:2412.06781
- **核心**: 在连续潜在空间进行推理，避免生成显式 token
- **方法**: 递归精炼潜在思考表示，通过 BPTT 训练
- **局限**: BPTT 计算成本高，训练时间随步数线性增长

### 2. Thinking States
- **来源**: Google, 2026.02, arxiv:2602.08332
- **核心**: O(1) 内存复杂度的监督式潜在推理
- **创新**:
  - Teacher-forcing 避免 BPTT，支持并行训练
  - Chunk-Recurrent Processing 架构
  - 自然语言思考保持可解释性
- **局限**: 仅适用于单查询，不支持多轮 Agent 交互

### 3. LatentMAS
- **来源**: Princeton/UIUC/Stanford, 2025, arxiv:2511.20639
- **核心**: 多智能体在潜在空间协作，training-free 部署
- **创新**:
  - KV Cache 直接传递，无需解码-重新编码
  - Input-Output Alignment 对齐技术
  - 471倍表达效率提升 (d_h / log|V|)
- **效果**: Token 减少 70-80%，推理加速 4-7 倍

---

## 拟研究的创新架构: Thought-Hourglass

**核心思想**: 抛弃"文本历史"，维护持续进化的"潜在状态"

### 架构组件

```
┌─────────────────────────────────────────────────────────┐
│                    Thought-Hourglass                    │
├─────────────────────────────────────────────────────────┤
│  Input → [Encoder] → [Processor] → [Decoder] → Output   │
│            (压缩)      (推理)       (执行/坦白)          │
└─────────────────────────────────────────────────────────┘
```

1. **Encoder (压缩层)**: VQ-VAE 将 Observation 与 State 融合，压入 Latent Bottleneck
2. **Processor (推理层)**: Latent Tokens 在 Transformer 内部递归自注意力计算
3. **Decoder (执行/坦白层)**: 输出 Action + 可读"思考内容"

### 四大技术流派融合

| 流派 | 代表工作 | 在 Thought-Hourglass 中的体现 |
|------|----------|------------------------------|
| Meta派 | Gist Tokens | 压缩至 Bottleneck 的设计 |
| 顿悟派 | CoConut | 隐式潜在推理 |
| 反思派 | Quiet-STaR | 递归内部处理 |
| 坦白派 | VQ-VAE | 可解释的潜在编码 |

---

## 当前项目状态

**已实现**:
- ✅ Baseline 单 Agent 推理
- ✅ TextMAS 文本多 Agent 系统
- ✅ LatentMAS 潜在空间多 Agent 协作
- ✅ vLLM 集成支持
- ✅ 9个基准测试 (GSM8K, MATH, GPQA, HumanEval, MBPP 等)

**待研究/实现**:
- ⏳ Thinking States 的 O(1) 状态机制集成到多 Agent 场景
- ⏳ Thought-Hourglass 的递归状态更新
- ⏳ VQ-VAE 可解释性层
- ⏳ 通用工具使用在潜在空间的表示

---

## 技术限制分析：为什么难以支持 Tool Calling？

### LatentMAS 的推理机制

LatentMAS 的工作流程分为两个阶段：

1. **Latent 推理阶段**：`generate_latent_batch()`
   - 执行 `latent_steps` 次循环
   - 每次循环：`last_hidden → latent_vec → 新的 hidden state`
   - **累积 KV Cache，不输出任何 token**

2. **Judger 解码阶段**：`generate_text_batch()`
   - 将 latent state 解码为最终文本

### Tool Calling 的核心矛盾

| 传统 Agent | LatentMAS |
|-----------|-----------|
| Thinking → 输出"调用工具" → 执行 → 继续思考 → 输出... | 连续 latent 推理 → 最后输出结果 |

**问题**：
- Latent 推理是**连续、自回归**的过程，模型一直在"思考"直到完成
- Tool calling 需要**中断-执行-恢复**的循环
- 在 latent 推理过程中，无法确定"何时需要调用工具"

### 关键技术难点

| 难点 | 说明 |
|------|------|
| **意图识别** | Latent space 中如何表示"这里需要调用工具"？ |
| **状态注入** | 如何将工具执行结果编码回 latent space？ |
| **KV Cache 中断** | Tool execution 会打断 KV cache 的连续累积 |
| **恢复推理** | 工具返回后，如何无缝衔接恢复 latent 推理？ |

### 类比理解

- **传统 CoT**：思考 → 说出来 → 听到反馈 → 继续思考 → 再说出来...
- **LatentMAS**：在脑子里连续想很久 → 最后才说出来

Tool calling 就像是在"脑子里连续思考"的过程中突然停下来去查资料，再无缝衔接回去继续刚才的思路——这在技术上非常困难。

### 可能的解决方向

1. **显式中断点**：在 latent 推理中设计强制中断机制
2. **Latent Tool Token**：引入特殊的 latent token 来表示 tool call
3. **两阶段混合**：平时用 latent reasoning，遇到 tool call 时切回传统模式
4. **Tool-aware Latent**：重新设计 latent 空间，区分"思考"和"行动"

但这些方案都会增加复杂度，可能违背 LatentMAS **简单高效**的设计初衷。

---

## 预期实验基准

- **GAIA**: 通用 AI 助手基准 (多工具推理)
- **AgentBench**: 多领域 Agent 评估
- **WebShop**: 电商 Agent 导航 (长会话)
- **ALFWorld**: 具身 Agent 任务
