# LatentMAS + Tool-Use Research Memory

## 1. 背景与核心动机 (Context & Motivation)
**LatentMAS (2025.11)** 通过在潜空间传递 KV-Cache 实现了极高的推理效率（Token 减少 70-80%），但其核心缺陷在于：**不支持离散的工具调用 (Tool Calling)**。
现有的 Agent 依赖文本 Token 作为动作接口，这破坏了 Latent 空间的连续思维流。

## 2. 核心挑战 (Core Challenges)
- **连续 vs 离散的冲突**：Latent 推理是连续自回归的，而工具调用是离散中断的。
- **中断与恢复**：如何在潜空间识别“何时需要工具”，以及如何将工具结果（Observation）无缝注入潜空间。
- **Token 税与延迟**：传统的“想清楚再写出来”模式在复杂决策中响应过慢。

## 3. 拟选研究方向 (Proposed Research Directions)

### 方向 1：潜空间动作量化 (Latent Action Quantization) —— **当前重点**
*   **核心理念**：动作本身就是潜空间的一个特殊流形（Manifold）。
*   **技术路径**：
    - 引入 **VQ-VAE Codebook**：将工具意图和参数映射为潜空间中的量化向量（Action Embeddings）。
    - **触发机制**：当潜空间推理轨迹进入特定的“动作簇（Action Cluster）”时，自动触发动作，无需经过文本解码。
*   **顶会价值**：挑战“文本即接口”范式，提出 **Reasoning-Action Isomorphism (推理-动作同构)**。

### 方向 2：递归世界模型 (Recursive World Model - O(1) Memory)
*   **核心理念**：结合 **Thinking States (2026.02)** 的 O(1) 递归机制。
*   **技术路径**：维护固定大小的 **World State Vector**。工具返回结果作为状态更新函数（State Update Function）的输入。
*   **顶会价值**：实现无限长程（Infinite Context）的工具 Agent。

### 方向 3：异步双流协作 (Asynchronous Dual-Stream)
*   **核心理念**：模仿人类 System 1 (直觉/行动) 与 System 2 (逻辑/规划) 的并行。
*   **技术路径**：潜空间流（Slow/Continuous）负责长程规划，Token 流（Fast/Discrete）负责环境交互。

## 4. 实验路线图 (Experimental Roadmap)

### Phase 1: 现象观察与基准测试
- **数据集**：GSM8K-Python, MATH, AIME (竞赛级数学题，迫使模型产生“计算渴望”)。
- **目标**：观察模型在遇到复杂计算时，潜空间向量是否会偏离正常的“推理流形”。
- **当前进度**：已创建 data/test_math_dataset.json。

### Phase 2: 动作触发器 (Action Trigger) 开发
- **任务**：在 latent_steps 循环中增加一个 Action Head 或 VQ 层。
- **验证**：模型能否在不显式输出文本的情况下，通过潜空间向量的变化识别出“该用 Python 了”。

### Phase 3: 闭环验证
- **目标**：在 ToolBench 或 GAIA 数据集上实现端到端的潜空间推理 + 工具调用。

---
**Last Updated**: 2026-03-10
**Status**: 正在进行 Phase 1 的环境与脚本搭建。