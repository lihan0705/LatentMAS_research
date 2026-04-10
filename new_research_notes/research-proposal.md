# Research Proposal: Latent-Space Tool Embeddings for Multi-Agent Systems

## 1. Background (背景与动机)
现代多智能体系统（MAS）在解决复杂推理任务时展现出巨大潜力，而 LatentMAS 进一步通过“隐层空间通信（Latent Space Communication）”消除了文本解码的瓶颈，实现了显著的速度与性能提升。
然而，当前的 LatentMAS 被禁锢在模型的内部权重中，缺乏与外部世界交互的能力。传统的工具调用（Tool Calling）依赖于在 Prompt 中硬编码工具描述，并要求模型输出特定格式的文本（如 JSON），这与 LatentMAS 纯粹的连续隐空间通信格格不入。
**核心问题**：如何在不退回低效的“Token 空间”的前提下，赋予连续隐层多智能体网络执行离散外部工具（如搜索引擎、计算器、代码解释器）的能力？

## 2. Related Work (相关工作)
* **Latent Communication**: `LatentMAS`, `Interlat` 证明了共享 KV-Cache/Hidden States 比传递文本更高效。但它们均未涉及外部环境交互。
* **Tool Use & Agents**: `CoTools` 开始尝试用 Hidden States 来预测是否需要调用工具，但其仍局限于单体模型，且最终的工具参数依然依赖文本生成。
* **Continuous Reasoning**: `COCONUT` 提出了隐空间内的连续思考，但主要用于替代 CoT（Chain of Thought），未解决“连续-离散”的模态跨越问题。

## 3. Proposed Method (提出方法)
我们提出 **LTE-MAS (Latent Tool Embeddings for MAS)** 框架。
1. **Tool Vectorization (工具预编码)**: 在推理前，将所有可用工具（描述、签名）通过一个 Encoder 映射到 LatentMAS 共享的潜层空间中，形成一组静态的 `Tool Vectors`。
2. **Latent Triggering (隐式触发机制)**: 当 Agent 网络在进行 Latent 沟通时，系统实时计算当前群体的聚合 Hidden State 与所有 `Tool Vectors` 的相似度（如 Cosine Similarity）。当最高相似度超过动态阈值 $\tau$ 时，系统自动中断纯隐层通信，触发“工具调用事件”。
3. **Latent-to-Action Decoding (潜层动作解码)**: 触发后，将当前的 Hidden State 喂给一个极轻量级的 Projection Head（或复用原始模型的 LM Head），使其一次性解码出离散的 API 参数，执行外部调用。
4. **Result Re-injection (结果再注入)**: 工具返回的离散结果文本，通过 Embedding 层重新编码为 Latent States，并作为上下文历史追加到所有 Agent 的 KV-Cache 中，供网络继续推理。

**三大核心贡献**:
1. 第一个允许隐层多智能体网络进行外部工具交互的框架。
2. 提出基于隐状态余弦相似度的无额外文本开销的工具触发机制。
3. 在需要外部工具支持的复杂推理基准上实现效率与性能的双重 SOTA。

## 4. Experiment Plan (实验计划)
* **数据集**:
    * **GPQA / HotpotQA**: 需要大量开放域知识，验证 Search 工具的调用。
    * **Math / AIME**: 需要复杂计算，验证 Calculator / Python Interpreter 的调用。
* **基线 (Baselines)**:
    * 传统 Text-MAS (如 AutoGen) + 文本 Tool Calling。
    * LatentMAS (无工具) 作为下界。
    * 单体大模型 + CoTools。
* **核心评价指标**:
    * 任务准确率 (Accuracy)。
    * 总体 Token 消耗与 Wall-clock 延迟 (Efficiency)。
    * 工具调用的准确率（是否在正确的时间调用了正确的工具）。

## 5. Expected Results (预期结果)
* 相比于传统的文本多智能体工具调用（Text-MAS with Tools），**Token 消耗降低 60% 以上**，端到端延迟降低至少 **3 倍**。
* 由于跳过了冗长的文本推理描述，避免了模型在复杂上下文中的注意力涣散，在 GPQA 等任务上的**准确率提升 5-10%**。
* 余弦相似度触发机制能以 >95% 的准确率替代传统的基于 Prompt 的工具意图预测。

## 6. Risks and Mitigations (风险与应对)
* **风险 1：参数解码不准确**。连续的隐状态虽然能准确定位“需要用搜索”，但可能很难精确解码出搜索的关键词字符串。
    * *Mitigation*: 引入混合架构。触发依然用相似度，但触发后，允许系统临时“解冻”输出文本 token，让原模型根据当前的隐状态显式地生成 API JSON。
* **风险 2：外部结果的上下文破坏**。离散文本重新 Encode 后，其潜空间分布可能与当前 Agent 正在进行的连续沟通产生“语义断层 (Semantic Gap)”。
    * *Mitigation*: 参考 `LatentMAS` 的 *Latent-space Alignment*，对重新注入的 Embedding 进行一次轻量级的对齐映射，平滑过渡。
