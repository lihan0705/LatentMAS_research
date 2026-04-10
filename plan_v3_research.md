# Plan V3: Research-Centric Roadmap — The $\mathcal{L}$-Tool Framework

## 1. Research Vision: Bridging Continuous-Discrete Modality Crossing
**Objective**: Transform LatentMAS from an efficiency-focused engineering tool into a theoretically grounded reasoning framework that can interact with the discrete world (API/Tools) without breaking the continuous latent manifold.

**Core Academic Story**: 
Traditional tool-calling (ReAct/Toolformer) forces a discrete "sampling break" in reasoning. $\mathcal{L}$-Tool maintains the **Reasoning Manifold** topology by treating tool interactions as a latent retrieval and alignment problem.

---

## 2. Key Research Pillars (The "Three Pillars" Strategy)

### Pillar A: Action-Space Superposition (Theory-Heavy)
*   **Problem**: Premature hard-triggering of tools leads to redundancy and lack of consensus among agents (e.g., Planner/Critic repeating same calls).
*   **Insight**: Tool intent should be a "momentum accumulation" in the latent space, not a threshold jump.
*   **Method**: Represent APIs as orthogonal basis vectors in $\mathbb{R}^d$. Hidden states accumulate projections onto these vectors across latent steps. Execution only occurs when "wavefunction collapse" is triggered by a consensus momentum.
*   **Novelty**: Extends the "Chain of Continuous Thought" (CoConut) into the Action domain.

### Pillar B: Manifold-Aligned Re-entry (LRP) (Analysis-Heavy)
*   **Problem**: "Semantic Gap" between discrete tool results (text) and the highly abstract $N$-th step latent state.
*   **Insight**: Discrete embeddings are "poison" to the reasoning manifold if injected without alignment.
*   **Method**: **Latent Re-entry Projection (LRP)**. A learned projector (or contrastive alignment head) that maps "tool-informed text features" onto the local tangent space of the current latent reasoning trajectory.
*   **Validation**: UMAP/t-SNE visualization of trajectory continuity vs. discontinuity.

### Pillar C: Native Latent API (Architecture-Heavy)
*   **Problem**: The "Hourglass Bottleneck" (Latent -> Text -> Exec -> Text -> Latent) limits expressivity.
*   **Insight**: Some tools (Calculator, Search Index) can be bypassed by "Latent Gateways."
*   **Method**: Direct hidden-state to hidden-state transformation for fixed tools, bypassing discrete decoding entirely for deterministic operations.

---

## 3. Evaluation & Rigor (Top-Tier Standards)

### I. Stress Testing (Beyond Benchmarks)
- **Robustness against Truncation**: Compare $\mathcal{L}$-Tool vs. Text-Agent when output tokens are restricted to < 50. Show $\mathcal{L}$-Tool recovers intent while Text-Agent fails due to syntax errors.
- **Long-Horizon Error Correction**: A 5-step tool-chain task. Measure "Representation Drift" over 50+ latent steps.

### II. Theoretical Analysis
- **Information Bottleneck**: Use Mutual Information metrics to prove the network actually "absorbs" tool results into its reasoning flow.
- **Complexity Analysis**: Formalize the $O(1)$ memory complexity vs. $O(N)$ KV-cache growth in traditional MAS.

---

## 4. Implementation Steps (Phased)

### Phase 1: Manifold Alignment (The LRP Module)
- [ ] Implement `LatentReentryProjector` in `models_tool.py`.
- [ ] Train alignment weights using a contrastive loss on Tool Calling datasets.
- [ ] **Goal**: Ensure accuracy doesn't drop when injecting tool results.

### Phase 2: Consensus-based Triggering (Superposition)
- [ ] Replace `tool_threshold` with `consensus_momentum` (accumulated sim across agents).
- [ ] Implement `ToolResultCache` (Shared Blackboard) to avoid redundant execution during the "buildup" phase.

### Phase 3: Qualitative Analysis (The "Paper Figures")
- [ ] Generate trajectory visualizations (UMAP).
- [ ] Conduct "Cross-Model Alignment" tests (Qwen hidden states to Llama tool triggers).

---

## 5. Target Venues
- **NeurIPS 2026** (Main Track or Workshop on Foundation Models)
- **ICLR 2027** (Deep Learning and Reasoning tracks)
- **ICML 2027** (Multi-Agent Systems / Tool Use)

---
*Created: 2026-04-08*
*Status: Research-Focus Activation*
