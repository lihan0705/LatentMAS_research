# LatentMAS Research Plan V2: LTE-MAS Framework

## **1. Project Vision: The "Whole Idea"**
**LatentMAS** aims to solve the efficiency and intelligence bottleneck of multi-agent systems by moving communication from the **discrete Token Space** to the **continuous Latent Space**. 

**LTE-MAS (Latent Tool Embeddings)** is the critical evolution that connects this continuous reasoning network to the discrete physical world (Tools/APIs). Instead of using slow, error-prone text prompts to "ask" for a tool, the network **retrieves** tools via high-dimensional vector similarity, effectively making external tools a "native" extension of the model's own hidden reasoning states.

---

## **2. Current Status: Milestone 1 Recap**
*   **Accuracy**: Improved from 30% (Baseline) to **65%** (LatentMAS Tool V1).
*   **Infrastructure**: Functional 20-step latent reasoning loop with **Cross-Attention Injection**.
*   **Success**: Multiplication and basic arithmetic are successfully offloaded to Python.
*   **Pain Point**: High redundancy (agents repeat same calls) and "Semantic Gap" between tool results and latent flow.

---

## **3. Short-Term Implementation Plan (Target: >80% Accuracy)**

### **Phase 1: Latent-Native Triggering (Tool Vectorization)**
- **Goal**: Replace MLP/Keyword heuristics with semantic **Latent Retrieval**.
- **Mechanism**: Trigger = $cos\_sim(H_{latent}, V_{tool\_anchor}) > \tau$.

### **Phase 2: State-Aware Hybrid Translator**
- **Goal**: Inform code generation with the **Latent Consensus** (Thinking States).
- **Mechanism**: Use the `last_hidden_state` as a prefix/soft-prompt for the code generator.

### **Phase 3: Anti-Redundancy Knowledge Sharing**
- **Goal**: Stop redundant calls by Planner, Critic, and Refiner.
- **Mechanism**: Implement a session-level `ToolResultCache` shared across the batch.

### **Phase 4: Manifold-Aligned Re-injection**
- **Goal**: Bridge the **Semantic Gap**.
- **Mechanism**: Implement a lightweight **Alignment Projection** ($W_{align}$) to map results onto the reasoning manifold.

---

## **4. Long-Term Research Roadmap (2026-2027)**

### **Mid-Term: Scaling & Cross-Model Alignment (3-6 Months)**
*   **Heterogeneous MAS**: Enable Latent Communication between different model families (e.g., Llama-3 and Qwen-2) using training-free latent alignment.
*   **Domain Expansion**: Scale from GSM8K/Math to **GPQA** (Scientific Reasoning) and **SWE-bench** (Software Engineering).

### **Long-Term: Advanced Reasoning Architectures (6-12 Months)**
*   **[C1.1] Action-Space Superposition**: Move from discrete tool-calling "jumps" to a continuous probability distribution of actions that "collapses" into an execution only when confidence is maximized.
*   **Long-Horizon Latent Memory**: Develop a persistent, indexable latent memory store to prevent "Representation Drift" in tasks requiring 100+ reasoning steps.

---

## **5. Immediate Next Steps**
1.  **Redundancy Fix**: Implement `ToolResultCache` in `methods/latent_mas_tool.py`.
2.  **Native Trigger**: Build `tools/tool_vectorizer.py` and integrate into `models_tool.py`.
3.  **Evaluation**: Re-run the 20-sample `toolcalling` dataset to verify >75% accuracy.
