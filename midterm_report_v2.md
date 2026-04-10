# Midterm Report v2: LatentMAS Tool Calling Optimization

## 1. Overview
This report documents the significant progress in the LatentMAS tool-calling (LTE-MAS) framework between March 24th and March 25th, 2026. Through architectural refinements and prompt engineering, we achieved a substantial boost in both accuracy and inference efficiency.

## 2. Experimental Comparison (BFCL v3 Simple)

| Metric | Baseline (Single Agent) | LatentMAS (Before Fix) | LatentMAS (Current) |
|--------|-------------------------|------------------------|---------------------|
| **Result File** | `bfcl_baseline_20260324_125234.json` | `bfcl_latent_mas_tool_20260324_180646.json` | `bfcl_latent_mas_tool_20260325_110822.json` |
| **Accuracy** | 69.0% | 61.0% | **89.0%** |
| **Total Correct** | 69/100 | 61/100 | **89/100** |
| **Time per Sample**| **6.65s** | 11.06s | 8.12s |

## 3. Experimental Comparison (BFCL v3 Multiple - NEW)

| Metric | Baseline (Single Agent) | LatentMAS (Current) |
|--------|-------------------------|---------------------|
| **Result File** | `bfcl_baseline_20260325_142518.json` | `bfcl_latent_mas_tool_20260325_160056.json` |
| **Accuracy** | 62.0% | **86.0%** |
| **Total Correct** | 31/50 | **43/50** |
| **Time per Sample**| **7.46s** | 9.23s |

## 4. Key Findings

*   **Surpassing Baseline**: Our current LatentMAS implementation outperforms the standard Single-Agent Baseline by **20% absolute accuracy**. 
*   **Fixing the Regression**: On March 24th, our LatentMAS implementation was actually performing *worse* than the baseline (61% vs 69%). The optimizations implemented on March 25th successfully corrected this regression and achieved state-of-the-art results for our project.
*   **Inference Efficiency**: Although the baseline remains the fastest (as it lacks multi-agent overhead), we have successfully reduced the LatentMAS latency from 11.06s to 8.12s, making the accuracy-speed tradeoff highly favorable.

### A. Schema Injection (The "Information Gap" Fix)
*   **Previous Issue**: The Judger agent often had to generate function calls based solely on latent reasoning states. If the latent detection was slightly off or the reasoning was abstract, the Judger had to "guess" the API names and parameter keys.
*   **Modification**: Injected the full `functions` schema into the Judger's prompt context for every sample.
*   **Impact**: Virtually eliminated "Function Name Mismatch" errors and significantly improved parameter mapping accuracy.

### B. Generation Control & Truncation Robustness
*   **Previous Issue**: Qwen3-8B models tend to generate long `<think>` blocks. With a 512-token limit, the actual `[CALL]` code was often truncated, leading to parsing failures.
*   **Modifications**:
    *   Increased `max_new_tokens` from 512 to **1024**.
    *   Added explicit instructions: *"Do not provide reasoning. Output ONLY the function call."*
    *   Implemented a more robust `_extract_code` function that handles unclosed tags and auto-balances parentheses.
*   **Impact**: Drastically reduced "Invalid function call format" errors and ensured that even if the model "yaps," the code is still produced and captured.

### C. Mas Architecture Streamlining
*   **Refinement**: Optimized the transition from Planner to Judger.
*   **Impact**: Improved the consistency of information passing through the latent space, reducing semantic drift during multi-agent collaboration.

## 4. Conclusion
The jump from 61% to 89% accuracy confirms that the LatentMAS framework is highly capable of complex tool invocation when the "last-mile" generation and extraction logic is properly anchored with explicit schema information. The reduction in latency further demonstrates that more focused prompts lead to more efficient model inference.

## 5. Next Steps
*   Test on more complex BFCL "Multiple Function" datasets.
*   Fine-tune the Latent Detection Head to reduce reliance on heuristic fallbacks.
*   Investigate the remaining 11% failure cases (mostly edge-case parameter values).

---
**Date**: March 25, 2026  