# Evolution Report - Cycle: LTE-MAS Optimization
**Date**: 2026-03-25
**Trigger**: Experiment Success (89% Accuracy on BFCL Simple)

## Extracted Strategies

### 1. Data Processing: Schema Injection
- **Strategy**: Injecting `functions` schema into the Judger's context.
- **Evidence**: Accuracy jumped from 61% to 89% after providing the API list to the Judger.
- **Generality**: Broadly applicable to any tool-calling task where the final generation agent is separate from the retrieval agent.

### 2. Model Training: Generation Constraints
- **Strategy**: "No-Reasoning" constraint + 1024 Max Tokens.
- **Evidence**: Resolved truncation issues where Qwen models would "yap" too long in the `<think>` block, pushing the actual code out of the context window.
- **Generality**: Critical for Reasoner-style models (Qwen-DeepSeek style) when used for structured tool output.

### 3. Debugging: Robust Extraction
- **Strategy**: Regex with unclosed tag support and parenthesis balancing.
- **Evidence**: Successfully extracted `calculate_mean` calls even when the model omitted the closing parenthesis or the generator cut off.
- **Generality**: Essential for handle-ing stochastic model behavior in structured generation.

## Impact on Future Cycles
Future iterations of LatentMAS should default to 1024 tokens for tool generation and ensure all agents in the chain have access to the function schema, even if they communicate via latent hidden states.

---
**Next Research Direction**: Moving from Keyword-based detection to Latent-Native Retrieval (Similarity-based triggering) as per Plan V2 Phase 1.
