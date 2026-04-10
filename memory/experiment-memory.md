# Experimentation Memory (M_E) - LatentMAS

## Data Processing & Tool Result Strategies
- **Judger Override (Direct Injection)**: For mathematical/exact results, skip the Judger and use the Tool Result directly as the final answer to prevent Judger misinterpretation (Resolved Debug #12).
- **Result Re-injection (Semantic Gap)**: When re-encoding tool results, apply a lightweight alignment mapping (projection) to prevent breaking the latent manifold's alignment.
- **Judger Schema Injection (LTE-MAS)**: In tool-calling tasks, inject the full `functions` schema into the Judger's prompt via the `context` parameter. This prevents "blind guessing" of API names when latent detection is slightly off. (Added 2026-03-25).

## Model Training & Architecture Strategies
- **Tool Vectorization (Retrieval)**: Pre-encode API signatures and descriptions into the latent space. Triggering = max(cos_sim(H, Tool_Vectors)) > tau.
- **Hybrid Decoding (Translator)**: Use a separate head or un-frozen text tokens for *parameter generation* even if *triggering* is latent.
- **Dtype Consistency (bfloat16)**: All projection heads (Detection/Translator) MUST be cast to the model's native dtype (bfloat16) to avoid mat1/mat2 mismatches.
- **Generation Control (No-Reasoning)**: Force tool generators (Judger/Planner) to skip reasoning text using explicit "No reasoning" constraints in the system prompt. Combined with increased `max_new_tokens` (1024), this prevents truncation of complex function calls. (Added 2026-03-25).

## Execution & Debugging Strategies
- **Exec Namespace Management**: Always use exec(code, safe_globals, safe_globals) to allow cross-function calls in generated Python scripts.
- **Max Tokens for Tools**: Set max_tokens >= 1024 for tool-call generators to avoid truncating complex code (e.g., prime number algorithms). (Updated 2026-03-25).
- **Robust Tool Extraction**: Use regex that handles unclosed tags (truncation) and implement a "balanced parenthesis" checker for multi-line or partially generated calls. (Added 2026-03-25).
