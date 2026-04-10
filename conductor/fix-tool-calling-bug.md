# Plan: Fix LatentMAS Tool Calling "System" Bug and Robustness Improvements

## Objective
Fix the issue where LatentMAS incorrectly identifies tool calls as "system" and fails to extract valid function calls. Improve the tool detection heuristic and ensure proper tool vectorization.

## Key Files
- `models_tool.py`: Core tool calling and detection logic.
- `methods/latent_mas_tool.py`: Agent-level tool handling.
- `run_test_bfcl.py`: Evaluation script with DummyExecutor.

## Implementation Steps

### 1. Robust Tool Detection in `models_tool.py`
- Update `ModelWrapperWithTools.detect_tool_intent_heuristic`:
    - Increase default `tool_threshold` to `0.5`.
    - Change default `neural_score` to `0.3` when head is not trained.
    - Expand `tool_keywords` to include more relevant math/code terms (e.g., 'odds', 'solve', 'find', 'expression').
    - Incorporate `ToolVectorizer` similarity into the confidence score.
- Update `ToolVectorizer`:
    - Add `clear_anchors()` method to reset registered tools between samples.

### 2. Precise Tool Call Extraction in `models_tool.py`
- Update `generate_tool_call_from_text`:
    - Set `add_special_tokens=False` in `tokenizer.encode`.
    - Improve extraction logic:
        - Use regex to find `function_name(args)` patterns.
        - Handle ChatML boilerplates (e.g., stripping "system", "user", "assistant" tags).
        - Fallback to `None` if no valid call is found instead of returning garbage.

### 3. Evaluation Script Refinement in `run_test_bfcl.py`
- Clear `ToolVectorizer` anchors before each sample processing to avoid cross-contamination.
- Fix `DummyExecutor`:
    - Return `False` for `success` if no valid function call regex match is found.
    - Return a clear error message instead of garbage.

### 4. Agent Method Improvements in `methods/latent_mas_tool.py`
- Ensure `tool_results_for_judger` is only populated on true success.
- Ensure the Judger is invoked correctly when tool execution fails or is skipped.

## Verification
- Run `run_test_bfcl.py` on the simple dataset.
- Check logs to ensure tool calls are correctly extracted (no more "system").
- Verify that `accuracy` in `results/bfcl_latent_mas_tool_*.json` improves significantly.
