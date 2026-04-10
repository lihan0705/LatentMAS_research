# LatentMAS Agent Debug Log

每次调试或尝试新方案时更新此文件。

---

## Debug #1: prompts.py method 检查失败
**时间**: 2026-03-16
**错误信息**:
```
AssertionError: this prompt only for latent_mas method
```

**问题分析**:
- `prompts.py` 中 `build_agent_message_sequential_latent_mas` 函数检查 `method in ["latent_mas"]`
- 新方法 `latent_mas_tool` 没有被添加到允许列表中

**修复方案**:
- 修改 `prompts.py` 第 6 行和第 122 行
- 将 `method in ["latent_mas"]` 改为 `method in ["latent_mas", "latent_mas_tool"]`
- 添加 `toolcalling` 任务到支持的 task 列表

---

## Debug #2: PythonExecutor import 错误
**时间**: 2026-03-16
**错误信息**:
```
Execution error: ImportError: __import__ not found
```

**问题分析**:
- `PythonExecutor` 的 `safe_globals` 中没有包含 `__import__`
- 导致 `from math import sqrt` 这类语句无法执行

**修复方案**:
- 在 `safe_globals['__builtins__']` 中添加 `'__import__': __import__`
- 预导入常用模块（math, random, statistics, itertools, numpy, sympy）到全局命名空间

---

## Debug #3: dtype 不匹配 (ToolDetectionHead)
**时间**: 2026-03-16
**错误信息**:
```
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```

**问题分析**:
- 模型的 hidden_state 是 bfloat16
- `ToolDetectionHead` 默认是 float32
- 在 `detect_tool_intent_heuristic` 中直接调用导致 dtype 不匹配

**修复方案**:
```python
# models_tool.py detect_tool_intent_heuristic()
hidden_fp32 = hidden_state.to(torch.float32)
prob = self._tool_detection_head(hidden_fp32).item()
```

---

## Debug #4: 工具没有被调用
**时间**: 2026-03-16
**症状**:
- 运行时没有看到 `[ToolDetection]` 消息
- 模型直接输出错误答案

**问题分析**:
1. `input_text` 传入空字符串，关键词匹配无效
2. 神经网络检测头未训练，输出约 0.5
3. 阈值 0.7 太高，`confidence = 0.3 * 0 + 0.7 * 0.5 = 0.35 < 0.7`

**修复方案**:
1. 降低阈值: `tool_threshold` 从 0.7 → 0.3
2. 传入问题文本到检测函数
3. 优化关键词匹配逻辑：如果匹配到关键词，confidence 直接跳到 0.8-1.0

**代码修改**:
```python
# methods/latent_mas_tool.py
questions = [item["question"] for item in items]
past_kv, tool_info = self.model.generate_latent_batch_with_tools(
    ...,
    questions=questions,  # 新增参数
)

# models_tool.py detect_tool_intent_heuristic()
if keyword_match:
    confidence = 0.8 + 0.2 * neural_score  # 0.8-1.0 range
else:
    confidence = 0.7 * neural_score  # 0-0.7 range
```

---

## Debug #5: dtype 不匹配 (decode_to_tool_call)
**时间**: 2026-03-16
**错误信息**:
```
ToolError] Exception: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

**问题分析**:
- `decode_to_tool_call` 中强制转换为 float32
- 但 LM head 和模型参数是 bfloat16

**修复方案**:
```python
# models_tool.py decode_to_tool_call()
# 动态获取模型 dtype
model_dtype = next(source_model.parameters()).dtype
hidden_typed = hidden_state.to(model_dtype)
logits = lm_head(hidden_typed)

# embedding 也要转换
token_embed = source_model.get_input_embeddings()(
    torch.tensor([generated_tokens[-1]], device=self.device)
).to(model_dtype)
```

---

## Debug #6: Tool Call 解码生成垃圾代码
**时间**: 2026-03-16
**症状**:
- `[ToolDetection]` 正常工作，confidence ~0.89
- 但 `[ToolCall]` 输出垃圾: `dd))))))))))))...`
- 所有 tool calls 都报 `Syntax error: unmatched ')'`

**问题分析**:
1. `decode_to_tool_call` 从 latent hidden state 解码工具调用
2. 但 hidden state 是为推理优化的，不是为生成可执行代码优化的
3. 模型没有训练过 "从 latent state 生成 Python 代码"
4. LM head 从 latent state 解码出的 token 是无意义的

**根本原因**:
Latent space reasoning 产生的 hidden states 表示抽象推理，不是具体代码。
直接用 LM head 解码会得到无意义的输出。

**修复方案**:
方案 B (Tool Calling Agent): 不从 latent state 解码工具调用，
而是:
1. 保留原始问题文本
2. 使用正常的文本生成模式生成工具调用
3. 工具结果再编码回 latent space

```python
# 改进思路
def generate_tool_call_from_text(question: str) -> str:
    """使用正常文本生成工具调用，而不是从 latent 解码"""
    prompt = f"Generate Python code to solve: {question}\n```python\n"
    # 正常自回归生成
    response = model.generate(prompt, max_tokens=100)
    return extract_code(response)
```

---

## Debug #7: 复杂问题代码生成被截断
**时间**: 2026-03-16
**症状**:
- 简单问题（乘法计算）工作正常
- 复杂问题（质数求和）代码被截断
- 错误: `is_prime(candid` 处截断，导致 `NameError: name 'is_prime' is not defined`

**问题分析**:
1. `max_tokens=300` 不够复杂问题的代码生成
2. 质数求和需要定义多个函数，代码更长
3. 代码在函数调用中途被截断

**修复方案**:
增加 `max_tokens` 从 300 → 600:
```python
def generate_tool_call_from_text(self, question: str, max_tokens: int = 600) -> str:
```

**状态**: 已验证，修复有效

---

## Debug #8: complex 函数未定义
**时间**: 2026-03-16
**症状**:
- Problem #18 (复数运算) 失败
- 错误: `NameError: name 'complex' is not defined`

**问题分析**:
- `PythonExecutor` 的 `safe_globals` 中没有包含 `complex` 函数
- 导致 `z = complex(1, 2)` 无法执行

**修复方案**:
在 `tools/python_executor.py` 的 `safe_globals['__builtins__']` 中添加:
```python
'complex': complex,
```

**状态**: 已修复

---

## Debug #9: Tool结果未传递给Judger
**时间**: 2026-03-16
**症状**:
- Tool执行成功，结果正确 (如 `2.6457513111`)
- 但最终答案错误 (如 `7`)
- 多个问题受影响: #3, #7, #12, #13, #15, #17, #20

**问题分析**:
1. Tool结果通过cross-attention注入到latent space
2. 但Judger生成最终答案时，prompt中没有包含tool结果
3. Judger需要从latent space"猜测"答案，导致错误

**根本原因**:
Latent space的信息不能被可靠地解码为具体数值。
Judger需要直接看到tool结果文本。

**修复方案**:
在 `methods/latent_mas_tool.py` 中：
1. 添加 `tool_results_for_judger` 列表保存成功的tool结果
2. 在构建Judger prompt时，将tool结果注入到prompt中:
```python
tool_info = f"\n\n[Tool Result]\nThe following calculation result was obtained using Python: {tool_results_for_judger[idx]}\n\nPlease use this result in your final answer.\n"
judger_prompts_raw[idx] = prompts[idx] + tool_info
```

**状态**: 已修复，待验证

---

## Debug #10: 数据集Gold答案错误
**时间**: 2026-03-16
**症状**:
- 模型Tool计算正确，但与Gold不匹配
- Problem #3: Tool=75553.43, Gold=75553.31
- Problem #5: Tool=602, Gold=505
- Problem #6: Tool=595, Gold=415

**问题分析**:
经过手动验证，数据集的Gold答案本身有错误：
1. **Problem #3 复利**: `50000 * (1.035)^12 = 75553.43`，数据集写的75553.31是错的
2. **Problem #5 等差数列**: 第100项应为 `a + 99d = 8 + 594 = 602`，数据集的505是错的
3. **Problem #6 组合**: `C(6,2)*C(8,2) + C(6,3)*C(8,1) + C(6,4)*C(8,0) = 420+160+15 = 595`，数据集的415是错的

**修复方案**:
修改 `data/test_dataset_toolcalling.json`:
- Problem #3: 75553.31 → 75553.43
- Problem #5: 505 → 602
- Problem #6: 415 → 595

**状态**: 已修复

---

## Debug #11: 函数调用 NameError (exec namespace 问题)
**时间**: 2026-03-16
**症状**:
- 生成的代码完整正确
- 但执行时报 `NameError: name 'is_prime' is not defined`
- 问题涉及所有定义多个函数且函数互相调用的代码

**问题分析**:
在 `tools/python_executor.py` 中：
```python
exec(code, safe_globals, {})
```

问题在于 `exec` 的第三个参数 `{}` 是一个**空的 local namespace**：
1. 函数定义被存储在 local namespace (`{}`) 中
2. 但当函数内部调用另一个函数时，Python 先在函数的局部作用域查找
3. 找不到后去 **global namespace** (`safe_globals`) 查找
4. 结果：`is_prime` 在 locals 中定义，但 `sum_of_first_n_primes` 去 globals 中找，找不到

**修复方案**:
修改 `exec` 调用，让 globals 和 locals 指向同一个字典：
```python
exec(code, safe_globals, safe_globals)
```

这样函数定义和函数调用都在同一个命名空间中，可以互相访问。

**状态**: 已修复

---

## Debug #12: Tool结果被Judger误解
**时间**: 2026-03-16
**症状**:
- Tool 执行成功，返回正确结果
- 但最终答案错误，因为 Judger 解析错误
- 例如 #14: Tool 返回 `[3, 3, 3607, 3803]`，但 Pred 只有 `3`

**问题分析**:
1. Tool 结果通过 cross-attention 注入到 latent space
2. Judger 需要从 latent space 或文本提示中提取答案
3. 但 Judger 可能误解或只取部分结果

**修复方案**:
如果 Tool 调用成功，直接使用 Tool 结果作为最终答案，不经过 Judger：
```python
# methods/latent_mas_tool.py
if tool_results_for_judger[idx] is not None:
    tool_success_for_sample[idx] = True
    final_texts[idx] = str(tool_results_for_judger[idx])
```

同时增加 `max_tokens` 从 600 → 1000，避免复杂代码被截断。

**状态**: 已修复

---

## 统计

| 问题类型 | 次数 |
|---------|------|
| dtype 不匹配 | 2 |
| 配置/检查失败 | 1 |
| 逻辑错误 | 1 |
| import 错误 | 1 |
| 架构设计问题 | 1 |
| 参数不足 | 1 |
| 函数未定义 | 1 |
| 信息传递失败 | 1 |
| 数据集标注错误 | 1 |
| exec namespace 问题 | 1 |

---

## 经验总结

1. **dtype 问题**: PyTorch 模型使用 bfloat16 时，所有中间操作都需要注意 dtype 一致性。最佳实践是动态获取模型 dtype，而不是硬编码。

2. **阈值调优**: 对于未训练的神经网络组件，阈值设置需要保守。最好结合规则和神经网络输出。

3. **向后兼容**: 添加新功能时，确保原有代码路径不受影响。使用 `enable_tools` 开关。

4. **Latent Space vs Text Generation**: Latent hidden states 编码的是抽象推理，不是可执行代码。工具调用应该使用文本生成模式。

5. **exec() namespace 问题**: 当使用 `exec(code, globals, locals)` 时，如果 globals 和 locals 是不同的字典，函数定义会存入 locals，但函数内部的调用会去 globals 查找。解决方案是使用同一个字典：`exec(code, safe_globals, safe_globals)`。

---

*Last Updated: 2026-03-16*
