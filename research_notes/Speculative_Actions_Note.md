# 深度学术笔记：Speculative Actions (Ye et al., 2025)

> **Citation**: Ye, N., Ahuja, A., Liargkovas, G., Lu, Y., Kaffes, K., & Peng, T. (2025). Speculative Actions: A Lossless Framework for Faster Agentic Systems. arXiv:2510.04371. Columbia University.
>
> **核心一句话**：把 CPU 推测执行的思想移植到 AI Agent，用一个"快手小模型"预测下一步 API 调用并提前发射，等"权威大模型"验证后直接提交或回滚，从而把串行等待变成并行执行，实现无损加速。

---

## 1. 背景与动机

现代 AI Agent 执行任务极慢，根本原因是严格串行的 API 调用链：

```
思考 → 调用API → 等待响应 → 思考 → 调用API → 等待响应 → ...
              ↑ 每步都必须等上一步，形成大量"流水线气泡"
```

**实际数据（论文 Table 1）：**

| 任务类型 | 典型耗时 |
|----------|----------|
| OS 任务 | 10–20 分钟 |
| Deep Research | 5–30 分钟 |
| 数据流水线 | 30–45 分钟 |
| Kaggle 国际象棋 | 1 小时 |

**核心问题**：Agent 必须严格串行地与环境交互吗？**本文回答：不必须。**

---

## 2. 灵感来源

本文从两个"前辈"技术类比移植：

| 前辈技术 | 领域 | 核心思路 |
|----------|------|----------|
| 微处理器推测执行（Tomasulo, 1967） | CPU 架构 | 分支预测 + 预执行 + 错误回滚 |
| LLM 推测解码（Leviathan et al., 2023） | LLM 推理 | 小模型起草 token，大模型批量验证 |
| **本文：Speculative Actions** | **Agent 系统** | **Speculator 预测 API 调用，Actor 验证提交** |

---

## 3. 核心框架：形式化定义

### 3.1 MDP 建模

Agent 被形式化为马尔可夫决策过程 (MDP)，每步动作 = 一次 **API 调用**：

```
(h_t, q_t) ← π(s_t)           # 策略决定调用哪个API及其参数
ā_t ⇝ h_t(q_t)                # 异步发射API调用，ā_t 是"pending future"
a_t ← await(ā_t)              # 等待结果到达
s_{t+1} ← f(s_t, a_t)        # 状态转移
```

这个抽象涵盖：
- **LLM 调用**：每次调用 GPT/Claude 等模型
- **工具 / MCP 调用**：Web 搜索、代码执行、外部 API
- **人类响应**：把人类回复也抽象成延迟较高的 API

### 3.2 双角色模型

| 角色 | 特征 | 例子 |
|------|------|------|
| **Actor（权威执行者）** | 慢但准确，决定 ground truth | 强力 LLM (GPT-5 high reasoning)、外部 API、人类 |
| **Speculator（推测者）** | 快且廉价，预测下一步 | 小模型、低推理预算的同款模型、规则启发式 |

---

## 4. 核心算法（Algorithm 1）及伪代码实现

### 4.1 论文原版算法（Algorithm 1：k-way 并行推测）

```
Require: 初始状态 s0, 步数 T, 状态转移 f, 策略 π, 预测器 ĝ, 缓存 C

for t = 0 to T-1:
    (h_t, q_t) ← π(s_t)               # Step 1: 确定当前该调哪个API

    if (h_t, q_t) ∈ C:                 # Step 2: 缓存命中（上一步预测成功）
        ā_t ← C[(h_t, q_t)]
        a_t ← await(ā_t)              # 直接等pending结果，无需重发请求
        s_{t+1} ← f(s_t, a_t)
        continue

    # Step 3: 缓存未命中，正常发射 + 同时推测
    ā_t ⇝ h_t(q_t)                    # Actor: 发射真实API调用（异步，non-blocking）
    {â_t^(i)}_{i=1}^k ← await(ĝ(s_t, (h_t, q_t)))  # Speculator: 预测k个候选回应

    # Step 4: 基于每个预测，提前发射下一步的API调用
    for i = 1 to k:
        ŝ_{t+1}^(i) ← f(s_t, â_t^(i))              # 模拟状态转移
        (ĥ_{t+1}^(i), q̂_{t+1}^(i)) ← π(ŝ_{t+1}^(i))  # 预测下一步策略
        ā_{t+1}^(i) ⇝ ĥ_{t+1}^(i)(q̂_{t+1}^(i))    # 提前发射！（non-blocking）
        C[(ĥ_{t+1}^(i), q̂_{t+1}^(i))] ← ā_{t+1}^(i)  # 存入缓存

    # Step 5: 等Actor返回真实结果，验证并提交
    a_t ← await(ā_t)
    s_{t+1} ← f(s_t, a_t)
    # 若 (h_{t+1}, q_{t+1}) 命中缓存，下一轮直接走 Step 2（加速成功）
    # 否则，缓存中的推测结果被自然丢弃
```

### 4.2 Python 风格伪代码实现

```python
import asyncio
from typing import Any, Dict, List, Tuple, Optional

# ── 类型别名 ────────────────────────────────────────────────────────────
State   = Any                          # Agent 状态（对话历史、棋盘等）
APICall = Tuple[str, dict]             # (api_name, params)
Future  = asyncio.Future               # 异步 pending action
Cache   = Dict[APICall, Future]        # 推测缓存：API调用 → pending future


# ── 核心组件（接口定义）──────────────────────────────────────────────────

async def actor_api_call(h: str, q: dict) -> Any:
    """Actor: 发射真实 API 调用，返回权威结果（慢但准确）"""
    # 实际是: response = await real_api_or_llm(h, q)
    ...

async def speculator_predict(state: State, current_call: APICall,
                              k: int) -> List[Any]:
    """Speculator: 预测当前 API 调用的 k 个可能返回值（快但不一定准）"""
    # 实际是: guesses = await fast_model.predict(state, current_call, k)
    ...

def policy(state: State) -> APICall:
    """Agent 策略：根据当前状态决定调用哪个 API 及其参数"""
    ...

def transition(state: State, action: Any) -> State:
    """状态转移函数"""
    ...


# ── Speculative Actions 主循环 ────────────────────────────────────────────

async def speculative_agent_loop(
    initial_state: State,
    horizon: int,
    k: int = 3,                        # 每步推测 k 个候选
) -> List[Any]:
    """
    无损推测执行主循环。
    每步同时：
      1. Actor  异步发射真实 API 调用
      2. Speculator 预测响应 → 提前发射下一步 k 个候选调用
      3. 等 Actor 返回 → 检查缓存命中 → 提交或继续
    """
    state   = initial_state
    cache: Cache = {}                  # 推测缓存
    trajectory = []

    for t in range(horizon):
        # ── Step 1: 确定当前API调用 ────────────────────────────────────
        api_call = policy(state)       # (h_t, q_t)

        # ── Step 2: 缓存命中（上轮推测成功）──────────────────────────────
        if api_call in cache:
            pending = cache.pop(api_call)
            action = await pending     # 可能已经返回，await 几乎零延迟
            print(f"[t={t}] 缓存命中！跳过实际API调用，直接提交")
        else:
            # ── Step 3: 并行发射 Actor + Speculator ──────────────────────
            # Actor 异步发射真实调用（non-blocking）
            actor_future = asyncio.create_task(
                actor_api_call(*api_call)
            )

            # Speculator 同步预测（或也可异步，与 Actor 并行）
            guessed_responses = await speculator_predict(state, api_call, k)

            # ── Step 4: 基于每个猜测，提前发射下一步 API ─────────────────
            for guess in guessed_responses:
                speculated_next_state = transition(state, guess)
                next_api_call = policy(speculated_next_state)

                if next_api_call not in cache:  # 避免重复发射
                    # 提前发射！non-blocking，结果存入缓存等待命中
                    cache[next_api_call] = asyncio.create_task(
                        actor_api_call(*next_api_call)
                    )

            # ── Step 5: 等 Actor 返回真实结果 ─────────────────────────────
            action = await actor_future
            print(f"[t={t}] Actor 返回，验证推测缓存...")

        # ── Step 6: 状态转移，进入下一轮 ────────────────────────────────
        state = transition(state, action)
        trajectory.append(action)

        # 注：若本轮是缓存命中，下一轮的 policy(state) 大概率也命中缓存
        # 形成连续加速链；若未命中，缓存中的推测 future 自然超时/丢弃

    # 清理未命中的推测 future（取消未使用的并行调用）
    for pending in cache.values():
        pending.cancel()

    return trajectory
```

### 4.3 时序对比图示

```
非推测（串行）：
─────────────────────────────────────────────────────────►
  [Actor t0]──►wait──►[Actor t1]──►wait──►[Actor t2]──►...
       ^                   ^                   ^
       延迟1               延迟2               延迟3

推测执行（并行）：
─────────────────────────────────────────────────────────►
  [Actor t0]─────────────────────────────►[Actor t2]
       └──[Spec]──[pre-launch t1 A]──►✓ 命中缓存！
                 [pre-launch t1 B]──►✗ 取消
                 [pre-launch t1 C]──►✗ 取消
                       [pre-launch t2...]──►...

  Actor t0 在等待时，t1 的调用已经在运行了 → 时间节省 = 重叠等待时间
```

---

## 5. 无损性保障三机制

### 5.1 语义守卫（Semantic Guards）

```python
def semantic_guard(speculated_call: APICall, context: State) -> bool:
    """
    检查推测的 API 调用是否与当前上下文语义一致。
    如果猜测严重偏离上下文逻辑，提前拦截，避免执行无意义的调用。
    """
    # 例：当前在处理"查询订单"，推测到"删除用户" → 拒绝
    if is_semantically_inconsistent(speculated_call, context):
        return False
    return True

# 在 pre-launch 前调用
if semantic_guard(next_api_call, state):
    cache[next_api_call] = asyncio.create_task(actor_api_call(*next_api_call))
```

### 5.2 安全包络（Safety Envelopes）

```python
# 定义哪些 API 可以被推测（白名单）
SPECULATABLE_APIS = {
    "web_search",          # ✅ 无副作用，幂等
    "get_order_info",      # ✅ 只读
    "check_return_eligibility",  # ✅ 只读
    "llm_call",            # ✅ 无外部副作用
}

NON_SPECULATABLE_APIS = {
    "place_order",         # ❌ 不可逆，会真正下单
    "delete_record",       # ❌ 不可逆
    "send_email",          # ❌ 不可逆
    "execute_payment",     # ❌ 不可逆
}

def can_speculate(api_name: str) -> bool:
    return api_name in SPECULATABLE_APIS
```

### 5.3 修复路径（Repair Paths / Rollback）

```python
async def speculative_step_with_rollback(state, api_call, cache):
    """
    执行推测步骤，若推测结果已被执行但后来发现错误，则回滚。
    """
    action = await actor_api_call(*api_call)  # 获取真实结果

    # 检查缓存中哪些推测 future 已经被提前执行
    committed_speculations = [
        (call, fut) for call, fut in cache.items() if fut.done()
    ]

    for speculated_call, fut in committed_speculations:
        speculated_result = fut.result()

        # 若推测导致了可逆的副作用，现在验证并回滚
        if needs_rollback(speculated_call, action):
            await rollback_action(speculated_call, speculated_result)
            # 例：已经预取的数据缓存失效；已经创建的临时购物车清空
    
    return action

def needs_rollback(speculated_call: APICall, true_action: Any) -> bool:
    """判断推测动作是否需要回滚（基于真实结果）"""
    # 若推测的调用对应的状态分支与真实分支不同，则需要回滚
    ...
```

---

## 6. 理论分析：加速比公式（Proposition 1）

**假设**：
- 推测准确率为 $p$（每步独立）
- Speculator 延迟 $\sim \text{Exp}(\alpha)$，API 调用延迟 $\sim \text{Exp}(\beta)$，$\beta < \alpha$

**推导出加速比**：

$$\frac{E[T^s]}{E[T^{seq}]} = 1 - \frac{1}{T} \cdot \frac{\alpha}{\alpha+\beta} \left[\frac{(T-1)p}{1+p} + \frac{p^2}{(1+p)^2} - \frac{p^2}{(1+p)^2}(-p)^{T-1}\right]$$

**极限情形**（$T \to \infty$）：

$$\frac{E[T^s]}{E[T^{seq}]} \xrightarrow{T\to\infty} 1 - \frac{p}{1+p} \cdot \frac{\alpha}{\alpha+\beta}$$

**特殊情况（理论上界）**：当 $p=1$，$\alpha \to \infty$（推测完全准确且瞬时）：

$$\text{加速比} \to 1 - \frac{1}{2} = 50\%$$

**直觉理解**：即使推测完全准确，由于每次命中后下一步不能立即推测（需等 Actor 确认），理论上限是节省 50% 时间，而非 100%。

---

## 7. 四个实验环境及关键结果

### 7.1 国际象棋（Chess）

- **Speculator**：GPT-5 低推理 effort + 快速预测 prompt
- **Actor**：GPT-5 高推理 effort
- **机制**：B 玩家在 A 玩家思考时，提前预测 A 的走法并开始分析应对

| 预测数量 k | 预测准确率 | 节省时间 |
|------------|-----------|----------|
| 1 prediction | 31.3% | 11.8% |
| 2 predictions | 41.3% | 15.0% |
| 3 predictions | 54.7% | 19.5% |

### 7.2 电商对话（E-Commerce, τ-bench）

- **机制**：用户发消息时，Speculator 预测用户意图 + 提前调用所需 API
- **结果**：22%–38% 的 API 调用被正确预测；多模型并联（GPT + Gemini）优于单模型

```python
# E-Commerce 推测逻辑伪代码
async def ecommerce_speculative_pipeline(conversation_history):
    # 1. Speculator 预测用户下一条消息
    predicted_user_query = await speculator.predict_user_intent(conversation_history)
    
    # 2. 基于预测，推测需要调用哪些 API
    predicted_api_calls = await speculator.predict_api_calls(
        conversation_history, predicted_user_query
    )
    
    # 3. 提前发射所有预测的 API 调用（非阻塞）
    pending_calls = {
        call: asyncio.create_task(execute_api(*call))
        for call in predicted_api_calls
        if can_speculate(call[0])  # 安全包络过滤
    }
    
    # 4. 等用户真实消息到来（30s 平均用户输入时间）
    actual_user_query = await wait_for_user_input()
    
    # 5. 验证哪些预测命中
    actual_api_calls = determine_required_apis(actual_user_query)
    for call in actual_api_calls:
        if call in pending_calls:
            result = await pending_calls[call]  # 几乎零等待！
        else:
            result = await execute_api(*call)   # 正常执行
```

### 7.3 多跳 Web 搜索（HotpotQA）

- **机制**：在等 Wikipedia API 响应时，Speculator 从自身知识库猜测结果，并提前规划下一次搜索
- **结果**：Top-3 预测准确率高达 **46%**（严格匹配标准）

```python
# ReAct + 推测执行 伪代码
async def speculative_react_loop(question: str):
    state = {"question": question, "reasoning_trace": "", "retrieved_info": []}
    
    while not is_finished(state):
        # Actor 决定下一步动作
        action_type, query = actor_decide(state)   # e.g., Search("Einstein")
        
        # Speculator 预测 API 响应
        speculated_responses = await speculator.predict_api_response(
            state, action_type, query, k=3
        )
        
        # 提前基于每个猜测生成后续推理路径
        speculated_next_calls = []
        for resp in speculated_responses:
            next_state = simulate_transition(state, resp)
            next_action = actor_decide(next_state)
            speculated_next_calls.append(next_action)
            # 提前发射！
            cache[next_action] = asyncio.create_task(execute_search(*next_action))
        
        # 等真实响应
        true_response = await execute_search(action_type, query)
        state = update_state(state, true_response)
        # 若 cache 命中，下一步几乎无等待
```

### 7.4 OS 超参调优（Lossy Extension，有损扩展）

这是一个**放宽无损约束**的扩展实验，适用于延迟敏感系统：

- **场景**：调优 Linux CFS 调度器参数 `min_granularity_ns`
- **机制**：Speculator 每秒快速调整，Actor 每 10–15 秒做深度分析并覆盖
- **"last-write-wins"**：Actor 的决定直接覆盖 Speculator，无需回滚

```python
async def speculative_os_tuner():
    """有损推测：Speculator 快速试探，Actor 定期覆盖"""
    
    async def speculator_loop():
        """每 1s 快速调整一次"""
        while True:
            metrics = read_system_metrics()
            new_param = speculator.quick_propose(metrics)
            apply_cfs_param(new_param)  # 立即生效（last-write-wins）
            await asyncio.sleep(1)
    
    async def actor_loop():
        """每 10-15s 深度分析一次"""
        while True:
            history = get_compressed_speculator_history()
            new_param = await actor.deep_analyze(history)  # 慢但深度
            apply_cfs_param(new_param)  # 覆盖 Speculator 的值
            await asyncio.sleep(10)
    
    # 两者并行运行
    await asyncio.gather(speculator_loop(), actor_loop())
```

**结果对比**：

| 系统 | 平均 p95 延迟 | 收敛时间 |
|------|--------------|----------|
| 未调优 | 102.97 ms | — |
| Actor-only | 54.00 ms | ~200s |
| Actor + Speculator | **37.93 ms** | **~15s** |
| Speculator-only | 36.24 ms（次优） | ~20s（局部最优） |

---

## 8. 关键洞察总结

### 8.1 为什么推测能奏效？（两个假设）

**Assumption 1（推测准确性）**：API 响应通常是可预测的，因为：
- 任务有结构化逻辑（搜索 A 后大概率搜 B）
- LLM 的行为在给定上下文下有规律
- 弱模型/简化 prompt 已足够预测高层意图

**Assumption 2（并发可逆）**：大多数 API 调用可以并发且副作用可逆：
- Read-only API（搜索、查询）：天然幂等
- 沙箱化操作（OS 参数）：可直接覆盖
- 购物车操作（加入购物车）：可撤销

### 8.2 与 Speculative Decoding 的区别

| 维度 | Speculative Decoding（LLM 推理） | Speculative Actions（Agent 系统） |
|------|----------------------------------|-----------------------------------|
| 预测对象 | 下一个 token | 下一个 API 调用及其参数 |
| 验证者 | 目标大模型（批量并行验证） | Actor（权威 LLM 或环境响应） |
| 粒度 | token 级（毫秒） | 步骤级（秒到分钟） |
| 副作用 | 无（纯计算） | 有（工具调用、状态变化） |
| 回滚 | 不需要（只是重新采样） | 需要（安全包络 + 修复路径） |

---

## 9. 局限性

1. **不可逆操作**：下订单、删数据库、发邮件等操作无法推测，必须等 Actor 确认
2. **推测成本**：额外的 token 消耗和 API 调用费用（k=3 时 token 成本增加约 40%）
3. **理论上限 50%**：由于连续命中后存在"推测窗口"限制，加速比天然受限
4. **准确率依赖上下文**：强模型有时因表述多样性反而准确率低于弱模型

---

## 10. 对 LatentMAS 的启示

本文的推测执行思路对 LatentMAS 有以下潜在应用方向：

### 10.1 潜在空间的零成本推测
LatentMAS 的 Processor 在隐藏空间操作，**回滚只需重置 KV Cache**，比真实 API 调用的回滚代价低几个数量级。可以在隐藏空间并行探索多条推理路径（类似 Beam Search），只提交最优的一条。

```python
# LatentMAS + 推测执行的概念伪代码
async def latent_speculative_reasoning(prompt, latent_steps=10, k=3):
    initial_kv_cache = encode_prompt(prompt)
    
    # 同时探索 k 条潜在推理路径
    async def latent_branch(kv_cache_snapshot):
        for step in range(latent_steps):
            new_hidden = model_forward(kv_cache_snapshot)
            kv_cache_snapshot = update_kv_cache(kv_cache_snapshot, new_hidden)
        return kv_cache_snapshot, score_path(kv_cache_snapshot)
    
    branches = await asyncio.gather(*[
        latent_branch(copy_kv_cache(initial_kv_cache)) for _ in range(k)
    ])
    
    # 选最优路径
    best_kv_cache = max(branches, key=lambda x: x[1])[0]
    return decode_final_answer(best_kv_cache)
```

### 10.2 多 Agent 的隐藏状态预取
在 LatentMAS 的 Composer 层，可以让下游 Agent 提前开始处理上游 Agent 的"预测隐藏状态"，而不是等待真实的 KV Cache 传递完成。

### 10.3 Theorem 3.3 的扩展
论文 Theorem 3.3（KV Cache 传递 ≡ 显式输入传递）意味着潜在空间的推测执行在信息论上是无损的，这为 LatentMAS 引入推测执行提供了理论基础。

---

## 参考文献核心

- **Algorithm 1**（论文 p.4–5）：核心推测动作算法
- **Proposition 1**（论文 p.4 + Appendix A）：加速比形式化证明
- **Figure 1**（论文 p.3）：国际象棋推测执行时序图
- **Figure 5**（论文 p.10）：OS 调优 Speculator vs Actor vs 联合系统对比
