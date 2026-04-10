# LLMLingua 论文分析

## 基本信息
- **论文**: LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
- **作者**: Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, Lili Qiu
- **机构**: Microsoft Corporation
- **发表**: arXiv:2310.05736v2 [cs.CL] 6 Dec 2023

## 核心概念
LLMLingua是一种粗到细的提示压缩方法，通过预算控制器、token级迭代压缩算法和指令调优方法来在高压缩比下保持语义完整性。

### 关键洞察
1. **自然冗余性**: 自然语言具有内在冗余性，可在保持意义的同时压缩
2. **信息熵原理**: 低困惑度(perplexity)的token对整体熵增贡献较小
3. **分布对齐**: 压缩模型与目标LLM之间存在分布差异需要校正

## 方法详解

### 三级压缩框架

#### 1. 预算控制器(Budget Controller)
- 动态分配不同组件(指令、示例、问题)的压缩比例
- 粗粒度演示级压缩保持语义完整性

#### 2. Token级迭代压缩算法
- 考虑token间条件依赖关系
- 迭代优化压缩决策
- 相比Selective-Context更好保留关键信息

#### 3. 分布对齐方法
- 基于指令调优的对齐技术
- 减小小语言模型与目标LLM的分布差异

## 实验结果
- **压缩比**: 高达20倍压缩
- **性能损失**: 极小性能损失
- **数据集**: GSM8K, BBH, ShareGPT, Arxiv-March23
- **状态**: 达到SOTA性能

## 伪代码实现

```python
class LLMLinguaCompressor:
    def __init__(self, small_model, target_model, compression_ratio=0.5):
        self.small_model = small_model  # 用于压缩的小模型
        self.target_model = target_model  # 目标大模型
        self.compression_ratio = compression_ratio
        
    def compress_prompt(self, prompt, budget=None):
        """粗到细的提示压缩"""
        # 1. 预算控制 - 组件级分配
        components = self.parse_components(prompt)
        budgets = self.budget_controller(components, budget)
        
        compressed_parts = []
        
        for component, text in components.items():
            comp_budget = budgets[component]
            
            # 2. 粗粒度压缩 (演示级)
            if component == "demonstrations":
                compressed = self.coarse_compress(text, comp_budget)
            else:
                # 3. Token级精细压缩
                compressed = self.iterative_compress(text, comp_budget)
                
            compressed_parts.append(compressed)
            
        return "\n".join(compressed_parts)
    
    def budget_controller(self, components, total_budget):
        """动态预算分配"""
        if total_budget is None:
            total_budget = int(sum(len(c) for c in components.values()) * self.compression_ratio)
            
        # 根据重要性分配预算
        importance_weights = {
            "instruction": 0.3,  # 指令最重要
            "demonstrations": 0.5,  # 示例次重要
            "question": 0.2  # 问题相对不重要
        }
        
        budgets = {}
        remaining_budget = total_budget
        
        # 按比例分配
        for comp, weight in importance_weights.items():
            if comp in components:
                comp_budget = int(total_budget * weight)
                budgets[comp] = min(comp_budget, remaining_budget)
                remaining_budget -= budgets[comp]
                
        return budgets
    
    def iterative_compress(self, text, budget):
        """Token级迭代压缩"""
        tokens = text.split()
        target_len = int(len(tokens) * (budget / len(text)))
        
        # 计算token重要性 (基于小模型)
        importance_scores = self.compute_token_importance(tokens)
        
        # 迭代优化
        compressed = []
        for i, token in enumerate(tokens):
            if len(compressed) >= target_len:
                break
                
            # 考虑上下文依赖
            context_score = self.context_dependency_score(token, tokens[max(0,i-3):i+3])
            final_score = importance_scores[i] * context_score
            
            if final_score > self.threshold:
                compressed.append(token)
                
        return " ".join(compressed)
    
    def compute_token_importance(self, tokens):
        """计算token重要性分数"""
        # 使用小模型计算自信息
        scores = []
        for token in tokens:
            log_prob = self.small_model.log_probability(token)
            importance = -log_prob  # 信息量 = -log(P)
            scores.append(importance)
        return scores
    
    def distribute_align(self, compressed_prompt):
        """分布对齐微调"""
        # 使用目标模型的反馈微调压缩模型
        target_logits = self.target_model(compressed_prompt)
        small_logits = self.small_model(compressed_prompt)
        
        # KL散度损失对齐分布
        kl_loss = F.kl_div(
            F.log_softmax(small_logits, dim=-1),
            F.softmax(target_logits, dim=-1),
            reduction='batchmean'
        )
        
        return kl_loss

def compress_for_llm(long_prompt, target_llm, compression_ratio=0.3):
    """LLMLingua压缩主函数"""
    # 初始化压缩器
    compressor = LLMLinguaCompressor(
        small_model=GPT2Small(),  # 小模型
        target_model=target_llm,   # 目标模型
        compression_ratio=compression_ratio
    )
    
    # 执行压缩
    compressed_prompt = compressor.compress_prompt(long_prompt)
    
    # 可选：分布对齐微调
    if hasattr(compressor, 'distribute_align'):
        compressor.distribute_align(compressed_prompt)
        
    return compressed_prompt
```

## 在LatentMAS中的应用潜力

### 多代理通信优化
1. **消息压缩**: 代理间通信的冗长推理链压缩
2. **知识共享**: 大型知识库的紧凑表示
3. **上下文管理**: 限制latent space中的信息冗余

### 与Tool Use集成
```python
class CompressedToolInterface:
    def __init__(self, llm_lingua_compressor):
        self.compressor = llm_lingua_compressor
        
    def compress_tool_call(self, tool_description, parameters):
        prompt = f"Tool: {tool_description}\nParams: {parameters}"
        return self.compressor.compress_prompt(prompt)
```

## 优势与局限

### 优势
- 无需访问目标LLM参数(支持API场景)
- 高压缩比下保持语义完整性
- 考虑token间依赖关系

### 局限
- 压缩质量依赖小模型能力
- 可能丢失细微语义差别
- 实时压缩增加计算开销

## 相关技术对比
- **Selective-Context**: 忽略token依赖，LLMLingua考虑更全面
- **Gist Tokens**: 学习压缩vs启发式压缩，各有优势
- **Quiet-STaR**: 内部思维压缩vs外部提示压缩