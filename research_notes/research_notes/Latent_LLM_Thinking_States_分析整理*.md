# Latent Reasoning with Supervised Thinking States 论文分析

## 基本信息
- **论文**: Latent Reasoning with Supervised Thinking States
- **作者**: Ido Amos, Avi Caciularu, Mor Geva, Amir Globerson, Jonathan Herzig, Lior Shani, Idan Szpektor
- **机构**: Google Research, Hebrew University of Jerusalem, Tel Aviv University
- **发表**: arXiv:2602.08332v1 [cs.CL] 9 Feb 2026

## 核心概念
Thinking States是一种在输入处理过程中执行推理的方法，通过定期生成思考token序列，将其转换回embedding空间并添加到后续输入token中。

### 关键洞察
1. **计算共享**: 使用现有token的表示生成思维
2. **循环机制**: 思维作为反馈输入给未来token（类似CoT但不断开上下文）
3. **自然语言监督**: 思维token可通过自然语言监督，保持可解释性
4. **避免BPTT**: 不需要通过时间反向传播进行优化

## 方法详解

### 三步流程
1. **Thinking**: 每几个输入token生成一次思维状态
2. **Embedding**: 将思维token转换回embedding空间
3. **Integration**: 添加到后续输入token的表示中

### 与CoT对比优势
- **上下文长度**: 不增加上下文长度（思维不在显式上下文中）
- **并行训练**: 支持teacher-forcing并行化训练
- **推理延迟**: 显著低于CoT的推理延迟
- **监督信号**: 可使用自然语言形式的监督信号

## 实验结果
- **数学推理**: 缩小了与CoT的性能差距
- **2跳问答**: 匹配CoT性能且延迟更低
- **状态追踪**: 超越CoT的推理行为，成功外推到更长序列
- **训练速度**: 比需要BPTT的隐式思维模型快得多

## 伪代码实现

```python
class ThinkingStates(nn.Module):
    def __init__(self, llm, thought_interval=3, thought_length=16):
        super().__init__()
        self.llm = llm
        self.thought_interval = thought_interval  # 每多少个token生成一个思维
        self.thought_length = thought_length      # 思维token数量
        self.thought_projection = nn.Linear(llm.hidden_size, llm.vocab_size)
        
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        outputs = []
        accumulated_thoughts = []
        
        for i in range(0, seq_len, self.thought_interval):
            # 处理当前片段
            chunk = input_ids[:, i:i+self.thought_interval]
            chunk_embeds = self.llm.get_embeddings(chunk)
            
            # 生成思维状态
            if len(accumulated_thoughts) > 0:
                thought_context = torch.cat([accumulated_thoughts[-1], chunk_embeds], dim=1)
            else:
                thought_context = chunk_embeds
                
            thought_logits = self.thought_projection(thought_context)
            thought_tokens = self.sample_thoughts(thought_logits)
            
            # 将思维转换为embedding并累积
            thought_embeds = self.llm.get_embeddings(thought_tokens)
            accumulated_thoughts.append(thought_embeds)
            
            # 为下一个片段准备输入（包含累积的思维）
            next_input = self.integrate_thoughts(chunk_embeds, accumulated_thoughts)
            outputs.append(next_input)
            
        return torch.cat(outputs, dim=1)
    
    def sample_thoughts(self, logits, temperature=0.7):
        """采样思维token"""
        probs = F.softmax(logits / temperature, dim=-1)
        thought_tokens = torch.multinomial(probs, self.thought_length)
        return thought_tokens
    
    def integrate_thoughts(self, current_embeds, accumulated_thoughts):
        """整合思维到当前表示"""
        if accumulated_thoughts:
            # 取最新的思维状态
            latest_thoughts = accumulated_thoughts[-1]
            # 残差连接方式整合
            integrated = current_embeds + 0.1 * latest_thoughts.mean(dim=1, keepdim=True)
            return integrated
        return current_embeds

def train_thinking_states(model, dataset, thought_supervision):
    """训练Thinking States模型"""
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    for batch in dataset:
        input_ids = batch["input_ids"]
        targets = batch["targets"]
        
        # Teacher forcing: 使用真实的思维标签
        if thought_supervision:
            thought_labels = batch["thought_labels"]
            
            # 前向传播
            outputs = model(input_ids, thought_labels=thought_labels)
            
            # 计算损失：主任务 + 思维生成
            main_loss = F.cross_entropy(outputs.main_logits, targets)
            thought_loss = F.cross_entropy(outputs.thought_logits, thought_labels)
            
            total_loss = main_loss + 0.3 * thought_loss
        else:
            # 自回归训练
            outputs = model(input_ids)
            total_loss = F.cross_entropy(outputs.logits, targets)
            
        # 反向传播
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 在LatentMAS中的应用

### 与LatentMAS集成
```python
class LatentThinkingStates(LatentMASMethod):
    def __init__(self, base_model):
        super().__init__(base_model)
        self.thinking_states = ThinkingStates(base_model)
        
    def run_batch(self, messages_list):
        # 使用Thinking States进行隐式推理
        latent_representations = []
        
        for messages in messages_list:
            # 转换为输入格式
            input_text = self.format_messages(messages)
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # 生成思维增强的表示
            thought_enhanced = self.thinking_states(input_ids)
            latent_representations.append(thought_enhanced)
            
        return latent_representations
```

### 优势
1. **计算效率**: 比显式CoT更高效
2. **知识保留**: 思维过程不占用上下文窗口
3. **可扩展性**: 支持更深层的隐式推理链

## 局限性与挑战
1. **思维质量**: 依赖于思维生成的监督信号质量
2. **调试困难**: 隐式思维过程较难分析和调试
3. **理论保证**: 缺乏严格的收敛性理论分析

## 与LatentMAS Tool Use的联系
Thinking States为解决"tensor mismatch"问题提供了新思路：
- 将离散工具调用转换为连续的思维状态
- 通过embedding空间对齐实现隐式工具选择
- 为异步交叉注意力注入提供理论基础