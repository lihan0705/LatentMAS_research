# Gist Tokens 论文分析

## 基本信息
- **论文**: Learning to Compress Prompts with Gist Tokens
- **作者**: Jesse Mu, Xiang Lisa Li, Noah Goodman (Stanford University)
- **发表**: NeurIPS 2023, arXiv:2304.08467v3

## 核心概念
Gist Tokens是一种提示压缩技术，将长提示压缩为更小的"要点"令牌集合，可以缓存和重用以提高计算效率。

### 关键洞察
1. **提示效率低**: 重复编码相同提示在上下文窗口中占用宝贵空间
2. **微调成本高**: 传统微调需要为每个任务重新训练模型
3. **压缩优势**: 通过元学习方法训练模型将任意提示压缩为短token集合

## 方法详解

### Gisting过程
1. 在指令微调期间插入虚拟"gist tokens"
2. 修改注意力掩码，防止gist tokens后的token关注gist tokens前的token
3. 模型同时学习提示压缩和指令遵循

### 训练策略
- 无额外训练成本：在标准指令微调基础上简单修改注意力掩码
- 元学习方法：给定提示预测gist前缀，零样本泛化到未见指令

## 实验结果
- **压缩比**: 高达26倍压缩
- **计算节省**: 高达40% FLOPs减少
- **延迟提升**: 4.2%墙钟时间加速
- **质量保持**: 输出质量损失最小

## 伪代码实现

```python
class GistTokenizer:
    def __init__(self, model, num_gist_tokens=8):
        self.model = model
        self.num_gist_tokens = num_gist_tokens
        
    def compress_prompt(self, prompt):
        """将长提示压缩为gist tokens"""
        # 在提示后插入gist token占位符
        gist_placeholders = [f"<G{i}>" for i in range(self.num_gist_tokens)]
        extended_prompt = prompt + " " + " ".join(gist_placeholders)
        
        # 前向传播获取gist representations
        outputs = self.model(extended_prompt, output_hidden_states=True)
        
        # 提取gist token位置的激活作为压缩表示
        gist_embeddings = outputs.hidden_states[-1][:, -self.num_gist_tokens:]
        
        return gist_embeddings
    
    def reconstruct_response(self, gist_embeddings, input_text):
        """使用gist embeddings生成响应"""
        # 将gist embeddings与输入文本拼接
        combined_input = self._combine_embeddings(gist_embeddings, input_text)
        
        # 自回归生成
        response = self.model.generate(combined_input)
        return response

def train_gist_model(model, dataset, epochs=3):
    """训练gist模型"""
    optimizer = AdamW(model.parameters())
    
    for epoch in range(epochs):
        for prompt, input_text, target_output in dataset:
            # 插入gist token位置
            gist_start = len(prompt.split())
            
            # 修改注意力掩码
            attention_mask = create_gist_attention_mask(
                len(prompt.split()), len(input_text.split()), model.config.num_gist_tokens
            )
            
            # 前向传播
            outputs = model(
                prompt + " " + input_text,
                attention_mask=attention_mask,
                labels=target_output
            )
            
            # 反向传播
            loss = outputs.loss
            loss.backward()
            optimizer.step()

def create_gist_attention_mask(prompt_len, input_len, num_gist=8):
    """创建gist专用注意力掩码"""
    total_len = prompt_len + input_len + num_gist
    mask = torch.ones(total_len, total_len)
    
    # gist tokens不能关注前面的内容
    gist_start = prompt_len
    for i in range(gist_start, gist_start + num_gist):
        for j in range(i):
            mask[i, j] = 0
    
    return mask
```

## 在LatentMAS中的应用潜力
1. **上下文压缩**: 将多轮对话压缩为gist tokens存储
2. **工具调用优化**: 将复杂工具描述压缩为紧凑表示
3. **跨代理通信**: 代理间传递gist而非完整推理链

## 局限性
- 压缩质量依赖训练数据分布
- 极端压缩可能影响推理准确性
- 需要额外的gist token管理逻辑