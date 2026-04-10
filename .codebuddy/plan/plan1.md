# LatentMAS 异步Cross-Attention注入方案实施计划

## 项目背景

**LatentMAS** 是在潜在空间进行多智能体协作的推理框架，核心优势：
- Token减少70-80%
- 推理加速4-7倍
- 无需训练即可部署

**当前实验结果**：
- Baseline: 30%准确率，44.05秒/样本
- LatentMAS: 30%准确率，42.34秒/样本

## 核心挑战：Latent空间Tool Use

Latent空间无法直接支持Tool Use的三大技术难点：

1. **离散性断裂** - API需要精确字符，Latent→Discrete转换不可导
2. **观察值注入污染** - 工具返回离散结果编码回Latent空间引发语义偏移
3. **因果链严苛性** - 工具调用对Token精度零容忍

## 解决方案：融合Gist机制的异步Cross-Attention注入

### 理论基础

结合两篇论文的思想：
1. **research-directions.md的异步Cross-Attention注入** - 工具结果作为Side-input注入
2. **Gist Tokens论文(arXiv:2304.08467)** - 通过注意力掩码压缩信息

### 核心思想

将工具调用的离散指令压缩成gist-like representations，通过Cross-Attention注入到latent推理中：

```
工具调用流程:
1. 检测工具调用意图 → 2. 执行工具 → 3. 结果编码为gist向量 → 4. Cross-Attention注入
```

## 技术方案

### 方案架构

```
LatentMAS + Tool Support
├── Tool Detection Layer    # 检测是否需要工具调用
├── Tool Executor          # 执行Python/计算工具
├── Gist Encoder           # 将工具结果编码为连续向量
├── Cross-Attention Injector # 注入工具结果到latent stream
└── Latent Backbone        # 保持原有latent推理流程
```

### 关键组件设计

#### 1. Tool Detection Layer
```python
# 在latent推理过程中检测工具调用意图
class ToolDetector:
    def detect_tool_intent(self, hidden_state):
        # 通过训练一个分类头或使用阈值判断
        # 返回: (need_tool, tool_type)
        pass
```

#### 2. Gist Encoder
```python
# 参考gist论文，将离散工具结果编码为连续表示
class GistEncoder:
    def encode_tool_result(self, result_text):
        # 使用模型的embedding层 + 轻量级编码器
        # 返回: gist_vectors [1, D] 或 [k, D]
        pass
```

#### 3. Cross-Attention Injector
```python
# 异步注入工具结果
class CrossAttentionInjector:
    def inject(self, H_latent, K_tool, V_tool):
        # Cross-Attention: query=H_latent, key=K_tool, value=V_tool
        H_refined = cross_attention(H_latent, K_tool, V_tool)
        return H_refined
```

## 实施计划

### Phase 1: 基础框架搭建

**目标**: 实现最小可行版本

**修改文件**:
- `models.py`: 添加GistEncoder和CrossAttentionInjector类
- 新建 `tool_executor.py`: 实现安全的Python代码执行器

**步骤**:
1. 实现ToolExecutor类 - 安全执行Python代码
   - 使用subprocess隔离执行
   - 设置超时限制
   - 捕获stdout输出

2. 实现GistEncoder类
   - 使用模型的tokenizer编码工具结果
   - 通过embedding层获取初始表示
   - 可选：添加小型MLP投影层

3. 实现CrossAttentionInjector
   - 使用PyTorch的MultiheadAttention
   - 残差连接保持latent稳定性

### Phase 2: 模型扩展

**修改文件**:
- `models.py`: 扩展ModelWrapper类

**步骤**:
1. 修改`generate_latent_batch`函数
   - 添加工具检测逻辑
   - 支持异步工具执行
   - 集成Cross-Attention注入

```python
def generate_latent_batch_with_tools(self, ..., tool_executor=None):
    for step in range(latent_steps):
        # 标准latent推理
        last_hidden = self._latent_step(...)
        
        # 工具检测
        if tool_executor:
            need_tool, tool_type = self.detect_tool(last_hidden)
            if need_tool:
                # 执行工具
                tool_result = tool_executor.execute(...)
                # 编码为gist向量
                gist_vec = self.gist_encoder(tool_result)
                # Cross-Attention注入
                last_hidden = self.cross_attn_injector(
                    last_hidden, gist_vec
                )
```

### Phase 3: 方法集成

**修改文件**:
- `methods/latent_mas.py`: 扩展LatentMASMethod类

**步骤**:
1. 在`__init__`中初始化工具相关组件
2. 修改`run_batch`方法
   - 为需要计算的agent启用工具调用
   - 管理工具调用队列

3. 调整agent提示词
   - 告知agent可以使用Python计算工具
   - 指定工具调用格式

### Phase 4: 测试验证

**测试文件**:
- `run_test_toolcalling.py`: 已有测试脚本

**步骤**:
1. 运行基础测试
   - 验证工具执行安全性
   - 测试gist编码质量

2. 端到端测试
   - 运行test_dataset_toolcalling.json
   - 对比改进前后准确率

3. 性能评估
   - 测量推理延迟变化
   - 评估token节省效果

## 预期成果

### 性能指标
- 准确率提升: 30% → 50%+ (目标)
- 推理时间: 保持42-45秒区间
- Token使用: 维持70-80%减少

### 技术贡献
1. 首次在Latent空间实现工具调用
2. 融合Gist压缩机制与Cross-Attention注入
3. 保持Latent连续性同时支持离散交互

## 风险与缓解

### 潜在风险
1. **Gist编码质量** - 工具结果可能无法准确编码
   - 缓解: 使用多层编码器，增加gist token数量

2. **工具检测准确性** - 可能误触发或不触发
   - 缓解: 使用明确的触发规则，如检测"计算"关键词

3. **Cross-Attention干扰** - 可能破坏latent推理流
   - 缓解: 使用残差连接和层归一化

### 备选方案
如果效果不佳，可回退到方案B（潜空间工具索引）：
- 将工具向量化
- 通过相似度匹配选择工具
- 避免显式的工具结果编码

## 关键文件清单

```
LatentMAS/
├── models.py                    # 核心修改: 添加工具支持
├── tool_executor.py             # 新建: 工具执行器
├── methods/
│   └── latent_mas.py           # 修改: 集成工具调用
├── run_test_toolcalling.py     # 测试脚本
├── data/
│   └── test_dataset_toolcalling.json  # 测试数据
└── plan/
    └── plan1.md                # 本计划文件
```

## 参考资料

1. research-directions.md - 四种解决方案详述
2. arXiv:2304.08467 - Gist Tokens论文
3. arXiv:2302.13971 - LLaMA论文（模型架构参考）

---

*计划创建时间: 2026-03-13*
