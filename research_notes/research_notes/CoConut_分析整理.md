# CoConut (Chain of Continuous Thought) 分析整理

## 论文信息
- **实际文件**: `/research_notes/paper/coconut-2412.06781.txt` 
- **内容**: "Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation"
- **真实主题**: 基于扩散模型和流匹配的视觉地理定位方法
- **arXiv**: 2412.06781

**注意**: 文件名可能为误标，实际内容与连续思维链无关。

## 核心方法

### 1. 地理扩散 (Geographic Diffusion)
```python
# 扩散过程
xt = sqrt(1-κ(t)) * x0 + sqrt(κ(t)) * ε
# 训练目标: 预测噪声 ε
loss = E[||ψ(xt|c) - ε||²]
```

### 2. 流匹配 (Flow Matching)
```python
# R³空间流匹配
xt = (1-κ(t))*x0 + κ(t)*ε
v(xt) = dxt/dt = κ'(t)*(ε-x0)
loss = E[||ψ(xt|c) - v(xt)||²]
```

### 3. 黎曼流匹配 (Riemannian Flow Matching)
```python
# 球面上直接操作
xt = exp_x0(κ(t) * log_x0(ε))
v(xt) = κ'(t) * D(xt)
```

## 实验结果

| 数据集 | 方法 | GeoScore | 距离(km) | 准确率(%) |
|--------|------|----------|----------|-----------|
| OSV-5M | RFM-S² | 3767 | 1069 | 76.2 |
| YFCC | RFM-S² | 2889 | 2461 | 23.7 |
| iNat21 | RFM-S² | 3210 | 2058 | 33.5 |

## 创新点
1. **首个生成式视觉地理定位方法**
2. **直接在球面上操作** (黎曼流匹配)
3. **概率地理定位任务** (预测位置分布而非单点)

---

# LatentMAS 相关文献真实分析

## 已确认的核心论文

### 1. LatentMAS: Latent Collaboration in Multi-Agent Systems
- **文件**: `compact-context-2511.20639.pdf/.txt`
- **作者**: Zou et al. (Princeton/UIUC/Stanford, 2025)
- **核心**: 多智能体在潜在空间协作，KV Cache 直接传递

### 2. Quiet-STaR: Recursive Internal Processing  
- **文件**: `quiet-star-2403.09629.pdf/.txt`
- **作者**: Zelikman et al. (Stanford, 2024)
- **核心**: O(1)内存的递归内部推理

### 3. Gist Tokens: Learning to Compress Prompts
- **文件**: `gist-2304.08467.pdf/.txt` 
- **作者**: Ge et al. (Meta, 2023)
- **核心**: 信息压缩到紧凑令牌

### 4. Communicate Latent Space
- **文件**: `communicate_latent_space.pdf`
- **推测**: 潜在空间通信机制

## 技术分析总结

LatentMAS 的核心创新是将离散的符号推理转化为连续的潜在推理，通过 KV Cache 的无损传递实现高效的多智能体协作。主要挑战是如何在保持潜在空间连续性的同时支持离散的工具调用。