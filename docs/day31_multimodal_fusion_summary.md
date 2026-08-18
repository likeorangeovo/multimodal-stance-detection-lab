# Day31：多模态融合范式总结

本篇压缩总结早期融合、晚期融合、cross-attention、单流结构和双流结构的关系，目标是为后续多模态立场检测实验确定一条稳妥路线。

完成情况如下：

- [x] 梳理 early fusion / middle fusion / late fusion 的区别
- [x] 理清 cross-attention 与 fusion stage 的关系
- [x] 对比 single-stream 与 dual-stream 结构
- [x] 总结多模态立场检测中的推荐建模路线
- [x] 明确后续实验优先级

## 1. 核心概念

| 概念 | 回答的问题 | 一句话理解 |
| --- | --- | --- |
| 早期融合 | 什么时候融合 | 不同模态很早进入同一建模过程，交互充分但计算和噪声风险更高 |
| 中期融合 | 什么时候融合 | 各模态先独立编码一部分，再进行跨模态交互，是常见折中方案 |
| 晚期融合 | 什么时候融合 | 各模态先独立判断或提取高级语义，最后合并，简单稳定 |
| cross-attention | 怎么融合 | 一个模态作为 query，选择性读取另一个模态的 key/value 信息 |
| 单流结构 | 网络怎么组织 | 文本 token、图像 token 等进入同一个共享主干 |
| 双流结构 | 网络怎么组织 | 文本和图像各有自己的 encoder，再做交互或融合 |

这几个概念不在同一个维度上。更准确的拆法是：

```text
融合阶段：early / middle / late
融合机制：concat / gating / cross-attention / co-attention / bilinear / MoE
网络结构：single-stream / dual-stream / hybrid
```

## 2. 早期融合与晚期融合

早期融合让不同模态尽早互相看到。例如把文本 token 与图像 patch token 拼在一起送入 Transformer：

```text
[CLS], target tokens, text tokens, image patches
        -> multimodal Transformer
        -> classifier
```

优点是细粒度交互强，适合图文问答、图文推理、讽刺/冲突理解等需要词-区域对齐的任务。缺点是计算开销大，对数据量和模态质量要求高，图像噪声可能过早影响文本判断。

晚期融合则让每个模态先独立编码：

```text
target + text -> text encoder  -> text_repr
image         -> image encoder -> image_repr
text_repr + image_repr -> fusion classifier
```

优点是实现简单、训练稳定、方便替换 encoder，也更适合数据量不大或图像信号不稳定的场景。缺点是跨模态细节交互弱，可能只能学到全局向量拼接关系。

## 3. Cross-Attention

cross-attention 是一种融合机制，不是融合阶段。

标准形式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

文本关注图像：

```text
Q = text tokens
K = image tokens
V = image tokens
```

含义是每个文本 token 主动询问图像中哪些区域、patch 或视觉线索对自己有用。也可以反过来做图像关注文本：

```text
Q = image tokens
K = text tokens
V = text tokens
```

在立场检测中，更贴合任务的做法是 target-aware cross-attention：

```text
target tokens -> Q
text tokens   -> K,V -> target-aware text_repr

target tokens -> Q
image tokens  -> K,V -> target-aware image_repr
```

因为 stance detection 不是单纯判断文本情绪，而是判断文本和图像对某个 target 的态度。

## 4. 单流与双流

单流结构把不同模态放进同一个共享主干：

```text
target tokens + text tokens + image tokens
        -> shared multimodal Transformer
        -> stance classifier
```

它通常交互充分，适合深层图文理解。但在小数据任务中，训练成本和过拟合风险更高。

双流结构保留各模态自己的 encoder：

```text
target + text -> text encoder
image         -> image encoder
        -> fusion / cross-attention / classifier
```

它更模块化，方便使用 BERT/RoBERTa/DeBERTa、ResNet/ViT/CLIP 等预训练模型。CLIP 这类双编码器主要通过图文相似度或共享语义空间对齐，而不一定做深层 cross-attention。

需要注意：

```text
单流不等于早期融合。
双流不等于晚期融合。
```

单流可以先提取特征再统一建模；双流也可以在中间层加入多层 cross-attention。

## 5. 对多模态立场检测的判断

多模态立场检测通常有几个特点：

- 文本往往是最强主信号。
- 图像可能提供人物、事件、场景、符号、截图文字或梗图线索。
- 图像也可能只是装饰或噪声。
- target 很关键，同一句话面对不同 target 可能有不同 stance。
- 数据规模通常不足以从零训练重型多模态模型。

因此不建议一上来就做很重的单流早期融合。更稳妥的路线是先把文本 baseline 做扎实，再逐步引入图像特征和跨模态交互。

推荐实验优先级：

```text
1. 文本单模态 baseline
2. 图像单模态 baseline
3. BERT/RoBERTa + ResNet/ViT/CLIP late fusion
4. gated fusion，控制图像噪声影响
5. CLIP 图文相似度特征增强
6. target-aware cross-attention
7. 单流 multimodal Transformer 或更重的 VLM 方法
```

## 6. 推荐结构

后续项目可优先实现一个双流 + 门控 + target-aware cross-attention 的中等复杂度模型：

```text
target + text -> text encoder  -> text tokens, text_cls
image         -> image encoder -> image tokens, image_cls

target_text  = CrossAttention(Q=target, K=text,  V=text)
target_image = CrossAttention(Q=target, K=image, V=image)

fused = concat(text_cls, target_text, image_cls, target_image, clip_similarity)
gate  = sigmoid(MLP(fused))

final_repr = gated fusion
final_repr -> stance classifier
```

这个结构的好处是：

- 保留文本强基线，不让图像噪声完全主导。
- 图像以补充信息进入模型。
- target 显式参与融合，贴合 stance detection。
- 比完整单流 Transformer 更轻，适合当前项目阶段。
- 方便做消融实验：去掉图像、去掉 gate、去掉 cross-attention、去掉 CLIP 相似度。

## 7. 阶段结论

本阶段完成了多模态融合范式的理论梳理。当前项目下一步应进入可运行实验阶段：先做 CLIP/OpenCLIP 图文特征提取，再实现 late fusion baseline，随后加入 gated fusion 和 target-aware cross-attention 做对比。

最终路线保持为：

```text
文本 baseline
    -> 图像 baseline
    -> late fusion
    -> CLIP 相似度增强
    -> gated fusion
    -> target-aware cross-attention
    -> 消融实验与错误分析
```
