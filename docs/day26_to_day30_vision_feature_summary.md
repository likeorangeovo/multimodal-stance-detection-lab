# Day26-Day30：视觉特征提取阶段总结

这一阶段的目标是理解 ViT 的基本原理，掌握用预训练视觉模型提取图像特征的方法，并知道 CNN、ViT 和图像增强在后续多模态立场检测中的作用。

完成情况如下：

- [x] Day26 学习 ViT 原理，理解 Patch Embedding -> [ViT 官方实现](https://github.com/google-research/vision_transformer)
- [x] Day27 用 HuggingFace ViTModel 提取特征
- [x] Day28 对比 CNN 与 ViT 特征的差异
- [x] Day29 练习图像增强技术
- [x] Day30 打卡输出：一份视觉特征提取方法对比笔记

## 1. 为什么要做视觉特征提取

多模态立场检测通常输入不只有文本，还包含图片。文本模型负责理解观点、目标和情绪倾向，视觉模型负责提取图片中的对象、场景、符号、人物或事件线索。

在多模态模型中，图像不会直接以原始像素参与分类，而是先转成特征向量：

```text
image
-> image encoder
-> image feature
-> fusion module
-> stance classifier
```

因此这一阶段最重要的不是从零训练一个强视觉模型，而是理解并使用预训练视觉 backbone，例如 ResNet、ViT、CLIP image encoder。

## 2. ViT 核心思想

ViT 全称是 Vision Transformer。它把图像切成一系列 patch，然后把每个 patch 当作类似 NLP 里的 token，送入 Transformer Encoder。

整体流程：

```text
image
-> split into patches
-> patch embedding
-> add [CLS] token
-> add position embedding
-> Transformer Encoder
-> image representation
```

一句话理解：

```text
ViT 把图片变成 patch 序列，再用 Transformer 建模 patch 与 patch 之间的关系。
```

### Patch Embedding

Patch Embedding 是 ViT 最关键的入口。假设输入图像大小是 `224 x 224`，patch 大小是 `16 x 16`，那么图像会被切成：

```text
(224 / 16) * (224 / 16) = 14 * 14 = 196 个 patch
```

每个 patch 会被展平并映射成一个向量：

```text
patch: 16 x 16 x 3
-> flatten
-> linear projection
-> patch embedding
```

最后得到一串视觉 token：

```text
[patch_1, patch_2, ..., patch_196]
```

再加上 `[CLS]` token 和位置编码：

```text
[CLS], patch_1, patch_2, ..., patch_196
-> Transformer Encoder
```

`[CLS]` 的输出通常作为整张图像的全局表示，类似 BERT 中 `[CLS]` 表示整句语义。

## 3. 用 HuggingFace ViTModel 提取特征

HuggingFace 的 `ViTModel` 可以直接作为图像特征提取器。常见流程如下：

```text
image
-> image processor
-> pixel_values
-> ViTModel
-> last_hidden_state / pooler_output
```

核心输出：

| 输出 | 含义 | 常见用途 |
| --- | --- | --- |
| `last_hidden_state` | 每个 patch token 的 hidden state | 做局部 patch 分析或 cross-attention |
| `pooler_output` | 图像级表示 | 做分类、匹配、晚期融合 |
| `[CLS]` hidden state | 第一个 token 的表示 | 常作为整图特征 |

用于多模态立场检测时，最简单的做法是取图像级向量：

```text
image
-> ViTModel
-> image_feature
```

然后和文本特征拼接：

```text
text_feature + image_feature
-> fusion classifier
-> AGAINST / FAVOR / NONE
```

## 4. CNN 与 ViT 的差异

CNN 和 ViT 都可以作为图像 backbone，但它们建模图像的方式不同。

| 对比项 | CNN | ViT |
| --- | --- | --- |
| 基本单位 | 局部卷积窗口 | 图像 patch |
| 归纳偏置 | 强，天然关注局部纹理和平移不变性 | 弱，更依赖数据和预训练 |
| 全局关系 | 需要多层堆叠逐渐扩大感受野 | self-attention 可以直接建模远距离 patch 关系 |
| 小数据场景 | 通常更稳 | 更依赖预训练 |
| 可解释视角 | feature map / Grad-CAM | attention map / patch token |
| 多模态接入 | 取 CNN 全局池化特征 | 取 `[CLS]` 或 patch token 特征 |

直观理解：

```text
CNN 更像从局部纹理逐层组合出高级语义；
ViT 更像把图片切成视觉 token 后做全局关系建模。
```

在多模态立场检测中：

- 如果数据量小，ResNet 这类 CNN backbone 往往是稳定 baseline。
- 如果想和文本 Transformer 做结构统一，ViT 更自然。
- 如果想利用图文预训练能力，CLIP image encoder 通常比单独 ViT 更贴近任务。

## 5. 图像增强的作用

图像增强用于提升模型鲁棒性，降低模型对颜色、裁剪、尺度、亮度等偶然因素的依赖。

常见增强方式：

| 增强方式 | 作用 |
| --- | --- |
| Resize / CenterCrop | 统一输入尺寸 |
| RandomCrop | 提升对局部裁剪的鲁棒性 |
| RandomHorizontalFlip | 增加左右翻转样本 |
| ColorJitter | 增强对亮度、对比度、颜色变化的适应 |
| Normalize | 匹配预训练模型输入分布 |

但在立场检测任务中，图像增强要谨慎。过强增强可能破坏图像中的文字、标志、人物表情或事件线索。

建议：

```text
训练阶段：使用轻量增强，如 resize、random crop、horizontal flip、color jitter。
验证/测试阶段：只做确定性预处理，如 resize、center crop、normalize。
```

如果图片里包含文字、标语或截图，尽量避免过强裁剪和模糊处理。

## 6. 视觉特征提取方法对比

| 方法 | 适合场景 | 优点 | 注意点 |
| --- | --- | --- | --- |
| ResNet 特征 | 快速建立图像 baseline | 稳定、简单、小数据友好 | 全局语义可能不够细 |
| ViT 特征 | Transformer 风格视觉建模 | 能建模全局 patch 关系 | 更依赖预训练和规范输入 |
| CLIP image encoder | 图文相关任务 | 图文空间天然对齐 | 需要注意文本 prompt 和相似度设计 |
| Patch token 特征 | cross-attention 融合 | 保留局部区域信息 | 计算量比单个全局向量更大 |

后续项目中可以按复杂度逐步推进：

```text
ResNet global feature
-> ViT [CLS] feature
-> CLIP image/text embedding
-> patch token + text token cross-attention
```

## 7. 对多模态立场检测的接入方式

最简单的多模态 baseline：

```text
tweet + target
-> BERT / RoBERTa / DeBERTa
-> text_feature

image
-> ResNet / ViT
-> image_feature

[text_feature, image_feature]
-> MLP classifier
-> stance label
```

如果加入 CLIP 相似度：

```text
image
-> CLIP image encoder
-> image_embedding

tweet / target prompt
-> CLIP text encoder
-> text_embedding

cosine_similarity(image_embedding, text_embedding)
-> 作为额外特征加入分类器
```

如果使用 cross-attention：

```text
text tokens
-> text encoder
-> text hidden states

image patches
-> ViT
-> patch hidden states

text hidden states <-> patch hidden states
-> cross-attention fusion
-> classifier
```

## 8. 阶段结论

这一阶段的核心收获：

- ViT 的关键是把图像切成 patch，并把 patch 当作 token 交给 Transformer Encoder。
- Patch Embedding 负责把局部图像块映射到模型能处理的向量空间。
- CNN 和 ViT 都能提取视觉特征，CNN 更稳，ViT 更适合全局关系建模。
- 图像增强能提升鲁棒性，但在立场检测中不能破坏文字、标志和关键视觉线索。
- 后续多模态立场检测不需要继续单独深挖 ViT，从能用预训练视觉 backbone 提取特征开始即可。

最短复习版：

```text
ViT = image patches + patch embedding + position embedding + Transformer Encoder。
CNN 偏局部纹理和稳定 baseline，ViT 偏全局 patch 关系。
多模态立场检测中，视觉模型主要负责把 image 转成 image_feature。
下一步重点不是继续训练视觉模型，而是把 image_feature 和 text_feature 融合起来。
```
