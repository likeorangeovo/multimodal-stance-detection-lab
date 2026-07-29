# Day12 学习总结：Transformer 与 BERT 架构核心

## 1. Transformer 核心思想

Transformer 是一种基于注意力机制的序列建模架构。它不依赖 RNN 的顺序递归，而是通过 self-attention 让序列中任意两个 token 可以直接建立联系。

核心目标：

- 建模 token 与 token 之间的依赖关系。
- 支持并行计算，提高训练效率。
- 更好地捕捉长距离语义关系。

一句话理解：

```text
Transformer 用 self-attention 替代循环结构，让每个 token 根据上下文重新生成自己的表示。
```

## 2. Transformer 整体架构

原始 Transformer 是 Encoder-Decoder 架构，常用于机器翻译、摘要等序列到序列任务。

```text
输入序列
→ Transformer Encoder
→ 编码后的上下文表示
→ Transformer Decoder
→ 输出序列
```

Encoder 负责理解输入，Decoder 负责生成输出。

## 3. Transformer Encoder 架构

Transformer Encoder 由多个相同的 Encoder Block 堆叠而成。

单个 Encoder Block：

```text
输入向量
→ Multi-Head Self-Attention
→ Add & Norm
→ Feed-Forward Network
→ Add & Norm
→ 输出向量
```

主要模块：

| 模块 | 作用 |
|---|---|
| Multi-Head Self-Attention | 让每个 token 关注输入序列中的其他 token |
| Add & Norm | 残差连接 + 层归一化，稳定训练 |
| Feed-Forward Network | 对每个 token 的表示进行非线性变换 |

Encoder 的特点：

- 每个 token 可以看到完整输入序列。
- 输出每个 token 的上下文表示，也就是一组高维特征向量。
- 适合理解类任务，如分类、匹配、实体识别。

## 4. Transformer Decoder 架构

Transformer Decoder 也由多个 Decoder Block 堆叠而成。

单个 Decoder Block：

```text
目标序列输入
→ Masked Multi-Head Self-Attention
→ Add & Norm
→ Encoder-Decoder Attention
→ Add & Norm
→ Feed-Forward Network
→ Add & Norm
→ 输出向量
```

主要模块：

| 模块 | 作用 |
|---|---|
| Masked Self-Attention | 只允许当前位置关注之前的 token，防止看到未来信息 |
| Encoder-Decoder Attention | 让 Decoder 读取 Encoder 的输出信息 |
| Feed-Forward Network | 增强每个位置的表示能力 |
| Add & Norm | 稳定训练，加快收敛 |

Decoder 的特点：

- 生成任务中按顺序预测下一个 token。
- 使用 mask 保证自回归生成。
- 适合文本生成、翻译、摘要等任务。

## 5. Self-Attention 核心机制

Self-attention 的作用是计算同一序列中不同 token 之间的相关性。

每个 token 会生成三个向量：

| 向量 | 含义 |
|---|---|
| Query | 当前 token 想查询什么信息 |
| Key | 每个 token 提供的匹配特征 |
| Value | 每个 token 真正提供的信息内容 |

计算过程：

```text
1. 用 Query 和 Key 计算相关性分数
2. 对分数做缩放和 softmax，得到注意力权重
3. 用注意力权重对 Value 加权求和
4. 得到当前 token 的上下文表示
```

公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

直观理解：

```text
一个 token 会根据与其他 token 的相关性，从整个序列中收集有用信息。
```

## 6. Multi-Head Attention

Multi-head attention 是多个 attention head 的并行组合。

每个 head 都会独立学习一组 Q、K、V 投影，从不同角度捕捉关系。

可能学习到的关系：

- 语法依赖。
- 指代关系。
- 关键词关联。
- 长距离语义关系。
- 局部上下文模式。

核心意义：

```text
单个注意力头只能从一个表示空间建模关系，多头注意力可以同时从多个子空间理解文本。
```

## 7. Positional Encoding

Transformer 的 self-attention 本身不包含顺序信息。如果不加入位置编码，模型无法区分 token 的先后顺序。

因此需要把位置信息加入输入表示：

```text
Transformer 输入 = Token Embedding + Positional Encoding
```

位置编码的作用：

- 表示 token 在序列中的位置。
- 帮助模型理解词序。
- 让模型区分相同 token 出现在不同位置时的意义。

## 8. Transformer 训练流程

原始 Transformer 常用于序列到序列任务。训练时通常使用 teacher forcing，也就是 Decoder 在每一步接收真实目标序列的前缀，而不是自己上一步生成的结果。

训练流程：

```text
源序列
→ Token Embedding + Positional Encoding
→ Encoder
→ Encoder Hidden States

目标序列右移一位
→ Token Embedding + Positional Encoding
→ Masked Decoder Self-Attention
→ Encoder-Decoder Attention
→ Decoder Hidden States
→ Linear + Softmax
→ 预测下一个 token
→ 与真实目标 token 计算 Cross-Entropy Loss
→ 反向传播更新参数
```

训练时 Decoder 的输入和目标：

```text
真实目标：      I love deep learning [EOS]
Decoder 输入： [BOS] I love deep learning
Decoder 目标： I love deep learning [EOS]
```

核心目的：

```text
让模型学习在给定源序列和目标前缀的条件下，预测下一个 token。
```

## 9. Transformer 推理流程

推理时没有真实目标序列，因此 Decoder 必须自回归生成。

推理流程：

```text
源序列
→ Encoder
→ Encoder Hidden States

[BOS]
→ Decoder
→ 预测第 1 个 token
→ 将预测 token 拼回输入
→ Decoder 再预测下一个 token
→ 重复直到生成 [EOS] 或达到最大长度
```

推理特点：

- 一次生成一个 token。
- 每一步只能看到已经生成的 token。
- 通过 mask 保证不会使用未来信息。
- 常见解码策略包括 greedy search、beam search、top-k sampling、top-p sampling。

## 10. BERT 核心思想

BERT 全称是 Bidirectional Encoder Representations from Transformers，即“来自 Transformer 的双向编码器表示”。

BERT 的核心是：

```text
BERT = 多层 Transformer Encoder 堆叠
```

它只使用 Transformer 的 Encoder，不使用 Decoder。

BERT 的主要特点：

- Encoder-only 架构。
- 双向上下文建模。
- 基于大规模无标注文本预训练。
- 通过微调用于下游 NLP 任务。
- 输出上下文相关的 token 表示。

一句话理解：

```text
BERT 是一个通用文本特征提取器，Encoder 输出高维特征，任务 Head 把特征映射成具体概率。
```

## 11. BERT 架构

BERT 的输入先转成 embedding，然后经过多层 Transformer Encoder。

整体结构：

```text
输入文本
→ Tokenization
→ Token Embedding + Segment Embedding + Position Embedding
→ 多层 Transformer Encoder
→ 每个 token 的 hidden state
→ 任务 Head
→ logits / probabilities
```

BERT 输入表示由三部分组成：

| Embedding | 作用 |
|---|---|
| Token Embedding | 表示每个 token 本身 |
| Segment Embedding | 区分句子 A 和句子 B |
| Position Embedding | 表示 token 的位置 |

因此：

```text
BERT 输入表示 = Token Embedding + Segment Embedding + Position Embedding
```

## 12. BERT 特殊符号

| 符号 | 作用 |
|---|---|
| `[CLS]` | 放在序列开头，常用于整句分类 |
| `[SEP]` | 分隔句子或文本片段 |
| `[MASK]` | 预训练时遮蔽 token，用于 MLM 预测 |
| `[PAD]` | 补齐序列长度 |

单句输入：

```text
[CLS] 句子 A [SEP]
```

文本对输入：

```text
[CLS] 句子 A [SEP] 句子 B [SEP]
```

其中 `[CLS]` 的最终 hidden state 常用于文本分类或句子关系判断。

## 13. BERT 的 Head：从特征到概率

Transformer Encoder 输出的是特征，也就是每个 token 对应的高维 hidden state。例如 BERT-base 中，每个位置通常输出一个 768 维向量。

这些特征本身不能直接回答“这个词是什么”“这句话是什么类别”。需要在 Encoder 上方接一个任务 Head，把特征空间映射到标签空间。

核心形式：

```text
hidden state
→ Linear
→ logits
→ Softmax
→ probabilities
```

可以理解为：

```text
Encoder 是通用大脑，输出抽象特征；
Head 是任务嘴巴，把抽象特征翻译成具体预测。
```

## 14. BERT 预训练流程

BERT 预训练阶段同时学习 MLM 和 NSP。整体流程不是“Encoder 直接给答案”，而是先构造预训练样本，再把样本转成模型需要的输入张量，最后通过两个任务 Head 计算损失。

完整流程：

```text
原始语料
→ 构造句子 A / 句子 B
→ 添加 [CLS] 和 [SEP]
→ 随机 mask 一些 token
→ 转成 input_ids / token_type_ids / attention_mask
→ BERT Encoder
→ MLM Head 预测被 mask 的词
→ NSP Head 判断 B 是否为 A 下一句
→ loss = MLM loss + NSP loss
→ 反向传播更新 BERT
```

此时 BERT Encoder 上方接两个任务 Head：

- MLM Head：预测被 `[MASK]` 遮住的 token。
- NSP Head：判断句子 B 是否是句子 A 的下一句。

### 预训练样本构造

BERT 原始预训练会从语料中构造两个句子：

- 句子 A：第一段文本。
- 句子 B：可能是真实下一句，也可能是随机抽取的其他句子。

然后加入特殊符号：

```text
[CLS] 句子 A [SEP] 句子 B [SEP]
```

如果句子 B 确实是句子 A 的下一句，NSP 标签是 `IsNext`；否则标签是 `NotNext`。

接着随机 mask 一部分 token，用于 MLM 任务。

示例：

```text
原始输入：[CLS] 我 吃 苹果 [SEP] 这是水果 [SEP]
mask 后： [CLS] 我 [MASK] 苹果 [SEP] 这是水果 [SEP]
MLM 标签：被 mask 的真实词是“吃”
NSP 标签：IsNext 或 NotNext
```

### 模型输入张量

构造好文本后，需要转成 BERT 真正接收的输入：

| 输入 | 作用 |
|---|---|
| `input_ids` | 每个 token 在词表中的编号 |
| `token_type_ids` | 区分 token 属于句子 A 还是句子 B，也叫 segment ids |
| `attention_mask` | 标记哪些是真实 token，哪些是 padding |

输入表示仍然由三类 embedding 相加：

```text
Token Embedding + Segment Embedding + Position Embedding
```

其中：

- `input_ids` 进入 Token Embedding。
- `token_type_ids` 进入 Segment Embedding。
- position ids 进入 Position Embedding。
- `attention_mask` 控制 attention 是否关注 padding 位置。

### Encoder 前向传播

输入张量经过多层 Transformer Encoder 后，每个位置都会输出一个 hidden state。

```text
input_ids / token_type_ids / attention_mask
→ Token Embedding + Segment Embedding + Position Embedding
→ 多层 Transformer Encoder
→ 每个位置输出 hidden state
```

输出特征示例：

```text
H_[CLS]
H_我
H_[MASK]
H_苹果
H_[SEP]
...
```

### MLM Head：预测被 mask 的词

MLM 取出被遮蔽位置的 hidden state，例如 `H_[MASK]`。

假设 hidden size 是 768，词表大小是 30000：

```text
H_[MASK]：1 × 768
W_mlm：768 × 30000

logits = H_[MASK] · W_mlm
logits 维度 = 1 × 30000
```

再经过 softmax，得到词表中每个词的概率。

```text
H_[MASK]
→ Linear
→ 30000 个 logits
→ Softmax
→ 30000 个词的概率分布
→ 预测概率最高的词
```

训练时，用预测分布和真实被遮住的词计算交叉熵损失。

### NSP Head：判断句子 B 是否为 A 的下一句

NSP 取出 `[CLS]` 位置的 hidden state，即 `H_[CLS]`。

假设 hidden size 是 768，NSP 是二分类：

```text
H_[CLS]：1 × 768
W_nsp：768 × 2

logits = H_[CLS] · W_nsp
logits 维度 = 1 × 2
```

再经过 softmax，得到两个概率：

```text
IsNext
NotNext
```

训练时，用预测分布和真实 NSP 标签计算交叉熵损失。

### Loss 与反向传播

BERT 原始预训练总损失：

```text
total loss = MLM loss + NSP loss
```

反向传播会同时更新：

- MLM Head 的参数。
- NSP Head 的参数。
- BERT Encoder 的全部参数。

核心理解：

```text
Head 学会把特征映射到具体标签；
Encoder 学会产生更适合这些任务的上下文特征。
```

## 15. BERT 预训练推理

预训练推理常见形式是完形填空，也就是输入带 `[MASK]` 的句子，让 MLM Head 输出最可能的词。

流程：

```text
输入带 [MASK] 的句子
→ BERT Encoder
→ 取 [MASK] 位置的 hidden state
→ MLM Head
→ Softmax
→ 输出词表概率
→ 选择概率最高的 token
```

注意：

- 预训练推理不需要反向传播。
- 它只是前向计算并输出概率。
- 这不是 BERT 最常见的实际业务使用方式，更多是用来检查语言理解能力。

## 16. BERT 微调流程

预训练结束后，MLM Head 和 NSP Head 通常会被丢弃。下游任务会保留预训练好的 Encoder，然后换上新的任务 Head。

微调核心：

```text
预训练 Encoder
→ 换上新任务 Head
→ 用标注数据训练
→ 更新新 Head，并通常轻微更新 Encoder
```

以情感分类为例：

```text
输入：[CLS] 这家餐厅的菜真难吃 [SEP]
→ BERT Encoder
→ 得到 H_[CLS]
→ 分类 Head
→ logits
→ Softmax
→ 好评 / 差评概率
→ 与真实标签计算 loss
→ 反向传播更新参数
```

如果是二分类，分类 Head 可以理解为：

```text
H_[CLS]：1 × 768
W_cls：768 × 2

logits = H_[CLS] · W_cls
```

微调时会训练：

- 新任务 Head 的参数。
- BERT Encoder 的参数，通常是在预训练参数基础上小幅调整。

这个过程让通用语言特征适应具体领域和任务。

## 17. 不同任务的 Head

不同任务使用不同位置的 hidden state，也接不同形式的 Head。

| 任务 | 使用的特征 | Head 结构 | 输出 |
|---|---|---|---|
| 文本分类 / 情感分类 | `[CLS]` 的 hidden state | Linear → 类别数 | 每个类别的概率 |
| 命名实体识别 | 每个 token 的 hidden state | 每个 token 接 Linear → 标签数 | 每个 token 的标签 |
| 抽取式问答 | 所有 token 的 hidden state | 两个 Linear 分别预测 start / end | 答案起止位置 |
| 句子相似度 / 文本匹配 | `[CLS]` 的 hidden state | Linear → 类别或分数 | 相似度或匹配标签 |

核心结论：

```text
BERT Encoder 负责抽特征；
任务 Head 负责把特征变成任务需要的概率或分数。
```

## 18. BERT 微调推理流程

微调完成后，实际使用时通常不再输入 `[MASK]`，而是输入真实文本。

文本分类推理：

```text
输入文本
→ Tokenizer
→ [CLS] 文本 [SEP]
→ BERT Encoder
→ 取 H_[CLS]
→ 分类 Head
→ Softmax
→ 输出类别概率
→ 选择概率最大的类别
```

序列标注推理：

```text
输入文本
→ Tokenizer
→ BERT Encoder
→ 每个 token 的 hidden state
→ token 分类 Head
→ 输出每个 token 的标签
```

抽取式问答推理：

```text
[CLS] 问题 [SEP] 文章 [SEP]
→ BERT Encoder
→ 所有 token 的 hidden state
→ start Head + end Head
→ 选择概率最高的起止位置
→ 输出文章中的答案片段
```

推理特点：

- 只做前向传播，不反向传播。
- 不更新模型参数。
- 输出依赖下游任务 Head。
- BERT 不像 GPT 那样逐 token 自回归生成，而是一次性编码完整输入。

## 19. BERT 训练与推理总结

| 阶段 | 输入 | Head | 是否反向传播 | 输出 |
|---|---|---|---|---|
| 预训练训练 | 带 `[MASK]` 的文本对 | MLM Head + NSP Head | 是 | MLM loss + NSP loss |
| 预训练推理 | 带 `[MASK]` 的文本 | MLM Head | 否 | 被遮蔽词的概率 |
| 微调训练 | 下游任务标注数据 | 新任务 Head | 是 | 任务 loss |
| 微调推理 | 真实业务输入 | 新任务 Head | 否 | 标签、概率、答案位置等 |

最关键的闭环：

```text
预训练：学通用特征。
微调：学任务映射。
推理：用 Encoder 抽特征，用 Head 输出结果。
```

## 20. Transformer 与 BERT 的关系

| 对比项 | Transformer | BERT |
|---|---|---|
| 架构类型 | 原始结构是 Encoder-Decoder | Encoder-only |
| 核心机制 | self-attention | self-attention |
| 是否生成文本 | Decoder 可生成 | 原始 BERT 不适合直接生成 |
| 上下文方向 | Encoder 双向，Decoder 单向 mask | 双向 |
| 主要用途 | 翻译、生成、理解 | 文本理解 |
| 训练方式 | 根据任务训练 | 先预训练，再微调 |
| 推理方式 | Decoder 可自回归逐 token 生成 | 一次性编码完整输入，再由 Head 输出 |

关系总结：

```text
Transformer 是基础架构；
BERT 是基于 Transformer Encoder 的预训练语言模型。
```

## 21. 最短复习版

```text
Transformer：
由 Encoder 和 Decoder 组成，核心是 self-attention。
Self-attention 通过 Q、K、V 计算 token 间关系。
Multi-head attention 从多个子空间学习不同关系。
Positional encoding 提供顺序信息。
训练时常用 teacher forcing，用目标序列前缀预测下一个 token。
推理时 Decoder 自回归逐 token 生成。

BERT：
只使用 Transformer Encoder。
输入 = Token Embedding + Segment Embedding + Position Embedding。
Encoder 输出每个 token 的 hidden state，也就是上下文特征。
特征经过任务 Head 的 Linear + Softmax 后，映射成概率。
预训练时使用 MLM Head 和 NSP Head，loss = MLM loss + NSP loss。
微调时丢掉预训练 Head，换成新任务 Head。
推理时只做前向传播，不更新参数，输出分类、标签或答案位置。
```
