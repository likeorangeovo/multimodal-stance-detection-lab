# Day13-Day20：BERT 立场检测阶段总结

这一阶段的学习重点从立场检测任务理解，推进到基于 BERT 的文本立场分类实验。整体路线是：先探索 SemEval-2016 Task 6 数据集，再用 HuggingFace 搭建 BERT 分类 baseline，随后围绕类别不平衡、模型替换、Prompt-based 方法和论文阅读继续扩展。

完成情况如下：

- [x] Day13 下载并探索 SemEval-2016 Task6 数据集 -> [SemEval2016 Task6](https://alt.qcri.org/semeval2016/task6/)
- [x] Day14 阅读文本分类教程，加载 BERT 分类模型 -> [HuggingFace 文本分类教程](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [x] Day15 微调 BERT 做立场分类，跑通 baseline，记录 F1 打卡
- [x] Day16 分析错误样本，尝试加权损失处理不平衡
- [x] Day17 替换 RoBERTa/BERTweet 等模型对比 -> [HuggingFace Models](https://huggingface.co/models)
- [x] Day18 尝试 Prompt-based 立场检测（简单模板）
- [x] Day19 阅读一篇 BERT 立场检测改进论文
- [ ] Day20 阶段实战：整理代码，输出可视化小脚本

## 1. SemEval-2016 Task 6 数据集

Day13 使用 SemEval-2016 Task 6 作为立场检测实验数据集。任务形式为：

```text
输入：Target + Tweet
输出：AGAINST / FAVOR / NONE
```

项目中的数据路径：

```text
data/semeval2016_task6/semeval2016-task6-trainingdata.txt
data/semeval2016_task6/testdata-gold/SemEval2016-Task6-subtaskA-testdata-gold.txt
```

这个任务和普通情感分类不同。情感分类关注文本整体情绪，立场检测关注文本对特定目标的态度。文本本身可能是正面情绪，但对目标却是反对立场；也可能文本中没有直接出现目标，却仍然表达了明确倾向。

例如：

```text
Tweet: Great, this policy is finally cancelled.
Target: this policy
Stance: AGAINST
```

这类样本说明，立场检测需要同时理解文本内容和目标对象，不能只依赖情绪词判断。

## 2. BERT 文本分类基础

Day14 主要参考 HuggingFace 的 sequence classification 教程，完成 BERT 文本分类模型加载。

核心组件：

```python
AutoTokenizer
AutoModelForSequenceClassification
```

`AutoModelForSequenceClassification` 可以理解为：

```text
BERT encoder + classification head
```

在立场检测任务中，模型结构为：

```text
Target + Tweet
    -> tokenizer
    -> BERT encoder
    -> [CLS] representation
    -> classification head
    -> AGAINST / FAVOR / NONE
```

分类头负责把 BERT 输出的隐藏表示映射到 3 个立场标签。这个流程是后续 BERT baseline 的基础。

## 3. BERT Stance Baseline

Day15 完成了第一个可训练的 BERT 立场检测 baseline：

[day15_bert_stance_baseline.py](D:/Pycharm/multimodal-stance-detection-lab/src/day15_bert_stance_baseline.py)

使用模型：

```text
prajjwal1/bert-tiny
```

主要流程：

```text
读取 SemEval 数据
    -> 编码 Target + Tweet
    -> 构建 DataLoader
    -> 微调 BERT 分类模型
    -> 计算 accuracy 和 F1
    -> 输出错误样本
```

脚本中包含几个关键函数：

| 函数 | 作用 |
|---|---|
| `read_data()` | 读取 TSV 数据 |
| `build_dataset()` | 对 target 和 tweet 进行 tokenizer 编码 |
| `evaluate()` | 计算 accuracy、各类别 F1、macro F1、SemEval F1 |
| `main()` | 组织训练和评估流程 |

立场检测不能只看 accuracy。由于 `AGAINST`、`FAVOR`、`NONE` 的分布不均衡，各类别 F1 更能反映模型实际表现。SemEval 任务中常用的指标是：

```text
SemEval F1 = (F1_AGAINST + F1_FAVOR) / 2
```

这个指标更关注明确支持和明确反对的识别效果。

## 4. 错误样本与类别不平衡

Day16 在 Day15 baseline 基础上分析错误样本，并尝试使用加权损失处理类别不平衡。

SemEval 数据集中，不同立场标签数量并不完全均衡。训练过程中，模型容易偏向样本更多的类别，导致某些类别的召回率和 F1 偏低。

加权交叉熵的实现方式如下：

```python
class_weights = torch.tensor([
    len(train_label_ids) / (3 * counts[0]),
    len(train_label_ids) / (3 * counts[1]),
    len(train_label_ids) / (3 * counts[2]),
], dtype=torch.float).to(device)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
```

权重根据训练集中各类别数量计算。样本较少的类别会获得更高权重，从而在训练时对模型产生更强约束。

这一部分的重点在于建立基本实验分析流程：

```text
整体 accuracy
各类别 F1
SemEval F1
错误样本
类别分布
```

只有结合这些信息，才能判断模型是整体能力不足，还是对某些类别存在明显偏置。

## 5. RoBERTa 模型替换实验

Day17 将 BERT baseline 替换为 RoBERTa 系列模型：

[day17_roberta_model_demo.py](D:/Pycharm/multimodal-stance-detection-lab/src/day17_roberta_model_demo.py)

使用模型：

```text
distilbert/distilroberta-base
```

模型流程保持不变：

```text
Target + Tweet
    -> DistilRoBERTa
    -> classification head
    -> AGAINST / FAVOR / NONE
```

与 BERT 版本相比，RoBERTa 的主要实现差异是通常不使用 `token_type_ids`。因此 Day17 的 dataset 返回：

```text
input_ids
attention_mask
labels
```

这个实验的意义在于建立统一的模型对比模板。同一数据集、同一训练流程下替换 encoder，可以观察不同预训练模型在立场检测任务上的表现差异。后续如果继续尝试 BERTweet、DeBERTa 或其他模型，也可以沿用这个结构。

## 6. Prompt-based 立场检测

Day18 尝试了一个 zero-shot prompt-based 立场检测版本：

[day18_prompt_based_stance_detection.py](D:/Pycharm/multimodal-stance-detection-lab/src/day18_prompt_based_stance_detection.py)

配套笔记：

[day18_prompt_based_stance.md](D:/Pycharm/multimodal-stance-detection-lab/docs/day18_prompt_based_stance.md)

传统分类方式是：

```text
Target + Tweet
    -> encoder
    -> classification head
    -> stance label
```

Prompt-based 方法将分类任务改写为填空任务：

```text
Tweet: ...
Target: ...
The stance toward the target is <mask>.
```

然后使用 masked language model 预测 `<mask>` 位置的词。Day18 使用的模型为：

```text
distilbert/distilroberta-base
```

加载方式为：

```python
AutoModelForMaskedLM
```

因此这里使用的是：

```text
RoBERTa encoder + masked language modeling head
```

而不是：

```text
RoBERTa encoder + stance classification head
```

标签词映射如下：

```python
label_words = {
    "AGAINST": "against",
    "FAVOR": "favor",
    "NONE": "neutral",
}
```

预测过程：

```text
1. 构造包含 <mask> 的 prompt
2. 找到 <mask> 在输入序列中的位置
3. 取出该位置对整个词表的 logits
4. 只比较 against / favor / neutral 三个词的分数
5. 将分数最高的词映射回立场标签
```

需要注意的是，BERT/RoBERTa 并不是 next-token prediction 模型，而是 masked language model。Prompt-based BERT 预测的是 `<mask>` 位置的词，而不是预测下一个词。

当前 Day18 是 zero-shot evaluation，没有训练循环，也没有更新模型参数。因此运行时只会完成一次测试集推理，不会像 fine-tuning 脚本那样输出多个 epoch。

前 200 条测试样本的结果：

```text
accuracy: 0.1250
F1 AGAINST: 0.0000
F1 FAVOR: 0.0000
F1 NONE: 0.2222
macro F1: 0.0741
SemEval F1: 0.0000
wrong examples: 175
```

这个结果说明，简单 prompt 和简单 verbalizer 下，模型明显偏向 `NONE`。这也体现了 prompt-based 方法的敏感性：模板设计、标签词选择、是否加入示例、是否进行 prompt tuning，都会影响最终效果。

## 7. BERT 立场检测改进论文

Day19 阅读的论文是：

[Infusing Knowledge from Wikipedia to Enhance Stance Detection](https://ar5iv.labs.arxiv.org/html/2204.03839?_immersive_translate_auto_translate=1)

论文提出 WS-BERT，即 Wikipedia Stance Detection BERT。核心思想是给立场检测模型加入 target 的 Wikipedia 背景知识。

普通 BERT 立场检测通常只使用：

```text
Text + Target
```

WS-BERT 使用：

```text
Text + Target + Wikipedia Knowledge
```

论文认为，立场检测中很多判断依赖目标背景。尤其在政治人物、社会议题、公共卫生政策等话题中，只给模型一个 target 名称，并不一定足够。

论文提出两个模型版本：

| 模型 | 适用场景 | 结构 |
|---|---|---|
| WS-BERT-Single | 正式文本 | Text、Target、Wikipedia 一起输入一个 BERT |
| WS-BERT-Dual | 社交媒体文本 | 一个 encoder 编码 tweet + target，另一个 encoder 编码 Wikipedia，再拼接分类 |

WS-BERT 的主要结论：

```text
1. target 背景知识能够增强立场检测模型
2. Wikipedia summary 是简单有效的外部知识来源
3. 在 cross-target、zero-shot、few-shot 场景中提升更明显
4. 对训练集中没见过或样本很少的 target，外部知识尤其有帮助
```

这篇论文为后续实验提供了一个自然方向：在 SemEval target 上补充简短背景知识，将输入从 `Tweet + Target` 扩展为 `Tweet + Target + Knowledge`，再与 Day15 BERT baseline 对比。

一个简化版 target knowledge 可以先手写：

```python
target_knowledge = {
    "Atheism": "Atheism is the absence of belief in the existence of deities.",
    "Climate Change is a Real Concern": "Climate change refers to long-term shifts in temperatures and weather patterns.",
    "Feminist Movement": "The feminist movement advocates for women's rights and gender equality.",
    "Hillary Clinton": "Hillary Clinton is an American politician and former Democratic presidential candidate.",
    "Legalization of Abortion": "Abortion legalization concerns whether abortion should be permitted by law.",
}
```

## 阶段收获

这一阶段已经形成了一个比较完整的文本立场检测实验闭环：

```text
数据集探索
    -> BERT baseline
    -> 指标评估
    -> 错误分析
    -> 类别不平衡处理
    -> RoBERTa 模型替换
    -> Prompt-based 方法
    -> 论文改进方向
```

几个关键概念也逐渐明确：

| 概念 | 含义 |
|---|---|
| 立场检测 | 判断文本对某个 target 是支持、反对还是无明确立场 |
| BERT baseline | 使用 `[CLS]` 表示接分类头做三分类 |
| 加权损失 | 缓解类别不平衡导致的模型偏置 |
| RoBERTa 替换 | 保持训练流程基本不变，只替换 encoder |
| Prompt-based | 把分类任务改写成 `<mask>` 填空任务 |
| MLM head | 预测 `<mask>` 位置的词表分数 |
| Verbalizer | 将标签映射为自然语言词 |
| WS-BERT | 使用 Wikipedia 背景知识增强 target 理解 |

后续路线可以继续围绕两个方向展开：

```text
Knowledge-enhanced BERT：给 target 加背景知识
Prompt-based 改进：尝试多模板、换 verbalizer、加入 few-shot 示例
```

如果将两条路线结合，还可以进一步尝试：

```text
Tweet + Target + Knowledge + Prompt
```

也就是把 Wikipedia 背景知识写入 prompt，再用 masked language model 或分类模型进行立场检测。

