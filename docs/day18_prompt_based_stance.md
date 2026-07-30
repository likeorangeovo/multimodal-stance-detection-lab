# Day18 Prompt-based 立场检测：简单模板

## 1. 今天要做什么

Day15 和 Day17 的做法是常规分类：

```text
Target + Tweet -> BERT / RoBERTa -> 分类头 -> AGAINST / FAVOR / NONE
```

Day18 换一个思路：不新训练分类头，而是把立场检测改写成一个填空题：

```text
Tweet: ...
Target: ...
The stance toward the target is <mask>.
```

然后让 masked language model 判断 `<mask>` 位置更像哪个词：

```text
AGAINST -> against
FAVOR   -> favor
NONE    -> neutral
```

这就是最简单的 prompt-based stance detection。

本实验默认直接使用：

```text
distilbert/distilroberta-base
```

加载方式是 `AutoModelForMaskedLM`，也就是复用 RoBERTa 的 masked language modeling head，不额外训练 `AGAINST / FAVOR / NONE` 分类头。

## 2. 核心流程

输入仍然是 SemEval-2016 Task 6 的一条样本：

```text
Target: Atheism
Tweet: ...
Gold: AGAINST
```

脚本会做四步：

```text
1. 构造 prompt
2. 找到 prompt 里的 <mask> 位置
3. 取出 masked language model 在该位置的 logits
4. 只比较 against / favor / neutral 三个词的分数
```

谁的分数最高，就预测成对应的立场标签。

## 3. 运行方式

从项目根目录运行：

```powershell
D:\Anaconda\envs\env_3.11\python.exe src\day18_prompt_based_stance_detection.py
```

默认只跑测试集前 200 条，方便快速观察。

如果要跑完整测试集：

```powershell
D:\Anaconda\envs\env_3.11\python.exe src\day18_prompt_based_stance_detection.py --limit 0
```

如果只想检查流程：

```powershell
D:\Anaconda\envs\env_3.11\python.exe src\day18_prompt_based_stance_detection.py --limit 5
```

如果想看 prompt 之后三个标签词的分数：

```powershell
D:\Anaconda\envs\env_3.11\python.exe src\day18_prompt_based_stance_detection.py --limit 5 --show-scores
```

## 4. 重点看什么

不要只盯 accuracy。这个实验的重点是理解 prompt-based 方法：

- 模板会影响结果。
- 标签词会影响结果。
- `NONE` 不一定等价于 `neutral`，这只是一个近似 verbalizer。
- 没有微调时，模型并不一定懂 SemEval 的立场定义。
- 错误样例比最终分数更有学习价值。

## 5. 可以怎么改进

第一种：换模板。

```text
Tweet: ...
Question: What is the author's stance toward TARGET?
Answer: <mask>.
```

第二种：换 verbalizer。

```text
AGAINST -> oppose / against
FAVOR   -> support / favor
NONE    -> neutral / unrelated
```

第三种：多模板投票。

```text
template_1 预测 AGAINST
template_2 预测 NONE
template_3 预测 AGAINST
最终预测 AGAINST
```

第四种：后面可以进入 prompt tuning / P-tuning。

Day18 先不要急着做复杂优化。今天的目标是把“分类任务可以改写成语言模型填空题”这件事真正跑通。
