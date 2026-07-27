# Day10 Image-Text Matching

Day10 做的是一个很小的图文匹配练习。数据是自己合成的，不追求效果多好，主要是把“图片 + 文本 -> 二分类”的流程跑通。

## 任务

给一张图和一句文本，判断它们是否匹配。

例子：

```text
image: red circle
text:  red circle
label: 1

image: red circle
text:  blue rectangle
label: 0
```

目前图片由颜色和形状组成：

```text
colors = red, green, blue, yellow, magenta, cyan, black
shapes = rectangle, circle, ellipse
```

## 目录

```text
day10_image_text_matching/
  __init__.py
  checkpoint.py
  config.py
  dataset.py
  eval.py
  main.py
  model.py
  train.py
  utils.py
  README.md
```

现在主要用到：

- `dataset.py`：生成合成图片和文本，构建 Dataset。
- `model.py`：ResNet18 图像编码器 + LSTM 文本编码器 + MLP 分类器。
- `train.py`：训练一轮的逻辑。
- `eval.py`：验证准确率和单样本预测。
- `checkpoint.py`：保存和加载模型。
- `main.py`：把完整流程串起来。

## 模型结构

大致流程：

```text
image -> pretrained ResNet18 -> image feature [B, 512]
text  -> Embedding + LSTM   -> text feature  [B, 128]

[image feature, text feature]
-> concatenate
-> MLP
-> one logit
```

因为最后输出的是一个二分类 logit，所以 loss 用：

```python
nn.BCEWithLogitsLoss()
```

不要用 `CrossEntropyLoss`，除非模型输出改成 `[batch_size, 2]` 这种两类分数。

## 运行

在仓库根目录运行：

```powershell
conda activate env_3.11
python src\day10_image_text_matching\main.py
```

如果当前终端没有初始化 conda，也可以直接用完整路径：

```powershell
D:\Anaconda\envs\env_3.11\python.exe src\day10_image_text_matching\main.py
```

运行后会训练 10 个 epoch，并保存验证集准确率最好的模型：

```text
src/day10_image_text_matching/checkpoints/best.pt
```

最后会加载 best checkpoint，跑一个正样本和一个负样本的预测 demo。

## 目前的问题

这个练习的数据量很小，只有几十条样本，所以验证集准确率会波动，预测概率也不一定特别分开。

这不是主要问题。这个阶段更重要的是把下面这些环节连起来：

```text
Dataset
DataLoader
image encoder
text encoder
feature fusion
binary classifier
loss
train / eval
checkpoint
predict demo
```

后面如果要继续改，可以考虑：

- 增加合成样本数量。
- 让负样本只改颜色或只改形状，难度更细。
- 把 LSTM 换成简单 Transformer 或 BERT。
- 把 ResNet18 换成 CLIP image encoder。
- 加 TensorBoard 记录训练曲线。
