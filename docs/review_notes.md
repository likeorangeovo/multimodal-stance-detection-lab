# 复习总结

这份文件只放复习用的内容。写法尽量短一点，主要记我这几天真正用到的东西和踩过的坑。

---

## Day1：Tensor

`Tensor` 是 PyTorch 里的基本数据结构，可以理解成支持 GPU 和自动求导的多维数组。

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.randn(2, 3)
z = torch.zeros(4, 1, 28, 28)
```

图像 batch 常见形状：

```text
[batch_size, channels, height, width]
```

比如 MNIST：

```text
[64, 1, 28, 28]
```

要记住 PyTorch 图像默认是 `NCHW`，通道维在前面。

---

## Day2：自动求导和线性回归

线性回归形式：

```text
y = Xw + b
```

手写训练循环里，最容易混的是梯度：

```python
loss.backward()
param -= lr * param.grad / batch_size
param.grad.zero_()
```

几个点：

- `backward()` 只算梯度，不更新参数。
- 参数更新可以手写，也可以交给 optimizer。
- 梯度默认会累加，所以每次更新后要清零。
- `torch.manual_seed(42)` 可以固定随机性，方便复现。

小批量数据可以用随机索引：

```python
indices = torch.randperm(num_examples)
batch_indices = indices[i: i + batch_size]
batch_x = features[batch_indices]
```

---

## Day3：Dataset、DataLoader、nn.Module

这一天主要是把“数据”和“模型”拆清楚。

```python
class MyDataset(Dataset):
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
```

我的理解：

- `Dataset` 管一条样本怎么取。
- `DataLoader` 管怎么组成 batch、是否 shuffle。
- `nn.Module` 管数据进来以后怎么算。

自定义模型：

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)
```

注意：自己写模型时，`__init__` 和 `forward` 这两个方法名不能写错。

---

## Day4：MLP 分类器

MNIST 可以先用 MLP 跑通分类流程：

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
```

多分类常用：

```python
loss_fn = nn.CrossEntropyLoss()
```

这里不用手动加 `Softmax`。`CrossEntropyLoss` 里面已经处理了。

形状要对：

```text
模型输出: [batch_size, num_classes]
标签:     [batch_size]
```

准确率：

```python
preds = y_hat.argmax(dim=1)
acc = (preds == y).sum().item() / y.numel()
```

训练和评估：

```python
model.train()
model.eval()

with torch.no_grad():
    ...
```

`eval()` 和 `no_grad()` 不是一回事。`eval()` 切模型状态，`no_grad()` 关闭梯度记录。

---

## Day5：训练脚本整理

Day5 没有单独文件，主要内容合到后面几天了。

现在先记住一个固定顺序：

```text
前向传播 -> 计算 loss -> optimizer.zero_grad() -> loss.backward() -> optimizer.step()
```

常见模板：

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        ...
```

---

## Day6：CNN 和 FashionMNIST

CNN 比 MLP 更适合图像，因为卷积会利用局部空间结构。

几个概念：

- `Conv2d` 提取局部特征。
- `MaxPool2d` 压缩空间尺寸。
- `Flatten` 把特征图展平成向量。
- 最后接全连接层做分类。

图像输入还是：

```text
[B, C, H, W]
```

卷积层输出通道数由 `out_channels` 决定。池化一般会让 `H` 和 `W` 变小。

Day6 的重点不是把 FashionMNIST 训到很高，而是把 CNN 的训练流程跑通。

---

## Day7：字符级 LSTM 文本生成

LSTM 用来处理序列。字符级文本生成的任务是：

```text
根据前面的字符，预测下一个字符
```

文本要先转成索引：

```python
chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
corpus = [char_to_idx[ch] for ch in text]
```

训练样本里，`y` 是 `x` 往后错一位：

```text
text = hello
num_steps = 3

x: h e l
y: e l l
```

`num_steps` 是序列长度，不是 epoch 数，也不是反向传播次数。

如果模型输出：

```text
[num_steps * batch_size, vocab_size]
```

标签也要拉平：

```python
y = y.T.reshape(-1)
```

生成文本时分两步：

```text
先读 prefix 更新 LSTM 状态
再用上一个字符逐个预测下一个字符
```

LSTM 训练时可以加梯度裁剪，防止梯度爆炸。

---

## Day8：checkpoint、scheduler、TensorBoard

这一天是把训练脚本写得更像真实项目。

保存模型时，推荐保存一个字典：

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "test_acc": test_acc,
    },
    path,
)
```

加载时：

```python
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
```

注意：

- `model.state_dict()` 保存模型权重。
- `optimizer.state_dict()` 保存优化器状态，方便断点续训。
- `map_location=device` 可以避免 CPU/GPU 不匹配。
- best model 应该看验证集指标，不看训练集指标。

学习率衰减：

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
scheduler.step()
```

TensorBoard：

```python
writer = SummaryWriter(log_dir="src/day08_training_tools/runs/day8_mnist_mlp")
writer.add_scalar("epoch/train_loss", train_loss, epoch)
writer.add_scalar("epoch/test_acc", test_acc, epoch)
writer.close()
```

启动：

```powershell
tensorboard --logdir runs
```

---

## Day9：预训练 ResNet 特征提取

Day9 不是重新训练 ResNet，而是把预训练 ResNet 当图像特征提取器。

流程：

```text
图片 -> transforms -> ResNet18 -> avgpool -> flatten -> embedding
```

核心代码：

```python
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
model = resnet18(weights=weights)
```

如果想取中间层：

```python
extractor = create_feature_extractor(
    model,
    return_nodes={
        "layer4": "feature_map",
        "avgpool": "embedding",
    },
)
```

如果只要最后的图像向量，也可以：

```python
model.fc = torch.nn.Identity()
embedding = model(x)
```

形状：

```text
preprocess(image): [3, H, W]
unsqueeze(0):      [1, 3, H, W]
avgpool:           [1, 512, 1, 1]
flatten:           [1, 512]
```

保存特征前可以转 CPU：

```python
torch.save(embedding.cpu(), path)
```

---

## Day10：图文匹配小项目

Day10 是一个完整小 pipeline：

```text
合成图片和文本
-> Dataset/DataLoader
-> ResNet18 图像编码器
-> LSTM 文本编码器
-> 拼接特征
-> MLP 二分类
-> 保存 best checkpoint
-> 单样本预测
```

### 1. 魔法方法别写错

Python 里要写：

```python
def __init__(self):
    super().__init__()
```

不是：

```python
def **init**
```

`__name__` 和 `"__main__"` 也是一样。复制到 Markdown 或聊天里时，双下划线有时会被显示成加粗，要自己看清楚源码。

### 2. torchvision 参数名是 weights

ResNet18 预训练权重要这样写：

```python
weights = ResNet18_Weights.DEFAULT
self.image_encoder = resnet18(weights=weights)
```

不要写成：

```python
resnet18(wights=wights)
```

这个拼写错误会报：

```text
unexpected keyword argument 'wights'
```

### 3. LSTM 要注意 batch_first

文本输入是：

```text
[batch_size, seq_len]
```

经过 embedding 后是：

```text
[batch_size, seq_len, embedding_dim]
```

所以 LSTM 要写：

```python
self.lstm = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    batch_first=True,
)
```

否则 LSTM 会把第一维当成 `seq_len`，后面和图片特征拼接时 batch 对不上。

### 4. 冻结 ResNet 只冻图像编码器

更稳的写法：

```python
for param in self.image_encoder.parameters():
    param.requires_grad = False
```

不要随手写成：

```python
for param in self.parameters():
    param.requires_grad = False
```

不然后面如果模块定义顺序变了，可能会把整个模型都冻住。

### 5. 二分类单 logit 用 BCEWithLogitsLoss

Day10 模型最后输出：

```text
[batch_size]
```

每个样本只有一个 logit，所以 loss 用：

```python
nn.BCEWithLogitsLoss()
```

`CrossEntropyLoss` 更适合这种输出：

```text
[batch_size, num_classes]
```

也就是多分类或者二分类两列分数的写法。

### 6. 运行要用虚拟环境 Python

系统 Python 可能没有装 torch。这个项目运行时用：

```powershell
.\.venv\Scripts\python.exe src\day10_image_text_matching\main.py
```

如果直接：

```powershell
python src\day10_image_text_matching\main.py
```

可能会报：

```text
ModuleNotFoundError: No module named 'torch'
```

### 7. 当前数据很小，不要太纠结准确率

Day10 的数据是颜色和形状合成的，样本数很少。现在重点不是模型效果，而是把流程跑通：

```text
Dataset -> Model -> Train -> Eval -> Checkpoint -> Predict
```

后面要提升，可以先增加样本数量，再考虑换 CLIP 或 BERT。
