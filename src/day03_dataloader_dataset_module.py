# 注释：导入 PyTorch 主库，用于张量计算、自动求导和模型训练。
import torch
# 注释：导入 nn 模块，用于搭建神经网络层和损失函数。
from torch import nn
# 注释：导入 DataLoader 和 Dataset，用于自定义数据集并按 batch 读取。
from torch.utils.data import DataLoader, Dataset


# 注释：定义线性回归练习用的自定义数据集类。
class SyntheticLinearDataset(Dataset):
    # 注释：用文档字符串说明当前函数或类的用途。
    """A tiny Dataset for y = Xw + b + noise."""

    # 注释：定义 `__init__` 函数，封装一段可复用逻辑。
    def __init__(self, w, b, num_examples, noise_std=0.01):
        # 注释：生成标准正态分布随机张量并保存到 `self.x`。
        self.x = torch.randn(num_examples, len(w))
        # 注释：将 `torch.matmul(self.x, w.reshape(-1, 1)) + b` 这一步的结果保存到 `y`，供后续代码使用。
        y = torch.matmul(self.x, w.reshape(-1, 1)) + b
        # 注释：将 `y + torch.normal(0, noise_std, size=y.shape)` 这一步的结果保存到 `self.y`，供后续代码使用。
        self.y = y + torch.normal(0, noise_std, size=y.shape)

    # 注释：定义 `__len__` 函数，封装一段可复用逻辑。
    def __len__(self):
        # 注释：返回数据集中样本数量，DataLoader 会用它计算可迭代长度。
        return len(self.x)

    # 注释：定义 `__getitem__` 函数，封装一段可复用逻辑。
    def __getitem__(self, index):
        # 注释：返回指定索引的一条样本，供 DataLoader 组成 batch。
        return self.x[index], self.y[index]


# 注释：定义线性回归模型类，只包含一个线性层。
class LinearRegressionModel(nn.Module):
    # 注释：定义 `__init__` 函数，封装一段可复用逻辑。
    def __init__(self, in_features):
        # 注释：调用父类 nn.Module 的初始化逻辑，让 PyTorch 能正确管理子模块和参数。
        super().__init__()
        # 注释：创建输出线性层，把 LSTM 隐状态转成各个字符的预测分数。
        self.linear = nn.Linear(in_features, 1)

    # 注释：定义模型前向传播逻辑，说明输入如何变成预测输出。
    def forward(self, x):
        # 注释：返回指定索引的一条样本，供 DataLoader 组成 batch。
        return self.linear(x)


# 注释：定义 `train_epoch` 函数，封装一段可复用逻辑。
def train_epoch(model, data_iter, loss_fn, optimizer):
    # 注释：切换到训练模式，启用训练阶段行为。
    model.train()
    # 注释：把 `total_loss` 初始化为 0，用于后续累计统计量。
    total_loss = 0.0
    # 注释：把 `total_examples` 初始化为 0，用于后续累计统计量。
    total_examples = 0

    # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
    for batch_x, batch_y in data_iter:
        # 注释：执行模型前向传播，得到预测结果并保存到 `y_hat`。
        y_hat = model(batch_x)
        # 注释：根据模型预测和真实标签计算当前 batch 的损失。
        loss = loss_fn(y_hat, batch_y)

        # 注释：清空上一次累积的梯度，避免梯度叠加。
        optimizer.zero_grad()
        # 注释：根据损失反向传播，计算每个参数的梯度。
        loss.backward()
        # 注释：让优化器根据当前梯度更新模型参数。
        optimizer.step()

        # 注释：把当前 batch 的统计量累加到 `total_loss` 中。
        total_loss += loss.item() * len(batch_x)
        # 注释：把当前 batch 的统计量累加到 `total_examples` 中。
        total_examples += len(batch_x)

    # 注释：返回当前 epoch 的平均训练损失。
    return total_loss / total_examples


# 注释：定义脚本主入口，把当前练习的数据、模型和训练流程串起来。
def main():
    # 注释：执行 `torch.manual_seed(39)` 这行代码，完成当前函数中的对应步骤。
    torch.manual_seed(39)

    # 注释：把 Python 数据转换成 PyTorch 张量并保存到 `true_w`。
    true_w = torch.tensor([3.2, 0.7])
    # 注释：将 `0.9` 这一步的结果保存到 `true_b`，供后续代码使用。
    true_b = 0.9

    # 注释：将 `SyntheticLinearDataset(true_w, true_b, num_examples=1000)` 这一步的结果保存到 `dataset`，供后续代码使用。
    dataset = SyntheticLinearDataset(true_w, true_b, num_examples=1000)
    # 注释：把数据集封装成可按 batch 迭代的 `data_iter`。
    data_iter = DataLoader(dataset, batch_size=32, shuffle=True)

    # 注释：将 `next(iter(data_iter))` 这一步的结果保存到 `first_x, first_y`，供后续代码使用。
    first_x, first_y = next(iter(data_iter))
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("first batch x shape:", first_x.shape)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("first batch y shape:", first_y.shape)

    # 注释：将 `LinearRegressionModel(in_features=2)` 这一步的结果保存到 `model`，供后续代码使用。
    model = LinearRegressionModel(in_features=2)
    # 注释：将 `nn.MSELoss()` 这一步的结果保存到 `loss_fn`，供后续代码使用。
    loss_fn = nn.MSELoss()
    # 注释：创建优化器，用于根据梯度更新模型参数。
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

    # 注释：按 epoch 多轮训练模型。
    for epoch in range(10):
        # 注释：将 `train_epoch(model, data_iter, loss_fn, optimizer)` 这一步的结果保存到 `train_loss`，供后续代码使用。
        train_loss = train_epoch(model, data_iter, loss_fn, optimizer)
        # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
        print(f"epoch {epoch + 1:02d}, loss {train_loss:.6f}")

    # 注释：将 `model.linear.weight.data.reshape(-1)` 这一步的结果保存到 `learned_w`，供后续代码使用。
    learned_w = model.linear.weight.data.reshape(-1)
    # 注释：将 `model.linear.bias.data.item()` 这一步的结果保存到 `learned_b`，供后续代码使用。
    learned_b = model.linear.bias.data.item()

    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("true w:", true_w)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("learned w:", learned_w)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("true b:", true_b)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("learned b:", learned_b)


# 注释：判断当前文件是否被直接运行，直接运行时才调用 main。
if __name__ == "__main__":
    # 注释：调用主函数，启动当前脚本的完整练习流程。
    main()
