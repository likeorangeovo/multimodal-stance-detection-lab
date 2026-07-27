# 注释：导入 time，用于统计训练耗时或处理速度。
import time

# 注释：导入 PyTorch 主库，用于张量计算、自动求导和模型训练。
import torch
# 注释：导入 nn 模块，用于搭建神经网络层和损失函数。
from torch import nn
# 注释：导入 DataLoader 和 Subset，用于批量加载数据并截取小数据集练习。
from torch.utils.data import DataLoader, Subset
# 注释：导入 torchvision 的数据集和图像预处理工具。
from torchvision import datasets, transforms


# 注释：定义设备选择函数，优先使用 GPU，没有 GPU 就使用 CPU。
def try_gpu():
    # 注释：用文档字符串说明当前函数或类的用途。
    """Return GPU if available, otherwise return CPU."""
    # 注释：检查当前环境是否可以使用 CUDA GPU。
    if torch.cuda.is_available():
        # 注释：返回 CUDA 设备，让模型和数据在 GPU 上运行。
        return torch.device("cuda")
    # 注释：返回 CPU 设备，在没有 GPU 时仍然能运行。
    return torch.device("cpu")


# 注释：定义准确率统计函数，统计预测类别和真实标签相等的样本数。
def accuracy(y_hat, y):
    # 注释：用文档字符串说明当前函数或类的用途。
    """Compute the number of correct predictions."""
    # 注释：执行 `if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:` 这行代码，完成当前函数中的对应步骤。
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 注释：将 `y_hat.argmax(axis=1)` 这一步的结果保存到 `y_hat`，供后续代码使用。
        y_hat = y_hat.argmax(axis=1)
    # 注释：执行 `cmp = y_hat.type(y.dtype) == y` 这行代码，完成当前函数中的对应步骤。
    cmp = y_hat.type(y.dtype) == y
    # 注释：返回 `float(cmp.type(y.dtype).sum())` 的结果，供调用这个函数的代码继续使用。
    return float(cmp.type(y.dtype).sum())


# 注释：定义 GPU 评估函数，在给定数据集上计算准确率。
def evaluate_accuracy_gpu(net, data_iter, device=None):
    # 注释：用文档字符串说明当前函数或类的用途。
    """Compute the accuracy for a model on a dataset using a GPU if available."""
    # 注释：执行 `if isinstance(net, nn.Module):` 这行代码，完成当前函数中的对应步骤。
    if isinstance(net, nn.Module):
        # 注释：切换到评估模式，关闭训练阶段专用行为。
        net.eval()
        # 注释：执行 `if device is None:` 这行代码，完成当前函数中的对应步骤。
        if device is None:
            # 注释：将 `next(iter(net.parameters())).device` 这一步的结果保存到 `device`，供后续代码使用。
            device = next(iter(net.parameters())).device

    # 注释：将 `[0.0, 0.0]` 这一步的结果保存到 `metric`，供后续代码使用。
    metric = [0.0, 0.0]
    # 注释：进入无梯度模式，评估时不记录计算图以节省内存。
    with torch.no_grad():
        # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
        for x, y in data_iter:
            # 注释：把数据或模型移动到当前训练设备，并保存到 `x`。
            x = x.to(device)
            # 注释：把数据或模型移动到当前训练设备，并保存到 `y`。
            y = y.to(device)
            # 注释：把当前 batch 的统计量累加到 `metric[0]` 中。
            metric[0] += accuracy(net(x), y)
            # 注释：把当前 batch 的统计量累加到 `metric[1]` 中。
            metric[1] += y.numel()
    # 注释：返回 `metric[0] / metric[1]` 的结果，供调用这个函数的代码继续使用。
    return metric[0] / metric[1]


# 注释：定义 FashionMNIST 加载函数，可按需要缩放图片和截取子集。
def load_data_fashion_mnist(batch_size, resize=None, train_limit=None, test_limit=None):
    # 注释：用文档字符串说明当前函数或类的用途。
    """Download FashionMNIST and return train/test DataLoader objects."""
    # 注释：将 `[transforms.ToTensor()]` 这一步的结果保存到 `trans`，供后续代码使用。
    trans = [transforms.ToTensor()]
    # 注释：执行 `if resize:` 这行代码，完成当前函数中的对应步骤。
    if resize:
        # 注释：把 Resize 预处理插到 ToTensor 前面，先缩放图片再转张量。
        trans.insert(0, transforms.Resize(resize))
    # 注释：将 `transforms.Compose(trans)` 这一步的结果保存到 `trans`，供后续代码使用。
    trans = transforms.Compose(trans)

    # 注释：加载 FashionMNIST 数据集并保存到 `mnist_train`。
    mnist_train = datasets.FashionMNIST(
        # 注释：设置 `root` 参数，控制上方函数调用的行为。
        root="data",
        # 注释：设置 `train` 参数，控制上方函数调用的行为。
        train=True,
        # 注释：设置 `transform` 参数，控制上方函数调用的行为。
        transform=trans,
        # 注释：设置 `download` 参数，控制上方函数调用的行为。
        download=True,
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )
    # 注释：加载 FashionMNIST 数据集并保存到 `mnist_test`。
    mnist_test = datasets.FashionMNIST(
        # 注释：设置 `root` 参数，控制上方函数调用的行为。
        root="data",
        # 注释：设置 `train` 参数，控制上方函数调用的行为。
        train=False,
        # 注释：设置 `transform` 参数，控制上方函数调用的行为。
        transform=trans,
        # 注释：设置 `download` 参数，控制上方函数调用的行为。
        download=True,
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )

    # 注释：执行 `if train_limit is not None:` 这行代码，完成当前函数中的对应步骤。
    if train_limit is not None:
        # 注释：截取数据集的一部分样本，得到更快的练习数据 `mnist_train`。
        mnist_train = Subset(mnist_train, range(train_limit))
    # 注释：执行 `if test_limit is not None:` 这行代码，完成当前函数中的对应步骤。
    if test_limit is not None:
        # 注释：截取数据集的一部分样本，得到更快的练习数据 `mnist_test`。
        mnist_test = Subset(mnist_test, range(test_limit))

    # 注释：将多个结果组合返回，便于调用处同时拿到训练和测试迭代器。
    return (
        # 注释：将 `True, num_workers=0),` 这一步的结果保存到 `DataLoader(mnist_train, batch_size, shuffle`，供后续代码使用。
        DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0),
        # 注释：将 `False, num_workers=0),` 这一步的结果保存到 `DataLoader(mnist_test, batch_size, shuffle`，供后续代码使用。
        DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0),
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )


# 注释：定义 D2L 第 6 章风格的 CNN 训练函数。
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    # 注释：用文档字符串说明当前函数或类的用途。
    """Train a model with a GPU, following the D2L chapter 6 style."""
    # 注释：执行 `net.to(device)` 这行代码，完成当前函数中的对应步骤。
    net.to(device)

    # 注释：定义 `init_weights` 函数，封装一段可复用逻辑。
    def init_weights(m):
        # 注释：执行 `if type(m) == nn.Linear or type(m) == nn.Conv2d:` 这行代码，完成当前函数中的对应步骤。
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # 注释：执行 `nn.init.xavier_uniform_(m.weight)` 这行代码，完成当前函数中的对应步骤。
            nn.init.xavier_uniform_(m.weight)

    # 注释：对网络所有层应用初始化函数，设置可训练权重的初始值。
    net.apply(init_weights)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("training on", device)

    # 注释：创建交叉熵损失函数，用于多分类任务。
    loss = nn.CrossEntropyLoss()
    # 注释：创建优化器，用于根据梯度更新模型参数。
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 注释：按 epoch 多轮训练模型。
    for epoch in range(num_epochs):
        # 注释：将 `[0.0, 0.0, 0.0]` 这一步的结果保存到 `metric`，供后续代码使用。
        metric = [0.0, 0.0, 0.0]
        # 注释：将 `time.time()` 这一步的结果保存到 `start`，供后续代码使用。
        start = time.time()
        # 注释：切换到训练模式，启用训练阶段行为。
        net.train()

        # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
        for x, y in train_iter:
            # 注释：清空上一次累积的梯度，避免梯度叠加。
            optimizer.zero_grad()
            # 注释：把数据或模型移动到当前训练设备，并保存到 `x`。
            x = x.to(device)
            # 注释：把数据或模型移动到当前训练设备，并保存到 `y`。
            y = y.to(device)
            # 注释：执行模型前向传播，得到预测结果并保存到 `y_hat`。
            y_hat = net(x)
            # 注释：根据模型预测和真实标签计算当前 batch 的损失。
            l = loss(y_hat, y)
            # 注释：根据损失反向传播，计算每个参数的梯度。
            l.backward()
            # 注释：让优化器根据当前梯度更新模型参数。
            optimizer.step()

            # 注释：进入无梯度模式，评估时不记录计算图以节省内存。
            with torch.no_grad():
                # 注释：把当前 batch 的统计量累加到 `metric[0]` 中。
                metric[0] += l * x.shape[0]
                # 注释：把当前 batch 的统计量累加到 `metric[1]` 中。
                metric[1] += accuracy(y_hat, y)
                # 注释：把当前 batch 的统计量累加到 `metric[2]` 中。
                metric[2] += x.shape[0]

        # 注释：将 `metric[0] / metric[2]` 这一步的结果保存到 `train_loss`，供后续代码使用。
        train_loss = metric[0] / metric[2]
        # 注释：将 `metric[1] / metric[2]` 这一步的结果保存到 `train_acc`，供后续代码使用。
        train_acc = metric[1] / metric[2]
        # 注释：将 `evaluate_accuracy_gpu(net, test_iter)` 这一步的结果保存到 `test_acc`，供后续代码使用。
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # 注释：将 `metric[2] / (time.time() - start)` 这一步的结果保存到 `examples_per_sec`，供后续代码使用。
        examples_per_sec = metric[2] / (time.time() - start)

        # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
        print(
            # 注释：执行 `f"epoch {epoch + 1}, "` 这行代码，完成当前函数中的对应步骤。
            f"epoch {epoch + 1}, "
            # 注释：执行 `f"loss {train_loss:.3f}, "` 这行代码，完成当前函数中的对应步骤。
            f"loss {train_loss:.3f}, "
            # 注释：执行 `f"train acc {train_acc:.3f}, "` 这行代码，完成当前函数中的对应步骤。
            f"train acc {train_acc:.3f}, "
            # 注释：执行 `f"test acc {test_acc:.3f}, "` 这行代码，完成当前函数中的对应步骤。
            f"test acc {test_acc:.3f}, "
            # 注释：执行 `f"{examples_per_sec:.1f} examples/sec"` 这行代码，完成当前函数中的对应步骤。
            f"{examples_per_sec:.1f} examples/sec"
        # 注释：结束或承接上方的多行表达式，保持代码结构完整。
        )


# 注释：定义脚本主入口，把当前练习的数据、模型和训练流程串起来。
def main():
    # 注释：固定 PyTorch 随机种子，让数据打乱和参数初始化尽量可复现。
    torch.manual_seed(33)

    # 注释：设置 `batch_size` 参数，控制上方函数调用的行为。
    batch_size = 128
    # 注释：将 `0.01` 这一步的结果保存到 `lr`，供后续代码使用。
    lr = 0.01
    # 注释：将 `5` 这一步的结果保存到 `num_epochs`，供后续代码使用。
    num_epochs = 5
    # 注释：将 `224` 这一步的结果保存到 `resize`，供后续代码使用。
    resize = 224

    # CPU training with AlexNet is slow, so use a subset while learning.
    # Change these two lines to None to train/evaluate on the full dataset.
    # 注释：将 `5000` 这一步的结果保存到 `train_limit`，供后续代码使用。
    train_limit = 5000
    # 注释：将 `1000` 这一步的结果保存到 `test_limit`，供后续代码使用。
    test_limit = 1000

    # 注释：创建顺序神经网络模型并保存到 `net`。
    net = nn.Sequential(
        # 注释：添加卷积层，用卷积核提取图像局部特征。
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加最大池化层，降低特征图尺寸并保留强响应。
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 注释：添加卷积层，用卷积核提取图像局部特征。
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加最大池化层，降低特征图尺寸并保留强响应。
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 注释：添加卷积层，用卷积核提取图像局部特征。
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加卷积层，用卷积核提取图像局部特征。
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加卷积层，用卷积核提取图像局部特征。
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加最大池化层，降低特征图尺寸并保留强响应。
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 注释：添加展平层，把多维特征图转换成全连接层输入。
        nn.Flatten(),
        # 注释：添加全连接层，完成特征维度变换或类别打分。
        nn.Linear(6400, 4096),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加 Dropout，训练时随机丢弃部分神经元以缓解过拟合。
        nn.Dropout(p=0.5),
        # 注释：添加全连接层，完成特征维度变换或类别打分。
        nn.Linear(4096, 4096),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加 Dropout，训练时随机丢弃部分神经元以缓解过拟合。
        nn.Dropout(p=0.5),
        # 注释：添加全连接层，完成特征维度变换或类别打分。
        nn.Linear(4096, 10),
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )

    # 注释：生成标准正态分布随机张量并保存到 `x`。
    x = torch.randn(1, 1, resize, resize)
    # 注释：执行 `for layer in net:` 这行代码，完成当前函数中的对应步骤。
    for layer in net:
        # 注释：将 `layer(x)` 这一步的结果保存到 `x`，供后续代码使用。
        x = layer(x)
        # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
        print(layer.__class__.__name__, "output shape:\t", x.shape)

    # 注释：将 `load_data_fashion_mnist(` 这一步的结果保存到 `train_iter, test_iter`，供后续代码使用。
    train_iter, test_iter = load_data_fashion_mnist(
        # 注释：作为多行函数调用或数据结构中的一个参数。
        batch_size,
        # 注释：将 `resize,` 这一步的结果保存到 `resize`，供后续代码使用。
        resize=resize,
        # 注释：将 `train_limit,` 这一步的结果保存到 `train_limit`，供后续代码使用。
        train_limit=train_limit,
        # 注释：将 `test_limit,` 这一步的结果保存到 `test_limit`，供后续代码使用。
        test_limit=test_limit,
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )
    # 注释：启动训练流程，将模型、数据、训练轮数和学习率传入训练函数。
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())


# 注释：判断当前文件是否被直接运行，直接运行时才调用 main。
if __name__ == "__main__":
    # 注释：调用主函数，启动当前脚本的完整练习流程。
    main()
