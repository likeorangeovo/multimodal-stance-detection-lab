# 注释：导入 PyTorch 主库，用于张量计算、自动求导和模型训练。
import torch
# 注释：导入 nn 模块，用于搭建神经网络层和损失函数。
from torch import nn
# 注释：导入当前脚本需要的组件。
from torch.utils.data import DataLoader,Subset
# 注释：导入 torchvision 的数据集和图像预处理工具。
from torchvision import datasets, transforms

# 注释：定义准确率统计函数，统计预测类别和真实标签相等的样本数。
def accuracy(y_hat,y):
    # 注释：将多个结果组合返回，便于调用处同时拿到训练和测试迭代器。
    return (y_hat.argmax(dim=1) == y).sum().item()

# 注释：定义脚本主入口，把当前练习的数据、模型和训练流程串起来。
def main():
    # 注释：固定 PyTorch 随机种子，让数据打乱和参数初始化尽量可复现。
    torch.manual_seed(33)
    # 注释：设置 `batch_size` 参数，控制上方函数调用的行为。
    batch_size = 64
    # 注释：将 `10` 这一步的结果保存到 `num_epochs`，供后续代码使用。
    num_epochs = 10
    # 注释：将 `0.001` 这一步的结果保存到 `lr`，供后续代码使用。
    lr = 0.001
    # 注释：将 `28*28` 这一步的结果保存到 `num_inputs`，供后续代码使用。
    num_inputs = 28*28
    # 注释：将 `256` 这一步的结果保存到 `num_hidden`，供后续代码使用。
    num_hidden = 256
    # 注释：将 `10` 这一步的结果保存到 `num_outputs`，供后续代码使用。
    num_outputs = 10

    # 注释：设置 `transform` 参数，控制上方函数调用的行为。
    transform = transforms.ToTensor()

    # 注释：加载 MNIST 数据集并保存到 `train_dataset`。
    train_dataset = datasets.MNIST(
        # 注释：设置 `root` 参数，控制上方函数调用的行为。
        root = "data",
        # 注释：设置 `train` 参数，控制上方函数调用的行为。
        train = True,
        # 注释：设置 `transform` 参数，控制上方函数调用的行为。
        transform = transform,
        # 注释：设置 `download` 参数，控制上方函数调用的行为。
        download = True
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )

    # 注释：加载 MNIST 数据集并保存到 `test_dataset`。
    test_dataset = datasets.MNIST(
        # 注释：设置 `root` 参数，控制上方函数调用的行为。
        root = "data",
        # 注释：设置 `train` 参数，控制上方函数调用的行为。
        train = False,
        # 注释：设置 `transform` 参数，控制上方函数调用的行为。
        transform = transform,
        # 注释：设置 `download` 参数，控制上方函数调用的行为。
        download = True
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )

    # 注释：截取数据集的一部分样本，得到更快的练习数据 `train_data`。
    train_data = Subset(train_dataset, range(5000))
    # 注释：截取数据集的一部分样本，得到更快的练习数据 `test_data`。
    test_data = Subset(test_dataset, range(1000))

    # 注释：把数据集封装成可按 batch 迭代的 `train_loader`。
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    # 注释：把数据集封装成可按 batch 迭代的 `test_loader`。
    test_loader = DataLoader(test_data, batch_size, shuffle=True)

    # 注释：创建顺序神经网络模型并保存到 `model`。
    model = nn.Sequential(
        # 注释：添加展平层，把多维特征图转换成全连接层输入。
        nn.Flatten(),
        # 注释：添加全连接层，完成特征维度变换或类别打分。
        nn.Linear(num_inputs, num_hidden),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加全连接层，完成特征维度变换或类别打分。
        nn.Linear(num_hidden, num_outputs),
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )
    # 注释：创建交叉熵损失函数，用于多分类任务。
    loss_fn = nn.CrossEntropyLoss()
    # 注释：创建优化器，用于根据梯度更新模型参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 注释：按 epoch 多轮训练模型。
    for epoch in range(num_epochs):
        # 注释：把 `total_loss` 初始化为 0，用于后续累计统计量。
        total_loss = 0
        # 注释：把 `total_correct` 初始化为 0，用于后续累计统计量。
        total_correct = 0
        # 注释：把 `total_num` 初始化为 0，用于后续累计统计量。
        total_num = 0

        # 注释：切换到训练模式，启用训练阶段行为。
        model.train()
        # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
        for batch_x, batch_y in train_loader:
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
            total_loss += loss.item()*batch_y.shape[0]
            # 注释：把当前 batch 的统计量累加到 `total_correct` 中。
            total_correct += accuracy(y_hat, batch_y)
            # 注释：把当前 batch 的统计量累加到 `total_num` 中。
            total_num += batch_y.shape[0]
        # 注释：将 `total_loss/total_num` 这一步的结果保存到 `train_loss`，供后续代码使用。
        train_loss = total_loss/total_num
        # 注释：将 `total_correct/total_num` 这一步的结果保存到 `train_correct`，供后续代码使用。
        train_correct = total_correct/total_num

        # 注释：切换到评估模式，关闭训练阶段专用行为。
        model.eval()
        # 注释：把 `test_num` 初始化为 0，用于后续累计统计量。
        test_num = 0
        # 注释：把 `test_correct` 初始化为 0，用于后续累计统计量。
        test_correct = 0

        # 注释：进入无梯度模式，评估时不记录计算图以节省内存。
        with torch.no_grad():
            # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
            for batch_x, batch_y in test_loader:
                # 注释：执行模型前向传播，得到预测结果并保存到 `y_hat`。
                y_hat = model(batch_x)
                # 注释：把当前 batch 的统计量累加到 `test_correct` 中。
                test_correct += accuracy(y_hat, batch_y)
                # 注释：把当前 batch 的统计量累加到 `test_num` 中。
                test_num += batch_y.shape[0]
            # 注释：将 `test_correct/test_num` 这一步的结果保存到 `test_acc`，供后续代码使用。
            test_acc = test_correct/test_num

        # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
        print(
            # 注释：执行 `f"epoch {epoch + 1}, "` 这行代码，完成当前函数中的对应步骤。
            f"epoch {epoch + 1}, "
            # 注释：执行 `f"loss {train_loss:.4f}, "` 这行代码，完成当前函数中的对应步骤。
            f"loss {train_loss:.4f}, "
            # 注释：执行 `f"train acc {train_correct:.3f}, "` 这行代码，完成当前函数中的对应步骤。
            f"train acc {train_correct:.3f}, "
            # 注释：执行 `f"test acc {test_acc:.3f}"` 这行代码，完成当前函数中的对应步骤。
            f"test acc {test_acc:.3f}"
        # 注释：结束或承接上方的多行表达式，保持代码结构完整。
        )

# 注释：判断当前文件是否被直接运行，直接运行时才调用 main。
if __name__ == "__main__":
    # 注释：调用主函数，启动当前脚本的完整练习流程。
    main()

