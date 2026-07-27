# 注释：导入 os，用于创建目录和拼接 checkpoint 路径。
import os
from pathlib import Path

# 注释：导入 PyTorch 主库，用于张量计算、自动求导和模型训练。
import torch
# 注释：导入 nn 模块，用于搭建神经网络层和损失函数。
from torch import nn
# 注释：导入 DataLoader 和 Subset，用于批量加载数据并截取小数据集练习。
from torch.utils.data import DataLoader, Subset
# 注释：导入 SummaryWriter，用于把 loss、acc 和学习率写入 TensorBoard。
from torch.utils.tensorboard import SummaryWriter
# 注释：导入 torchvision 的数据集和图像预处理工具。
from torchvision import datasets, transforms


# 注释：定义准确率统计函数，统计预测类别和真实标签相等的样本数。
def accuracy(y_hat, y):
    # 注释：将多个结果组合返回，便于调用处同时拿到训练和测试迭代器。
    return (y_hat.argmax(dim=1) == y).sum().item()


# 注释：定义评估函数，在测试集上计算模型准确率。
def evaluate(model, data_loader, device):
    # 注释：切换到评估模式，关闭训练阶段专用行为。
    model.eval()
    # 注释：把 `total_correct` 初始化为 0，用于后续累计统计量。
    total_correct = 0
    # 注释：把 `total_num` 初始化为 0，用于后续累计统计量。
    total_num = 0

    # 注释：进入无梯度模式，评估时不记录计算图以节省内存。
    with torch.no_grad():
        # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
        for x, y in data_loader:
            # 注释：把数据或模型移动到当前训练设备，并保存到 `x`。
            x = x.to(device)
            # 注释：把数据或模型移动到当前训练设备，并保存到 `y`。
            y = y.to(device)
            # 注释：执行模型前向传播，得到预测结果并保存到 `y_hat`。
            y_hat = model(x)
            # 注释：把当前 batch 的统计量累加到 `total_correct` 中。
            total_correct += accuracy(y_hat, y)
            # 注释：把当前 batch 的统计量累加到 `total_num` 中。
            total_num += y.shape[0]

    # 注释：返回测试集中预测正确的比例。
    return total_correct / total_num


# 注释：定义设备选择函数，返回 cuda 或 cpu。
def get_device():
    # 注释：检查当前环境是否可以使用 CUDA GPU。
    if torch.cuda.is_available():
        # 注释：返回 CUDA 设备，让模型和数据在 GPU 上运行。
        return torch.device("cuda")
    # 注释：返回 CPU 设备，在没有 GPU 时仍然能运行。
    return torch.device("cpu")


# 注释：定义数据加载函数，返回训练集和测试集的 DataLoader。
def get_data_loaders(batch_size):
    # 注释：设置 `transform` 参数，控制上方函数调用的行为。
    transform = transforms.ToTensor()

    # 注释：加载 MNIST 数据集并保存到 `train_dataset`。
    train_dataset = datasets.MNIST(
        # 注释：设置 `root` 参数，控制上方函数调用的行为。
        root="data",
        # 注释：设置 `train` 参数，控制上方函数调用的行为。
        train=True,
        # 注释：设置 `transform` 参数，控制上方函数调用的行为。
        transform=transform,
        # 注释：设置 `download` 参数，控制上方函数调用的行为。
        download=True,
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )
    # 注释：加载 MNIST 数据集并保存到 `test_dataset`。
    test_dataset = datasets.MNIST(
        # 注释：设置 `root` 参数，控制上方函数调用的行为。
        root="data",
        # 注释：设置 `train` 参数，控制上方函数调用的行为。
        train=False,
        # 注释：设置 `transform` 参数，控制上方函数调用的行为。
        transform=transform,
        # 注释：设置 `download` 参数，控制上方函数调用的行为。
        download=True,
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )

    # 注释：截取数据集的一部分样本，得到更快的练习数据 `train_data`。
    train_data = Subset(train_dataset, range(5000))
    # 注释：截取数据集的一部分样本，得到更快的练习数据 `test_data`。
    test_data = Subset(test_dataset, range(1000))

    # 注释：把数据集封装成可按 batch 迭代的 `train_loader`。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # 注释：把数据集封装成可按 batch 迭代的 `test_loader`。
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # 注释：返回训练和测试 DataLoader，供训练与评估使用。
    return train_loader, test_loader


# 注释：定义模型构建函数，返回一个简单的 MLP 网络。
def build_model():
    # 注释：返回按顺序堆叠的神经网络模型。
    return nn.Sequential(
        # 注释：添加展平层，把多维特征图转换成全连接层输入。
        nn.Flatten(),
        # 注释：添加全连接层，完成特征维度变换或类别打分。
        nn.Linear(28 * 28, 256),
        # 注释：添加 ReLU 激活函数，引入非线性表达能力。
        nn.ReLU(),
        # 注释：添加全连接层，完成特征维度变换或类别打分。
        nn.Linear(256, 10),
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )


# 注释：定义 checkpoint 保存函数，把模型、优化器、学习率调度器和指标一起保存。
def save_checkpoint(path, model, optimizer, scheduler, epoch, test_acc):
    # 注释：调用 torch.save 把 checkpoint 字典写入磁盘，后面可以用 torch.load 恢复。
    torch.save(
        # 注释：结束或承接上方的多行表达式，保持代码结构完整。
        {
            # 注释：作为多行函数调用或数据结构中的一个参数。
            "model_state_dict": model.state_dict(),
            # 注释：作为多行函数调用或数据结构中的一个参数。
            "optimizer_state_dict": optimizer.state_dict(),
            # 注释：作为多行函数调用或数据结构中的一个参数。
            "scheduler_state_dict": scheduler.state_dict(),
            # 注释：作为多行函数调用或数据结构中的一个参数。
            "epoch": epoch,
            # 注释：作为多行函数调用或数据结构中的一个参数。
            "test_acc": test_acc,
        # 注释：作为多行函数调用或数据结构中的一个参数。
        },
        # 注释：作为多行函数调用或数据结构中的一个参数。
        path,
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )


# 注释：定义模型加载函数，用 checkpoint 恢复模型并切换到评估模式。
def load_model_for_eval(path, device):
    # 注释：创建新的 MLP 模型并移动到指定设备，用于加载 checkpoint 后评估。
    model = build_model().to(device)
    # 注释：将 `torch.load(path, map_location=device)` 这一步的结果保存到 `checkpoint`，供后续代码使用。
    checkpoint = torch.load(path, map_location=device)
    # 注释：把 checkpoint 里保存的模型参数加载回新建的模型中。
    model.load_state_dict(checkpoint["model_state_dict"])
    # 注释：切换到评估模式，关闭训练阶段专用行为。
    model.eval()
    # 注释：返回 `model, checkpoint` 的结果，供调用这个函数的代码继续使用。
    return model, checkpoint


# 注释：定义脚本主入口，把当前练习的数据、模型和训练流程串起来。
def main():
    # 注释：固定 PyTorch 随机种子，让数据打乱和参数初始化尽量可复现。
    torch.manual_seed(33)

    # 注释：设置 `batch_size` 参数，控制上方函数调用的行为。
    batch_size = 64
    # 注释：将 `5` 这一步的结果保存到 `num_epochs`，供后续代码使用。
    num_epochs = 5
    # 注释：将 `0.01` 这一步的结果保存到 `lr`，供后续代码使用。
    lr = 0.01
    base_dir = Path(__file__).resolve().parent
    # 注释：Day8 的输出都放在当前小项目目录下，避免散在仓库根目录。
    checkpoint_dir = base_dir / "checkpoints"
    log_dir = base_dir / "runs" / "day8_mnist_mlp"
    # 注释：将 `os.path.join(checkpoint_dir, "day8_best_mlp.pt")` 这一步的结果保存到 `best_model_path`，供后续代码使用。
    best_model_path = checkpoint_dir / "day8_best_mlp.pt"

    # 注释：创建保存 checkpoint 的目录，如果目录已经存在就不报错。
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 注释：将 `get_device()` 这一步的结果保存到 `device`，供后续代码使用。
    device = get_device()
    # 注释：将 `get_data_loaders(batch_size)` 这一步的结果保存到 `train_loader, test_loader`，供后续代码使用。
    train_loader, test_loader = get_data_loaders(batch_size)
    # 注释：创建新的 MLP 模型并移动到指定设备，用于加载 checkpoint 后评估。
    model = build_model().to(device)

    # 注释：创建交叉熵损失函数，用于多分类任务。
    loss_fn = nn.CrossEntropyLoss()
    # 注释：创建优化器，用于根据梯度更新模型参数。
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 注释：创建优化器，用于根据梯度更新模型参数。
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    # 注释：创建 TensorBoard 写入器，用于记录训练曲线。
    writer = SummaryWriter(log_dir=log_dir)

    # 注释：把 `best_acc` 初始化为 0，用于后续累计统计量。
    best_acc = 0.0
    # 注释：把 `global_step` 初始化为 0，用于后续累计统计量。
    global_step = 0

    # 注释：按 epoch 多轮训练模型。
    for epoch in range(num_epochs):
        # 注释：切换到训练模式，启用训练阶段行为。
        model.train()
        # 注释：把 `total_loss` 初始化为 0，用于后续累计统计量。
        total_loss = 0.0
        # 注释：把 `total_correct` 初始化为 0，用于后续累计统计量。
        total_correct = 0
        # 注释：把 `total_num` 初始化为 0，用于后续累计统计量。
        total_num = 0

        # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
        for x, y in train_loader:
            # 注释：把数据或模型移动到当前训练设备，并保存到 `x`。
            x = x.to(device)
            # 注释：把数据或模型移动到当前训练设备，并保存到 `y`。
            y = y.to(device)

            # 注释：执行模型前向传播，得到预测结果并保存到 `y_hat`。
            y_hat = model(x)
            # 注释：根据模型预测和真实标签计算当前 batch 的损失。
            loss = loss_fn(y_hat, y)

            # 注释：清空上一次累积的梯度，避免梯度叠加。
            optimizer.zero_grad()
            # 注释：根据损失反向传播，计算每个参数的梯度。
            loss.backward()
            # 注释：让优化器根据当前梯度更新模型参数。
            optimizer.step()

            # 注释：把当前 batch 的统计量累加到 `total_loss` 中。
            total_loss += loss.item() * y.shape[0]
            # 注释：把当前 batch 的统计量累加到 `total_correct` 中。
            total_correct += accuracy(y_hat, y)
            # 注释：把当前 batch 的统计量累加到 `total_num` 中。
            total_num += y.shape[0]

            # 注释：把当前指标写入 TensorBoard，后面可以画曲线查看训练过程。
            writer.add_scalar("batch/train_loss", loss.item(), global_step)
            # 注释：把当前 batch 的统计量累加到 `global_step` 中。
            global_step += 1

        # 注释：将 `total_loss / total_num` 这一步的结果保存到 `train_loss`，供后续代码使用。
        train_loss = total_loss / total_num
        # 注释：将 `total_correct / total_num` 这一步的结果保存到 `train_acc`，供后续代码使用。
        train_acc = total_correct / total_num
        # 注释：将 `evaluate(model, test_loader, device)` 这一步的结果保存到 `test_acc`，供后续代码使用。
        test_acc = evaluate(model, test_loader, device)
        # 注释：将 `optimizer.param_groups[0]["lr"]` 这一步的结果保存到 `current_lr`，供后续代码使用。
        current_lr = optimizer.param_groups[0]["lr"]

        # 注释：把当前指标写入 TensorBoard，后面可以画曲线查看训练过程。
        writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
        # 注释：把当前指标写入 TensorBoard，后面可以画曲线查看训练过程。
        writer.add_scalar("epoch/train_acc", train_acc, epoch + 1)
        # 注释：把当前指标写入 TensorBoard，后面可以画曲线查看训练过程。
        writer.add_scalar("epoch/test_acc", test_acc, epoch + 1)
        # 注释：把当前指标写入 TensorBoard，后面可以画曲线查看训练过程。
        writer.add_scalar("epoch/learning_rate", current_lr, epoch + 1)

        # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
        print(
            # 注释：执行 `f"epoch {epoch + 1}, "` 这行代码，完成当前函数中的对应步骤。
            f"epoch {epoch + 1}, "
            # 注释：执行 `f"lr {current_lr:.5f}, "` 这行代码，完成当前函数中的对应步骤。
            f"lr {current_lr:.5f}, "
            # 注释：执行 `f"loss {train_loss:.4f}, "` 这行代码，完成当前函数中的对应步骤。
            f"loss {train_loss:.4f}, "
            # 注释：执行 `f"train acc {train_acc:.3f}, "` 这行代码，完成当前函数中的对应步骤。
            f"train acc {train_acc:.3f}, "
            # 注释：执行 `f"test acc {test_acc:.3f}"` 这行代码，完成当前函数中的对应步骤。
            f"test acc {test_acc:.3f}"
        # 注释：结束或承接上方的多行表达式，保持代码结构完整。
        )

        # 注释：如果当前测试准确率超过历史最好结果，就保存新的最佳模型。
        if test_acc > best_acc:
            # 注释：将 `test_acc` 这一步的结果保存到 `best_acc`，供后续代码使用。
            best_acc = test_acc
            # 注释：保存当前最优模型的 checkpoint，包含权重、优化器状态和评估指标。
            save_checkpoint(
                best_model_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                test_acc,
            )
            print(f"saved best model to {best_model_path}")

        # 注释：让学习率调度器进入下一步，必要时衰减学习率。
        scheduler.step()

    # 注释：关闭 TensorBoard 写入器，确保日志写入磁盘。
    writer.close()

    # 注释：将 `load_model_for_eval(best_model_path, device)` 这一步的结果保存到 `loaded_model, checkpoint`，供后续代码使用。
    loaded_model, checkpoint = load_model_for_eval(best_model_path, device)
    # 注释：将 `evaluate(loaded_model, test_loader, device)` 这一步的结果保存到 `loaded_acc`，供后续代码使用。
    loaded_acc = evaluate(loaded_model, test_loader, device)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print(
        # 注释：执行 `f"loaded epoch {checkpoint['epoch']} checkpoint, "` 这行代码，完成当前函数中的对应步骤。
        f"loaded epoch {checkpoint['epoch']} checkpoint, "
        # 注释：执行 `f"saved acc {checkpoint['test_acc']:.3f}, "` 这行代码，完成当前函数中的对应步骤。
        f"saved acc {checkpoint['test_acc']:.3f}, "
        # 注释：执行 `f"loaded acc {loaded_acc:.3f}"` 这行代码，完成当前函数中的对应步骤。
        f"loaded acc {loaded_acc:.3f}"
    # 注释：结束或承接上方的多行表达式，保持代码结构完整。
    )
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print(f"tensorboard log dir: {log_dir}")


# 注释：判断当前文件是否被直接运行，直接运行时才调用 main。
if __name__ == "__main__":
    # 注释：调用主函数，启动当前脚本的完整练习流程。
    main()
