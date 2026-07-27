# 注释：导入 math，用于计算困惑度等数学指标。
import math

# 注释：导入 PyTorch 主库，用于张量计算、自动求导和模型训练。
import torch
# 注释：导入 nn 模块，用于搭建神经网络层和损失函数。
from torch import nn
# 注释：导入函数式接口 F，用于 one-hot 等无参数操作。
from torch.nn import functional as F
# 注释：导入 DataLoader 和 TensorDataset，用于把张量数据按 batch 送入模型。
from torch.utils.data import DataLoader, TensorDataset


# 注释：定义设备选择函数，优先使用 GPU，没有 GPU 就使用 CPU。
def try_gpu():
    # 注释：检查当前环境是否可以使用 CUDA GPU。
    if torch.cuda.is_available():
        # 注释：返回 CUDA 设备，让模型和数据在 GPU 上运行。
        return torch.device("cuda")
    # 注释：返回 CPU 设备，在没有 GPU 时仍然能运行。
    return torch.device("cpu")


# 注释：定义字符级数据集构造函数，把文本切成输入序列和目标序列。
def build_dataset(text, num_steps):
    # 注释：用文档字符串说明当前函数或类的用途。
    """Turn text into many (input sequence, target sequence) pairs."""
    # 注释：从文本中取出不重复字符并排序，得到固定顺序的字符表。
    chars = sorted(list(set(text)))
    # 注释：建立字符到数字索引的映射，让文本能转成模型可处理的数字。
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    # 注释：建立数字索引到字符的映射，用于把模型预测还原成文本。
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    # 注释：把整段文本按字符表转成索引序列，作为语言模型语料。
    corpus = [char_to_idx[ch] for ch in text]

    # 注释：创建空列表 `x_list`，后面用来收集数据。
    x_list = []
    # 注释：创建空列表 `y_list`，后面用来收集数据。
    y_list = []
    # 注释：沿着文本滑动窗口，逐个构造长度为 num_steps 的训练样本。
    for i in range(0, len(corpus) - num_steps):
        # 注释：把当前位置的输入序列片段加入训练输入列表。
        x_list.append(corpus[i : i + num_steps])
        # 注释：把输入序列后移一位的目标片段加入标签列表。
        y_list.append(corpus[i + 1 : i + num_steps + 1])

    # 注释：把 Python 数据转换成 PyTorch 张量并保存到 `x`。
    x = torch.tensor(x_list)
    # 注释：把 Python 数据转换成 PyTorch 张量并保存到 `y`。
    y = torch.tensor(y_list)
    # 注释：返回可被 DataLoader 使用的数据集，以及字符和索引的双向映射。
    return TensorDataset(x, y), char_to_idx, idx_to_char


# 注释：定义字符级 LSTM 语言模型类。
class CharLSTM(nn.Module):
    # 注释：用文档字符串说明当前函数或类的用途。
    """A small character-level LSTM language model."""

    # 注释：定义 `__init__` 函数，封装一段可复用逻辑。
    def __init__(self, vocab_size, num_hiddens):
        # 注释：调用父类 nn.Module 的初始化逻辑，让 PyTorch 能正确管理子模块和参数。
        super().__init__()
        # 注释：保存字符表大小，后面 one-hot 和输出层都要用它确定维度。
        self.vocab_size = vocab_size
        # 注释：创建 LSTM 层，让模型按时间步读取字符序列并维护隐状态。
        self.lstm = nn.LSTM(vocab_size, num_hiddens)
        # 注释：创建输出线性层，把 LSTM 隐状态转成各个字符的预测分数。
        self.linear = nn.Linear(num_hiddens, vocab_size)

    # 注释：定义模型前向传播逻辑，说明输入如何变成预测输出。
    def forward(self, x, state=None):
        # x shape: [batch_size, num_steps]
        # 注释：把字符索引转成 one-hot 向量，作为 LSTM 每个时间步的输入。
        x = F.one_hot(x.T.long(), self.vocab_size).float()
        # one-hot x shape: [num_steps, batch_size, vocab_size]
        # 注释：把 one-hot 序列送入 LSTM，得到每个时间步的隐状态和最新记忆状态。
        y, state = self.lstm(x, state)
        # y shape: [num_steps, batch_size, num_hiddens]
        # 注释：将 LSTM 的所有时间步输出拉平后送入线性层，得到下一字符分数。
        output = self.linear(y.reshape(-1, y.shape[-1]))
        # output shape: [num_steps * batch_size, vocab_size]
        # 注释：同时返回所有时间步的预测分数和 LSTM 最新隐状态。
        return output, state


# 注释：定义文本生成函数，用前缀预热 LSTM 状态后逐字符生成。
def predict(prefix, num_preds, net, char_to_idx, idx_to_char, device):
    # 注释：用文档字符串说明当前函数或类的用途。
    """Generate text after a prefix."""
    # 注释：切换到评估模式，关闭训练阶段专用行为。
    net.eval()
    # 注释：把前缀的第一个字符索引放入生成结果列表，作为起点。
    output = [char_to_idx[prefix[0]]]
    # 注释：把 `state` 初始化为空状态，表示还没有历史记忆。
    state = None

    # 注释：逐字符读取前缀，用真实前缀更新 LSTM 隐状态。
    for ch in prefix[1:]:
        # 注释：把 Python 数据转换成 PyTorch 张量并保存到 `x`。
        x = torch.tensor([[output[-1]]], device=device)
        # 注释：执行模型前向传播，得到预测结果并保存到 `_, state`。
        _, state = net(x, state)
        # 注释：把真实前缀字符或模型预测字符加入生成结果列表。
        output.append(char_to_idx[ch])

    # 注释：循环生成指定数量的新字符。
    for _ in range(num_preds):
        # 注释：把 Python 数据转换成 PyTorch 张量并保存到 `x`。
        x = torch.tensor([[output[-1]]], device=device)
        # 注释：执行模型前向传播，得到预测结果并保存到 `y, state`。
        y, state = net(x, state)
        # 注释：从模型输出分数中取最大值的索引，作为下一个生成字符。
        next_idx = int(y.argmax(dim=1))
        # 注释：把真实前缀字符或模型预测字符加入生成结果列表。
        output.append(next_idx)

    # 注释：把生成的字符索引还原成字符串并返回。
    return "".join(idx_to_char[i] for i in output)


# 注释：定义梯度裁剪函数，防止 LSTM 训练时梯度爆炸。
def grad_clipping(net, theta):
    # 注释：将 `[p for p in net.parameters() if p.requires_grad]` 这一步的结果保存到 `params`，供后续代码使用。
    params = [p for p in net.parameters() if p.requires_grad]
    # 注释：将 `torch.sqrt(sum(torch.sum(p.grad**2) for p in params if p.grad is not None))` 这一步的结果保存到 `norm`，供后续代码使用。
    norm = torch.sqrt(sum(torch.sum(p.grad**2) for p in params if p.grad is not None))
    # 注释：如果整体梯度范数超过阈值，就进行梯度裁剪。
    if norm > theta:
        # 注释：遍历所有可训练参数，准备按比例缩小梯度。
        for param in params:
            # 注释：执行 `if param.grad is not None:` 这行代码，完成当前函数中的对应步骤。
            if param.grad is not None:
                # 注释：将 `theta / norm` 这一步的结果保存到 `param.grad[:] *`，供后续代码使用。
                param.grad[:] *= theta / norm


# 注释：定义训练函数，完成多轮前向传播、反向传播和参数更新。
def train(net, data_iter, char_to_idx, idx_to_char, device, num_epochs, lr):
    # 注释：创建交叉熵损失函数，用于多分类任务。
    loss = nn.CrossEntropyLoss()
    # 注释：创建优化器，用于根据梯度更新模型参数。
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # 注释：执行 `net.to(device)` 这行代码，完成当前函数中的对应步骤。
    net.to(device)

    # 注释：按 epoch 多轮训练模型。
    for epoch in range(num_epochs):
        # 注释：把 `total_loss` 初始化为 0，用于后续累计统计量。
        total_loss = 0.0
        # 注释：把 `total_num` 初始化为 0，用于后续累计统计量。
        total_num = 0
        # 注释：切换到训练模式，启用训练阶段行为。
        net.train()

        # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
        for x, y in data_iter:
            # 注释：把数据或模型移动到当前训练设备，并保存到 `x`。
            x = x.to(device)
            # 注释：把数据或模型移动到当前训练设备，并保存到 `y`。
            y = y.T.reshape(-1).to(device)

            # 注释：执行模型前向传播，得到预测结果并保存到 `y_hat, _`。
            y_hat, _ = net(x)
            # 注释：根据模型预测和真实标签计算当前 batch 的损失。
            l = loss(y_hat, y)

            # 注释：清空上一次累积的梯度，避免梯度叠加。
            optimizer.zero_grad()
            # 注释：根据损失反向传播，计算每个参数的梯度。
            l.backward()
            # 注释：在优化器更新前裁剪梯度，防止梯度爆炸导致训练不稳定。
            grad_clipping(net, 1)
            # 注释：让优化器根据当前梯度更新模型参数。
            optimizer.step()

            # 注释：把当前 batch 的统计量累加到 `total_loss` 中。
            total_loss += l.item() * y.numel()
            # 注释：把当前 batch 的统计量累加到 `total_num` 中。
            total_num += y.numel()

        # 注释：执行 `if (epoch + 1) % 10 == 0 or epoch == 0:` 这行代码，完成当前函数中的对应步骤。
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # 注释：将 `math.exp(total_loss / total_num)` 这一步的结果保存到 `ppl`，供后续代码使用。
            ppl = math.exp(total_loss / total_num)
            # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
            print(f"epoch {epoch + 1}, perplexity {ppl:.2f}")
            # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
            print(predict("time traveller", 50, net, char_to_idx, idx_to_char, device))


# 注释：定义脚本主入口，把当前练习的数据、模型和训练流程串起来。
def main():
    # 注释：固定 PyTorch 随机种子，让数据打乱和参数初始化尽量可复现。
    torch.manual_seed(33)

    # 注释：将 `(` 这一步的结果保存到 `text`，供后续代码使用。
    text = (
        # 注释：执行 `"time traveller the time machine moves through time "` 这行代码，完成当前函数中的对应步骤。
        "time traveller the time machine moves through time "
        # 注释：执行 `"the traveller sees strange worlds in the future "` 这行代码，完成当前函数中的对应步骤。
        "the traveller sees strange worlds in the future "
    # 注释：执行 `) * 80` 这行代码，完成当前函数中的对应步骤。
    ) * 80

    # 注释：设置 `batch_size` 参数，控制上方函数调用的行为。
    batch_size = 32
    # 注释：将 `20` 这一步的结果保存到 `num_steps`，供后续代码使用。
    num_steps = 20
    # 注释：将 `64` 这一步的结果保存到 `num_hiddens`，供后续代码使用。
    num_hiddens = 64
    # 注释：将 `20` 这一步的结果保存到 `num_epochs`，供后续代码使用。
    num_epochs = 20
    # 注释：将 `1.0` 这一步的结果保存到 `lr`，供后续代码使用。
    lr = 1.0

    # 注释：将 `build_dataset(text, num_steps)` 这一步的结果保存到 `dataset, char_to_idx, idx_to_char`，供后续代码使用。
    dataset, char_to_idx, idx_to_char = build_dataset(text, num_steps)
    # 注释：把数据集封装成可按 batch 迭代的 `data_iter`。
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 注释：将 `try_gpu()` 这一步的结果保存到 `device`，供后续代码使用。
    device = try_gpu()
    # 注释：将 `CharLSTM(vocab_size=len(char_to_idx), num_hiddens=num_hiddens)` 这一步的结果保存到 `net`，供后续代码使用。
    net = CharLSTM(vocab_size=len(char_to_idx), num_hiddens=num_hiddens)

    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("vocab size:", len(char_to_idx))
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("training on:", device)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("before training:")
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print(predict("time traveller", 50, net.to(device), char_to_idx, idx_to_char, device))

    # 注释：启动训练流程，将模型、数据、训练轮数和学习率传入训练函数。
    train(net, data_iter, char_to_idx, idx_to_char, device, num_epochs, lr)

    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("after training:")
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print(predict("time traveller", 100, net, char_to_idx, idx_to_char, device))


# 注释：判断当前文件是否被直接运行，直接运行时才调用 main。
if __name__ == "__main__":
    # 注释：调用主函数，启动当前脚本的完整练习流程。
    main()
