# 注释：导入 PyTorch 主库，用于张量计算、自动求导和模型训练。
import torch


# 注释：定义 `synthetic_data` 函数，封装一段可复用逻辑。
def synthetic_data(w, b, num_examples):
    # 注释：生成标准正态分布随机张量并保存到 `x`。
    x = torch.randn(num_examples, len(w))
    # 注释：将 `torch.matmul(x, w) + b` 这一步的结果保存到 `y`，供后续代码使用。
    y = torch.matmul(x, w) + b
    # 注释：把当前 batch 的统计量累加到 `y` 中。
    y += torch.normal(0,0.01,y.shape)
    # 注释：返回 `x,y.reshape(-1,1)` 的结果，供调用这个函数的代码继续使用。
    return x,y.reshape(-1,1)

# 注释：定义 `iter_data` 函数，封装一段可复用逻辑。
def iter_data(batch_size,features,labels):
    # 注释：将 `len(features)` 这一步的结果保存到 `num_examples`，供后续代码使用。
    num_examples = len(features)
    # 注释：将 `torch.randperm(num_examples)` 这一步的结果保存到 `indices`，供后续代码使用。
    indices = torch.randperm(num_examples)
    # 注释：执行 `for i in range(0,num_examples,batch_size):` 这行代码，完成当前函数中的对应步骤。
    for i in range(0,num_examples,batch_size):
        # 注释：把当前 batch 的输入和标签交给训练循环使用。
        yield features[indices[i:i+batch_size]],labels[indices[i:i+batch_size]]

# 注释：定义 `linear_regression` 函数，封装一段可复用逻辑。
def linear_regression(x,w,b):
    # 注释：返回 `torch.matmul(x,w) + b` 的结果，供调用这个函数的代码继续使用。
    return torch.matmul(x,w) + b

# 注释：定义 `loss` 函数，封装一段可复用逻辑。
def loss(y_hat,y):
    # 注释：将多个结果组合返回，便于调用处同时拿到训练和测试迭代器。
    return (y_hat - y.reshape(y_hat.shape))**2/2

# 注释：定义 `sgd` 函数，封装一段可复用逻辑。
def sgd(params,lr,batch_size):
    # 注释：进入无梯度模式，评估时不记录计算图以节省内存。
    with torch.no_grad():
        # 注释：遍历所有可训练参数，准备按比例缩小梯度。
        for param in params:
            # 注释：按学习率和平均梯度手动更新参数，完成一次梯度下降。
            param -= lr * param.grad / batch_size
            # 注释：把当前参数的梯度清零，防止下一次反向传播累加旧梯度。
            param.grad.zero_()

# 注释：定义脚本主入口，把当前练习的数据、模型和训练流程串起来。
def main():
    # 注释：执行 `torch.manual_seed(39)` 这行代码，完成当前函数中的对应步骤。
    torch.manual_seed(39)

    # 注释：把 Python 数据转换成 PyTorch 张量并保存到 `true_w`。
    true_w = torch.tensor([3.2, 0.7])
    # 注释：把 Python 数据转换成 PyTorch 张量并保存到 `true_b`。
    true_b = torch.tensor(0.9)


    # 注释：将 `synthetic_data(true_w,true_b,100)` 这一步的结果保存到 `features , labels`，供后续代码使用。
    features , labels = synthetic_data(true_w,true_b,100)
    # 注释：将 `torch.normal(0, 0.01, size=(2,1), requires_grad=True)` 这一步的结果保存到 `w`，供后续代码使用。
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    # 注释：将 `torch.zeros(1,requires_grad=True)` 这一步的结果保存到 `b`，供后续代码使用。
    b = torch.zeros(1,requires_grad=True)
    # 注释：将 `0.01` 这一步的结果保存到 `lr`，供后续代码使用。
    lr = 0.01
    # 注释：将 `50` 这一步的结果保存到 `num_epochs`，供后续代码使用。
    num_epochs = 50
    # 注释：设置 `batch_size` 参数，控制上方函数调用的行为。
    batch_size = 10

    # 注释：按 epoch 多轮训练模型。
    for epoch in range(num_epochs):
        # 注释：从 DataLoader 中取出一个 batch 的输入和标签。
        for batch_x, batch_y in iter_data(batch_size,features,labels):
            # 注释：将 `linear_regression(batch_x,w,b)` 这一步的结果保存到 `y_hat`，供后续代码使用。
            y_hat = linear_regression(batch_x,w,b)
            # 注释：根据模型预测和真实标签计算当前 batch 的损失。
            l = loss(y_hat,batch_y).sum()
            # 注释：根据损失反向传播，计算每个参数的梯度。
            l.backward()
            # 注释：执行 `sgd([w,b],lr,batch_size)` 这行代码，完成当前函数中的对应步骤。
            sgd([w,b],lr,batch_size)

        # 注释：进入无梯度模式，评估时不记录计算图以节省内存。
        with torch.no_grad():
            # 注释：根据模型预测和真实标签计算当前 batch 的损失。
            train_loss = loss(linear_regression(features,w,b),labels)
            # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
            print("epoch:",epoch+1,"loss:",train_loss.mean().item())

    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("w",w)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print("b",b)

# 注释：判断当前文件是否被直接运行，直接运行时才调用 main。
if __name__ == "__main__":
    # 注释：调用主函数，启动当前脚本的完整练习流程。
    main()
