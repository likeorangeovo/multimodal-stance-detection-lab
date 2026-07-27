# 注释：导入 PyTorch 主库，用于张量计算、自动求导和模型训练。
import torch

# 注释：定义脚本主入口，把当前练习的数据、模型和训练流程串起来。
def main():
    # print("11111")
    # x = torch.tensor([[1,2,3],[4,5,6]])
    # y = torch.zeros(2,3)
    # z = torch.ones(2,3)
    # r = torch.rand(2,3)
    #
    #
    # print(x)
    # print(y)
    # print(z)
    # print(r)
    #
    # print(x+y)
    # print(x*y)
    # print(x.sum())
    # print(r.mean())

    # 注释：生成随机张量并保存到 `m1`。
    m1 = torch.rand(2, 3)
    # 注释：生成随机张量并保存到 `m2`。
    m2 = torch.rand(3, 4)
    # 注释：打印当前结果、模型信息或生成文本，方便观察运行状态。
    print(m1 @ m2)

# 注释：判断当前文件是否被直接运行，直接运行时才调用 main。
if __name__ == '__main__':
    # 注释：调用主函数，启动当前脚本的完整练习流程。
    main()
