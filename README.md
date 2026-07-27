# Multimodal Stance Detection Lab

这个仓库是我的多模态立场检测学习记录。现在还在前期打基础，主要围绕 PyTorch、CNN、LSTM、预训练 ResNet 和一个简单图文匹配 pipeline 展开。

后面的方向是文本立场检测、多模态融合、CLIP、MMSD 数据集和完整项目工程化。

## 目录

```text
.
├── src/
│   ├── day01_tensors.py
│   ├── day02_autograd.py
│   ├── day02_linear_regression.py
│   ├── day03_dataloader_dataset_module.py
│   ├── day04_mlp_classifier.py
│   ├── day06_alexnet_fashionmnist.py
│   ├── day07_lstm_text_generation.py
│   ├── day08_training_tools/
│   │   ├── main.py
│   │   ├── checkpoints/
│   │   └── runs/
│   ├── day09_resnet_feature_extractor.py
│   └── day10_image_text_matching/
├── docs/
│   └── review_notes.md
├── data/
└── requirements.txt
```

`src/` 放每天的代码，`docs/review_notes.md` 放复习总结。

`checkpoints/` 是模型权重保存目录，现在放在对应的小项目里面。比如 Day8 的 MLP checkpoint 在 `src/day08_training_tools/checkpoints/`，Day10 的图文匹配 checkpoint 在 `src/day10_image_text_matching/checkpoints/`。

`runs/` 是 TensorBoard 日志目录，不是源码。训练时用 `SummaryWriter` 写入 loss、accuracy、learning rate 等曲线，之后可以用 TensorBoard 打开看训练过程。它也放在对应的小项目目录里。

## 计划进度

原计划是 16 周，从 PyTorch 基础一路推进到多模态立场检测项目。这里先放一个总表，方便看目前走到哪里。

| 阶段 | 时间 | 内容 | 当前状态 |
| --- | --- | --- | --- |
| 阶段一 | 第 1-2 周 | PyTorch、Dataset、MLP、CNN、LSTM、训练工具、ResNet、图文匹配 | 进行中，Day1-Day10 已有代码初版 |
| 阶段二 | 第 3-4 周 | 文本立场检测，BERT baseline，SemEval 数据集 | 未开始 |
| 阶段三 | 第 5-6 周 | 视觉特征提取，CNN / EfficientNet / ViT 对比 | 未开始 |
| 阶段四 | 第 7-8 周 | 多模态学习基础，融合策略、跨模态注意力、CLIP | 未开始 |
| 阶段五 | 第 9-11 周 | 多模态立场检测核心方法，MMSD、BERT+ResNet、Transformer 融合 | 未开始 |
| 阶段六 | 第 12-14 周 | 项目工程化，实验管理、断点续训、Gradio Demo、GitHub 发布 | 未开始 |
| 阶段七 | 第 15-16 周 | 多模态大模型和研究拓展，LLaVA、Qwen-VL、论文阅读、研究计划 | 未开始 |

当前更细一点的进度：

| Day | 内容 | 文件 | 状态 |
| --- | --- | --- | --- |
| Day1 | Tensor 基础 | `src/day01_tensors.py` | 已完成 |
| Day2 | 自动求导、线性回归 | `src/day02_autograd.py`, `src/day02_linear_regression.py` | 已完成 |
| Day3 | Dataset、DataLoader、nn.Module | `src/day03_dataloader_dataset_module.py` | 已完成 |
| Day4 | MLP 分类器 | `src/day04_mlp_classifier.py` | 已完成 |
| Day5 | 训练和验证脚本整理 | 暂无单独文件 | 跳过/合并到后续 |
| Day6 | CNN / AlexNet / FashionMNIST | `src/day06_alexnet_fashionmnist.py` | 已完成 |
| Day7 | 字符级 LSTM 文本生成 | `src/day07_lstm_text_generation.py` | 已完成 |
| Day8 | checkpoint、scheduler、TensorBoard | `src/day08_training_tools/main.py` | 已完成 |
| Day9 | 预训练 ResNet 特征提取 | `src/day09_resnet_feature_extractor.py` | 已完成 |
| Day10 | 图文匹配小项目 | `src/day10_image_text_matching/` | 已完成初版 |

## 运行

建议从仓库根目录运行，并使用 conda 环境 `env_3.11`：

```powershell
conda activate env_3.11

python src\day01_tensors.py
python src\day02_autograd.py
python src\day02_linear_regression.py
python src\day03_dataloader_dataset_module.py
python src\day04_mlp_classifier.py
python src\day06_alexnet_fashionmnist.py
python src\day07_lstm_text_generation.py
python src\day08_training_tools\main.py
python src\day09_resnet_feature_extractor.py
python src\day10_image_text_matching\main.py
```

如果当前终端没有初始化 conda，也可以直接用完整路径运行：

```powershell
D:\Anaconda\envs\env_3.11\python.exe src\day10_image_text_matching\main.py
```

不要直接用系统的 `python` 或 conda base，里面可能没装 `torch`，会报 `ModuleNotFoundError: No module named 'torch'`。

## 备注

Day6 默认只用 FashionMNIST 的小子集，主要是为了快速跑通。

Day7 用内置字符文本，不需要下载数据。

Day9 第一次运行会下载 torchvision 的 ResNet18 预训练权重。

Day10 用合成的颜色和形状图片做正负样本，数据量很小，重点是把图文匹配流程连起来，不是追求很高准确率。
