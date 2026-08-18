# Multimodal Stance Detection Lab

这是一个多模态立场检测学习与实验仓库。当前项目已经完成 PyTorch 基础、图像/文本基础模型、图文匹配小练习、基于 SemEval-2016 Task 6 的文本立场检测 baseline，以及多模态融合范式的理论梳理。


```text
PyTorch 基础
    -> 文本立场检测 baseline
    -> 图像与图文匹配基础
    -> CLIP / ViT / 文本模型特征提取
    -> 多模态 late fusion baseline
    ----------------------------------------
    -> CLIP 相似度增强 / cross-attention 融合
    -> 消融实验与错误分析
    -> Gradio Demo / README / 实验报告
```

## 当前状态

| 模块 | 内容 | 状态 |
| --- | --- | --- |
| PyTorch 基础 | Tensor、autograd、Dataset、DataLoader、`nn.Module`、MLP | 已完成 |
| 经典模型练习 | AlexNet/FashionMNIST、字符级 LSTM、checkpoint、scheduler、TensorBoard | 已完成 |
| 图像特征 | 预训练 ResNet18 特征提取、ViT 基础理解与特征提取练习 | 已完成 |
| 图文匹配 | 合成图文匹配数据，ResNet18 + LSTM + MLP 二分类 pipeline | 已完成初版 |
| 文本立场检测 | SemEval-2016 Task 6、BERT baseline、RoBERTa 对比、prompt-based 方法 | 已完成初版 |
| 多模态融合 | 早期/晚期融合、cross-attention、单流/双流结构、立场检测选型路线 | 理论梳理已完成 |
| 项目收束 | README、阶段笔记、进度记录 | 已完结 |

## 目录结构

```text
.
├── data/
│   └── semeval2016_task6/
├── docs/
│   ├── day1_to_day10_notes.md
│   ├── day11_stance_detection.md
│   ├── day12_transformer_bert_summary.md
│   ├── day13_to_day20_week4_summary.md
│   ├── day18_prompt_based_stance.md
│   ├── day26_to_day30_vision_feature_summary.md
│   └── day31_multimodal_fusion_summary.md
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
│   ├── day10_image_text_matching/
│   │   ├── checkpoint.py
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── eval.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── README.md
│   ├── day14_bert_sequence_classification.py
│   ├── day15_bert_stance_baseline.py
│   ├── day17_roberta_model_demo.py
│   └── day18_prompt_based_stance_detection.py
├── requirements.txt
└── README.md
```

`src/` 放每天的代码和小实验，`docs/` 放阶段笔记、论文/方法总结和复盘材料。`checkpoints/` 和 `runs/` 是训练过程生成的模型权重与 TensorBoard 日志。

## 已完成内容

| Day | 内容 | 文件/文档 | 状态 |
| --- | --- | --- | --- |
| Day1 | Tensor 创建、索引、运算 | `src/day01_tensors.py` | 已完成 |
| Day2 | 自动求导、线性回归 | `src/day02_autograd.py`, `src/day02_linear_regression.py` | 已完成 |
| Day3 | Dataset、DataLoader、`nn.Module` | `src/day03_dataloader_dataset_module.py` | 已完成 |
| Day4 | MLP 分类器 | `src/day04_mlp_classifier.py` | 已完成 |
| Day5 | 完整训练/验证脚本整理 | 合并到 Day8/Day10 | 已合并 |
| Day6 | AlexNet / FashionMNIST | `src/day06_alexnet_fashionmnist.py` | 已完成 |
| Day7 | 字符级 LSTM 文本生成 | `src/day07_lstm_text_generation.py` | 已完成 |
| Day8 | checkpoint、scheduler、TensorBoard | `src/day08_training_tools/main.py` | 已完成 |
| Day9 | 预训练 ResNet18 特征提取 | `src/day09_resnet_feature_extractor.py` | 已完成 |
| Day10 | 图文匹配小项目 | `src/day10_image_text_matching/` | 已完成初版 |
| Day11 | 立场检测任务与综述 | `docs/day11_stance_detection.md` | 已完成 |
| Day12 | Transformer / BERT 总结 | `docs/day12_transformer_bert_summary.md` | 已完成 |
| Day13 | SemEval-2016 Task 6 数据探索 | `docs/day13_to_day20_week4_summary.md` | 已完成 |
| Day14 | HuggingFace BERT 分类模型加载 | `src/day14_bert_sequence_classification.py` | 已完成 |
| Day15 | BERT 立场检测 baseline | `src/day15_bert_stance_baseline.py` | 已完成 |
| Day16 | 错误样本与类别不平衡分析 | `docs/day13_to_day20_week4_summary.md` | 已完成 |
| Day17 | RoBERTa / DistilRoBERTa 对比实验 | `src/day17_roberta_model_demo.py` | 已完成 |
| Day18 | Prompt-based 立场检测 | `src/day18_prompt_based_stance_detection.py`, `docs/day18_prompt_based_stance.md` | 已完成 |
| Day19 | BERT 立场检测论文阅读 | `docs/day13_to_day20_week4_summary.md` | 已完成 |
| Day20 | 阶段整理与可视化脚本 | `docs/day13_to_day20_week4_summary.md` | 基础总结已完成，脚本可继续补 |
| Day26-Day30 | 视觉特征提取方法总结 | `docs/day26_to_day30_vision_feature_summary.md` | 已完成 |
| Day31 | 多模态融合范式总结 | `docs/day31_multimodal_fusion_summary.md` | 已完成 |

## 环境

基础依赖见 `requirements.txt`：

```text
torch
torchvision
torchaudio
matplotlib
tensorboard
```

Day14 之后的 HuggingFace 实验还需要：

```powershell
pip install transformers
```

## 运行方式

建议从仓库根目录运行，并使用已经配置好的 conda 环境：

```powershell
conda activate env_3.11
```

基础阶段脚本：

```powershell
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

文本立场检测阶段脚本：

```powershell
python src\day14_bert_sequence_classification.py
python src\day15_bert_stance_baseline.py
python src\day17_roberta_model_demo.py
python src\day18_prompt_based_stance_detection.py
```

如果当前终端没有初始化 conda，也可以直接用环境里的 Python：

```powershell
D:\Anaconda\envs\env_3.11\python.exe src\day15_bert_stance_baseline.py
```

不要直接使用系统 Python 或 conda base，里面可能没有安装 `torch`、`transformers` 等依赖。

## 数据说明

当前文本立场检测实验使用 SemEval-2016 Task 6，默认路径为：

```text
data/semeval2016_task6/semeval2016-task6-trainingdata.txt
data/semeval2016_task6/testdata-gold/SemEval2016-Task6-subtaskA-testdata-gold.txt
```

Day10 图文匹配数据是代码合成的小数据集，主要用于跑通：

```text
Dataset -> DataLoader -> image encoder -> text encoder -> feature fusion -> classifier -> train/eval/checkpoint
```

多模态数据集选型已在阶段笔记中讨论过，MMSD、PHEME 或其他图文谣言/立场相关数据集可作为参考方向。

## 备注

Day6 默认使用 FashionMNIST 的小子集，目的是快速验证 CNN 训练流程。

Day7 使用内置字符文本，不需要额外下载数据。

Day9 第一次运行会下载 torchvision 的 ResNet18 预训练权重。

Day10 使用合成颜色和形状图片做正负样本，重点是打通图文匹配流程，不追求高准确率。

Day14-Day18 依赖 SemEval 数据和 HuggingFace 模型下载，首次运行需要联网下载 tokenizer / model 权重。
