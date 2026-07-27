"""
Day 9: 使用预训练 ResNet 提取图像特征

目标：
1. 学会加载 torchvision 模型库里的预训练 ResNet。
2. 使用与预训练权重匹配的图像预处理。
3. 提取分类头之前的图像特征，为后续图文多模态任务做准备。

运行：
    python day9_resnet_feature_extractor.py

第一次运行会自动下载 ResNet18 的预训练权重。
"""

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


def get_device():
    """选择可用设备：有 CUDA 就用 GPU，否则用 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_demo_image(image_path):
    """
    生成一张简单的 RGB 示例图。

    这样 demo 不依赖外部图片文件，下载好模型权重后就能直接跑。
    如果你想换成自己的图片，只需要把 main() 里的 image_path 改掉即可。
    """
    image_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 320, 240
    image = Image.new("RGB", (width, height), color=(245, 247, 250))
    draw = ImageDraw.Draw(image)

    # 画一个简单的“桌面上的杯子”场景，让预训练 CNN 能看到颜色、边缘和形状。
    draw.rectangle((0, 170, width, height), fill=(205, 222, 210))
    draw.ellipse((110, 70, 210, 180), fill=(81, 125, 180), outline=(32, 64, 110), width=4)
    draw.ellipse((128, 58, 192, 86), fill=(236, 242, 248), outline=(32, 64, 110), width=3)
    draw.arc((185, 95, 245, 155), start=-70, end=80, fill=(32, 64, 110), width=8)
    draw.rectangle((36, 142, 95, 162), fill=(228, 169, 96))
    draw.ellipse((245, 45, 282, 82), fill=(242, 197, 66))

    image.save(image_path)
    return image_path


def load_image(image_path):
    """读取图片并转成 RGB，避免 PNG 透明通道或灰度图导致通道数不匹配。"""
    return Image.open(image_path).convert("RGB")


def build_feature_extractor(device):
    """
    加载预训练 ResNet，并把它改造成特征提取器。

    ResNet 原始结构大致是：
        输入图片 -> conv/bn/relu/pool -> layer1~layer4 -> avgpool -> fc 分类头

    我们这里取两个中间节点：
    - layer4: 最后一组卷积特征图，形状通常是 [B, 512, 7, 7]
    - avgpool: 全局平均池化后的向量，形状通常是 [B, 512, 1, 1]

    多模态里常用的是 avgpool 展平后的向量，因为它可以当作一张图的 embedding。
    """
    weights = ResNet18_Weights.DEFAULT

    # weights.transforms() 会返回与这套预训练权重匹配的 resize、crop、归一化等预处理。
    preprocess = weights.transforms()

    # 如果想换成 ResNet50：
    # from torchvision.models import ResNet50_Weights, resnet50
    # weights = ResNet50_Weights.DEFAULT
    # model = resnet50(weights=weights)
    model = resnet18(weights=weights)
    model.eval()
    model.to(device)

    # create_feature_extractor 会截住指定节点的输出，不需要手动拆模型层。
    extractor = create_feature_extractor(
        model,
        return_nodes={
            "layer4": "feature_map",
            "avgpool": "embedding",
        },
    )
    extractor.eval()
    extractor.to(device)

    return extractor, preprocess, weights


def extract_features(image, extractor, preprocess, device):
    """
    把单张 PIL 图片送入 ResNet 特征提取器。

    注意：
    - preprocess(image) 输出形状是 [3, H, W]
    - unsqueeze(0) 增加 batch 维度，变成 [1, 3, H, W]
    """
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # 推理阶段不需要梯度，关闭梯度能节省内存并加快运行。
    with torch.no_grad():
        outputs = extractor(image_tensor)

    feature_map = outputs["feature_map"]
    embedding = outputs["embedding"].flatten(start_dim=1)
    return feature_map, embedding


def show_top_predictions(image, weights, device):
    """
    用原始分类头看一下预训练模型“觉得”图片像什么。

    这一步不是特征提取必须的，只是帮助你确认预处理和权重都工作正常。
    """
    model = resnet18(weights=weights).eval().to(device)

    # 这里为了演示分类结果，重新使用完整模型做一次前向传播。
    # 更正式的项目里可以复用同一个 backbone，避免重复计算。
    preprocess = weights.transforms()
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = logits.softmax(dim=1)
        top_probs, top_indices = probabilities.topk(5, dim=1)

    categories = weights.meta["categories"]
    print("\nTop-5 ImageNet 预测，仅用于 sanity check：")
    for rank, (prob, class_index) in enumerate(zip(top_probs[0], top_indices[0]), start=1):
        label = categories[class_index.item()]
        print(f"{rank}. {label:20s} probability={prob.item():.4f}")


def save_embedding(embedding, output_path):
    """把图像 embedding 保存下来，后续可以接文本特征、检索模型或分类器。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding.cpu(), output_path)


def main():
    device = get_device()
    print(f"Using device: {device}")

    image_path = Path("data/day9_demo_image.png")
    if not image_path.exists():
        build_demo_image(image_path)
        print(f"Created demo image: {image_path}")
    else:
        print(f"Using existing image: {image_path}")

    image = load_image(image_path)
    extractor, preprocess, weights = build_feature_extractor(device)
    feature_map, embedding = extract_features(image, extractor, preprocess, device)

    print("\nResNet 特征提取结果：")
    print(f"feature_map shape: {tuple(feature_map.shape)}")
    print(f"embedding shape:   {tuple(embedding.shape)}")
    print(f"embedding 前 8 个数: {embedding[0, :8].cpu().tolist()}")

    output_path = Path("data/day9_resnet18_embedding.pt")
    save_embedding(embedding, output_path)
    print(f"\nSaved embedding tensor to: {output_path}")

    show_top_predictions(image, weights, device)


if __name__ == "__main__":
    main()
