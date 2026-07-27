"""Dataset and synthetic data generation for Day10.

TODO:
- build_vocab()
- encode_text()
- draw_shape_image()
- build_samples()
- ImageTextMatchDataset
- build_dataloaders()
"""
import torch
from PIL import Image, ImageDraw
import random
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights

colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black']
shapes = ['rectangle', 'circle', 'ellipse']

def build_text():
    res = []
    for color in colors:
        for shape in shapes:
           res.append((color, shape))
    return res

def build_samples():
    texts = build_text()
    res = []
    for color, shape in texts:  # 直接解包元素
        image = Image.new('RGB', (400, 400), color='white')
        draw = ImageDraw.Draw(image)
        if shape == 'rectangle':
            draw.rectangle((0, 0, 400, 400), fill=color)
        elif shape == 'circle':
            draw.ellipse((100, 100, 300, 300), fill=color, outline='black', width=2)
        else:
            draw.ellipse((100, 50, 300, 200), fill=color, outline='black', width=2)
        res.append({"color": color, "shape": shape, "text": color + " " + shape, "image": image, "label": 1})

        other_colors = [c for c in colors if c != color]
        other_shapes = [s for s in shapes if s != shape]
        neg_color = random.choice(other_colors)
        neg_shape = random.choice(other_shapes)
        res.append({"color": neg_color, "shape": neg_shape, "text": neg_color + " " + neg_shape, "image": image, "label": 0})
    return res


def build_vocab():
    res = {}
    for i in range(len(colors)):
        res[colors[i]] = i
    for j in range(len(shapes)):
        res[shapes[j]] = j+len(colors)
    return  res

def encode_text(text,vocab):
    res = []
    tokens = text.split()
    for token in tokens:
        res.append(vocab[token])
    return torch.tensor(res, dtype=torch.long)

class ImageTextMatchDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab
        weights = ResNet18_Weights.DEFAULT
        self.image_transform = weights.transforms()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample["image"]
        text = sample["text"]
        label = sample["label"]

        image_tensor = self.image_transform(image)
        text_tensor = encode_text(text, self.vocab)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image_tensor, text_tensor, label_tensor

# from torch.utils.data import DataLoader
# if __name__ == '__main__':
#     samples = build_samples()
#     vocab = build_vocab()
#     dataset = ImageTextMatchDataset(samples, vocab)
#     loader = DataLoader(dataset, batch_size=4, shuffle=True)
#
#     for images, texts, labels in loader:
#         print(images.shape)
#         print(texts.shape)
#         print(labels.shape)
#         print(texts)
#         print(labels)
#         break