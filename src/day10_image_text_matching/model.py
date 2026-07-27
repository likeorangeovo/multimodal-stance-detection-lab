"""Model definitions for Day10 image-text matching.

TODO:
- TextEncoder
- ImageTextMatchingModel
- optional freeze/unfreeze helpers for ResNet
"""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights,resnet18


class TextLstmEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        output, (h_n, c_n) = self.lstm(x)
        text_feature = h_n[-1]

        return text_feature

class  ImageTextMatchingModel(nn.Module):
    def __init__(self, vocab_size, freeze_resnet=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.image_encoder = resnet18(weights=weights)
        self.image_encoder.fc = nn.Identity()

        if freeze_resnet:
            for param in self.parameters():
                param.requires_grad = False

        self.text_encoder = TextLstmEncoder(vocab_size)
        self.classify = nn.Sequential(
            nn.Linear(512+128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, image, text):
        text_featrue = self.text_encoder(text)
        image_featrue = self.image_encoder(image)
        features = torch.cat((text_featrue, image_featrue), dim=1)
        logits = self.classify(features)
        logits = logits.squeeze(1)
        return logits

# if __name__ == "__main__":
#     model = ImageTextMatchingModel(vocab_size=10)
#
#     images = torch.randn(4, 3, 224, 224)
#     texts = torch.randint(0, 10, (4, 2))
#
#     logits = model(images, texts)
#
#     print(logits)