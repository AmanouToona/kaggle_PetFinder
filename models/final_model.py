from turtle import forward
import timm
import torch
from torch import nn


class UseMeta(nn.Module):
    def __init__(
        self,
        base_model: str = "resnet50d",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.1,
        fc_dim: int = 1,
    ):
        super().__init__()

        if hasattr(timm.models, base_model):
            self.backbone = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels, num_classes=0)
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.dense1 = nn.LazyLinear(out_features=128)
        self.dense2 = nn.Linear(140, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, fc_dim)

    def forward(self, img, meta):
        x = self.backbone(img)
        x = self.dropout(x)
        feature = self.dense1(x)

        x = torch.cat([feature, meta], dim=1)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return output, feature, meta


class NoUseMeta(nn.Module):
    def __init__(
        self,
        base_model: str = "resnet50d",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.3,
        fc_dim: int = 1,
    ):
        super().__init__()

        if hasattr(timm.models, base_model):
            self.backbone = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels, num_classes=0)
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.dense1 = nn.LazyLinear(out_features=128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, fc_dim)

    def forward(self, img):
        x = self.backbone(img)
        x = self.dropout(x)
        x = self.dense1(x)

        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return output


if __name__ == "__main__":
    model = UseMeta()
    print(model)
