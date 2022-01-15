from turtle import forward
import timm
import torch
from torch import nn


class UseMeta(nn.Module):
    def __init__(
        self, base_model: str = "resnet50d", pretrained: bool = True, in_channels: int = 3, fc_dim: int = 1,
    ):
        super().__init__()

        if hasattr(timm.models, base_model):
            self.model = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels, num_classes=0)
        else:
            raise NotImplementedError

        self.dense1 = nn.LazyLinear(out_features=128)
        self.dense2 = nn.Linear(140, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, fc_dim)

        # nn.init.xavier_normal_(self.dense1.weight)
        nn.init.xavier_normal_(self.dense2.weight)
        nn.init.xavier_normal_(self.dense3.weight)
        nn.init.xavier_normal_(self.dense4.weight)

        nn.init.constant_(self.dense2.bias, 0)
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.constant_(self.dense4.bias, 0)

    def forward(self, img, meta):
        x = self.model(img)
        feature = self.dense1(x)

        x = torch.cat([feature, meta], dim=1)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return output, feature, meta


class NoUseMeta(nn.Module):
    def __init__(
        self, base_model: str = "resnet50d", pretrained: bool = True, in_channels: int = 3, fc_dim: int = 1,
    ):
        super().__init__()

        if hasattr(timm.models, base_model):
            self.model = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels, num_classes=0)
        else:
            raise NotImplementedError

        self.dense1 = nn.LazyLinear(out_features=128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, fc_dim)

        # nn.init.xavier_normal_(self.dense1.weight)
        nn.init.xavier_normal_(self.dense2.weight)
        nn.init.xavier_normal_(self.dense3.weight)
        nn.init.xavier_normal_(self.dense4.weight)

        nn.init.constant_(self.dense2.bias, 0)
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.constant_(self.dense4.bias, 0)

    def forward(self, img):
        x = self.model(img)
        x = self.dense1(x)

        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return output


class NoMetaSwa(nn.Module):
    def __init__(
        self,
        base_model: str = "swin_large_patch4_window7_224",
        pretrained: bool = True,
        in_channels: int = 3,
        fc_dim: int = 1,
    ):
        super().__init__()

        if hasattr(timm.models, base_model):
            self.model = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels)
        else:
            raise NotImplementedError

        self.dense1 = nn.Linear(self.model.head.out_features, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, fc_dim)

        nn.init.xavier_normal_(self.dense1.weight)
        nn.init.xavier_normal_(self.dense2.weight)
        nn.init.xavier_normal_(self.dense3.weight)
        nn.init.xavier_normal_(self.dense4.weight)

        nn.init.constant_(self.dense1.bias, 0)
        nn.init.constant_(self.dense2.bias, 0)
        nn.init.constant_(self.dense3.bias, 0)
        nn.init.constant_(self.dense4.bias, 0)

    def forward(self, img):
        x = self.model(img)
        x = self.dense1(x)

        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        return output


if __name__ == "__main__":
    model = NoMetaSwa()
    print(model)
