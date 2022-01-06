import timm
from torch import nn


class SimpleModel(nn.Module):
    def __init__(
        self,
        base_model: str = "resnet50d",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.0,
        fc_dim: int = 1,
    ):
        super().__init__()

        if hasattr(timm.models, base_model):
            self.backbone = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels)
            final_in_features = self.backbone.num_features
            if base_model == "swin_tiny_patch4_window7_224":
                final_in_features = self.backbone.head.out_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError

        self.fc = nn.Linear(final_in_features, fc_dim)

    def forward(self, img):
        feature = self.backbone(img)
        output = self.fc(feature)
        return output

    def feature(self, img):
        feature = self.backbone(img)
        return feature


class SimpleModelSig(nn.Module):
    def __init__(
        self,
        base_model: str = "resnet50d",
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.0,
        fc_dim: int = 1,
    ):
        super().__init__()

        if hasattr(timm.models, base_model):
            self.backbone = timm.create_model(base_model, pretrained=pretrained, in_chans=in_channels)
            final_in_features = self.backbone.num_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError

        self.fc = nn.Linear(final_in_features, fc_dim)
        self.sig = nn.Sigmoid()

    def forward(self, img):
        feature = self.backbone(img)
        x = self.fc(feature)
        output = self.sig(x)
        return output

    def feature(self, img):
        feature = self.backbone(img)
        return feature

if __name__ == "__main__":
    model = SimpleModel()
    print(model)
