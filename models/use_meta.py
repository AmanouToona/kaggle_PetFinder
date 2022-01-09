import timm
import torch
from torch import nn


class PawpularMetaModel(nn.Module):
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
            final_in_features = self.backbone.fc.in_features
            if base_model == "swin_tiny_patch4_window7_224":
                final_in_features = self.backbone.head.out_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(final_in_features, 128)
        self.dense2 = nn.Linear(140, 64)
        self.dense3 = nn.Linear(64, 32)
        self.dense4 = nn.Linear(32, 1)

        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(final_in_features, fc_dim))

    def forward(self, img, meta):
        x = self.backbone(img)
        feature = self.dense1(x)

        x = torch.cat([feature, meta], dim=1)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)

        return output, feature, meta


if __name__ == "__main__":
    model = PawpularMetaModel()
    model = timm.create_model("resnet50d")
    print(model)
