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
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError

        self.fc = nn.Linear(final_in_features, fc_dim)

    def forward(self, img):
        feature = self.backbone(img)
        output = self.fc(feature)
        return output


if __name__ == "__main__":
    model = SimpleModel()
    print(model)
