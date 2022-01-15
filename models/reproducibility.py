import timm
from torch import nn


class Repro001(nn.Module):
    def __init__(
        self, model_name, out_features, inp_channels, pretrained,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, in_chans=inp_channels, num_classes=out_features
        )

    def forward(self, image):
        output = self.model(image)
        return output


if __name__ == "__main__":
    model = timm.create_model("resnet50d")
    print(model)
