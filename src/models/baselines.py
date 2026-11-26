import torch.nn as nn
import segmentation_models_pytorch as smp


class BaselineWrapper(nn.Module):
    def __init__(self, arch, **kwargs):
        super().__init__()
        self.net = arch(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=2,
            classes=1,
            **kwargs,
        )

    def forward(self, sar, env=None):
        return self.net(sar)


def build_baseline(name):
    if name == "unet":
        return BaselineWrapper(smp.Unet)
    if name == "deeplabv3plus":
        return BaselineWrapper(smp.DeepLabV3Plus)
    if name == "attn_unet":
        return BaselineWrapper(smp.Unet, decoder_attention_type="scse")
    raise ValueError(f"Unknown baseline model: {name}")
