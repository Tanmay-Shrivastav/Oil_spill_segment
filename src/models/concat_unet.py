import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class ConcatUNet(nn.Module):
    def __init__(self, env_dim=5, proj_dim=64):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=2,
            classes=1,
        )
        bottleneck_ch = self.unet.encoder.out_channels[-1]
        self.env_proj = nn.Sequential(
            nn.Linear(env_dim, proj_dim),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(bottleneck_ch + proj_dim, bottleneck_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, sar, env=None):
        features = list(self.unet.encoder(sar))
        b, _, h, w = features[-1].shape
        if env is None:
            env = torch.zeros(b, 5, device=sar.device, dtype=sar.dtype)
        env_feat = self.env_proj(env).view(b, -1, 1, 1).expand(-1, -1, h, w)
        features[-1] = self.fuse(torch.cat([features[-1], env_feat], dim=1))
        dec_out = self.unet.decoder(features)
        return self.unet.segmentation_head(dec_out)
