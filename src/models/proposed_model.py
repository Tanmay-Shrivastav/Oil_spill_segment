import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class FiLMUNet(nn.Module):
    def __init__(self, env_dim=5, hidden_dim=64, film_dropout=0.25):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=2,
            classes=1,
        )
        self.film_dropout = film_dropout
        self.stage_channels = list(self.unet.encoder.out_channels[1:])
        total_ch = sum(self.stage_channels)
        self.mlp = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * total_ch),
        )

    def forward(self, sar, env=None):
        features = list(self.unet.encoder(sar))
        if env is not None:
            b = sar.shape[0]
            params = self.mlp(env)
            gammas, betas = params.chunk(2, dim=1)
            gammas = 1.0 + gammas
            if self.training and self.film_dropout > 0:
                keep = (torch.rand(b, 1, device=sar.device) >= self.film_dropout).to(params.dtype)
                gammas = keep * gammas + (1.0 - keep)
                betas = keep * betas

            offset = 0
            for idx, ch in enumerate(self.stage_channels, start=1):
                g = gammas[:, offset : offset + ch].view(b, ch, 1, 1)
                be = betas[:, offset : offset + ch].view(b, ch, 1, 1)
                features[idx] = g * features[idx] + be
                offset += ch

        dec_out = self.unet.decoder(features)
        return self.unet.segmentation_head(dec_out)
