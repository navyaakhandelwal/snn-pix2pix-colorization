%%writefile /kaggle/working/step2_snn_generator.py
"""
STEP 2: SNN Generator Architecture (slim U-Net, 5 levels, 128px input)
Final layer outputs raw membrane potential — no spiking on output.
"""

import torch
import torch.nn as nn


class SlimDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True, use_dropout=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))  # replaced by LIF in Step 3
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)
        self.out_ch = out_ch

    def forward(self, x):
        return self.model(x)


class SlimUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),  # replaced by LIF in Step 3
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)
        self.out_ch = out_ch

    def forward(self, x):
        return self.model(x)


class SlimGeneratorUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        self.down1 = SlimDownBlock(in_ch,  64,  normalize=False)
        self.down2 = SlimDownBlock(64,  128, normalize=True)
        self.down3 = SlimDownBlock(128, 256, normalize=True)
        self.down4 = SlimDownBlock(256, 256, normalize=True)
        self.down5 = SlimDownBlock(256, 256, normalize=True)

        self.up1 = SlimUpBlock(256,     256, use_dropout=True)
        self.up2 = SlimUpBlock(256+256, 256, use_dropout=True)
        self.up3 = SlimUpBlock(256+256, 128, use_dropout=False)
        self.up4 = SlimUpBlock(128+128, 64,  use_dropout=False)

        self.final_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(64+64, out_ch, kernel_size=3, stride=1, padding=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))

        u5_in = torch.cat([u4, d1], dim=1)
        u5 = self.final_up(u5_in)
        out = self.final_conv(u5)
        out = self.final_act(out)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SlimGeneratorUNet(in_ch=1, out_ch=2)
    dummy_L = torch.randn(4, 1, 128, 128)
    output = model(dummy_L)
    print("Output shape:", output.shape)
    print(f"Total parameters: {model.count_parameters():,}")
    assert output.shape == (4, 2, 128, 128)
    print("Step 2 complete.")