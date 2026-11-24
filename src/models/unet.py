import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential( # 2 Conv layers for 1 block
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)  
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    # 2 downsampling levels, skip connections, output: sigmoid
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = ConvBlock(3, base) # full res
        self.enc2 = ConvBlock(base, base*2) # 1/2 res
        self.bott = ConvBlock(base*2, base*4) # 1/4 res
        self.dec2 = ConvBlock(base*4+base*2, base*2) # deconv after concat with enc2
        self.dec1 = ConvBlock(base*2+base, base) # deconv after concat with enc1
        self.out = nn.Conv2d(base, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bott(self.pool(e2))

        up2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        out = torch.sigmoid(self.out(d1))
        return out[..., :h, :w]