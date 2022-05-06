""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class adaptUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(adaptUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc1 = DoubleConv(n_channels, 64)
        self.inc2 = DoubleConv(n_channels+1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = OutConv(64, 1)
        # self.outv = OutConv(64, 1)

    def forward(self, xlist):
        if len(xlist) == 1:
            x = xlist[0]
            x1 = self.inc1(x)
        else:
            x = torch.concat([xlist[0], xlist[1]], 1)
            x1 = self.inc2(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out(x)
        # pvalue = self.outc(x * torch.sigmoid(logits))
        return logits

if __name__ == '__main__':
    tmodel = adaptUnet(n_channels=3, n_classes=2, bilinear=True)
    img = torch.rand([2, 3, 224, 224])
    feat = torch.rand([2, 1, 224, 224])
    f, pvalue = tmodel([img, feat])
    print (f.size(), pvalue.size())
    f, pvalue = tmodel([img])
    print (f.size(), pvalue.size())
