import torch
import torch.nn as nn
import torch.nn.functional as F

from dfpn import DFPN
from csem import CSEM
from torchvision import models


class ThermalCC(nn.Module):
    def __init__(self, pretrained=False):
        super(ThermalCC, self).__init__()
        
        last_inp_channels = 384
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(last_inp_channels, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, output_padding=0, bias=True),

        )
        
        self.reg_layer = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        indim_list = [256, 512, 512]
        outdim = 128
        self.fpn = DFPN(indim_list, outdim)

        self.context1 = CSEM(128)
        self.context2 = CSEM(256)
        self.context3 = CSEM(512)
        

        self._weight_init_()
        
        # Using VGG16 as backbone network
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        
        # Partition VGG16 into five encoder blocks
        features = list(vgg.features.children())
        self.features0 = nn.Sequential(*features[0:6])
        self.features1 = nn.Sequential(*features[6:13])
        self.features2 = nn.Sequential(*features[13:23])
        self.features3 = nn.Sequential(*features[23:33])
        self.features4 = nn.Sequential(*features[33:43])

    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, process=None):
        x0 = self.features0(x)
        
        x1 = self.features1(x0)
        x1 = self.context1(x1)
        
        x2 = self.features2(x1)
        x2 = self.context2(x2)
        
        x3 = self.features3(x2)
        x3 = self.context3(x3)
        
        x4 = self.features4(x3)
        

        point_feats, reg_feats = self.fpn([x2, x3, x4], None)
        
        # point
        p_h, p_w = x3.size(2), x3.size(3)
        pfs = []
        for f in point_feats[1:]:
            pfs.append(F.interpolate(f, size=(p_h, p_w), mode='bilinear'))
        pfs = torch.cat(pfs, 1)
        px = self.reg_layer(pfs)
        px = torch.abs(px)
        
        # regress
        r_h, r_w = x2.size(2), x2.size(3)
        rfs = []
        for f in reg_feats:
            rfs.append(F.interpolate(f, size=(r_h, r_w), mode='bilinear'))
        rfs = torch.cat(rfs, 1)
        rx = self.last_layer(rfs)

        return rx, px

if __name__ == '__main__':
    model = ThermalCC(pretrained=True)

    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(f"y.shapes={[t.shape for t in y]}")
