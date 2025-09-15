import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module('flatten', Flatten())

        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]

        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                'gate_c_fc_%d' % i,
                nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_bn_%d' % (i + 1),
                                   nn.BatchNorm1d(gate_channels[i + 1]))
            self.gate_c.add_module('gate_c_relu_%d' % (i + 1), nn.ReLU())

        self.gate_c.add_module('gate_c_fc_final',
                               nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        
        stri = x.shape[2].item() if type(x.shape[0]) == torch.Tensor else x.shape[2]
        avg_pool = F.avg_pool2d(x, stri, stride=stri)
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x)


class SpatialGate(nn.Module):
    def __init__(self,
                 gate_channel,
                 reduction_ratio=4):
        super(SpatialGate, self).__init__()
        self.branch1x1 = BasicConv2d(gate_channel, gate_channel//reduction_ratio // 4, kernel_size=3, padding = 1)

        self.branch3x1 = BasicConv2d(gate_channel, gate_channel//reduction_ratio // 4, kernel_size=3, dilation=(3,1), padding=(3,1))

        self.branch1x3 = BasicConv2d(gate_channel, gate_channel//reduction_ratio // 4, kernel_size=3, dilation=(1,3), padding=(1,3))

        self.branch3x3 = BasicConv2d(gate_channel, gate_channel//reduction_ratio // 4, kernel_size=3, dilation=3, padding=3)
    
        self.conv = nn.Sequential(
            BasicConv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, padding=1),
            nn.BatchNorm2d(gate_channel // reduction_ratio),
            nn.ReLU(),
            BasicConv2d(gate_channel//reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, x):
        outs = [self.branch1x1(x),self.branch3x1(x),self.branch1x3(x),self.branch3x3(x)]
        out = torch.cat(outs, 1)
        return self.conv(out).expand_as(x)


class BasicConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)

        return x
        
        
class SFIM(nn.Module):
    def __init__(self, in_dim=48):
        super(SFIM, self).__init__()
        
        self.channel_att_p = ChannelGate(gate_channel=in_dim)
        self.spatial_att_p = SpatialGate(gate_channel=in_dim)
        
        self.channel_att_r = ChannelGate(gate_channel=in_dim)
        self.spatial_att_r = SpatialGate(gate_channel=in_dim)
        
        self.mu = nn.Parameter(torch.tensor(float(10.0)))
        self.log_k = nn.Parameter(torch.log(torch.tensor(float(1.0))))
        
        self.proc = 1.0
 
    def forward(self, x_p, x_r):
        at_p = self.channel_att_p(x_p) * self.spatial_att_p(x_p)
        at_r = self.channel_att_r(x_r) * self.spatial_att_r(x_r)
        
        atten = at_p + at_r
        
        at = torch.sigmoid(atten)
        at_mean = at.mean()
        thres = at_mean if self.proc > 0.5 and self.proc < 0.6 else 1.0
        
        if thres >= 1.0:
            at_point = at
            at_edge = at
        else:
            at_point = 0.5 + (1.0 / torch.pi) * torch.atan(self.mu * (atten - 0.5))
            
            k = torch.exp(self.log_k) + 1e-8
            at_edge = (at - 0.5).abs() * 2 + 1e-8
            at_edge = at_edge ** k
            at_edge = 0.5 * torch.cos(torch.pi * at_edge)
       
        x_p = at_point * x_p + x_p + atten 
        x_r = at_edge * x_r + x_r + atten
        
        return x_p, x_r


class FPNBlock(nn.Module):
    def __init__(self, C_in, C_out):
        super(FPNBlock, self).__init__()
        # 横向连接的1x1卷积
        self.lateral1 = nn.Conv2d(C_in, C_out, 1)
        self.lateral2 = nn.Conv2d(C_in, C_out, 1)
        
        # 特征融合后的3x3卷积
        self.fuse1 = nn.Conv2d(C_out, C_out, 3, padding=1)
        self.fuse2 = nn.Conv2d(C_out, C_out, 3, padding=1)
        
        self.fscm = SFIM(C_out)
        
    def forward(self, x, upsampled1=None, upsampled2=None):
        # 横向连接
        lateral1 = self.lateral1(x)
        lateral2 = self.lateral2(x)
        
        sensitive1, sensitive2 = self.fscm(lateral1, lateral2)
        
        # 如果有上采样的特征，进行融合
        if upsampled1 is not None:
            lateral1 = lateral1 + upsampled1
            sensitive1 = sensitive1 + upsampled1
        if upsampled2 is not None:
            lateral2 = lateral2 + upsampled2
            sensitive2 = sensitive2 + upsampled2
            
        # 3x3卷积处理融合后的特征
        out1 = self.fuse1(sensitive1)
        out2 = self.fuse2(sensitive2)
        
        return out1, lateral1, out2, lateral2
        

class DFPN(nn.Module):
    def __init__(self, indim_list, outdime):
        super(DFPN, self).__init__()
      
        # FPN层
        self.fpn_c3 = FPNBlock(indim_list[0], outdime)
        self.fpn_c4 = FPNBlock(indim_list[1], outdime)
        self.fpn_c5 = FPNBlock(indim_list[2], outdime)
  
    def set_proc(self, proc):
        if proc is not None:
            self.fpn_c3.fscm.proc = proc
            self.fpn_c4.fscm.proc = proc
            self.fpn_c5.fscm.proc = proc
        
        
    def forward(self, x, proc=None):
        self.set_proc(proc)
            
        c3, c4, c5 = x
        # 自顶向下路径
        p5_p, lateral5_p, p5_r, lateral5_r = self.fpn_c5(c5)
        
        # 上采样p5并与c4融合
        up_5p = F.interpolate(lateral5_p, c4.shape[-2:], mode='nearest')
        up_5r = F.interpolate(lateral5_r, c4.shape[-2:], mode='nearest')
        
        p4_p, lateral4_p, p4_r, lateral4_r = self.fpn_c4(c4, up_5p, up_5r)
        
        # 上采样p4并与c3融合
        up_4p = F.interpolate(lateral4_p, c3.shape[-2:], mode='nearest')
        up_4r = F.interpolate(lateral4_r, c3.shape[-2:], mode='nearest')
        
        p3_p, _, p3_r, _ = self.fpn_c3(c3, up_4p, up_4r)
       
        return [p3_p, p4_p, p5_p], [p3_r, p4_r, p5_r]
        

if __name__ == '__main__':
    indim_list = [256, 512, 512]
    outdim = 128
    model = DFPN(indim_list, outdim)
    
    c3 = torch.randn(16, 256, 32, 32)
    c4 = torch.randn(16, 512, 16, 16)
    c5 = torch.randn(16, 512, 8, 8)
    x = [c3, c4, c5]
    y = model(x)
    
    print(f"[INFO]p.shapes={[t.shape for t in y[0]]}")
    print(f"[INFO]r.shapes={[t.shape for t in y[1]]}")
