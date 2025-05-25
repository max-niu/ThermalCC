import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSelection(nn.Module):
    
    def __init__(self, indim):
        super(ChannelSelection, self).__init__()
        
        self.conv = nn.Conv2d(indim, indim, 1, groups=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(indim)
        self.select_conv = nn.Conv2d(indim, indim, 1, groups=1)
        
        

    def forward(self, x):
        max_atten = F.adaptive_avg_pool2d(x, 1)
        #print(f"[ChannelSelection]: max_atten.shape = {max_atten.shape}")
        
        avg_atten = F.adaptive_max_pool2d(x, 1)
        #print(f"[ChannelSelection]: avg_atten.shape = {avg_atten.shape}")
        
        atten = F.adaptive_avg_pool2d(max_atten + avg_atten, 1)
        #print(f"[ChannelSelection]: atten.shape = {atten.shape}")   
        
        atten = self.conv(atten)
        #print(f"[ChannelSelection]: atten.shape = {atten.shape}")   
        
        atten = self.relu(atten)
        #print(f"[ChannelSelection]: atten.shape = {atten.shape}")   
        
        atten = self.bn(atten)
        #print(f"[ChannelSelection]: atten.shape = {atten.shape}")   
        
        atten = self.select_conv(atten)
        #print(f"[ChannelSelection]: atten.shape = {atten.shape}")   
        
        atten = F.softmax(atten, dim=1)
        #print(f"[ChannelSelection]: atten.shape = {atten.shape}")   
        
        
        x = x + x * atten
        #print(f"[ChannelSelection]: x.shape = {x.shape}")   
        
        return x
        

class SpatialSelect(nn.Module):
    def __init__(self, indim):
        super(SpatialSelect, self).__init__()
   
        self.conv = nn.Conv2d(2, 2, 7, padding=3)
        self.select_conv = nn.Conv2d(indim // 2, indim, 1) 

    def forward(self, s1, s2, x_c, x):
        avg_atten = torch.mean(x_c, dim=1, keepdim=True)
        #print(f"[SpatialSelect]: avg_atten.shape = {avg_atten.shape}")
        
        max_atten, _ = torch.max(x_c, dim=1, keepdim=True)
        #print(f"[SpatialSelect]: max_atten.shape = {max_atten.shape}")
        
        atten = torch.cat([avg_atten, max_atten], dim=1)
        #print(f"[SpatialSelect]: atten.shape = {atten.shape}")
        
        atten = self.conv(atten).sigmoid()
        #print(f"[SpatialSelect]: atten.shape = {atten.shape}")
        
        atten = s1 * atten[:, 0, :, :].unsqueeze(1) + s2 * atten[:, 1, :, :].unsqueeze(1)
        #print(f"[SpatialSelect]: atten.shape = {atten.shape}")
        
        atten = self.select_conv(atten)
        #print(f"[SpatialSelect]: atten.shape = {atten.shape}")
        
        x = x + x * atten
        #print(f"[SpatialSelect]: x.shape = {x.shape}")
        
        return x

        
class CSEM(nn.Module):
    def __init__(self, indim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(indim, indim, 5, padding=2, groups=indim)
        self.conv2 = nn.Conv2d(indim, indim, 7, stride=1, padding=9, groups=indim, dilation=3)
        
        self.conv1_1 = nn.Conv2d(indim, indim // 2, 1)
        self.conv2_1 = nn.Conv2d(indim, indim // 2, 1)
       
        self.c_select = ChannelSelection(indim)
        self.s_select = SpatialSelect(indim)


    def forward(self, x):
        #print(f"[CSEM]: x.shape = {x.shape}")
        
        x1 = self.conv1(x)
        #print(f"[CSEM]: x1.shape = {x1.shape}")
        
        x2 = self.conv2(x1)
        #print(f"[CSEM]: x2.shape = {x2.shape}")
      
        x1_1 = self.conv1_1(x1)
        #print(f"[CSEM]: x1_1.shape = {x1_1.shape}")
        
        x2_1 = self.conv2_1(x2)
        #print(f"[CSEM]: x2_1.shape = {x2_1.shape}")
        
        x_c = torch.cat((x1_1, x2_1), 1)
        #print(f"[CSEM]: x_c.shape = {x_c.shape}")
        
        x_channel = self.c_select(x_c)
        #print(f"[CSEM]: x_channel.shape = {x_channel.shape}")
        
        x_spatial = self.s_select(x1_1, x2_1, x_c, x)
        #print(f"[CSEM]: x_spatial.shape = {x_spatial.shape}")
        
        return x_channel + x_spatial
        


if __name__ == '__main__':
    model = CSEM(48)

    x = torch.randn(16, 48, 64, 64)

    y = model(x)
    
    print(y.shape)
