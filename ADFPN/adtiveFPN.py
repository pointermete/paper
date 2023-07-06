#   wz 2023-5-4
#   denseadFPN
#   利用3个空洞卷积和本体各执上采样相同的通道后，利用ca注意力机制进行通道数加权，再利用1x1卷积进行降维
#
#
import torch
import torch.nn as nn

from nets.attention import *

class DENFPN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DENFPN, self).__init__()

        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)


        self.attention = CoordAttention(in_channel)    #通道需要替换

        self.inplan =  in_channel

        self.softmax = nn.Softmax(dim=1)

        self.cv= CoordAttention2(in_channel*5,out_channel*5)

        self.zhen = nn.Conv2d(in_channel*5, out_channel, 1, 1)


    def forward(self, x,y):

        batch=x.shape[0]

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = torch.cat((atrous_block1, atrous_block6,atrous_block12, atrous_block18,y), dim=1)
        net = net.view(batch,5, self.inplan,net.shape[2],net.shape[3])

        x1 =  self.attention(atrous_block1)
        x2 =  self.attention(atrous_block6)
        x3 =  self.attention(atrous_block12)
        x4 =  self.attention(atrous_block18)
        x5 =  self.attention(y)

        x_cat = torch.cat((x1,x2,x3,x4,x5), dim=1)
        x_weigh = x_cat.view(batch,5,self.inplan,1,1)
        x_weigh = self.softmax(x_weigh)

        net_weight = net * x_weigh

        for i  in range(5):

            xout = net_weight[:,i,:,:]

            if i==0:
                out =xout
            else:
                out = torch.cat((xout ,out),1)


        out = self.cv(out)
        out = self.zhen(out)
        return out

