from __future__ import absolute_import, print_function
import torch
import torch.nn as nn
from torch.nn import init

from . import conv_3_1, up_conv

from segmentation.constant.config import IN_CHANNELS, OUT_CLASSES

class BMSU_Net(nn.Module):
    def __init__(self, img_ch = IN_CHANNELS, output_ch = OUT_CLASSES):
        super(BMSU_Net, self).__init__()

        filters_number = [32, 64, 128, 256, 512, 1024]# 1024 INSTEAD OF 16

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_3_1(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_3_1(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[3])
        self.Conv5 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[4])

        self.bridge = nn.Sequential(
            nn.Conv2d(filters_number[4], filters_number[3], kernel_size=1, stride=1),
            nn.Conv2d(filters_number[3], filters_number[3], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(filters_number[3], filters_number[5], kernel_size=1, stride=1),
            nn.BatchNorm2d(filters_number[5]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_number[5], filters_number[3], kernel_size=1, stride=1),
            nn.Conv2d(filters_number[3], filters_number[3], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(filters_number[3], filters_number[5], kernel_size=1, stride=1),
            nn.BatchNorm2d(filters_number[5]),
            nn.ReLU(inplace=True)
        )

        self.Up5 = up_conv(ch_in=filters_number[5], ch_out=filters_number[4])
        self.Up_conv5 = conv_3_1(ch_in=filters_number[5], ch_out=filters_number[4])

        self.Up4 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        self.Up_conv4 = conv_3_1(ch_in=filters_number[4], ch_out=filters_number[3])

        self.Up3 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv3 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[2])

        self.Up2 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv2 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[1])

        self.Up1 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv1 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[0])

        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4= self.Conv4(x4)
        x5 = self.Maxpool(x4)

        x5 = self.Conv5(x5)
        x = self.Maxpool(x5)

        x = self.bridge(x)

        d5 = self.Up5(x)
        d5 = torch.cat((x5,d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x4,d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x3,d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x2,d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((x1,d1), dim=1)
        d1 = self.Up_conv1(d1)

        output = self.Conv_1x1(d1)
        output = self.sigmoid(output)

        return output