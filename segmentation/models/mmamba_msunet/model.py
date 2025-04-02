import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from mamba_ssm import Mamba  # Make sure mamba_ssm package is installed
from segmentation.constant.config import IN_CHANNELS, OUT_CLASSES

#######################################
# Basic Convolution Blocks Definitions
#######################################

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class conv_block_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class conv_block_5(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# This block fuses features from two convolution streams (1x1 and 5x5) and reduces channels.
class conv_3_1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_3_1, self).__init__()
        self.conv_1 = conv_block_1(ch_in, ch_out)
        self.conv_5 = conv_block_5(ch_in, ch_out)
        self.conv = nn.Conv2d(ch_out * 2, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        x1 = self.conv_1(x)
        x5 = self.conv_5(x)
        x_cat = torch.cat((x1, x5), dim=1)
        return self.conv(x_cat)

# Upsampling block using either bilinear upsampling or transposed convolution.
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, bilinear=False):
        super(up_conv, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_in),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
    def forward(self, x):
        return self.up(x)

#######################################
# Mamba-based Block (PVMLayer)
#######################################

class PVMLayer(nn.Module):
    """
    A PVMLayer wraps a Mamba block to capture long-range dependencies.
    It first reshapes the 2D feature map into tokens, normalizes them,
    splits the channel dimension into 4 parts, processes each with the Mamba block,
    concatenates the outputs, and then projects back to the desired output dimension.
    """
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super(PVMLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # divide channels into 4 parts
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        n_tokens = H * W
        # Reshape to (B, n_tokens, C)
        x_flat = x.view(B, C, n_tokens).transpose(1, 2)
        x_norm = self.norm(x_flat)
        # Split the channel dimension into 4 chunks
        x_chunks = torch.chunk(x_norm, 4, dim=2)
        mamba_out_chunks = []
        for chunk in x_chunks:
            # Process each chunk with the Mamba block and add a scaled skip connection
            mamba_out = self.mamba(chunk) + self.skip_scale * chunk
            mamba_out_chunks.append(mamba_out)
        x_mamba = torch.cat(mamba_out_chunks, dim=2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        # Reshape back to (B, output_dim, H, W)
        out = x_mamba.transpose(1, 2).view(B, self.proj.out_features, H, W)
        return out

#######################################
# Modified MAMBA_MSU_Net Incorporating Mamba Block
#######################################


class MAMBA_MSU_Net(nn.Module):
    def __init__(self, img_ch=IN_CHANNELS, output_ch=OUT_CLASSES, use_mamba=True):
        """
        When use_mamba is True, the deepest encoder block (Conv5)
        and its corresponding decoder block (Up_conv5) are replaced
        with PVMLayer blocks that incorporate the Mamba block.
        """
        super(MAMBA_MSU_Net, self).__init__()
        filters_number = [32, 64, 128, 256, 512]
        self.use_mamba = use_mamba
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder pathway
        self.Conv1 = conv_3_1(ch_in=img_ch, ch_out=filters_number[0])
        self.Conv2 = conv_3_1(ch_in=filters_number[0], ch_out=filters_number[1])
        self.Conv3 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[2])
        self.Conv4 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[3])
        
        # Use Mamba-based PVMLayer for the deepest block if enabled
        if self.use_mamba:
            self.Conv5 = PVMLayer(input_dim=filters_number[3], output_dim=filters_number[4])
        else:
            self.Conv5 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[4])
        
        # Decoder pathway
        self.Up5 = up_conv(ch_in=filters_number[4], ch_out=filters_number[3])
        if self.use_mamba:
            self.Up_conv5 = PVMLayer(input_dim=filters_number[4], output_dim=filters_number[3])
        else:
            self.Up_conv5 = conv_3_1(ch_in=filters_number[4], ch_out=filters_number[3])
        
        self.Up4 = up_conv(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up_conv4 = conv_3_1(ch_in=filters_number[3], ch_out=filters_number[2])
        self.Up3 = up_conv(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up_conv3 = conv_3_1(ch_in=filters_number[2], ch_out=filters_number[1])
        self.Up2 = up_conv(ch_in=filters_number[1], ch_out=filters_number[0])
        self.Up_conv2 = conv_3_1(ch_in=filters_number[1], ch_out=filters_number[0])
        
        self.Conv_1x1 = nn.Conv2d(filters_number[0], output_ch, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        x1 = self.Conv1(x)                     # [B, 32, H, W]
        x2 = self.Conv2(self.Maxpool(x1))        # [B, 64, H/2, W/2]
        x3 = self.Conv3(self.Maxpool(x2))        # [B, 128, H/4, W/4]
        x4 = self.Conv4(self.Maxpool(x3))        # [B, 256, H/8, W/8]
        x5 = self.Conv5(self.Maxpool(x4))        # [B, 512, H/16, W/16]
        
        # Decoder
        d5 = self.Up5(x5)                      # Upsample to [B, 256, H/8, W/8]
        d5 = torch.cat((x4, d5), dim=1)          # Concatenate skip connection (channels: 256+256)
        d5 = self.Up_conv5(d5)                  # Refine features
        
        d4 = self.Up4(d5)                      # [B, 128, H/4, W/4]
        d4 = torch.cat((x3, d4), dim=1)          # Concatenate skip connection
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)                      # [B, 64, H/2, W/2]
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)                      # [B, 32, H, W]
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Conv_1x1(d2)
        output = self.sigmoid(d1)
        return output
