#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import ultralytics.nn.modules.block as block
import ultralytics.nn.tasks as tasks

class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM."""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Ensure hidden_planes is at least 1
        hidden_planes = max(1, in_planes // ratio)
        self.f1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_concat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

# Keep a reference to the original C2f class
OriginalC2f = block.C2f

class C2f_CBAM(OriginalC2f):
    """
    Custom C2f block that passes its output through the CBAM module.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # CBAM takes the output channels of C2f. 
        # Tối ưu hóa cho ổ gà: ratio=8 (giữ chi tiết), kernel_size=3 (tập trung đặc trưng vi mô)
        self.cbam = CBAM(c2, ratio=8, kernel_size=3)
        
    def forward(self, x):
        # Run standard C2f extraction
        out = super().forward(x)
        # Trả lại hàm forward gốc của CBAM (Attention Gate)
        return self.cbam(out)

def apply_cbam_patch():
    """Dynamically patch Ultralytics model blocks with CBAM."""
    print("🔧 Applying CBAM monkey-patch...")
    block.C2f = C2f_CBAM
    tasks.C2f = C2f_CBAM
