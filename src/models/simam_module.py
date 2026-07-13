#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import ultralytics.nn.modules.block as block
import ultralytics.nn.tasks as tasks

class SimAM(nn.Module):
    """
    SimAM: Simple Parameter-Free Attention Module
    https://proceedings.mlr.press/v139/yang21o.html
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        
        # Spatial energy calculation
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * self.activation(y)

# Keep a reference to the original C2f class
OriginalC2f = block.C2f

class C2f_SimAM(OriginalC2f):
    """
    Custom C2f block that passes its output through the SimAM module.
    Because SimAM has no parameters, this block is perfectly compatible
    with standard YOLOv8 pretrained weights!
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.simam = SimAM()
        
    def forward(self, x):
        # Run standard C2f extraction
        out = super().forward(x)
        # Apply SimAM attention
        return self.simam(out)

def apply_simam_patch():
    """Dynamically patch Ultralytics model blocks with SimAM."""
    print("🔧 Applying SimAM monkey-patch...")
    block.C2f = C2f_SimAM
    tasks.C2f = C2f_SimAM
