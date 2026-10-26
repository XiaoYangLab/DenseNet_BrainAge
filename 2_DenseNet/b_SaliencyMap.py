"""
Visualizing saliency map of networks predictiong using guided backpropagation   

Writted by Yang Xiao, Lei Zhao, Shuer Ye in 2025

@: xiaoyang9604@gmail.com

"""

import os
import copy
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import torch.nn.functional as F

class GuidedReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0 
        grad_input[grad_output <= 0] = 0 
        return grad_input

class GuidedReLU(nn.Module):
    def forward(self, x):
        return GuidedReluFunction.apply(x)

def replace_relu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedReLU())
        else:
            replace_relu(module)
    return model

class GuidedGradCAM:
    def __init__(self, model, target_layer):
        self.model = replace_relu(copy.deepcopy(model))
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                found = True
                break
        if not found:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model.")

    def __call__(self, x, device):
        x = x.clone().to(device)
        x.requires_grad = True

        output = self.model(x)
        self.model.zero_grad()
        output.sum().backward()

        guided_backprop = x.grad.abs()
        spatial_dims = tuple(range(2, self.gradients.ndim)) # 0 = batch, 1 = channels
        weights = self.gradients.mean(dim = spatial_dims, keepdim = True)
        grad_cam = (weights * self.activations).sum(dim = 1, keepdim = True)
        grad_cam = torch.relu(grad_cam)  

        grad_cam = F.interpolate(grad_cam, size=guided_backprop.shape[2:], mode='trilinear', align_corners=False)   

        grad_cam = grad_cam.cpu().numpy()
        guided_backprop = guided_backprop.cpu().numpy()

        guided_grad_cam = grad_cam * guided_backprop  
        
        return guided_grad_cam
