#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:52:29 2022

@author: lollier
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         CNN                                           | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import torch
import torch.nn as nn
from torch.nn import init

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

"""
CNN basic implementation
"""

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1 ,padding_mode='replicate', bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        bias=bias,
        groups=groups)


def batch_norm(in_channels):
    return nn.BatchNorm2d(in_channels)


class CNN(nn.Module):
    """
    A Module that performs N convolutions and 1 activation.
    """

    def __init__(self, in_channels, out_channels, filters_list=[32,64,128,16,8], activation='ReLU', device='cuda'):
        """
        Parameters
        ----------
        in_channels : int

        out_channels : int

        filters_list : list, optional
            size of the convolutional layers. The default is [32,64,128,16,8].
        activation : str, optional
            ReLU or Softmax. The default is 'ReLU'.

        """
        super(CNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = batch_norm(out_channels)#pas mis de batchnorm dans le forward
        self.filters_list=filters_list
        
        if activation in ('ReLU', 'Softmax', 'SiLU'):
            if activation=='Softmax' :
                self.activation = nn.Softmax(dim=1)
            elif activation=='SiLU':
                self.activation = nn.SiLU()
            else : 
                self.activation = nn.ReLU()
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "activation. Only \"Sofmax\" and "
                             "\"ReLU\" are allowed.".format(activation))
            
            
            
        self.convolution_layers=[]
        
        for i in range(len(self.filters_list)):
            ins = self.in_channels if i == 0 else outs
            outs = self.filters_list[i] if i!=len(self.filters_list)-1 else self.out_channels

            conv = conv3x3(ins, outs) 
            self.convolution_layers.append(conv)
            
        self.convolution_layers = nn.ModuleList(self.convolution_layers)
        
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):

        for i, module in enumerate(self.convolution_layers):
            x = module(x)
            
        x = self.activation(x)
         
        return x
            
            
            
            
            
            
            
if __name__ == "__main__":
    """
    testing
    """
    import numpy as np
    from torchsummary import summary
    model = CNN(1, 1, [16,32,16])
    model.to(device=1)
    #block_input=Input_Block(8, 32)
    #summary(model, input_size=(8,128,128))
    
    # input_names = ['Sentence']
    # output_names = ['yhat']
    a=torch.Tensor(np.random.random(((32,1,360,100)))).to(device=1)
    # torch.onnx.export(model, a, 'test0.onnx', input_names=input_names, output_names=output_names)

    a=torch.Tensor(np.random.random(((32,1,64,128)))).to(device=1)




          
            
            
            
            