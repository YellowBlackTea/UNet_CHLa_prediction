#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:48:36 2022

@author: lollier



"""


import torch
import numpy as np
from plot import imshow_area

class MSE_Loss_masked(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.mse=torch.nn.MSELoss(reduction='none')
        #self.mask=torch.Tensor(np.load('/usr/home/lollier/datatmp/npy_emergence/train/chla.npy')[0])
        
    def forward(self, prediction,target, reduction='mean'):
        
        # if (torch.isnan(target)==torch.isnan(self.mask)).all():
        #     pred_masked=torch.mul(prediction, ~torch.isnan(self.mask))
        # else:

        # pred_masked1=torch.mul(prediction, ~torch.isnan(target))
        # if bat:
        #     pred_masked1=torch.mul(prediction, ~torch.isnan(target))
        #     bath = bathymetry == 1 
        #     pred_masked = torch.mul(pred_masked1, bath)
        # else:
        pred_masked = torch.mul(prediction, ~torch.isnan(target))

        #ça va c'est pas trop long comme ligne de calcul, les étapes ça me connaît
        
        if reduction=='none':
            return self.mse(pred_masked, torch.nan_to_num(target))
        #jsuis un peu un tocard sur ce coup parce que faire la somme divisée par le total ça revient à faire un np.nanmean
        # a = self.mse(pred_masked, torch.nan_to_num(target)).cpu().detach()
        # imshow_area(prediction[0,0].cpu().detach(), land=False)
        return torch.sum(self.mse(pred_masked, torch.nan_to_num(target)))/(torch.flatten(prediction).size()[0]-torch.count_nonzero(torch.isnan(target)))



class RMSE_Loss_masked(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MSE_Loss_masked()
        
    def forward(self,prediction,target, reduction='mean'):
        return torch.sqrt(self.mse(prediction,target, reduction=reduction))

def MAE_Loss_Masked(prediction, target):
    pred_masked = torch.mul(prediction, ~torch.isnan(target))
    mae = torch.sum(torch.abs(pred_masked - torch.nan_to_num(target)))/(torch.flatten(prediction).size()[0]-torch.count_nonzero(torch.isnan(target)))
    return mae

def R2_Loss_Masked(prediction, target):
    pred_masked = torch.mul(prediction, ~torch.isnan(target))
    target_mean = torch.mean(torch.nan_to_num(target))
    var_tot = torch.sum((torch.nan_to_num(target) - target_mean) ** 2)
    var_res = torch.sum((torch.nan_to_num(target) - pred_masked) ** 2)
    r2 = 1 - var_res / var_tot
    return r2


def RMSE_pixelwise(prediction, target, mode):
    pred_masked = torch.mul(prediction, ~torch.isnan(target))
    mse = torch.nn.MSELoss(reduction='none')
    
    mse_elt_wise = mse(pred_masked, torch.nan_to_num(target))
    
    if mode=='spatiale':
        rmse_pixel_wise = torch.sqrt(torch.sum(mse_elt_wise, dim=0, keepdim=True)/(prediction.shape[0]))
    if mode=='temporelle':
        rmse_pixel_wise = torch.sqrt(torch.sum(mse_elt_wise, dim=(2,3), keepdim=True)/(100*360))
        # print(rmse_pixel_wise.shape)
    
    return rmse_pixel_wise


def CORRelation(prediction, target):
    # # version torch
    # pred_masked = torch.mul(prediction, ~torch.isnan(target))
    # target_modified = torch.nan_to_num(target)

    # pred_mean = torch.mean(pred_masked, dim=(2,3), keepdim=True)
    # target_mean = torch.mean(target_modified, dim=(2,3), keepdim=True)

    # vx = pred_masked - pred_mean
    # vy = target_modified - target_mean

    # corr = torch.sum(vx * vy, dim=0, keepdim=True) / (torch.sqrt(torch.sum(vx ** 2, dim=0, keepdim=True)) * torch.sqrt(torch.sum(vy ** 2, dim=0, keepdim=True)))
    
    # version numpy
    pred_masked = np.multiply(prediction, ~np.isnan(target))
    target_modified = np.nan_to_num(target)

    pred_mean = np.mean(pred_masked, axis=(2,3), keepdims=True)
    target_mean = np.mean(target_modified, axis=(2,3), keepdims=True)

    vx = pred_masked - pred_mean
    vy = target_modified - target_mean

    corr = np.sum(vx * vy, axis=0, keepdims=True) / (np.sqrt(np.sum(vx ** 2, axis=0, keepdims=True)) * np.sqrt(np.sum(vy ** 2, axis=0, keepdims=True)))
    return corr

if __name__=='__main__':
    
    pred=torch.rand((102,1,100,360)).float()
    bathy = torch.randint(0,2,(102,1,100,360))
    target=torch.rand((102,1,100,360)).float()
    target[1] = float('nan')

    loss=MSE_Loss_masked()
    
    b=loss(pred, target)
    c = RMSE_pixelwise(pred, target, mode='temporelle')
    # imshow_area(c[0,0], land=True)
    #c=loss(pred,target,reduction='none')
    
    #print(b.shape,c.shape)
   