#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:22:46 2023

@author: ezheng
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from another_UNet import UNet
from CNN import CNN
from train import create_dataloader
from losses import MSE_Loss_masked, RMSE_pixelwise, MAE_Loss_Masked, R2_Loss_Masked, CORRelation
from plot import imshow_area

import matplotlib.pyplot as plt

device = torch.device(1)

def load_model(path, modele):
    model = modele
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

def test(model, test_loader, test_loss_list):
    #numpy_pred = torch.empty(size=(102, 1, 100, 360))
    list_tensor_pred = []
    loss_function = MSE_Loss_masked()
    rmse_list_spatiale = []
    rmse_list_temporelle = []
    corr = []

    with torch.no_grad():
        model.eval()
        test_loss = 0
        mae = 0
        r2 = 0
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            
            #numpy_pred[batch_idx]=pred
            list_tensor_pred.append(pred)
            # sum up batch loss
            #print("batch 1",val_loss)
            test_loss += loss_function(pred, target).item()
            
            mae += MAE_Loss_Masked(pred, target)
            r2 += R2_Loss_Masked(pred, target)
            
            rmse_loss_spatiale = RMSE_pixelwise(pred, target, mode='spatiale')
            rmse_loss_temporelle = torch.flatten(RMSE_pixelwise(pred, target, mode='temporelle')).cpu().detach().numpy()
            # coeff_corr = CORRelation(pred, target)
            
            rmse_list_spatiale.append(rmse_loss_spatiale)
            rmse_list_temporelle.append(rmse_loss_temporelle)
            # corr.append(coeff_corr)
        
        mae /= len(test_loader)
        r2 /= len(test_loader)
        test_loss /= len(test_loader)
        
        test_loss_list.append(test_loss)
        if (batch_idx+1) % len(test_loader) == 0:
            print('\tTest set: Average MSE: {}, MAE: {}, R2: {}\n'.format(test_loss, mae, r2))   
            # print(sum(rmse_list).shape) 
        return test_loss_list, list_tensor_pred, sum(rmse_list_spatiale)/len(rmse_list_spatiale), rmse_list_temporelle

if __name__=='__main__':
    path = "/data/home/ezheng/projet-long/saved_model/new2/best_cnn1.pt"
    # modele = UNet(1, in_channels=10, depth=3, up_mode='upsample',\
    #               merge_mode='concat', activation='ReLU', transfo_field=False,\
    #                   residual_block=False)
        
    modele = CNN(10,1,filters_list=[16,32,64,128])
    modele.to(device)
    
    train_path = "/data/home/ezheng/data/train/"
    val_path = "/data/home/ezheng/data/val/"
    test_path = "/data/home/ezheng/data/test/"
    
    
    max_chl = list()
    min_chl = list()
    max_pred = list()
    min_pred = list()

    batch_size = 32
    train_loader, val_loader, test_loader, max_chl, min_chl, max_pred, min_pred = create_dataloader(train_path, val_path, test_path, batch_size)
    test_loss_list = list()
    
    model = load_model(path,modele)
   
    test_lost_list, list_pred, rmse_spatiale, rmse_temporelle = test(model, test_loader, test_loss_list)
    
    # pour batch size 32
    # big_tensor = torch.cat((list_pred[0], list_pred[1], list_pred[2], \
    #                         list_pred[3], list_pred[4], list_pred[5], \
    #                         list_pred[6], list_pred[7], list_pred[8], \
    #                         list_pred[9], list_pred[10], list_pred[11], \
    #                         list_pred[12], list_pred[13], list_pred[14], \
    #                         list_pred[15], list_pred[16], list_pred[17], \
    #                         list_pred[18], list_pred[19], list_pred[20], \
    #                         list_pred[21], list_pred[22], list_pred[23], \
    #                         list_pred[24]))
    
    # pour batch 32 test pareil sauf que 0 à 3
    
    # big_tensor = torch.cat((list_pred[0], list_pred[1], list_pred[2], \
    #                         list_pred[3], list_pred[4], list_pred[5], \
    #                         list_pred[6]))
    
    
    big_tensor = torch.cat((list_pred))
    # print(big_tensor.shape)
    # for i in range(1, len(list_tensor_pred)+1):
    #     big_tensor = torch.cat((torch.cat(list_tensor_pred[:i]), torch.cat(list_tensor_pred[i:])))
    #     print(big_tensor)
    #     print(len(big_tensor))
    #     print(big_tensor.shape)
    numpy_pred = big_tensor.detach().cpu().numpy()
    chla_pred_log = min_chl[0] + numpy_pred*(max_chl[0]-min_chl[0])
    chla_pred = 10 ** chla_pred_log
    
    
    chla_true = np.load(test_path+"chla.npy")
    hist_chla_pred = np.reshape(chla_pred, 92*360*100)
    hist_chla_true = np.reshape(chla_true, 92*360*100)
    
    imshow_area(rmse_spatiale[0,0].cpu(), land=False, title='Pixel-wise RMSE', vmax=0.15)
    plt.figure(0)
    plt.plot(rmse_temporelle)
    plt.ylabel("week-wise RMSE")
    plt.xlabel("Week i")
    plt.show()
    
    corr = CORRelation(chla_pred, chla_true)
    imshow_area(corr[0,0], cmap='seismic', land=False, title='Corrélation U-Net', vmin= -1, vmax=1)
    
    plt.figure(2)
    plt.xlim(-3,2)
    plt.hist(np.reshape(chla_pred_log, 92*360*100), bins=5000)
    plt.title("Prediction log(CHL)")
    plt.show()
    
    plt.figure(3)
    plt.xlim(-3,2)
    plt.hist(np.log10(hist_chla_true), bins=5000)
    plt.title("True log(CHL)")
    plt.show()
    
    #print(chla_pred)
    # print(chla_pred.shape)    
    
    #np.save("/data/home/ezheng/data/chla_pred_model3.npy", chla_pred)
    
    
    
    