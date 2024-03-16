#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:39:47 2023

@author: ezheng
"""
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms


from train import create_dataloader, train, val, plot_over_epochs
from another_UNet import UNet
from CNN import CNN
from losses import MSE_Loss_masked

import matplotlib.pyplot as plt

device = torch.device(1)

def training(model, model_name, lr, batch_s):
    train_path = "/data/home/ezheng/data/train/"
    val_path = "/data/home/ezheng/data/val/"
    test_path = "/data/home/ezheng/data/test/"
    PATH = "/data/home/ezheng/projet-long/saved_model/new2/"
    
    batch_size = batch_s
    
    # 0. creating some lists to store values   
    train_loss_list = list()
    val_loss_list = list()
    lr_list = list()
    
    max_chl = list()
    min_chl = list()
    max_pred = list()
    min_pred = list()
    
    # 1. create dataloader 
    train_loader, val_loader, test_loader, max_chl, min_chl, max_pred, min_pred = create_dataloader(train_path, val_path, test_path, batch_size)
    
    model.to(device)
    
    # 3. train model hyperparameters
    learning_rate = lr
    max_epochs = 500

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = MSE_Loss_masked()
    
    # 3.1 - Fine tune
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)

    # 4. Start training and val
    print("Start training and validation...")
    
    for epoch in range(1,max_epochs+1):
        train_loss_list = train(model, train_loader, optimizer, loss_function, epoch, train_loss_list, bat=False, noise=False)

        val_loss_list, best_model = val(model, val_loader, val_loss_list, lr_list, loss_function, optimizer, scheduler, bat=False)
        if epoch % 50 == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_list[-1],
                    }, PATH+"model_{}_for_{}_epochs" .format(model_name, epoch))
    # 5. Plot la courbe de train-val
    # plot_over_epochs(max_epochs, train_loss_list, val_loss_list, \
    #                  'Train', 'Val', 'Train-Val Curve de {}'.format(model_name), 'MSE Maked Loss')
    plot_over_epochs(max_epochs, train_loss_list, val_loss_list, \
                         'Train', 'Val', 'Train-Val Curve de {}'.format(model_name), 'MSE Maked Loss (log scale)', log=True)
    plot_over_epochs(max_epochs, lr_list, None, None, None, 'LR Curve', 'LR', log=True)
    # 6. Save model (weights)
    torch.save(model.state_dict(), PATH+"{}.pt".format(model_name))
    torch.save(best_model.state_dict(), PATH+"best_{}.pt".format(model_name))    
   

if __name__=='__main__':
    
    
    # sur le modèle du resnet de mercredi 15
    
    # model1 = UNet(1, in_channels=8, depth=4, up_mode='upsample',\
    #              merge_mode='concat', activation='SiLU', transfo_field=False,\
    #                  residual_block=False)
    # model2 = UNet(1, in_channels=8, depth=5, up_mode='upsample',\
    #              merge_mode='concat', activation='SiLU', transfo_field=False,\
    #                  residual_block=False)
    # model3 = UNet(1, in_channels=8, depth=4, up_mode='upsample',\
    #              merge_mode='concat', activation='SiLU', transfo_field=False,\
    #                  residual_block=True)
    
    
    # cnn + resnet
    model1 = CNN(4,1,filters_list=[16,32,64,8])
    model2 = CNN(4,1, [16,32,64])
    model3 = UNet(1, in_channels=4, depth=3, up_mode='upsample',\
                 merge_mode='concat', activation='ReLU', transfo_field=False,\
                     residual_block=False)
    # model4 = UNet(1, in_channels=10, depth=4, up_mode='upsample',\
    #              merge_mode='concat', activation='ReLU', transfo_field=False,\
    #                  residual_block=False)
    # model5 = UNet(1, in_channels=10, depth=5, up_mode='upsample',\
    #              merge_mode='concat', activation='ReLU', transfo_field=False,\
    #             residual_block=False)
    # model6 = UNet(1, in_channels=10, depth=3, up_mode='upsample',\
    #              merge_mode='concat', activation='ReLU', transfo_field=False,\
    #             residual_block=True)
    # model7 = UNet(1, in_channels=10, depth=4, up_mode='upsample',\
    #              merge_mode='concat', activation='ReLU', transfo_field=False,\
    #             residual_block=True)
    # model8 = UNet(1, in_channels=10, depth=5, up_mode='upsample',\
    #              merge_mode='concat', activation='ReLU', transfo_field=False,\
    #             residual_block=True)
    # model3 = UNet(1, in_channels=8, depth=5, merge_mode='concat', activation='SiLU', \
    #               transfo_field=False)
    training(model1, "4var_cnn1", 1e-3, 32)
    training(model2, '4var_cnn2', 1e-3, 32)
    training(model3, '4var_unet_3', 1e-3, 32)
    # training(model4, 'unet_4', 1e-3, 32)
    # training(model5, 'unet_5', 1e-3, 32)
    # training(model6, 'resnet_3', 1e-3, 32)
    # training(model7, 'resnet_4', 1e-3, 16)
    # training(model8, 'resnet_5', 1e-3, 16)


