#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:31:47 2023

@author: ezheng
"""
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

from Dataloaders import get_dataloaders
from another_UNet import UNet
from CNN import CNN
from losses import MSE_Loss_masked
from plot import imshow_area

import matplotlib.pyplot as plt

device = torch.device(1)

def create_dataloader(train_path, val_path, test_path, batch_size):
    dataloader_config={'train_dataset_path':[train_path+"predictors_corr.npy",train_path+"chla.npy"],
                   'valid_dataset_path':[val_path+"predictors_corr.npy",val_path+"chla.npy"],
                   'test_dataset_path':[test_path+"predictors_corr.npy",test_path+"chla.npy"],
                   #'transform':None,
                   'transform':transforms.Compose([transforms.ToTensor()]),
                    'batch_size': batch_size,
                    'normalize':True,
                    'log_chl':True}

    return get_dataloaders(dataloader_config)

def train(model, train_loader, optimizer, loss_function, epoch, train_loss_list):
    model.train()
    epoch_loss = 0
    
    for nb_batch, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # essai = data.cpu().detach().numpy()
        # imshow_area(essai[0,3], land=False)
        
        optimizer.zero_grad()
        pred = model(data)
        # print(torch.nonzero(pred))
        loss = loss_function(pred, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    epoch_loss /= len(train_loader)
    
    predi = pred.cpu().detach().numpy()
    imshow_area(target[0,0].cpu(), land=False, title='True')
    imshow_area(predi[0,0], land=False, title='Pred')
    
    train_loss_list.append(epoch_loss)
    if (nb_batch+1) % len(train_loader) == 0:
        print("\tTrain Epoch: {} \tLoss: {:.6f}".format(epoch, epoch_loss))
    return train_loss_list

def val(model, val_loader, optm, loss_function, val_loss_list):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        
        for nb_batch, (data,target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = loss_function(pred, target)
            val_loss += loss.item()
            
        val_loss /= len(val_loader)

        val_loss_list.append(val_loss)
        if (nb_batch+1) % len(val_loader) == 0:
            print('\tVal set: Average loss: {:.6f}\n'.format(val_loss))    
        return val_loss_list
    
if __name__=='__main__':
    train_path = "/data/home/ezheng/data/train/"
    val_path = "/data/home/ezheng/data/val/"
    test_path = "/data/home/ezheng/data/test/"
    PATH = "/data/home/ezheng/projet-long/saved_model/CNN/"
    
    batch_size = 32
    
    train_loss_list = []
    val_loss_list = []
 
    # 1. create dataloader 
    train_loader, val_loader, test_loader, _, _, _, _ = create_dataloader(train_path, val_path, test_path, batch_size)
    
    # 2. create model
    # model = UNet(1, in_channels=8, depth=3, up_mode='upsample',\
    #              merge_mode='concat', activation='SiLU', transfo_field=False,\
    #                  residual_block=True)
        
    model = CNN(4,1,filters_list=[16,32,64])
    model.to(device)

    # 3. train model hyperparameters
    learning_rate = 1e-3
    max_epochs = 100
    start_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = MSE_Loss_masked()
    
    # 4. Start training and val
    print("Start training and validation...")
    
    for epoch in range(start_epoch+1,start_epoch+max_epochs+1):
        train_loss_list = train(model, train_loader, optimizer, loss_function, epoch,train_loss_list)

        val_loss_list = val(model, val_loader, optimizer, loss_function, val_loss_list)
        
        plt.plot(train_loss_list, label='Train')
        plt.plot(val_loss_list, label='Val')
        plt.legend()
        plt.yscale('log')
        plt.show()