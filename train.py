#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:53:04 2023

@author: ezheng
"""
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import copy

from Dataloaders import get_dataloaders
from another_UNet import UNet
from CNN import CNN
from losses import MSE_Loss_masked
from plot import imshow_area

import matplotlib.pyplot as plt

device = torch.device(1)

def create_dataloader(train_path, val_path, test_path, batch_size):
    dataloader_config={
                    'train_dataset_path':[train_path+"predictors.npy",train_path+"chla.npy"],
                    'valid_dataset_path':[val_path+"predictors.npy",val_path+"chla.npy"],
                    'test_dataset_path':[test_path+"predictors.npy",test_path+"chla.npy"],
                    # 'train_dataset_path':[train_path+"predictors_corr.npy",train_path+"chla.npy"],
                    #   'valid_dataset_path':[val_path+"predictors_corr.npy",val_path+"chla.npy"],
                    #   'test_dataset_path':[test_path+"predictors_corr.npy",test_path+"chla.npy"],
                   #'transform':None,
                   'transform':transforms.Compose([transforms.ToTensor()]),
                    'batch_size': batch_size,
                    'normalize':True,
                    'log_chl':True}

    return get_dataloaders(dataloader_config)

def train(model, train_loader, optimizer, loss_function, epoch, train_loss_list, bat=False, noise=False):
    model.train()
    epoch_loss = 0

    # nb iteration = nb samples / len(loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # bathy = data[:,-1:]
        # data = data[:,:-1]
        if noise:
            noisy = np.random.normal(1e-6, 1e-5, size = data.shape)
            noisy = torch.from_numpy(noisy).float().to(device)
            data = data + noisy
        
        optimizer.zero_grad()               # set gradients to 0
        pred = model(data)                  # predicted chl-a
            
        loss = loss_function(pred, target)  # loss true vs pred
        epoch_loss += loss.item()           # accumulate loss per item in batch
        loss.backward()                     # compute gradients
        optimizer.step()                    # update new weights

    epoch_loss /= len(train_loader)
    predi = pred.cpu().detach().numpy()
    imshow_area(target[0,0].cpu(), land=False, title='True')
    imshow_area(predi[0,0], land=False, title='Pred')
    
    train_loss_list.append(epoch_loss)
    # if epoch % epochs_log_interval == 0:
    #     print("\tTrain Epoch = {} \tLoss : {:.4f}" .format(epoch, epoch_loss))

    if (batch_idx+1) % len(train_loader) == 0:
        print("\tTrain Epoch: {} \tLoss: {:.6f}".format(epoch, epoch_loss))
    return train_loss_list

def val(model, val_loader, val_loss_list, lr_list, loss_function, optim, scheduler, bat=False):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        best_loss = 10
        best_model = copy.deepcopy(model)
        
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            # bathy = data[:,-1:]
            # data = data[:,:-1]
            pred = model(data)
            
            # sum up batch loss
            val_loss_one_batch = loss_function(pred, target).item()
            val_loss += val_loss_one_batch
        

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        val_loss_list.append(val_loss)
        
        lr = optim.param_groups[0]['lr']
        lr_list.append(lr)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            
    
        if (batch_idx+1) % len(val_loader) == 0:
            print('\tVal set: Average loss: {:.6f}, LR = {}\n'.format(val_loss, lr))    
        return val_loss_list, best_model
  
def plot_over_epochs(max_epochs, list_y1, list_y2, label1, label2, title, ylabel, log=False):
        epochs = list(range(1, max_epochs+1))
        
        plt.plot(epochs, list_y1, label=label1)
        if list_y2:
            plt.plot(epochs, list_y2, label=label2)

        # add title and axis labels
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        
        if label1 and label2:
            plt.legend()
        
        if log:
            plt.yscale('log')
        else:
            plt.yscale('linear')
        # display the plot
        plt.show()
        
def resuming_train(path, model, optimzer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    
    return model,optimizer, epoch
    
    
if __name__=='__main__':
    train_path = "/data/home/ezheng/data/train/"
    val_path = "/data/home/ezheng/data/val/"
    test_path = "/data/home/ezheng/data/test/"
    PATH = "/data/home/ezheng/projet-long/saved_model/new2/"
    
    batch_size = 32
    
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
    
    # 2. create model
    model = UNet(1, in_channels=4, depth=3, up_mode='upsample',\
                  merge_mode='concat', activation='ReLU', transfo_field=False,\
                      residual_block=False)
        
    # model = CNN(10,1,filters_list=[16,32,64,128,8])
    model.to(device)

    # 3. train model hyperparameters
    learning_rate = 1e-3
    max_epochs = 500
    start_epoch = 0
    #epochs_log_interval = 5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = MSE_Loss_masked()
    
    # 3.1 - Fine tune
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    
    # 3.9 - Resuming training
    # model, optimizer, start_epoch = resuming_train(PATH+"CA_MARCHE_100epo.pt", model, optimizer)
    
    
    # 4. Start training and val
    print("Start training and validation...")
    
    for epoch in range(start_epoch+1,start_epoch+max_epochs+1):
        train_loss_list = train(model, train_loader, optimizer, loss_function, epoch, train_loss_list, bat=False, noise=False)

        val_loss_list, best_model = val(model, val_loader, val_loss_list, lr_list, loss_function, optimizer, scheduler, bat=False)
        
        if epoch % 50 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss_list[-1],
                        }, PATH+"4var_unet_3{}_epochs.pt" .format(epoch))
        
        plt.plot(train_loss_list, label='Train')
        plt.plot(val_loss_list, label='Val')
        plt.title('Train-Val Curve')
        plt.legend()
        plt.yscale('log')
        plt.show()
        
    
    
    # 5. Plot la courbe de train-val
    # plot_over_epochs(max_epochs, train_loss_list, val_loss_list, \
    #                   'Train', 'Val', 'Train-Val Curve', 'MSE Maked Loss')
    plot_over_epochs(max_epochs, train_loss_list, val_loss_list, \
                      'Train', 'Val', 'Train-Val Curve', 'MSE Maked Loss (log scale)', log=True)
    plot_over_epochs(max_epochs, lr_list, None, None, None, 'LR Curve', 'LR', log=True)

    # 6. Save model (weights)
    torch.save(model.state_dict(), PATH+"4var_unet_3.pt")
    torch.save(best_model.state_dict(), PATH+"4var_best_unet_3.pt")    
    
   
    # torch.save({
    #         'epoch': max_epochs,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': train_loss_list[-1],
    #         }, PATH)
   
    