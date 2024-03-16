#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:21:46 2023

@author: mnsi
"""

import torch
import numpy as np
import torch.optim as optim
from CNN import CNN
from losses import MSE_Loss_masked
from Dataloaders import get_dataloaders 
from torchvision import transforms
import matplotlib.pyplot as plt


import cmocean
import cmocean.cm as cmo
import matplotlib.pyplot as plt

# Définition des paramètres

model = CNN(4, 1, [16,32,64,128])
#model = CNN(2, 1, [16,32,16])
model.to(device=1)

optimizer = optim.SGD(model.parameters(), lr=1e-4)
batch_size=32
num_epoch=10
loss_function=MSE_Loss_masked()


# Fonctions d'entraînement 
"""
def train(epoch, train_loader):
    model.train()
    epoch_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device=0), target.to(device=0)
        bathy = data[:,8:]
        data = data[:,[0,3]]
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = loss_function(output, target, bathy)
        epoch_loss += loss.item()
        loss.backward()
        
        optimizer.step()
    epoch /= len(train_loader)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    
    
def val(test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device=0), target.to(device=0)
            bathy = data[:,8:]
            data = data[:,[0,3]]
            output = model(data)

            # sum up batch loss
            test_loss += loss_function(output, target, bathy).item()

        test_loss /= len(test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))
"""       

if __name__=='__main__':
    
    dataloader_config={'train_dataset_path':["/data/home/ezheng/data/train/predictors_corr.npy","/data/home/ezheng/data/train/chla_corr.npy"],
                       'valid_dataset_path':["/data/home/ezheng/data/val/predictors_corr.npy","/data/home/ezheng/data/val/chla_corr.npy"],
                       'test_dataset_path':["/data/home/ezheng/data/test/predictors_corr.npy","/data/home/ezheng/data/test/chla_corr.npy"],
                       #'transform':None,
                       #'transform':transforms.Compose([transforms.ToTensor(), transforms.Pad(padding=[6,8], padding_mode='edge')]),
                       'transform':transforms.Compose([transforms.ToTensor()]), 
                       'batch_size': 8,
                        'normalize':True,
                        'log_chl':False}
    

    test=get_dataloaders(dataloader_config)
    
    train_loader=test[0]
    test_loader=test[1]
    
    
"""  
    for epoch in range(num_epoch):
        train(epoch, train_loader)
        val(test_loader)
"""



num_epoch=200


def train(epoch, train_loader, train_losses):
    model.train()
    epoch_loss=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device=1), target.to(device=1)
        bathy = data[:,-1:]
        data = data[:,:4]
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = loss_function(output, target, bathy)
        epoch_loss += loss.item()
        loss.backward()
        
        optimizer.step()
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), epoch_loss))

    return train_losses

def val(test_loader, val_losses):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device=1), target.to(device=1)
            bathy = data[:,-1:]
            data = data[:,:4]
            output = model(data)

            # sum up batch loss
            test_loss += loss_function(output, target, bathy).item()

        test_loss /= len(test_loader)
        val_losses.append(test_loss)
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))
        
        return val_losses

train_losses = []
val_losses = []

for epoch in range(1, num_epoch + 1):
    train_losses = train(epoch, train_loader, train_losses)
    val_losses = val(test_loader, val_losses)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Avec les 4 variables')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    

    
    
    
    
    
    
    
    
torch.save(model.state_dict(), "/.autofs/home/mnsi/Téléchargements/model.pt")