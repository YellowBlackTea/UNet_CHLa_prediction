#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:35:23 2022

@author: lollier


"""
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         DATALOADERS                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms


# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #
from utils.functions import minmax_scaler

"""
Custom Dataset inheriting from Pytorch Dataset
Take a .npy image time series as dataset and create time series for training of lenght L_source+L_target
"""

class chl_predictors_dataset(Dataset):

    def __init__(self, datasets, transform=None):

        
        self.dataset_predictors = datasets[0]
        self.dataset_chl = datasets[1]
        self.transform = transform



    def __len__(self):
        return self.dataset_predictors.shape[0]
    
    def __getitem__(self, index):
        """
        Main function of the CustomDataset class. 
        """
        item=self.dataset_predictors[index]
        target=self.dataset_chl[index]
        if self.transform is not None:
            # print("hello1")
            item = np.moveaxis(item, 0, -1)
            target = np.moveaxis(target, 0, -1)

            item=self.transform(item)   
            target=self.transform(target)
          
        a, b = item.clone().detach(), target.clone().detach()
        a, b = a.float(), b.float()
        return a, b



def get_dataloaders(dataloader_config):
    """
    Parameters
    ----------
    dataloader_config : dict
    dataloader configuration (see config.py).

    Returns
    -------
    Pytorch Dataloader or dict of Dataloader 
    """
    max_chl_list = list()
    min_chl_list = list()
    max_pred = list()
    min_pred = list()
    train_datasets=[np.load(i) for i in dataloader_config['train_dataset_path']]
    valid_datasets=[np.load(i) for i in dataloader_config['valid_dataset_path']]
    test_datasets=[np.load(i) for i in dataloader_config['test_dataset_path']]
    # print(len(valid_datasets[1][1][0][0]))
    # print(len(valid_datasets[1]))
        
    if dataloader_config['log_chl']:
        train_datasets[1]=np.log10(train_datasets[1])
        valid_datasets[1]=np.log10(valid_datasets[1])
        test_datasets[1]=np.log10(test_datasets[1])
        
    if dataloader_config['normalize'] : 
        # train_datasets=[minmax_scaler(train_datasets[0], axis=0), minmax_scaler(train_datasets[1], axis=1)]
        # valid_datasets=[minmax_scaler(valid_datasets[0], axis=0), minmax_scaler(valid_datasets[1], axis=1)]
        train_datasets=[minmax_scaler(train_datasets[0], mod='0'), minmax_scaler(train_datasets[1], mod='chl')]
        valid_datasets=[minmax_scaler(valid_datasets[0], mod='0'), minmax_scaler(valid_datasets[1], mod='chl')]
        
        
        max_predictors=[np.nanmax(test_datasets[0][i]) for i in range(test_datasets[0].shape[0])]
        min_predictors=[np.nanmin(test_datasets[0][i]) for i in range(test_datasets[0].shape[0])]

        max_chl=np.nanmax(test_datasets[1])
        min_chl=np.nanmin(test_datasets[1])
        test_datasets=[minmax_scaler(test_datasets[0], mod='0'), minmax_scaler(test_datasets[1], mod='chl')]
        # test_datasets=[minmax_scaler(test_datasets[0], axis=0), minmax_scaler(test_datasets[1], axis=1)]
    
    training_set=chl_predictors_dataset(train_datasets, 
                                    dataloader_config['transform'])
    
    validation_set=chl_predictors_dataset(valid_datasets, 
                                    dataloader_config['transform'])
    testing_set=chl_predictors_dataset(test_datasets, 
                                    dataloader_config['transform'])
   
    training_generator   = DataLoader(training_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    
    validation_generator = DataLoader(validation_set,
                                      batch_size=dataloader_config['batch_size'],
                                      shuffle=True)
    testing_generator = DataLoader(testing_set,
                                      batch_size=1,
                                      shuffle=False)
    #print(len(training_generator))
    max_chl_list.append(max_chl)
    min_chl_list.append(min_chl)
    max_pred.append(max_predictors) 
    min_pred.append(min_predictors)
        
    return training_generator, validation_generator, testing_generator, max_chl_list, min_chl_list, max_pred, min_pred
    

if __name__=='__main__':
    dataloader_config={'train_dataset_path':["/data/home/ezheng/data/train/predictors_pasimp.npy","/data/home/ezheng/data/train/chla.npy"],
                       'valid_dataset_path':["/data/home/ezheng/data/val/predictors_pasimp.npy","/data/home/ezheng/data/val/chla.npy"],
                       'test_dataset_path':["/data/home/ezheng/data/test/predictors_pasimp.npy","/data/home/ezheng/data/test/chla.npy"],
                       #'transform':None,
                       #'transform':transforms.Compose([transforms.ToTensor(), transforms.Pad(padding=[6,8], padding_mode='edge')]),
                       'transform':transforms.Compose([transforms.ToTensor()]), 
                       'batch_size': 32,
                        'normalize':True,
                        'log_chl':True}
    
    test=get_dataloaders(dataloader_config)
    #print(test[0])
    #print(len(test[0]))
    

        
        
        
        
        
        
        
        
        
    
    
    
    