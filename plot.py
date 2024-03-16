#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:34:22 2023

@author: lollier
"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         Plot                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable

import numpy as np
import os
import xarray as xr
import cmocean.cm as cm
import torch

import cartopy
import cartopy.feature as cfeature
import cartopy.crs as ccrs
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                           PLOT FUNCTIONS TO SHOW DATA AND RESULT                      | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


"""
TO DO
--> ajouter un affichage en log, ie un param qui prend le log de la carte d'entrée mais sans changer la colorbar
--> pour les PSC gérer un triple affichage ?
         
"""

def check_shape(data):
    """
    Parameters
    ----------
    tensor : tensor or array
        check the size and the format of inputs and return the axes for plot.

    Returns
    -------
    tensor : TYPE
        DESCRIPTION.
    axes_label : TYPE
        DESCRIPTION.

    """
    if type(data) not in [np.ndarray, np.array, torch.Tensor]:
        raise ValueError('type not valid')
    
    if type(data)==torch.Tensor:
        data=data.numpy()
        
    if data.shape==(120,360):
        axes_label=[-179.5,
                    179.5,
                    -59.5,
                    59.5]   
        return data, axes_label
    
    if data.shape==(100,360):
        axes_label=[-179.5,
                    179.5,
                    -49.5,
                    49.5]   
        return data, axes_label
    
    
    else :
        raise ValueError('shape not valid')
        
        
 
def imshow_area(array, cmap='jet', fig=None, ax=None, 
                vmin=None, vmax=None, log=False, title=False,
                colorbar=True, land=True, idx=0):

    if (fig is None and not(ax is None)) or (ax is None and not(fig is None)):
        raise ValueError("You need to specify both ax and fig params")
        

    array, axes_label=check_shape(array)
    arr = np.copy(array)
    lon, lat = np.mgrid[axes_label[0]:axes_label[1]+0.5,
                        axes_label[2]:axes_label[3]+0.5] #ça marche parce qu'on a 1 degré de lib
    proj=ccrs.Mercator()
    
    if land:
        load_nan = np.load("/data/home/ezheng/data/test/chla.npy")
        land_nan = load_nan[idx,0]
        for nb_missing, (i,j) in enumerate(np.argwhere(np.isnan(land_nan))):
            arr[i,j] = np.nan
            
    array = np.copy(arr)
            
    if fig is None and ax is None:
        fig=plt.figure(figsize=(11,8.5))
        ax=plt.subplot(1,1,1, projection=proj)
        show=True
        
    else :
        fig, ax=fig, ax
        show=False
        
    if log :
        
        array=np.log(array)
        vmax,vmin=np.nanmax(array),np.nanmin(array)
        cb=LogNorm(10**-3,10**2)
    
    else:
        # vmin=-1
        # vmax=0.08
        pass
    img = ax.pcolormesh(lon, lat, np.flip(np.rollaxis(array,1),1), 
                        transform=ccrs.PlateCarree(),cmap=cmap,
                        vmin=vmin, vmax=vmax)
    
    ax.coastlines(alpha=0.5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=0.5, 
                      color='gray', 
                      alpha=0.2, 
                      linestyle='solid')
    
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='grey', zorder=0)  

    
    ax.set_extent(axes_label, crs=ccrs.PlateCarree())
    
    if colorbar:
        if log : 
            fig.colorbar(ScalarMappable(cb,cmap=cmap), orientation='vertical', shrink=0.4)
        else :
            fig.colorbar(img, orientation='vertical', shrink=0.4)
              
              
    ax.set(xlabel='Longitude', ylabel='Latitude')
    #ax.label_outer()
    
    
    if title:
        ax.set_title(title)
        
    if show :
        plt.show()
        
        
# if __name__=="__main__": 
    
    #PSC=xr.load_dataset("/data/home/lollier/emergence/data/PSC_19980101_20191231_8d_100_lat50.nc")
    # PSC=np.load("/datatmp/home/lollier/npy_emergence/psc.npy")
    # CHL=np.load("/datatmp/home/lollier/npy_emergence/chl.npy")
    
    # imshow_area(PSC[46*12,0], title="%PSC Micro 1-8 janvier 2010")
    
    # #CHL par année
    
    # # for i in range(0,CHL.shape[0],46):
    # #     imshow_area(np.nanmean(CHL[i:i+46,0],axis=0), log=True,title=f'chl year {1998+i/46}')
    
    # #CHL par mois
    
    # for i in range(12):
    #     imshow_area(np.nanmean(CHL[i::46,0],axis=0), log=True,title=f'chl month {i}')

    # #%PSC par année
    
    # # proj=ccrs.Mercator()

    # # for i in range(0,PSC.shape[0],46):
        
    # #     fig = plt.figure(figsize=(10,15))
    # #     gs = fig.add_gridspec(nrows=3, ncols=1,
    # #                           hspace=0.05, wspace=0.025, right=1)
    # #     axs=[]
    # #     axs.append(fig.add_subplot(gs[0],projection=proj))
    # #     axs.append(fig.add_subplot(gs[1],projection=proj))
    # #     axs.append(fig.add_subplot(gs[2],projection=proj))
        
    # #     imshow_area(np.nanmean(PSC[i:i+46,0],axis=0), cmap='BuGn', fig=fig, ax=axs[0], log=False,title=f'Micro year {1998+i/46}', colorbar=False)
    # #     imshow_area(np.nanmean(PSC[i:i+46,1],axis=0), cmap='BuGn', fig=fig, ax=axs[1], log=False,title=f'Nano year {1998+i/46}', colorbar=False)
    # #     imshow_area(np.nanmean(PSC[i:i+46,2],axis=0), cmap='BuGn', fig=fig, ax=axs[2], log=False,title=f'Pico year {1998+i/46}', colorbar=False)
        
    # #     cb=Normalize(0,100)
    # #     cax = plt.axes([1.1, 0.15, 0.01, 0.7])
    # #     fig.colorbar(ScalarMappable(cb,cmap='BuGn'),cax=cax, label='%')
    # #     plt.show()


    # #PSC*CHL par année
    
    
    # # proj=ccrs.Mercator()

    # # for i in range(0,PSC.shape[0],46):
        
    # #     fig = plt.figure(figsize=(10,15))
    # #     gs = fig.add_gridspec(nrows=3, ncols=1,
    # #                           hspace=0.05, wspace=0.025, right=1)
    # #     axs=[]
    # #     axs.append(fig.add_subplot(gs[0],projection=proj))
    # #     axs.append(fig.add_subplot(gs[1],projection=proj))
    # #     axs.append(fig.add_subplot(gs[2],projection=proj))
        
    # #     chl=np.nanmean(CHL[i:i+46,0],axis=0)
        
    # #     imshow_area(chl*np.nanmean(PSC[i:i+46,0],axis=0), cmap='hsv', fig=fig, ax=axs[0], log=True,title=f'Micro year {1998+i/46}', colorbar=False)
    # #     imshow_area(chl*np.nanmean(PSC[i:i+46,1],axis=0), cmap='hsv', fig=fig, ax=axs[1], log=True,title=f'Nano year {1998+i/46}', colorbar=False)
    # #     imshow_area(chl*np.nanmean(PSC[i:i+46,2],axis=0), cmap='hsv', fig=fig, ax=axs[2], log=True,title=f'Pico year {1998+i/46}', colorbar=False)
        
    # #     cb=LogNorm(10**-3,10**2)
    # #     cax = plt.axes([1.1, 0.15, 0.01, 0.7])
    # #     fig.colorbar(ScalarMappable(cb,cmap='hsv'),cax=cax, label='concentration PSC')
    # #     plt.show()
    
    
    
    # #PSC*CHL par mois
    
    
    # proj=ccrs.Mercator()

    # for i in range(12):
        
    #     fig = plt.figure(figsize=(10,15))
    #     gs = fig.add_gridspec(nrows=3, ncols=1,
    #                           hspace=0.05, wspace=0.025, right=1)
    #     axs=[]
    #     axs.append(fig.add_subplot(gs[0],projection=proj))
    #     axs.append(fig.add_subplot(gs[1],projection=proj))
    #     axs.append(fig.add_subplot(gs[2],projection=proj))
        
    #     chl=np.nanmean(CHL[i:i+46,0],axis=0)
        
    #     imshow_area(chl*np.nanmean(PSC[i::46,0],axis=0), cmap='hsv', fig=fig, ax=axs[0], log=True,title=f'Micro month {i}', colorbar=False)
    #     imshow_area(chl*np.nanmean(PSC[i::46,1],axis=0), cmap='hsv', fig=fig, ax=axs[1], log=True,title=f'Nano month {i}', colorbar=False)
    #     imshow_area(chl*np.nanmean(PSC[i::46,2],axis=0), cmap='hsv', fig=fig, ax=axs[2], log=True,title=f'Pico month {i}', colorbar=False)
        
    #     cb=LogNorm(10**-3,10**2)
    #     cax = plt.axes([1.1, 0.15, 0.01, 0.7])
    #     fig.colorbar(ScalarMappable(cb,cmap='hsv'),cax=cax, label='concentration PSC')
    #     plt.show()

    # #%nan par pixel pour la CHL et pour les PSC
    
    # #spatial
    # imshow_area(np.sum(np.isnan(CHL[:,0]),axis=0)/1012, cmap='BuGn',title='% nan chl')
    # imshow_area(np.sum(np.isnan(PSC[:,0]),axis=0)/1012, cmap='BuGn',title='% nan Micro')
    # imshow_area(np.sum(np.isnan(PSC[:,1]),axis=0)/1012, cmap='BuGn',title='% nan Nano')
    # imshow_area(np.sum(np.isnan(PSC[:,2]),axis=0)/1012, cmap='BuGn',title='% nan Pico')

    # #temporel

    # plt.plot(np.sum(np.isnan(CHL[:,0]),axis=(1,2))/36000)
    # plt.plot(np.sum(np.isnan(PSC[:,0]),axis=(1,2))/36000)
    # plt.plot(np.sum(np.isnan(PSC[:,1]),axis=(1,2))/36000)
    # plt.plot(np.sum(np.isnan(PSC[:,2]),axis=(1,2))/36000)
    
    # plt.legend(['chl','Micro','Nano','Pico'])
    # plt.show()
    
    
    
    
