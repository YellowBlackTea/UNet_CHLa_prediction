#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:47:46 2023

@author: ezheng
"""

import numpy as np
import matplotlib.pyplot as plt


path = "/datatmp/home/lollier/npy_emergence/"
new_path = "/data/home/ezheng/data/"

def hist_var(var, xlim1=None, xlim2=None, ylim1=None, ylim2=None):
    plt.xlim(xlim1,xlim2)
    plt.ylim(ylim1,ylim2)
    plt.hist(np.reshape(var,1012*100*360), bins=1000)
    plt.ylabel()

if __name__=='__main__':
    x = np.load(new_path+"dyn_8var_bathy01.npy")
    y = np.load(new_path+"dyn_10var_bathy01.npy")
    c = np.load(new_path+"dyn_v2.npy")
    b = np.load(path+"dyn.npy")
    chl_modif = np.load(new_path+"chla_with_threshold.npy")
    chl = np.load(path+"chl.npy")
    
    # bathy = b[:,9:]
    # bathy1 = np.where(bathy>1000,True,bathy)
    # bathy01 = np.where(bathy<1000,False,bathy1)
    # data_norm_rot = c
    # data_sans_norm = b
    
    # new_data_norm_rot = np.append(c, bathy01, axis=1)
    # new_data = np.append(b, bathy01, axis=1)
    # np.save(new_path+"dyn_8var+bathy01.npy",new_data_norm_rot)
    # np.save(new_path+"dyn_10var+bathy01.npy",new_data)
    
    # chla = np.load(path+"chl.npy")
    # new_chla = np.where(chla<1e-2, 1e-2, chla)
    # np.save(new_path+"chla_with_threshold.npy", new_chla)
    
    # new_data = b[:,[3,4,5,6]]
    # np.save(new_path+"dyn_corr.npy", new_data)
    
    
    mld = np.reshape(b[:,:1],1012*100*360)
    u0 = np.reshape(b[:,1:2],1012*100*360)
    v0 = np.reshape(b[:,2:3],1012*100*360)
    sst = np.reshape(b[:,3:4],1012*100*360)
    so = np.reshape(b[:,4:5],1012*100*360)
    zos = np.reshape(b[:,5:6],1012*100*360)
    u10 = np.reshape(b[:,6:7],1012*100*360)
    v10 = np.reshape(b[:,7:8],1012*100*360)
    cdir = np.reshape(b[:,8:9],1012*100*360)
    wmb = np.reshape(b[:,9:],1012*100*360)
    
    chla = np.log10(np.reshape(chl, 1012*100*360))
    
    norm = np.reshape(c[:,6:7],1012*100*360)
    rot = np.reshape(c[:,7:8],1012*100*360)
    
    # plt.xlim(-0.5,2.5)
    plt.hist((chla-np.nanmin(chla))/(np.nanmax(chla)-np.nanmin(chla)), bins=5000)
    plt.xlabel("Concentration de log(CHL-a)_scaled [ ]")
    plt.ylabel('Fréquences')
    plt.show()

    # plt.xlim(-5,300)
    # plt.hist(np.reshape(b[:,:1],1012*100*360), bins=1000) # mlost
    # plt.xlabel("Profondeur de la couche de mélange - MLD [m]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.xlim(-1.5, 1.5)
    # plt.hist(np.reshape(b[:,1:2],1012*100*360), bins=1000) # u0
    # plt.xlabel("Vitesse du courant océanique vers l’est - u0 [m/s]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.xlim(-1, 1)
    # plt.hist(np.reshape(b[:,2:3],1012*100*360), bins=1000) # v0
    # plt.xlabel("Vitesse du courant océanique vers le nord - v0 [m/s]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.xlim(-10, 40)
    # plt.hist(np.reshape(b[:,3:4],1012*100*360), bins=1000) # theta0 sst
    # plt.hist((sst-np.nanmin(sst))/(np.nanmax(sst)-np.nanmin(sst)), bins=1000)
    # plt.xlabel("Température à la surface de la mer - SST [°C]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.xlim(25, 45)
    # plt.hist(np.reshape(b[:,4:5],1012*100*360), bins=1000) # so
    # plt.xlabel("Salinité de la mer - so [psu]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.xlim(-2, 2)
    # plt.hist(np.reshape(b[:,5:6],1012*100*360), bins=1000) # zos
    # plt.xlabel("Hauteur de la surface de la mer - sla [m]")
    # plt.ylabel('Fréquences')
    # plt.show()

    # plt.xlim(-15, 20)
    # plt.hist(np.reshape(b[:,6:7],1012*100*360), bins=1000) # u10
    # plt.xlabel("Vitesse du vent vers l’est - u10 [m/s]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.xlim(-13, 13)
    # plt.hist(np.reshape(b[:,7:8],1012*100*360), bins=1000) # v10
    # plt.xlabel("Vitesse du vent vers le nord- v10 [m/s]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.xlim(0.5e5, 1.5e6)
    # plt.hist(np.reshape(b[:,8:9],1012*100*360), bins=1000) # cdir
    # plt.xlabel("Rayonnement solaire - cdir [W/m2]")
    # plt.ylabel('Fréquences')
    # plt.show()    
    
    # plt.ylim(0,1e6)
    # plt.hist(np.reshape(b[:,9:],1012*100*360), bins=100) # wmb bathy
    # plt.xlabel("Profondeur de la mer - wmb [m]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.hist(chla, bins=1000) # wmb bathy
    # plt.xlabel("Concentration de CHL-a [mg/m3]")
    # plt.ylabel('Fréquences')
    # plt.show()
    
    # plt.xlim(-0.5,2)
    # plt.hist(np.reshape(c[:,6:7],1012*100*360), bins=1000) # norme
    
    # plt.xlim(-3,3)
    # plt.hist(np.reshape(c[:,7:8],1012*100*360), bins=1000) # rot
    
    # np.mean(b[:,0], axis=(0,2,3))
    
    
    
    
    
    