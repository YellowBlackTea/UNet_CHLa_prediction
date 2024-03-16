#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:56:09 2022

@author: lollier


"""

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                                  FILL LANDS                                           | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

from scipy.signal import convolve2d
import numpy as np 
import os 
from tqdm import tqdm
import xarray as xr
    
    
def fill_lands(global_map, nb_step=1000):
    """
    Parameters
    ----------
    global_map : 2D numpy array
        global map of earth for a feature (sst,sla etc.)
        with nan for the lands
        
    nb_step : int, optional
        The default is 1000.

    Returns
    -------
    new_global_map : 2D numpy array
        corrected array where the nan values has been replaced by the features values extended 
        with a 2d diffusion eq*.
        
    *filling lands nan values with the heat diffusion equation 
    (ref Aubert et al, 2006) and http://www.u.arizona.edu/~erdmann/mse350/_downloads/2D_heat_equation.pdf
    basically it is a convolution of [0,1/4,0][1/4,-1,1/4][0,1/4,0]
    alpha=1 
    """
    
    nan_coords=np.where(np.isnan(global_map))
    new_global_map=np.nan_to_num(global_map)
    max_i,max_j=global_map.shape
    
    #solution discrète de l'équation de diffusion de la chaleur en 2D
    conv_mat=np.array([[0,1/4,0],[1/4,-1,1/4],[0,1/4,0]])
    
    for t in range(nb_step):
        for i,j in zip(nan_coords[0],nan_coords[1]):
            
            #creation de la matrice des voisins pour la propagation
            neighbors_mat=np.zeros((3,3))
            
            if i!=max_i-1:
                neighbors_mat[2,1]=new_global_map[i+1,j]
            if i!=0:
                neighbors_mat[0,1]=new_global_map[i-1,j]                    
            if j!=max_j-1:
                neighbors_mat[1,2]=new_global_map[i,j+1]
            if j!=0:
                neighbors_mat[1,0]=new_global_map[i,j-1]
                
            neighbors_mat[1,1]=new_global_map[i,j]
            new_global_map[i,j]=new_global_map[i,j]+convolve2d(neighbors_mat, conv_mat, mode='valid')[0,0]
            
    return new_global_map

def netcdf_to_numpy(reanalysis_inputs, processed_bathymetry):
    """
    Parameters
    ----------
    reanalysis_inputs : xarray dataset
        xarray dataset with the 9 predictors : 'thetao','uo','vo','zos','u10','v10','ssr','wmb','so'
    processed_bathymetry : xarray dataset
        xarray dataset with the correct bathymetry (key: wmb).

    Returns
    -------
    numpy_input : npy
        array of shape (predictors, time, lat, lon)

    """
    
    variables_name=['thetao','uo','vo','zos','u10','v10','ssr','wmb','so']
    numpy_input=np.empty((9,12,100,360))
    #first let's use the correct bathymetry values
    reanalysis_inputs['wmb'].values=processed_bathymetry['wmb'].values
    
    for i,variable in enumerate(variables_name):
        
        numpy_input[i]=np.squeeze(reanalysis_inputs[variable].values)
    
    return numpy_input


def make_dataset(years):
    
    path_data='/data/labo/data/DREAM/'
    inputs_list=[]
    chl_list=[]
    #making predictors dataset
    for year in years:
        
        reanalysis_inputs=xr.open_dataset(os.path.join(path_data, f'{year}/reanalysis_inputs.nc'))
        bathymetry=xr.open_dataset(os.path.join(path_data, f'{year}/bat_era5_reanalysis_processed.nc'))
        chl=xr.open_dataset(os.path.join(path_data, f'{year}/chla_processed.nc'))

        inputs_list.append(netcdf_to_numpy(reanalysis_inputs, bathymetry))
        chl_list.append(chl['CHL1_mean'].values)
        
    dataset_predictors=np.concatenate(inputs_list, axis=1)
    dataset_chl=np.concatenate(chl_list, axis=0)
    
    #fill lands
    
    nb_predictors,nb_timestep=dataset_predictors.shape[:2]
    
    for predictors in range(nb_predictors):#* -1 because we don't want to fill the land with bathymetry feature
        if predictors!=7:
            print(f"{predictors}")
            for time in range(nb_timestep):
                dataset_predictors[predictors, time]=fill_lands(dataset_predictors[predictors, time], nb_step=1000)
    
    #* we simply replace land nan by 0 for bathymetry
        else :
            dataset_predictors[7]=np.nan_to_num(dataset_predictors[7])

    
    return dataset_predictors,dataset_chl




if __name__ == '__main__':
    
    #test land fill function
    # test=np.random.randn(200,200)
    
    # nan_values=np.random.randint(0,200,(2000,2))
    
    # for coord in nan_values:
    #     i,j=coord
    #     test[i,j]=np.nan
        
    # res=fill_lands(test)
    
    
    #test netcdf numpy function
    # year=2008
    # path_data='/data/labo/data/DREAM/'
    # real_inp=xr.open_dataset(path_data+'2008/reanalysis_inputs.nc')  
    # bat=xr.open_dataset('/data/labo/data/DREAM/2008/bat_era5_reanalysis_processed.nc')  
    
    # test=netcdf_to_numpy(real_inp, bat)
    
    #test make dataset
    
    # year=[i for i in range(2003,2004)]

    # test_pred,test_chl=make_dataset(year)
    
    
    #valid 
    # print('_____________valid______________')
    # path_valid='/datatmp/home/lollier/npy_emergence/valid'
    # years=[i for i in range(1998,2002)]
    
    # valid_predictors_dataset,valid_chl_dataset=make_dataset(years)
    
    # np.save(os.path.join(path_valid, 'predictors.npy'),valid_predictors_dataset)
    # np.save(os.path.join(path_valid, 'chla.npy'),valid_chl_dataset)

    # #train 
    # print('_____________train______________')

    # path_train='/datatmp/home/lollier/npy_emergence/train'
    # years=[i for i in range(2003,2011)]
    
    # train_predictors_dataset,train_chl_dataset=make_dataset(years)
    
    # np.save(os.path.join(path_train, 'predictors.npy'),train_predictors_dataset)
    # np.save(os.path.join(path_train, 'chla.npy'),train_chl_dataset)
    
    # #test 
    # print('_____________test______________')

    # path_test='/datatmp/home/lollier/npy_emergence/test'
    # years=[i for i in range(2012,2017)]
    
    # test_predictors_dataset,test_chl_dataset=make_dataset(years)
    
    # np.save(os.path.join(path_test, 'predictors.npy'),test_predictors_dataset)
    # np.save(os.path.join(path_test, 'chla.npy'),test_chl_dataset)


    #others
    print('_____________autres______________')

    for years in [2002,2011,2017,2018,2019]:
        print(f'_____________{years}_______________')
        path_autres='/datatmp/home/lollier/npy_emergence/autres'
    
        
        autres_predictors_dataset,autres_chl_dataset=make_dataset([years])
        
        np.save(os.path.join(path_autres, f'{years}_predictors.npy'),autres_predictors_dataset)
        np.save(os.path.join(path_autres, f'{years}_chla.npy'),autres_chl_dataset)

    
    #***************************************************************************************************************#
    #j'ai fill la bathymetry comme un tocard alors que je peux juste tout mettre à 0, petit script pour corriger ça #
    #(et je corrige ça dans le script original suivi d'un commentaire '*')                                          #
    #***************************************************************************************************************#
    
    path='/datatmp/home/lollier/npy_emergence/'
    path_data='/data/labo/data/DREAM/'
    
    bathymetry=xr.open_dataset(os.path.join(path_data, '2008/bat_era5_reanalysis_processed.nc'))
    mask=np.isnan(bathymetry['wmb'].values[0])

    for folder in ['train','valid','test']:
        
        data=np.load(os.path.join(path,folder,'predictors.npy'))
        
        for time in range(data.shape[1]):
            temp=np.ma.masked_array(data[7,time],mask=mask)
            data[7,time]=temp.filled(0.0)
            
        np.save(os.path.join(path,folder,'predictors.npy'), data)
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    