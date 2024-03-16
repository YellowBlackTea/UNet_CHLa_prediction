import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import convolve2d

def minmax_scaler(X, mod):
    
    
    # for i in range(X.shape[1]):
    #     scaler = MinMaxScaler()
    #     X_reshaped = X[:,i].reshape(X.shape[0], X.shape[2]*X.shape[3])
    #     X_scaled_2d = scaler.fit_transform(X_reshaped)
    #     X_scaled = X_scaled_2d.reshape(X.shape[0, X.shape[2], X.shape[3])
    
    # X_scaled[mask] = np.nan
    
    # X_normed = np.append(X_scaled, bathy, axis=1)

    mask = np.isnan(X)
    X_min = np.nanmin(X, axis=(2,3), keepdims=True)
    X_max = np.nanmax(X, axis=(2,3), keepdims=True)
    # print(X_min, "===wow===\n", X_max)
    X_scaled = (X - X_min)/(X_max-X_min)
    # print(X_scaled.shape)
    if mod=='chl':
        X_scaled[mask] = np.nan

    
    return X_scaled


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

def impute_nan(missing_data, mode='mean'):
    if mode=='heat':
       # for i in range(missing_data.shape[1]):
       #     for j in range(missing_data.shape[0]):
       #         x = missing_data[j][i]
       #         res = fill_lands(x, nb_step=100)
       #         missing_data[j] = res
       res = fill_lands(missing_data, nb_step=1000)
       missing_data = res
    if mode=='-inf':
        # fill_nan = -30
        missing_data[~np.isnan(missing_data)] = -50
    else:
        mean = np.nanmean(missing_data, axis=(0,2,3))
        print(mean.shape)
        for nb_missing, (i,j,k,l) in enumerate(np.argwhere(np.isnan(missing_data))):
            missing_data[i,j,k,l] = mean[j]
        # missing_data[np.isnan(missing_data)] = mean
    
    return missing_data

def add_threshold(chla):
    return np.where(chla<1e-2, 1e-2, chla)

if __name__=="__main__":
    X = np.random.rand(800,10,100,360)
    impute_nan(X)