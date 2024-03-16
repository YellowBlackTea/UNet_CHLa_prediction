import numpy as np
from sklearn.model_selection import train_test_split
from utils.functions import impute_nan

path = "/datatmp/home/lollier/npy_emergence/"
new_path = "/data/home/ezheng/data/"

def train_val_test_split(X, y):
    # # séparation 80-10-10
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=309)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, shuffle=True, random_state=309)
    
    # # separation 80-15-5
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=True, random_state=309)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16, shuffle=True, random_state=309)
    
    # séparation 90-10 et test = 2 ans
    X_test = X[920:,]
    y_test = y[920:,]    
    
    X_new = X[:920,]
    y_new = y[:920,]
    
    X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=0.1, shuffle=True, random_state=309)
    
    X_train_imp = impute_nan(X_train, mode='mean')
    # bathy_train = X_train[:,-1:]
    # x_train = X_train[:,:-1]
    # X_train_imp = impute_nan(x_train, mode='mean')
    # X_train_final = np.append(X_train_imp, bathy_train, axis=1)
    # #y_train = impute_nan(y_train)
    
    X_test_imp = impute_nan(X_test, mode='mean')
    # bathy_test = X_test[:,-1:]
    # x_test = X_test[:,:-1]    
    # X_test_imp = impute_nan(x_test, mode='mean')
    # X_test_final = np.append(X_test_imp, bathy_test, axis=1)
    # #y_test = impute_nan(y_test)
    
    X_val_imp = impute_nan(X_val, mode='mean')
    # bathy_val = X_val[:,-1:]
    # x_val = X_val[:,:-1]    
    # X_val_imp = impute_nan(x_val, mode='mean')
    # X_val_final = np.append(X_val_imp, bathy_val, axis=1)
    # #y_val = impute_nan(y_val)
    
    np.save(new_path+'train/predictors_corr.npy', X_train_imp)
    np.save(new_path+'train/chla.npy', y_train)

    np.save(new_path+'test/predictors_corr.npy', X_test_imp)
    np.save(new_path+'test/chla.npy', y_test)

    np.save(new_path+'val/predictors_corr.npy', X_val_imp)
    np.save(new_path+'val/chla.npy', y_val)
    
    # np.save(new_path+'train/bathy01.npy', bathy_train)
    # np.save(new_path+'test/bathy01.npy', bathy_test)
    # np.save(new_path+'val/bathy01.npy', bathy_val)

if __name__=='__main__':
    X = np.load(new_path+"dyn_corr.npy")
    y = np.load(path+"chl.npy")
    
    # bathy = X[:,8:]
    # x = X[:,:8]
    # X_imp = impute_nan(x)
    # new_X = np.append(X_imp, bathy, axis=1)
    #np.save(new_path+"dyn_8var+bathy01_imput.npy", new_X)
    train_val_test_split(X, y)