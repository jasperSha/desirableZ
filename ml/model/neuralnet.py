# %% Define our model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def rmse_loss(predictions: np.array, targets: np.array) -> np.array:
    '''
    Root Mean Squared Error Loss Function, aka L2 norm
    RMSE = sqrt( avg(y - yhat)^2),
    where y is the observed value and yhat is the prediction.
    
    Measures the average magnitude of error.
    
    RMSE minimizes the squared deviations and finds the *mean*
    MAE minimizes the sum of absolute deviations resulting in the *median*
    
    
    Either we use MAE to handle the outliers, or we divide the dataset
    by zones/clusters, and run RMSE on each zone separately, as the
    model for the Beverly Hills zipcode is going to be essentially useless
    for the model for Northridge, and vice versa.
    
    Might be optimal for this particularly for finding good deals on houses.
    
    '''
    diff = predictions - targets
    diff_squared = np.square(diff)
    mean_diff_squared = np.mean(diff_squared)
    
    rmse = np.sqrt(mean_diff_squared)
    
    return rmse

def mae_loss(predictions: np.array, targets: np.array) -> np.array:
    '''
    Mean Absolute Loss Error Function, aka L1 norm
    MAE = avg(abs(y - yhat))
    
    Also measures the average magnitude of error, but uses absolute value
    to eliminate the direction of error. It also equally weights all data 
    points. If we run the model on all the zones/clusters together, the
    MAE will probably be optimal, and not be thrown off as much by the
    massive outliers resultant from the wealth disparity in LA.
    
    '''
    diff = predictions - targets
    abs_diff = abs(diff)
    
    mae = np.mean(abs_diff)
    return mae




class Net(nn.Module):
    '''
    
    Defining a feed-forward neural net.
    
    NonLinear Regression - ReLu function introduces nonlinearity, outputs
                           directly if positive, otherwise outputs zero.
    '''
    def __init__(self, D_in, D_out):
        super().__init__()
        #input layer has 43 columns, so 43 inputs
        self.fc1 = nn.Linear(D_in, 100)
        
        #1 hidden layers, (20 - 20) each
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        
        #output layer
        self.fcout = nn.Linear(100, D_out)
        
        #initialize weights for each layer to uniform non-zeros, and bias to zeros
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
        nn.init.xavier_uniform_(self.fcout.weight)
        nn.init.zeros_(self.fcout.bias)
    
    def forward(self, x):
        #using ReLu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        y_pred = self.fcout(x)
        
        return y_pred




if __name__ == '__main__':
    pass