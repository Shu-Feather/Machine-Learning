import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def closed_form_solution(X, y):
    '''
    Implement the closed form solution for linear regression

    Args:
        X (np.ndarray) (n, d): the input data
        y (np.ndarray) (n,): the ground truth label
    Returns:
        y_pred (np.ndarray) (n,): the predicted value
        theta (np.ndarray) (2,1): the weights
    '''
    # In this section, please implement the linear regression using the closed form solution 

    # TODO: firstly, add a column of 1s to the x_train as the bias term

    n, d = X.shape
    I = np.ones((n, 1))
    X_hat = np.concatenate((X, I), axis = 1)

    # TODO: secondly, use the closed form solution to calculate the best theta

    theta = np.linalg.inv(np.transpose(X_hat) @ X_hat) @ np.transpose(X_hat) @ y

    # TODO: finally, compute the y_pred using the best theta

    y_pred = X_hat @ theta


    return y_pred, theta

def predict(X, theta):
    ''' 
    Predict the output of the input data

    Args:
        X (np.ndarray) (n, d): the input data
        theta (np.ndarray) (2,1): the weights
    Returns:
        y_pred (np.ndarray) (n,): the predicted value
    '''
    # TODO: compute the y_pred using the input data and the weights

    n, d = X.shape
    I = np.ones((n, 1))
    X_hat = np.concatenate((X, I), axis = 1)
    y_pred = X_hat @ theta

    return y_pred

# Define the loss function
def compute_loss(y_pred, label):
    ''' 
    Compute the loss function for linear regression

    Args:
        y_pred (np.ndarray) (n,): the predicted value
        label (np.ndarray) (n,): the ground truth label
    Returns:
        loss (float): the loss value
    '''
    # TODO: compute the loss using the y_pred and the label
    
    n,  = y_pred.shape
    loss = 0
    for i in range(n):
        loss += pow(y_pred[i] - label[i], 2)
    loss = loss / n
    return loss
    
    raise NotImplementedError
