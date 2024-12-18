import numpy as np
from typing import Tuple
from vis_util import plot_progress


def sigmoid(x):
    ''' 
    Sigmoid function.
    '''
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
    ''' 
    Logistic Regression
    '''
    def __init__(
        self,
        plot: bool = False
    ) -> None:
        
        self.w = None # random intialize w
        self.lr = None # learning rate
        self.reg = None # regularization parameter
        self.plot = plot

    def predict(
        self,
        x: np.array
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        ''' 
        Logistic Regression (LR) prediction.
        
        Arguments:
            x : (n, d + 1), where n represents the number of samples, d the number of features

        Return:
            prob: (n,), LR probabilities, where prob[i] is the probability P(y=1|x,w) for x[i], from [0, 1]
            pred: (n,), LR predictions, where pred[i] is the prediction for x[i], from {-1, 1}

        '''
        # implement predict method,
        # !! Assume that : self.w is already given.
        
        # TODO: first, you should compute the probability by invoking sigmoid function
        
        n = x.shape[0]
        prob = np.zeros(n)
        pred = np.zeros(n)

        for i in range(n):
            xi = x[i,:]
            p = self.w @ np.transpose(xi)
            prob[i] = sigmoid(p)

            # TODO: second, you should compute the prediction (W^T * x >= 0 --> y = 1, else y = -1)
            if p>=0 :
                pred[i] = +1
            else :
                pred[i] = -1

        return prob, pred
    

    def fit(
        self,
        x: np.array,
        y: np.array,
        n_iter: int,
        lr: float,
        reg: float,
    ) -> None:
        ''' 
        Logistic Regression (LR) training.

        Arguments:
            x : (n, d), where n represents the number of training samples, d the number of features
            y : (n,), where n represents the number of samples
            n_iter : number of iteration
            lr : learning rate
            reg : regularization parameter
            
        Return:
            None
        '''
        self.lr = lr
        self.reg = reg

        # add bias term
        x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        # random intialize w
        self.w = np.random.normal(0, 1, x.shape[1]) 
        loss_history = []
        w_module_history = []

        for i in range(n_iter):
            # update the weight 
            self.update(x, y)
            
            # plot loss and w module every 10 iterations
            if i % 10 == 0:
                # compute the loss
                loss = self.calLossReg(x, y)
                
                loss_history.append(loss)

                w_module_history.append(np.linalg.norm(self.w))
                if self.plot:
                    print("iter: {}, loss: {}, w_module: {}".format(i, loss, w_module_history[-1]))
                    plot_progress(i, loss_history,w_module_history, x, y, self.w)
        return

    def update(
        self,
        x: np.array,
        y: np.array,
    ) -> None:
        
        '''
        Update the parameters--weight w
        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            None
        '''

        # implement gradient descent algorithm
        n = x.shape[0]
    
        # TODO: 1. compute the gradient
        dw = np.zeros(x.shape[1])

        for i in range(n):
            xi = x[i,:]
            dw = -xi * ( (y[i] + 1) / 2.0 - sigmoid(self.w @ np.transpose(xi)) )
        #print(dw)

            # TODO: 2. update the weight
        
            self.w -= self.lr * dw

        return


    def calLossReg(
        self,
        x: np.array,
        y: np.array,
    ):
        ''' 
        Compute the loss

        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            loss: float, the loss value
        '''
        # TODO: compute the Logistic Regression loss, including regularization term
        # !! Note that the label y is from {-1, 1}, and the exp index should be carefully considered

        loss = 0.0
        n = x.shape[0]

        for i in range(n):
            xi = x[i,:]
            loss -= (y[i] + 1) / 2.0 * np.log(sigmoid(self.w @ np.transpose(xi))) + (1 - y[i]) / 2.0 * np.log(1 - sigmoid(self.w @ np.transpose(xi)))
        # Adding the regularization term
        loss += self.reg * self.w @ np.transpose(self.w) 

        return loss



    def calLoss(
        self,
        x: np.array,
        y: np.array,
    ):
        ''' 
        Compute the loss

        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            loss: float, the loss value
        '''
        # TODO: compute the Logistic Regression loss, not including regularization term
        # !! Note that the label y is from {-1, 1}, we should project it to {0, 1} by linear transformation: y <- (y + 1) / 2

        loss = 0.0
        n = x.shape[0]

        for i in range(n):
            xi = x[i,:]
            loss -= (y[i] + 1) / 2.0 * np.log(sigmoid(self.w @ np.transpose(xi))) + (1 - y[i]) / 2.0 * np.log(1 - sigmoid(self.w @ np.transpose(xi)))
    
        # Adding the regularization term
        # loss += self.reg * self.w @ np.transpose(self.w) 

        return loss


def compute_accuracy(
    y_pred: np.array,
    y_true: np.array,
) -> float:
    '''
    Compute the accuracy

    Arguments:
        y_pred: (n,), where n represents the number of samples
        y_true: (n,), where n represents the number of samples
    Return:
        acc: float, the accuracy
    '''
    acc = 0.0

    # TODO: compute the accuracy

    n = y_pred.shape[0]

    for i in range(n):
        if abs(y_pred[i] - y_true[i]) < 1e-5 :
            acc += 1

    acc = acc / n

    return acc
 