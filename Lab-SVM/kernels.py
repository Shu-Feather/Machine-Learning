import numpy as np


class Base_kernel():
    
    def __init__(self):
        pass
    
    def __call__(self, x1, x2):
        """
        Base kernel function.

        Arguments:
            x1 : np.ndarray, shape (n1, d) - First input data array.
            x2 : np.ndarray, shape (n2, d) - Second input data array.
            
        Returns:
            y : np.ndarray, shape (n1, n2), where y[i, j] = kernel(x1[i], x2[j]).
        """
        pass


class Linear_kernel(Base_kernel):
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, x1, x2):
        # TODO: Implement the linear kernel function
        
        y = x1 @ np.transpose(x2)

        return y
    
    
class Polynomial_kernel(Base_kernel):
        
    def __init__(self, degree, c):
        super().__init__()
        self.degree = degree
        self.c = c
        
    def __call__(self, x1, x2):
        # TODO: Implement the polynomial kernel function

        y = pow(x1 @ np.transpose(x2) + self.c, self.degree)
        
        return y

class RBF_kernel(Base_kernel):
    
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma 
        
        
    def __call__(self, x1, x2):
        # TODO: Implement the RBF kernel function
        
        n1 = 0
        n2 = 0
        if len(x1.shape) == 1:
            n1 = 1
        else:
            n1 = x1.shape[0]
        if len(x2.shape) == 1:
            n2 = 1
        else:
            n2 = x2.shape[0]

        y = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):

                if n1 == 1:
                    xi = x1
                else:
                    xi = x1[i]
                if n2 == 1:
                    xj = x2
                else:
                    xj = x2[j]

                distance = (xi - xj) @ np.transpose(xi - xj)

                y[i, j] = np.exp(-1 / (2 * np.power(self.sigma, 2)) * distance)

        return y