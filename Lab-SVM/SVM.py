import numpy as np
from typing import Tuple

class SVM(object):
    """
    Support Vector Machine (SVM) classifier using kernel methods.

    This SVM class is designed to operate in the dual form, leveraging a specified kernel function to map input data
    into a higher-dimensional feature space. The model parameters include support vectors, dual variables (alpha),
    support labels, and a threshold (b).
    """

    def __init__(self, kernel_fn) -> None:
        """
        Initializes the SVM model with the given kernel function.

        Arguments:
            kernel_fn: Kernel function to compute similarity 
            between data points in feature space.
        """
        self.kernel_fn = kernel_fn  # Kernel function as one object in **kernels.py**.
        self.b = None  # SVM's threshold (bias term), shape: (1,)
        self.alpha = None  # Dual variables for support vectors, shape: (n_support,)
        self.support_labels = None  # Labels of support vectors, shape: (n_support,), values in {-1, 1}
        self.support_vectors = None  # Support vectors, shape: (n_support, d)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the class labels and scores for the input data.

        This method uses the kernelized decision function to compute scores for each input sample.
        
        Arguments:
            x : np.ndarray, shape (n, d) - Input samples where `n` is the number of samples and `d` is the number of features.

        Returns:
            scores: np.ndarray, shape (n,) - Decision scores, where `scores[i]` is the SVM score for `x[i]`.
            pred: np.ndarray, shape (n,) - Predicted labels, where `pred[i]` is the predicted label for `x[i]`, values in {-1, 1}.
        """
        
        # TODO: Implement the predict method.
        # Assume self.b, self.alpha, self.support_labels, self.support_vectors, and self.kernel_fn are already set.
        # These attributes will be assigned during SSMO optimization, implemented separately.

        n = 0

        if len(x.shape) == 1:
            n = 1
        else:
            n = x.shape[0]
        
        n_support = self.support_vectors.shape[0]

        scores = np.zeros(n)
        pred = np.zeros(n)

        for i in range(n):
            for j in range(n_support):
                if n == 1:
                    scores[i] += self.alpha[j] * self.support_labels[j] * self.kernel_fn(x, self.support_vectors[j])
                else:
                    scores[i] += self.alpha[j] * self.support_labels[j] * self.kernel_fn(x[i], self.support_vectors[j])
            scores[i] += self.b

            if scores[i] >= 0:
                pred[i] = +1
            else:
                pred[i] = -1
        
        return scores, pred