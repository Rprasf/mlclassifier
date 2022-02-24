# coding:utf-8
import numpy as np


class BaseEstimator:

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary.
        
        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like

        """
        print("base")
        print(X)
        print(y)
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if y.size == 0:
            raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y):
        self._setup_input(X, y)
        if self.max_features is None:
            self.max_features = int(np.sqrt(self.X.shape[1]))
        else:
            assert self.X.shape[1] > self.max_features  

    def predict(self):
        raise NotImplementedError()
 
        