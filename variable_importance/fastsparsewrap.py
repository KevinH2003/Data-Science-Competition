import numpy as np
import pandas as pd
import fastsparsegams
from sklearn.base import BaseEstimator, TransformerMixin

class FastSparseSklearn(BaseEstimator, TransformerMixin):
    def __init__(self, max_support_size, tol=1e-8, lambda_0=0.025, gamma=0):
        # self.data = data 
        # self.labels = labels
        # self.data = data.to_numpy() if not isinstance(data, np.ndarray) else data
        # self.labels = labels.to_numpy() if not isinstance(labels, np.ndarray) else labels
        # self.data = self.transform(data)
        # self.num_features = np.shape(data)[1]
        self.max_support_size = max_support_size
        # self.labels = self.labels[0].T
        self.tol = tol
        self.lambda_0 = lambda_0
        self.gamma = gamma
        
    def transform(self, data):
        data = data.to_numpy() if not isinstance(data, np.ndarray) else data
        data = data.astype(float)
        return data
    
    def fit(self, data, labels):
        data = self.transform(data)
        labels = self.transform(labels)
        self.model = fastsparsegams.fit(data, labels, penalty="L0", max_support_size=self.max_support_size, algorithm = "CDPSI")
        
        coefficients = self.model.coeff(lambda_0=self.lambda_0, gamma=self.gamma).toarray()
        self.feature_importances_ = coefficients #might need to do more processing later

    def predict(self, X):
        X = self.transform(X)
        return self.model.predict(X, lambda_0=self.lambda_0, gamma=self.gamma)