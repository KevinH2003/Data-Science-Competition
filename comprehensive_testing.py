import numpy as np
import pandas as pd
import sklearn
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from warnings import simplefilter
from dgp import DataGenerator
from variable_importance_scoring import importance_score, cross_validation_scores

param_grid_lasso = {
    'alpha': [1e-2, 1e-1, 1, 10],  # A range from very small to large alpha values
    'max_iter': [1000, 5000, 10000],  # Maximum number of iterations to converge
    'tol': [1e-4, 1e-3, 1e-2]  # Tolerance for the optimization
}

'''
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],  # Smaller values make the model more robust.
    'n_estimators': [100, 300, 500],  # More trees can be better, but at the risk of overfitting.
    'max_depth': [3, 5, 7]
}
'''

param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],  # Smaller values make the model more robust.
    'n_estimators': [100, 500, 1000],  # More trees can be better, but at the risk of overfitting.
    'max_depth': [3, 5, 7, 9],  # Depths greater than 10 might lead to overfitting.
    'min_child_weight': [1, 3, 5],  # Controls over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    'gamma': [0, 0.1, 0.2],  # Larger values make the algorithm more conservative.
    'subsample': [0.6, 0.8, 1.0],  # Values lower than 0.6 might lead to under-fitting.
    'colsample_bytree': [0.6, 0.8, 1.0],  # Considering a subset of features for each tree might make the model more robust.
    'reg_lambda': [1, 1.5, 2],  # L2 regularization term.
    'reg_alpha': [0, 0.1, 0.5],  # L1 regularization term, larger values specify stronger regularization.
    'scale_pos_weight': [1]  # Use only if you have a highly imbalanced class distribution.
}

n_iter = 10

dgp_names = ["Toy Dataset w/ Noise", "1000 Cols, Highly Correlated", "Corn Mimic", "100 Cols and 50 Interaction Terms"]
dgps = [
    DataGenerator(num_cols=100, num_rows=100, num_important=10, num_interaction_terms=0, effects='constant', noise=1),
    DataGenerator(num_cols=1000, num_rows=100, num_important=10, num_interaction_terms=50, effects='linear', correlation_range=[-0.9, 0.9], noise=5),
    DataGenerator(num_cols=50000, num_rows=160, num_important=10, num_interaction_terms=500, effects='all', correlation_range=[-0.95, 0.95]),
    DataGenerator(num_cols=100, num_rows=1000, num_important=10, num_interaction_terms=50, effects='all', correlation_range=[-1, 1])
]

model_names = ["LASSO", "XGBoost"]
models = [Lasso(), XGBRegressor()]
param_grids= [param_grid_lasso, param_grid_xgb]
importance_attrs = ['coef_', 'feature_importances_']

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore")

aggregated_scores = {}

for i in range(len(dgps)):
    dgp = dgps[i]
    dataset = dgp.generate_data()
    X = dataset.drop(["target"], axis=1)
    y = dataset["target"]

    model_scores = {}

    for i in range(len(models)):
        model = models[i]
        param_grid = param_grids[i]
        importance_attr = importance_attrs[i]

        rscv = GridSearchCV(model, param_grid, scoring='r2', verbose=0)
        model_scores[model_names[i]] = (cross_validation_scores(rscv, X, y, importance_attr=importance_attr, true_importances=dgp.importances, verbose=True))

    aggregated_scores[dgp_names[i]] = model_scores