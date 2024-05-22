import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from variable_importance.dgp import DataGenerator
from variable_importance.fastsparsewrap import FastSparseSklearn
from variable_importance.scoring import importance_score, model_importance_score, importance_testing
from variable_importance.cmr import CMR
from variable_importance.loco import LOCOImportance
from variable_importance.mr import MRImportance
import numpy as np

print("Starting...")

###Parameter Grids###
param_grid_lasso = {
    'alpha': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100], 
    'max_iter': [1000, 2500, 5000, 10000, 25000, 500000, 1000000],  
    'tol': [1e-4, 1e-3, 1e-2, 1e-1], 
}

param_grid_fastsparse = {
    "max_support_size": [5, 10, 15, 20, 25],
    "atol": [1e-9, 1e-8, 1e-7, 1e-6],
    "lambda_0": [0.001, 0.005, 0.01, 0.05, 0.1],
    "penalty": ["L0"],
}

param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1], 
    'n_estimators': [100, 300, 500], 
    'max_depth': [3, 5, 7], 
    'min_child_weight': [1, 3, 5, 7], 
    'gamma': [0.1, 0.2, 0.3],  
    'subsample': [0.8, 1.0],  
    'colsample_bytree': [0.6, 0.8, 1.0],  
    'reg_lambda': [1, 1.5, 2],  
    'reg_alpha': [0, 0.1, 0.5, 1], 
}

param_grid_xgb_pipeline = {
    'prediction__learning_rate': [0.01, 0.05, 0.1], 
    'prediction__n_estimators': [100, 300, 500],  
    'prediction__max_depth': [3, 5, 7], 
    'prediction__min_child_weight': [1, 3, 5, 7],  
    'prediction__gamma': [0.1, 0.2, 0.3], 
    'prediction__subsample': [0.8, 1.0], 
    'prediction__colsample_bytree': [0.6, 0.8, 1.0], 
    'prediction__reg_lambda': [1, 1.5, 2], 
    'prediction__reg_alpha': [0, 0.1, 0.5, 1], 
}

###DGPs###

dgps = {
    "Toy": DataGenerator(
        num_cols=100, num_rows=100, num_important=10, 
        num_interaction_terms=0, effects='linear', 
        noise_distribution='normal', noise_scale=0.1),
}

datasets = {name: dgp.generate_data() for name, dgp in dgps.items()}
true_importances = {name: dgps[name].bucket_importances for name in dgps.keys()}

def model_importance(model, true_importances, importance_attr, ranked=False, **kwargs):
    return model_importance_score(model, true_importances, importance_attr, ranked=ranked)

def mr_importance(X, y, model, true_importances, score_func='r2', ranked=False, **kwargs):
    mr = MRImportance(X, y, score_func, model)
    return importance_score(mr.get_importance(), true_importances, ranked=ranked)

def cmr_importance(X, y, model, true_importances, error_func=mean_squared_error, ranked=False, **kwargs):
    cmr = CMR(X, y, error_func, model)
    return importance_score(cmr.importance_all(), true_importances, ranked=ranked)

def loco_importance(X, y, model, true_importances, score_func='r2', cv=5, ranked=False, **kwargs):
    loco = LOCOImportance(X, y, score_func, model, cv=5)
    return importance_score(loco.get_importance(), true_importances, ranked=ranked)

score_functions = {
    "model_importance": model_importance,
    "mr_importance": mr_importance,
    #"cmr_importance": cmr_importance,
    #"loco_importance": loco_importance,
}

print("Datasets Generated...")

models = {"LASSO": Lasso, "FastSparse": FastSparseSklearn}
param_grids = {"LASSO": param_grid_lasso, "FastSparse": param_grid_fastsparse, "XGBoost": param_grid_xgb}
importance_attrs = {"LASSO": 'coef_', "FastSparse": 'coef_', "XGBoost": 'feature_importances_'}
n_iters= {"LASSO": 300, "FastSparse": 50, "XGBoost": 1000}

trimming_steps = {"LASSO": Lasso, "FastSparse": FastSparseSklearn,}

final_predictors = {"XGBoost": lambda: XGBRegressor()}

importance_testing(
    models, param_grids, datasets, true_importances, 
    score_functions=score_functions, importance_attrs=importance_attrs, 
    trimming_steps=trimming_steps, final_predictors=final_predictors,
    n_iters=n_iters, ranked=True, save_results=False
    )