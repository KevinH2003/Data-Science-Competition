import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from warnings import simplefilter
from variable_importance.dgp import DataGenerator
from variable_importance.variable_importance_scoring import importance_score, cross_validation_scores
from variable_importance.pipelining import VI_Pipeline, FeatureSelector
import os

PROTOTYPING = True
RESULTS_FOLDER = "results_folder"
n_iter = 50
num_folds = 3

if PROTOTYPING:
    n_iter = 1
    num_folds = 3

# Create the folder if it doesn't exist
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

print("Starting...")

###Parameter Grids###
param_grid_lasso = {
    'alpha': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100],  # A range from very small to large alpha values
    'max_iter': [1000, 2500, 5000, 10000, 25000, 500000],  # Maximum number of iterations to converge
    'tol': [1e-4, 1e-3, 1e-2],  # Tolerance for the optimization
    'selection': ['random', 'cyclic'],
}

param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],  # Smaller values make the model more robust.
    'n_estimators': [100, 300, 500],  # More trees can be better, but at the risk of overfitting.
    'max_depth': [3, 5, 7],  # Depths greater than 10 might lead to overfitting.
    'min_child_weight': [1, 3, 5, 7],  # Controls over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    'gamma': [0.1, 0.2, 0.3],  # Larger values make the algorithm more conservative.
    'subsample': [0.8, 1.0],  # Values lower than 0.6 might lead to under-fitting.
    'colsample_bytree': [0.6, 0.8, 1.0],  # Considering a subset of features for each tree might make the model more robust.
    'reg_lambda': [1, 1.5, 2],  # L2 regularization term.
    'reg_alpha': [0, 0.1, 0.5],  # L1 regularization term, larger values specify stronger regularization.
}

param_grid_lasso_xgb = {
    'feature_trimming__estimator__alpha': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100],  # A range from very small to large alpha values
    'feature_trimming__estimator__max_iter': [1000, 2500, 5000, 10000, 25000, 500000],  # Maximum number of iterations to converge
    'feature_trimming__estimator__tol': [1e-4, 1e-3, 1e-2],  # Tolerance for the optimization
    'feature_trimming__estimator__selection': ['random', 'cyclic'],

    'prediction__learning_rate': [0.01, 0.05, 0.1],  # Smaller values make the model more robust.
    'prediction__n_estimators': [100, 300, 500],  # More trees can be better, but at the risk of overfitting.
    'prediction__max_depth': [3, 5, 7],  # Depths greater than 10 might lead to overfitting.
    'prediction__min_child_weight': [1, 3, 5, 7],  # Controls over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    'prediction__gamma': [0.1, 0.2, 0.3],  # Larger values make the algorithm more conservative.
    'prediction__subsample': [0.8, 1.0],  # Values lower than 0.6 might lead to under-fitting.
    'prediction__colsample_bytree': [0.6, 0.8, 1.0],  # Considering a subset of features for each tree might make the model more robust.
    'prediction__reg_lambda': [1, 1.5, 2],  # L2 regularization term.
    'prediction__reg_alpha': [0, 0.1, 0.5],  # L1 regularization term, larger values specify stronger regularization.
}


###DGPs###

dgps = {
    "Toy_With_Noise": DataGenerator(
        num_cols=100, num_rows=100, num_important=10, 
        num_interaction_terms=0, effects='constant', 
        noise_distribution='normal', noise_scale=1),
    "1000_Cols_Highly_Correlated": DataGenerator(
        num_cols=1000, num_rows=100, num_important=10, num_interaction_terms=50, effects='linear', 
        correlation_scale=1, correlation_distribution='normal', 
        intercept=0, noise_distribution='normal', noise_scale=1),
    "Corn_Mimic": DataGenerator(
        num_cols=50000, num_rows=160, num_important=10, num_interaction_terms=5000, effects='linear', 
        correlation_scale=1, correlation_distribution='normal', 
        intercept=10, noise_distribution='normal', noise_scale=1),
    "100_Cols_50_Interaction_Terms": DataGenerator(
        num_cols=100, num_rows=1000, num_important=10, num_interaction_terms=50, effects='all', 
        correlation_scale=0.9, correlation_distribution='uniform', 
        intercept=10, noise_distribution='normal', noise_scale=2)
}

if PROTOTYPING:
    dgps = {
        "Toy_With_Noise": DataGenerator(
            num_cols=10, num_rows=10, num_important=5, 
            num_interaction_terms=0, effects='linear', 
            noise_distribution='normal', noise_scale=1),
    }

datasets = {name: dgp.generate_data() for name, dgp in dgps.items()}

print("Datasets Generated...")

pipeline_lasso_xgb = VI_Pipeline(steps=[
    ('feature_trimming', FeatureSelector(Lasso())),
    ('prediction', XGBRegressor())
], prediction_step=True, vi_step="prediction")

model_names = ["LASSO", "XGBoost", "LASSO + XGBoost"]
models = [Lasso(), XGBRegressor(), pipeline_lasso_xgb]
param_grids= [param_grid_lasso, param_grid_xgb, param_grid_lasso_xgb]
importance_attrs = ['coef_', 'feature_importances_', 'feature_importances_']

aggregated_scores = {}

for name, dataset in datasets.items():
    print(f"{name}:")
    X = dataset.drop(["target"], axis=1)
    y = dataset["target"]

    model_scores = {}

    for i in range(len(models)):
        model = models[i]
        print(f"Scoring {model_names[i]}...")

        param_grid = param_grids[i]
        importance_attr = importance_attrs[i]

        rscv = RandomizedSearchCV(model, param_grid, cv=num_folds, scoring='r2', verbose=0, n_iter=n_iter, n_jobs=-1)
        model_scores[model_names[i]] = (cross_validation_scores(rscv, X, y, importance_attr=importance_attr, true_importances=dgps[name].importances, verbose=False))

    aggregated_scores[name] = pd.DataFrame(model_scores)

    filename = name + "_results.csv"
    file_path = os.path.join(RESULTS_FOLDER, filename)
    
    aggregated_scores[name].to_csv(file_path, index=True)
    print(f"Results saved to {filename}")

print("All Done!")