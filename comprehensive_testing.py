import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from warnings import simplefilter
from variable_importance.dgp import DataGenerator
from variable_importance.fastsparsewrap import FastSparseSklearn
from variable_importance.variable_importance_scoring import importance_score, cross_validation_scores
from variable_importance.pipelining import VI_Pipeline, FeatureSelector
from datetime import datetime
import os

PROTOTYPING = True
RESULTS_FOLDER = "results_folder"
num_folds = 3
nrows=None

if PROTOTYPING:
    num_folds = 3
    nrows= None

    model_names_testing = ["LASSO", "FastSparse"]

print("Starting...")

now = datetime.now()
current_time = now.strftime("%Y-%m-%d_%H:%M:%S")

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

###Parameter Grids###
param_grid_lasso = {
    'alpha': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100],  # A range from very small to large alpha values
    'max_iter': [1000, 2500, 5000, 10000, 25000, 500000, 1000000],  # Maximum number of iterations to converge
    'tol': [1e-4, 1e-3, 1e-2, 1e-1],  # Tolerance for the optimization
}

param_grid_fastsparse = {
    "max_support_size": [5, 10, 15, 20, 25],
    "atol": [1e-9, 1e-8, 1e-7, 1e-6],
    "lambda_0": [0.001, 0.005, 0.01, 0.05, 0.1],
    "penalty": ["L0"],
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
    'reg_alpha': [0, 0.1, 0.5, 1],  # L1 regularization term, larger values specify stronger regularization.
}

param_grid_xgb_pipeline = {
    'prediction__learning_rate': [0.01, 0.05, 0.1],  # Smaller values make the model more robust.
    'prediction__n_estimators': [100, 300, 500],  # More trees can be better, but at the risk of overfitting.
    'prediction__max_depth': [3, 5, 7],  # Depths greater than 10 might lead to overfitting.
    'prediction__min_child_weight': [1, 3, 5, 7],  # Controls over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    'prediction__gamma': [0.1, 0.2, 0.3],  # Larger values make the algorithm more conservative.
    'prediction__subsample': [0.8, 1.0],  # Values lower than 0.6 might lead to under-fitting.
    'prediction__colsample_bytree': [0.6, 0.8, 1.0],  # Considering a subset of features for each tree might make the model more robust.
    'prediction__reg_lambda': [1, 1.5, 2],  # L2 regularization term.
    'prediction__reg_alpha': [0, 0.1, 0.5, 1],  # L1 regularization term, larger values specify stronger regularization.
}

'''
large_input_df = pd.read_table('test_files/big_dataset/Large_benchmarking_rice.012', header=None, low_memory=False, nrows=nrows)
large_pheno_df = pd.read_table('test_files/big_dataset/Rice.Pheno.txt', header=None, nrows=nrows).drop(columns=0, axis=1).reset_index(drop=True)
large_test_SNP_metadata_df = pd.read_csv('test_files/big_dataset/Rice.Test.SNP.metadata.csv')

large_input_df['target'] = large_pheno_df.iloc[:, 0]
large_dataset_importances = large_test_SNP_metadata_df["EffectSize"]
'''

small_input_df = pd.read_table('test_files/small_dataset/Input.txt', header=None, low_memory=False, nrows=nrows)
small_pheno_df = pd.read_table('test_files/small_dataset/Pheno.txt', header=None, nrows=nrows).drop(columns=0, axis=1).reset_index(drop=True)
small_test_SNP_metadata_df = pd.read_csv('test_files/small_dataset/Test.SNP.metadata.csv')

small_input_df['target'] = small_pheno_df.iloc[:, 0]
small_dataset_importances = small_test_SNP_metadata_df["EffectSize"]

###DGPs###

dgps = {
    "Toy": DataGenerator(
        num_cols=100, num_rows=100, num_important=10, 
        num_interaction_terms=0, effects='linear', 
        noise_distribution='normal', noise_scale=0.1),
    "Slightly More Challening": DataGenerator(
        num_cols=100, num_rows=100, num_important=10, num_interaction_terms=20, effects='all', 
        correlation_scale=1.5, correlation_distribution='normal', 
        intercept=10, noise_distribution='normal', noise_scale=0.3),
    "High_Dimensionality": DataGenerator(
        num_cols=10000, num_rows=100, num_important=10, num_interaction_terms=20, effects='all', 
        correlation_scale=1, correlation_distribution='normal', 
        intercept=0, noise_distribution='normal', noise_scale=0.1),
    "High_Correlation": DataGenerator(
        num_cols=1000, num_rows=1000, num_important=10, num_interaction_terms=200, effects='all', 
        correlation_scale=0.95, correlation_distribution='uniform', 
        intercept=0, noise_distribution='normal', noise_scale=0.1),
    "High_Noise": DataGenerator(
        num_cols=1000, num_rows=1000, num_important=10, num_interaction_terms=50, effects='all', 
        correlation_scale=1, correlation_distribution='normal', 
        intercept=0, noise_distribution='uniform', noise_scale=0.5),
    "All Three": DataGenerator(
        num_cols=10000, num_rows=100, num_important=10, num_interaction_terms=200, effects='all', 
        correlation_scale=0.95, correlation_distribution='uniform', 
        intercept=0, noise_distribution='uniform', noise_scale=0.5),
}

if PROTOTYPING:
    dgps = {
    "Toy_With_Noise": DataGenerator(
        num_cols=100, num_rows=100, num_important=10, 
        num_interaction_terms=0, effects='linear', 
        noise_distribution='normal', noise_scale=0.5),
    }

datasets = {name: dgp.generate_data() for name, dgp in dgps.items()}
true_importances = {name: dgps[name].bucket_importances for name in dgps.keys()}

'''
if not PROTOTYPING:
    datasets["Small_Avalo"] = small_input_df
    #datasets["Large_Avalo"] = large_input_df

    true_importances["Small_Avalo"] = {"constant": small_dataset_importances}
    #true_importances["Large_Avalo"] = large_dataset_importances
'''
print("Datasets Generated...")

model_names = ["LASSO", "FastSparse"]
models = {"LASSO": Lasso(), "FastSparse": FastSparseSklearn(), "XGBoost": XGBRegressor()}
param_grids = {"LASSO": param_grid_lasso, "FastSparse": param_grid_fastsparse, "XGBoost": param_grid_xgb}
importance_attrs = {"LASSO": 'coef_', "FastSparse": 'coef_', "XGBoost": 'feature_importances_'}
n_iters= {"LASSO": 300, "FastSparse": 100, "XGBoost": 1000}

trimming_steps = {"LASSO": lambda params: Lasso(**params) if params is not None else Lasso(), "FastSparse": lambda params: FastSparseSklearn(**params) if params is not None else FastSparseSklearn()}
slow_predictors = {"XGBoost": lambda: XGBRegressor()}

if PROTOTYPING:
    model_names = model_names_testing

for predictor_name in slow_predictors:
    new_grid = {}

    for param, values in param_grids[predictor_name].items():
        new_grid["prediction__" + param] = values

    param_grids[predictor_name + "_Pipeline"] = new_grid

aggregated_scores = {}

subfolder = RESULTS_FOLDER + "/" + current_time

# Create the folder if it doesn't exist
if not os.path.exists(subfolder):
    os.makedirs(subfolder)


for name, dataset in datasets.items():
    print(f"\n***###{name}:###***")
    X = dataset.drop(["target"], axis=1)
    y = dataset["target"]

    model_scores = {}

    for model_name in model_names:
        model = models[model_name]
        print(f"\n***Scoring {model_name}...***")

        param_grid = param_grids[model_name]
        importance_attr = importance_attrs[model_name]

        rscv = RandomizedSearchCV(model, param_grid, cv=num_folds, scoring='r2', verbose=0, n_iter=n_iters[model_name], n_jobs=-1)
        
        model_scores[model_name] = (cross_validation_scores(rscv, X, y, importance_attr=importance_attr, true_importances=true_importances[name], verbose=True))

        if model_name in trimming_steps.keys():
            for predictor_name in slow_predictors.keys():
                new_selector = trimming_steps[model_name](model_scores[model_name]["params"])
                new_predictor = slow_predictors[predictor_name]()

                new_pipeline = VI_Pipeline(steps=[
                    ('feature_trimming', FeatureSelector(new_selector)),
                    ('prediction', new_predictor)
                ], prediction_step=True, vi_step="prediction", vi_attr=importance_attrs[predictor_name])

                pipeline_name = model_name + " + " + predictor_name

                model_names.append(pipeline_name)
                models[pipeline_name] = new_pipeline
                param_grids[pipeline_name] = param_grids[predictor_name + "_Pipeline"]
                importance_attrs[pipeline_name] = 'feature_importances_'
                n_iters[pipeline_name] = n_iters[predictor_name]
        
    aggregated_scores[name] = pd.DataFrame(model_scores)

    filename = name + "_results_" + ".csv"
    file_path = os.path.join(subfolder, filename)
    
    aggregated_scores[name].to_csv(file_path, index=True)
    print(f"Results saved to {filename}")

print("All Done!")