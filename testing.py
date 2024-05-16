import numpy as np
import pandas as pd
import sklearn
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from warnings import simplefilter
from variable_importance.dgp import DataGenerator
from variable_importance.variable_importance_scoring import importance_score, cross_validation_scores
import os

RESULTS_FOLDER = "results_folder"
n_iter = 50

# Create the folder if it doesn't exist
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

print("Starting...")

dgps = {
    "Toy": DataGenerator(num_cols=100, num_rows=100, num_important=10, num_interaction_terms=0, effects='constant'),
}

datasets = {name: dgp.generate_data() for name, dgp in dgps.items()}

print("Datasets Generated...")

###Parameter Grids###
pipeline_param_grid = []

#pipeline = 

model_names = ["Pipeline"]
models = [Lasso()]
param_grids= [{'alpha': [1, 0.1, 0.01]}]
importance_attrs = ['coef_']

scoring_function_names = ['model_importance', 'shap_importance', 'cmr_importance', 'mr_importance', 'loco_importance']


aggregated_scores = {}


for name, dataset in datasets.items():
    print(f"{name}:")
    X = dataset.drop(["target"], axis=1)
    y = dataset["target"]

    model_scores = {}
    rank_scores = {}

    for i in range(len(models)):
        model = models[i]
        print(f"Scoring {model_names[i]}...")

        param_grid = param_grids[i]
        importance_attr = importance_attrs[i]

        rscv = RandomizedSearchCV(model, param_grid, scoring='r2', verbose=0, n_iter=n_iter, n_jobs=-1)
        model_scores[model_names[i]], rank_scores = cross_validation_scores(rscv, X, y, importance_attr=importance_attr, true_importances=dgps[name].importances, verbose=True, score_function_names=scoring_function_names)

        rank_scores_df = pd.DataFrame(rank_scores)
        rank_scores_df['average'] = rank_scores_df.loc[:, rank_scores_df.columns != 'features'].mean(axis=1)
        rank_scores_df = rank_scores_df.sort_values(by=['average'])

        filename = name + "_" + model_names[i] + "_ranks.csv"
        file_path = os.path.join(RESULTS_FOLDER, filename)
        
        rank_scores_df.to_csv(file_path, index=False)

    aggregated_scores[name] = pd.DataFrame(model_scores)

    filename = name + "_results.csv"
    file_path = os.path.join(RESULTS_FOLDER, filename)
    
    aggregated_scores[name].to_csv(file_path, index=True)
    print(f"Results saved to {filename}")

print("All Done!")