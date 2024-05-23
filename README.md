# Data-Science-Competition

Importance Testing Package - README
Welcome to the Importance Testing Package. This package is designed to assess the importance of variables in predictive models, using various importance metrics and cross-validation techniques.

Below are examples demonstrating how to use the provided functions.

## Example 1: Using importance_score and model_importance_score

```
import numpy as np
import pandas as pd
from sklearn.metrics import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from variable_importance.scoring import importance_score, model_importance_score


# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 10)
true_importances = np.random.rand(10)
model = Lasso(alpha=0.1).fit(X, np.random.rand(100))

# Calculate model importance score
pred_importances = model.coef_
importance_score(pred_importances, true_importances)

# Alternatively, use model_importance_score directly
model_importance_score(model, true_importances, importance_attr='coef_')
```

## Example 2: Using importance_testing and DGP

```
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from variable_importance.dgp import DataGenerator
from variable_importance.fastsparsewrap import FastSparseSklearn
from variable_importance.scoring import importance_testing

# Define parameter grids
param_grid_lasso = {
    'alpha': [0.1, 1, 10],
    'max_iter': [1000, 10000],
}

param_grid_fastsparse = {
    "max_support_size": [5, 10],
    "atol": [1e-8, 1e-7],
}

param_grid_xgb = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300],
}

# Generate synthetic datasets
datasets = {
    "Toy": DataGenerator(num_cols=100, num_rows=100, num_important=10).generate_data()
}

true_importances = {"Toy": datasets["Toy"].bucket_importances}

# Define models and parameters
models = {"Lasso": Lasso, "FastSparse": FastSparseSklearn}
param_grids = {"Lasso": param_grid_lasso, "FastSparse": param_grid_fastsparse, "XGBoost": param_grid_xgb}
importance_attrs = {"Lasso": 'coef_', "FastSparse": 'coef_', "XGBoost": 'feature_importances_'}

# Run importance testing
importance_testing(
    models=models,
    param_grids=param_grids,
    datasets=datasets,
    true_importances=true_importances,
    score_functions={"model_importance": model_importance_score},
    importance_attrs=importance_attrs,
    ranked=True,
    save_results=True,
    results_folder="importance_testing_results",
    verbose=True
)
```

## Function Overview
### importance_score
```
def importance_score(pred_importances, true_importances, 
                     score=spearmanr, scramble=True, num_scrambles=5, ranked=False):
    """
    Calculate the importance score by comparing predicted importances to true importances.
    
    Parameters:
    - pred_importances: List or array-like, predicted importances.
    - true_importances: List or array-like, true importances.
    - score: Function to compute the correlation (default: spearmanr).
    - scramble: Bool, whether to scramble non-important variables (default: True).
    - num_scrambles: Int, number of scrambles (default: 5).
    - ranked: Bool, whether to return ranks (default: False).
    
    Returns:
    - correlation: The calculated importance score.
    - ranks (optional): The ranks if `ranked=True`.
    """
```

### model_importance_score
```
def model_importance_score(model, true_importances, importance_attr=None, score=spearmanr, absolute_value=True, scramble=True, num_scrambles=5, ranked=False):
    """
    Calculate the importance score for a model.
    
    Parameters:
    - model: The predictive model.
    - true_importances: List or array-like, true importances.
    - importance_attr: String, attribute name for model importances (default: None).
    - score: Function to compute the correlation (default: spearmanr).
    - absolute_value: Bool, whether to use absolute values (default: True).
    - scramble: Bool, whether to scramble non-important variables (default: True).
    - num_scrambles: Int, number of scrambles (default: 5).
    - ranked: Bool, whether to return ranks (default: False).
    
    Returns:
    - correlation: The calculated importance score.
    - ranks (optional): The ranks if `ranked=True`.
    """
```

### importance_testing
```
def importance_testing(models, param_grids, datasets, true_importances, score_functions=None, importance_attrs=None, trimming_steps=None, final_predictors=None, n_iters=None, num_folds=3, ranked=False, grid_search=False, save_results=True, results_folder="importance_testing_results", verbose=True):
    """
    Perform importance testing on multiple models and datasets.
    
    Parameters:
    - models: Dict-like, models to test.
    - param_grids: Dict-like, parameter grids for models.
    - datasets: Dict-like, datasets to use for testing.
    - true_importances: Dict-like, true importances for datasets.
    - score_functions: Dict-like, scoring functions (default: None).
    - importance_attrs: Dict-like, importance attributes for models (default: None).
    - trimming_steps: Dict-like, trimming steps (default: None).
    - final_predictors: Dict-like, final predictors (default: None).
    - n_iters: Dict-like, number of iterations (default: None).
    - num_folds: Int, number of folds for cross-validation (default: 3).
    - ranked: Bool, whether to return ranks (default: False).
    - grid_search: Bool, whether to use grid search (default: False).
    - save_results: Bool, whether to save results (default: True).
    - results_folder: String, folder to save results (default: "importance_testing_results").
    - verbose: Bool, whether to print verbose output (default: True).
    
    Returns:
    - aggregated_scores: Dict-like, aggregated scores for each dataset.
    """
```