# Data-Science-Competition

This package provides a class for generating synthetic data with ground truths as well as methods for testing models and variable importance metrics on given datasets.

Below are examples demonstrating how to use the provided tools.

## Example 1: Using importance_score and model_importance_score

```
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from variable_importance.scoring import importance_score, model_importance_score


# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 10)
true_importances = np.random.rand(10) #Not accurate, just for demonstration
model = Lasso(alpha=0.1).fit(X, np.random.rand(100))

# Calculate model importance score
pred_importances = model.coef_
importance_score(pred_importances, true_importances)

# Alternatively, use model_importance_score directly
model_importance_score(model, true_importances, importance_attr='coef_')
```

## Example 2: Using the DGP

```
import pandas as pd
from variable_importance.dgp import DataGenerator

# Initialize the DataGenerator with custom settings
dgp = DataGenerator(
    num_cols=10,
    num_rows=1000,
    num_important=2,
    correlation_scale=0.7,
    correlation_distribution='normal',
    importance_ranking='scaled'
)

# Generate a dataset
data = dgp.generate_data()

# Display the dataset
print(data.head())

print(dgp.importances)
print(dgp.bucket_importances)

```

## Example 3: Using importance_testing and DGP

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

dgps = {"Toy": DataGenerator(num_cols=100, num_rows=100, num_important=10)}
datasets = {name: dgp.generate_data() for name, dgp in dgps.items()}
true_importances = {name: dgps[name].bucket_importances for name in dgps.keys()}

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