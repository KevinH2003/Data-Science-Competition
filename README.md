# Variable Importance

This package provides a class for generating synthetic data with ground truths as well as methods for testing models and variable importance metrics on given datasets.

Below are examples demonstrating how to use the provided tools.

## Scoring.py

### Example 1: Using importance_score and model_importance_score

```
import numpy as np
from sklearn.linear_model import Lasso
from variable_importance_testing.scoring import importance_score, model_importance_score

# Generate synthetic data
np.random.seed(42)
X = np.random.randint(2, size=(100, 5))
y = X[:, 0]  # Target is only 1 if the first variable is 1

true_importances = [1, 0, 0, 0, 0]
model = Lasso(alpha=0.1).fit(X, y)

# Calculate model importance score
pred_importances = model.coef_
print(importance_score(pred_importances, true_importances))

# Alternatively, use model_importance_score directly
# (the model_importance_score function will automatically try 'coef_'
# without needing to pass in an importance_attr, but for
# models with importance_attrs that aren't 'coef_' or 
# 'feature_importances_ you would need to pass it in')

print(model_importance_score(model, true_importances))
```

### Example 2: Using importance_scores
```
import numpy as np
from sklearn.linear_model import Lasso
from variable_importance_testing.scoring import importance_scores

# Generate synthetic data
np.random.seed(42)
X = np.random.randint(2, size=(100, 5))
y = X[:, 0]  # Target is only 1 if the first variable is 1

# Define true importances
true_importances = [1, 0, 0, 0, 0]

# Initialize the model
model = Lasso(alpha=0.1)

# Call the importance_scores function
results = importance_scores(model=model, 
                            X=X, 
                            y=y, 
                            true_importances=true_importances, 
                            test_size=0.3, 
                            verbose=True)

print(results)
```

### Example 3: Using importance_scores with cross-validate setting
```
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from scipy.stats import pearsonr
from sklearn.model_selection import RandomizedSearchCV
from variable_importance_testing.scoring import model_importance_score, importance_scores

# Generate synthetic data
np.random.seed(42)
X = np.random.randint(2, size=(100, 5))
y = X[:, 0]  # Target is only 1 if the first variable is 1

# Initialize the model and make a parameter grid for CV
model = Lasso()

param_grid = {
    'alpha': [0.1, 1, 10],
    'max_iter': [1000, 10000],
}

# Initialize CV object
cv = RandomizedSearchCV(model, param_grid, cv=5, scoring='r2', verbose=0, n_iter=5)

# Define a custom score function
def model_importance_pearsonr(model, true_importances, importance_attr, ranked, **kwargs):
    return model_importance_score(model, true_importances, importance_attr, score=pearsonr, ranked=ranked)
# score functions can currently only accept the following keyword arguments:
# model, X, y, true_importances, importance_attr, and ranked

# (they don't have to accept all of the above arguments, but they
# cannot accept arguments aside from the above unless they have default parameters)

# Passing in two score functions
score_functions = {
    "model_top_n": model_importance_score, 
    "model_pearson": model_importance_pearsonr
    }

# Define true importances
true_importances = {"true_importances": [1, 0, 0, 0, 0], "bad_importances": [0, 0, 0, 0, 1]}

# Call the importance_scores function
results = importance_scores(model=cv, 
                            X=X, 
                            y=y, 
                            true_importances=true_importances, 
                            test_size=0.3, 
                            score_functions=score_functions, 
                            cross_validate=True,
                            verbose=True)

print("\nRESULTS:")
for result in results:
    print(f"{result}: {results[result]}")
```

### Example 4: Using importance_testing
```
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from variable_importance_testing.dgp import DataGenerator
from variable_importance_testing.scoring import importance_testing

# Define parameter grids
param_grid_lasso = {
    'alpha': [0.1, 1, 10],
    'max_iter': [1000, 10000],
}

param_grid_xgb = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 300],
}

# Generate dummy datasets
np.random.seed(42)
X = np.random.randint(2, size=(100, 5))
y1 = np.expand_dims(X[:, 0].T, axis=1)
y2 = np.expand_dims(X[:, 1].T, axis=1)

# Two datasets, one with y = feature 1 and the other with y = feature 2
# The importance_testing function treats the last column as the target
datasets = {"dataset1": np.concatenate((X, y1), axis=1), "dataset2": np.concatenate((X, y2), axis=1)}
true_importances = {"dataset1": [1, 0, 0, 0, 0], "dataset2": [0, 1, 0, 0, 0]}

# Define models and parameters
models = {"Lasso": Lasso, "XGBoost": XGBRegressor}
param_grids = {"Lasso": param_grid_lasso, "XGBoost": param_grid_xgb}

# Make the CV do 3 iterations of RandomizedSearchCV for the LASSO model
# (It will automatically do 10% of the parameter space for the non-specified models)
n_iters = {"Lasso": 3}

# Importance attributes for each model 
# (technically not necessary for these two importance attributes)
importance_attrs = {"Lasso": 'coef_', "XGBoost": 'feature_importances_'}

# Run importance testing
importance_testing(
    models=models,
    param_grids=param_grids,
    datasets=datasets,
    true_importances=true_importances,
    importance_attrs=importance_attrs,
    save_results=False,
    verbose=True
)
```

### Example 5: Using importance_testing with automated pipelining
```
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from variable_importance_testing.dgp import DataGenerator
from variable_importance_testing.fastsparsewrap import FastSparseSklearn
from variable_importance_testing.scoring import importance_testing

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

param_grids = {"Lasso": param_grid_lasso, "FastSparse": param_grid_fastsparse, "XGBoost": param_grid_xgb}

# Generate dummy dataset
np.random.seed(42)
X = np.random.randint(2, size=(100, 5))
y = np.expand_dims(X[:, 0].T, axis=1)

# The importance_testing function treats the last column as the target
datasets = {"dataset": np.concatenate((X, y), axis=1)}
true_importances = {"dataset": [1, 0, 0, 0, 0], }

# Define models and parameters 
# if using trimming steps, you don't have to put predictors that are also trimming steps in models
# BUT, if you want any final predictors to be evaluated on their own you must put them in models

# the testing loop will automatically add lasso and fastsparse if they 
# aren't present in models, so including them in models doesn't change anything,
# but it won't test XGBoost by itself unless it's included in models
models = {"Lasso": Lasso, "FastSparse": FastSparseSklearn, "XGBoost": XGBRegressor} 

# Importance attributes for each model 
# (technically not necessary for these two importance attributes)
importance_attrs = {"Lasso": 'coef_', "FastSparse": 'coef_', "XGBoost": 'feature_importances_'}

# Make the CV do 4 iterations of RandomizedSearchCV for the XGBoost model
# (This number also applies to every pipeline made with XGBoost as the final predictor)
n_iters= {"XGBoost": 4}

# Define trimming steps and final predictors
trimming_steps = {"Lasso": Lasso, "FastSparse": FastSparseSklearn}
final_predictors = {"XGBoost": XGBRegressor,}

# Run importance testing
importance_testing(
    models=models,
    param_grids=param_grids,
    datasets=datasets,
    true_importances=true_importances,
    importance_attrs=importance_attrs,
    trimming_steps=trimming_steps,
    final_predictors=final_predictors,
    save_results=False,
    verbose=True
)
```
It's important to note that the two-step pipelines built in this testing loop use the already cross-validated trimming steps as the first step in the pipeline and only cross-validate the final prediction step when testing the pipelines.

Example: Lasso by itself is tested first in the loop. Once the cross-validation is finished, the loop takes the best parameters and constructs n "optimal" Lasso objects (where n is the number of final predictors). It then makes each new Lasso object the trimming step in a pipeline with a different final predictor and adds the pipeline to the testing queue with a param grid equivalent to the param grid passed in for the respective final predictor. 

## dgp.py

### Example 1: Basic Use

```
import pandas as pd
from variable_importance_testing.dgp import DataGenerator

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

### Example 2: Tweaking Parameters
```
import pandas as pd
from variable_importance_testing.dgp import DataGenerator

dgp = DataGenerator(
    num_cols=10,
    num_rows=1000,
    # to produce a 1000x10 dataset (not including the target)

    num_important=3, 
    # the first 3 features will affect the target

    frequencies={1: 1, 2: 0, 3: 0.5}, 
    # make feature 1 always be 1, feature 2 always be 0,
    # and feature 3 be 1 50% of the time

    effects={0: (lambda x: 100 if x == 1 else -100)}, 
    # make feature 0 add either 100 or -100 to the target
    # depending on its value

    num_interaction_terms=2, 
    correlation_scale=0.5, 
    correlation_distribution='normal', 
    # the last 2 features will each be correlated with
    # one of the important features, and the amount of
    # correlation will be chosen from a normal distribution
    # with mean of 0 and standard deviation equal to 0.5

    interactions={4: (3, -1)},
    # add feature 4 as a third interaction term
    # with perfect negative correlation with feature 3

    intercept=9000, 
    # shift the values of target up by 9000

    noise_distribution='normal',
    noise_scale=0.01
    # add noise chosen from a normal distribution with
    # mean of 0 and standard deviation equal to 0.01 * the maximum 
    # absolute value of the target (pre-noise and pre-intercept)
    # (in this case it'll be about 0.01* 100 because we set
    # the effect of 0 to be so large)
)

# Generate a dataset
data = dgp.generate_data()

print(data.head())
print(dgp.importances)

for feature in data.columns:
    if feature != 'target':
        print(f"Frequency of feature {feature}: {sum(data[feature])}")
```

## Using importance_testing and the DGP Together

### Example 1: 
```
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from variable_importance_testing.dgp import DataGenerator
from variable_importance_testing.scoring import importance_testing

# Define parameter grids
param_grid_lasso = {
    'alpha': [0.1, 1, 10],
    'max_iter': [1000, 10000],
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
models = {"Lasso": Lasso, "XGBoost": XGBRegressor}
param_grids = {"Lasso": param_grid_lasso,  "XGBoost": param_grid_xgb}
importance_attrs = {"Lasso": 'coef_', "XGBoost": 'feature_importances_'}

# Run importance testing
importance_testing(
    models=models,
    param_grids=param_grids,
    datasets=datasets,
    true_importances=true_importances,
    importance_attrs=importance_attrs,
    save_results=False,
    verbose=True
)
```

### Example 2: Real-World Example
```
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from variable_importance_testing.dgp import DataGenerator
from variable_importance_testing.fastsparsewrap import FastSparseSklearn
from variable_importance_testing.scoring import importance_score, model_importance_score, importance_testing
from variable_importance_testing.cmr import CMR
from variable_importance_testing.loco import LOCOImportance
from variable_importance_testing.mr import MRImportance

nrows = None
results_folder = None

print("Starting...")

###Parameter Grids###
param_grid_lasso = {
    'alpha': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100], 
    'max_iter': [1000, 2500, 5000, 10000, 25000, 500000, 1000000],  
    'tol': [1e-4, 1e-3, 1e-2, 1e-1], 
}

param_grid_fastsparse = {
    "max_support_size": [5, 10, 15],
    "atol": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
    "lambda_0": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
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

###DATA###

# Import outside dataset
small_input_df = pd.read_table('test_files/small_dataset/Input.txt', header=None, low_memory=False, nrows=nrows)
small_pheno_df = pd.read_table('test_files/small_dataset/Pheno.txt', header=None, nrows=nrows).drop(columns=0, axis=1).reset_index(drop=True)
small_test_SNP_metadata_df = pd.read_csv('test_files/small_dataset/Test.SNP.metadata.csv')

small_input_df['target'] = small_pheno_df.iloc[:, 0]
small_dataset_importances = small_test_SNP_metadata_df["EffectSize"]

# DGPs
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

# Generate Datasets 
datasets = {name: dgp.generate_data() for name, dgp in dgps.items()}
true_importances = {name: dgps[name].importances for name in dgps.keys()}

# Integrate outside data
datasets["Small_Real_World"] = small_input_df
true_importances["Small_Real_World"] = {"constant": small_dataset_importances}

print("Datasets Generated...")

# Scoring methods
def model_importance_spearmanr(model, true_importances, importance_attr, ranked=False, **kwargs):
    return model_importance_score(model, true_importances, importance_attr, score=spearmanr, scramble=True, ranked=ranked)

def model_importance_pearsonr(model, true_importances, importance_attr, ranked=False, **kwargs):
    return model_importance_score(model, true_importances, importance_attr, score=pearsonr, ranked=ranked)

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
    "model_importance_top_n": model_importance_score,
    "model_importance_spearmanr": model_importance_spearmanr,
    "model_importance_pearsonr": model_importance_pearsonr,
    "mr_importance": mr_importance,
    "cmr_importance": cmr_importance,
    "loco_importance": loco_importance,
}

# Set up testing loop
models = {"LASSO": Lasso, "FastSparse": FastSparseSklearn}
param_grids = {"LASSO": param_grid_lasso, "FastSparse": param_grid_fastsparse, "XGBoost": param_grid_xgb}
importance_attrs = {"LASSO": 'coef_', "FastSparse": 'coef_', "XGBoost": 'feature_importances_'}
n_iters= {"LASSO": 300, "FastSparse": 100, "XGBoost": 2000}

trimming_steps = {"LASSO": Lasso, "FastSparse": FastSparseSklearn,}
final_predictors = {"XGBoost": XGBRegressor,}

print("Parameters Initialized...")

importance_testing(
    models, param_grids, datasets, true_importances, 
    score_functions=score_functions, importance_attrs=importance_attrs, 
    trimming_steps=trimming_steps, final_predictors=final_predictors,
    n_iters=n_iters, ranked=True, 
    save_results=False,
    )
```
