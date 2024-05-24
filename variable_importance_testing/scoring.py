from collections import deque
from datetime import datetime
import inspect
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import random
import time
import warnings

from variable_importance_testing.pipelining import VI_Pipeline, FeatureSelector

def importance_ranks(importances):
    """
    Rank variables by importance and return a list of relative rankings

    Parameters:
    importances (list-like): Variable importances, with the index 
        corresponding to the variable and importances[x] corresponding to x's importance

    Returns:
    list: A list of variables and their ranking, with index being the variable and 
        the value being its relative ranking (ranks[i] = x means i is the xth most important variable)
    """
    linked = [(i, importances[i]) for i in range(len(importances))]
    linked = sorted(linked, key=lambda x: -x[1])
    ranks = [(linked[i][0], i+1) for i in range(len(linked))]
    ranks = sorted(ranks, key=lambda x: x[0])
    ranks = [val[1] for val in ranks]

    return ranks

def rank_variables(importances):
    return sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

def top_n_score(true_importances, pred_importances, n=None):
    """
    Given n important variables in the ground truth (or the top n variables), 
    score how many important variables appear in the top n of the
    predicted importances. 

    Parameters:
    - true_importances (list-like): The ground truth importances.
    - pred_importances (list-like): The predicted importances.
    - n (int, optional): How many variables to consider at the top of the rankings.
        (defaults to the number of variables with nonzero importance in the ground truth)

    Returns:
    - float: The importance score, defined as (correct_vars / n) where correct_vars
        is the number of variables shared between the top n of the ground truth 
        and the top n of the predicted importances.

    """
    if n is None:
        n = sum([1 if x !=0 else 0 for x in true_importances])
    
    ranked_true = rank_variables(true_importances)[:n]
    ranked_pred = rank_variables(pred_importances)[:n]

    correct_vars = 0

    for variable in ranked_pred:
        if variable in ranked_true:
            correct_vars += 1

    return correct_vars / n

def importance_score(pred_importances, true_importances, 
                     score=top_n_score, scramble=False, num_scrambles=5, ranked=False):
    """
    Calculate the efficacy of a variable importance prediction against a ground truth.

    Parameters:
    - pred_importances (list-like): The predicted importances.
    - true_importances (list-like): The ground truth importances.
    - score (function, optional): A function to compute the correlation (default is spearmanr).
    - scramble (bool, optional): Whether to scramble non-important variables in the true importances. 
        This primarily applies when using spearmanr. (default is False)
    - num_scrambles (int, optional): The number of scrambles to perform (default is 5).
    - ranked (bool, optional): Whether to return a separate ranking of the variables by importance 
        If True, will return a tuple of (importance score, ranks). (default is False).

    Returns:
    - float: The importance score.
    - list (optional): The ranks of the predicted importances (if ranked is True).
    """
    
    # Copy and scale inputs
    scaler = MinMaxScaler()
    
    pred_importances = np.array(pred_importances).reshape(-1, 1)
    true_importances = np.array(true_importances).reshape(-1, 1)

    pred_importances = scaler.fit_transform(pred_importances).flatten()
    true_importances = scaler.fit_transform(true_importances).flatten()

    #Scramble non-important variables in true importances and get score
    if scramble:
        min_importance = min(true_importances)
        random_coeff = min(min_importance * -0.1, -0.00000001)

        scrambled_correlations = []
        for i in range(num_scrambles):
            scrambled_true_importances = [importance if importance != 0 else random_coeff * random.random() for importance in true_importances]
            try:
                scrambled_correlation, _ = score(true_importances, scrambled_true_importances)
            except Exception as e:
                print(e)
                scrambled_correlation = 1
            finally:
                scrambled_correlations.append(scrambled_correlation)

        #Get max of scrambled scores as hypothetical "Max" score
        scrambled_max = max(scrambled_correlations)
    
    # Calculate the correlation between true and predicted importances
    try:
        correlation = score(true_importances, pred_importances)
        if isinstance(correlation, tuple):
            correlation = correlation[0]
            
        if scramble:
            correlation = correlation / scrambled_max

        if ranked:
            # Rank the predicted importances
            ranks = importance_ranks(pred_importances)

    except Exception as e:
        print(e)
        correlation = 0
        ranks = []

    finally:
        if ranked:
            return correlation, ranks
        return correlation

def model_importance_score(model, true_importances, importance_attr=None, score=top_n_score, 
                           absolute_value=True, scramble=False, num_scrambles=5, ranked=False):
    """
    Calculate the importance score of a model's variable importance predictions against the ground truth.

    Parameters:
    - model (object): The model object containing variable importance attributes.
    - true_importances (array-like): The ground truth importances.
    - importance_attr (str, optional): The attribute name for the model's predicted importances. If None, attempts to infer.
    - score (function, optional): A function to compute the correlation (default is spearmanr).
    - absolute_value (bool, optional): Whether to take the absolute value of the predicted importances (default is True).
    - scramble (bool, optional): Whether to scramble non-important variables in the true importances
        (see importance_score scramble parameter) (default is True).
    - num_scrambles (int, optional): The number of scrambles to perform (default is 5).
    - ranked (bool, optional): Whether to return the ranked importances (default is False).

    Returns:
    - float: The importance score.
    - list (optional): The ranks of the predicted importances (if ranked is True).
    """
    
    # Attempt to infer importance attribute if None provided
    if importance_attr is None:
        if hasattr(model, 'feature_importances_'):
            importance_attr = 'feature_importances_'
        elif hasattr(model, 'coef_'):
            importance_attr = 'coef_'
        else:
            raise ValueError("Cannot infer importance attribute, model does not have feature_importances_ or coef_ attributes")

    # Get predicted importances and take absolute value if specified
    pred_importances = list(getattr(model, importance_attr))
    if absolute_value:
        pred_importances = [abs(x) for x in pred_importances]

    # Return importance score
    return importance_score(pred_importances, true_importances=true_importances, score=score, scramble=scramble, num_scrambles=num_scrambles, ranked=ranked)

def importance_scores(model, 
                      X, 
                      y, 
                      true_importances, 
                      test_size=0.3, 
                      importance_attr=None, 
                      score_functions=None, 
                      cross_validate=False, 
                      include_results=False, 
                      ranked=False, 
                      verbose=False):
    """
    Evaluate a model on a dataset and then, using the fitted model, 
    evaluate one or more metrics of variable importance on one or more ground truths.

    Parameters:
    - model (object): The model object to evaluate. 
        (Or a CV object such as GridSearchCV or RandomizedSearchCV if cross-validate is True)
    - X (array-like): The feature dataset.
    - y (array-like): The target variable.
    - true_importances (list or dict): The ground truth importances.
    - test_size (float, optional): The proportion of the dataset to include in the test split (default is 0.3).
    - importance_attr (str, optional): The attribute name for the model's predicted importances. 
        If None, will attempt to infer (see model_importance_score)
    - score_functions (dict or callable, optional): A dictionary of scoring functions 
        or a single scoring function (default is {"model importance": model_importance_score}).
    - cross_validate (bool, optional): Whether to perform cross-validation. 
        (If True, model must be a CV object) (default is False)
    - include_results (bool, optional): Whether to include cross-validation results. 
        For disambiguation, this refers solely to the .cv_results_ parameter of the CV object.
        The best model and parameters from the CV will always be returned if cross-validate is True.
        (cross-validate must be True for this to have any effect) (default is False).
    - ranked (bool, optional): Whether to return the ranked importances (default is False).
    - verbose (bool, optional): Whether to print verbose output (default is False).

    Returns:
    - dict: A dictionary containing the importance scores, R^2 scores, completion times, 
        and optional cross-validation results and importance rankings.
    """
    
    score_functions = score_functions if score_functions is not None else {"model_top_n": model_importance_score}
    
    # Handle if single true_importance or single score_function
    if isinstance(true_importances, list):
        true_importances = {"": true_importances}
    if callable(score_functions):
        score_functions = {"standard_scoring": score_functions}

    scores = {}
    scores["times"] = {}

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Cross validation
    if cross_validate:
        cv = model

        if verbose:
            print(f"Starting Cross-Validation...")

        failed = False
        start_time = time.time() # Start timer
        try:
            warnings.simplefilter("ignore", DeprecationWarning)
            cv.fit(X_train, y_train)
        except Exception as e:
            # Exit with None values upon exception so as to not disturb loop 
            # if running importance testing on multiple models/datasets

            print("something bad happened: " + str(e))
            scores['model'] = None
            scores['params'] = None
            scores['training_r2'] = None
            scores['test_r2'] = None
            for importance in true_importances:
                for func_name in score_functions:
                    scores[importance + "_" + func_name + "importance_score"] = None

            failed = True
        finally:
            if failed:
                return scores
        
        end_time = time.time()  # Stop timer and record CV time
        cv_elapsed_time = end_time - start_time
        scores['times']['cv'] = time.strftime("%H:%M:%S", time.gmtime(cv_elapsed_time))

        if verbose:
            print(f"Finished Cross-Validating in {scores['times']['cv']}")

        best_model = cv.best_estimator_
        scores['model'] = best_model
        scores['params'] = cv.best_params_
        if include_results:
            scores['cv_results'] = cv.cv_results_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    if cross_validate:
        scores['cv_r2'] = cv.best_score_

    # Calculate predictions for the training set and the test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Training and test R^2 score
    scores['training_r2'] = r2_score(y_train, y_train_pred)
    scores['test_r2'] = r2_score(y_test, y_test_pred)

    if ranked:
        scores["ranks"] = {}
        for importance in true_importances:
            scores["ranks"][importance] = {}

    # Perform every scoring function on every ground truth
    for func_name in score_functions:
        if verbose:
            print(f"Starting {func_name} Scoring...")
            
        score_start_time = time.time() # Start timer
        for importance in true_importances:

            # Filter the kwargs to only include those valid for the function
            kwargs = {
                "model":best_model, 
                "X": X_train, 
                "y": y_train, 
                "true_importances": true_importances[importance], 
                "importance_attr": importance_attr, 
                "ranked": ranked,
            }

            sig = inspect.signature(score_functions[func_name])
            valid_params = sig.parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

            #note: maybe change this so it can accept all true importances at once to save time
            if ranked:
                scores[importance + "_" + func_name + "_score"], scores["ranks"][importance][func_name] = score_functions[func_name](**filtered_kwargs)
            else:
                scores[importance + "_" + func_name + "_score"] = score_functions[func_name](**filtered_kwargs)
        
        score_end_time = time.time() # End timer and record time to score
        score_elapsed_time = score_end_time - score_start_time
        scores["times"][func_name] = time.strftime("%H:%M:%S", time.gmtime(score_elapsed_time))

        if verbose:
            print(f"Finished Scoring in {scores['times'][func_name]}")

    # Print results if verbose
    if verbose:
        print(f"Scores For {best_model.__class__}")
        print(f"Training R^2 Score: {scores['training_r2']}")
        if cross_validate:
            print(f"CV R^2 Score: {scores['cv_r2']}")
        print(f"Test R^2 Score: {scores['test_r2']}")

        for importance in true_importances:
            for func_name in score_functions:
                score = scores[importance + "_" + func_name + "_score"]
                print(f"{importance}_{func_name} Score: {score}")

    return scores


def importance_testing(models, 
                       param_grids, 
                       datasets, 
                       true_importances, 
                       score_functions=None, 
                       importance_attrs=None, 
                       trimming_steps=None, 
                       final_predictors=None, 
                       n_iters=None, 
                       num_folds=3,
                       ranked=False,
                       grid_search=False,  
                       save_results=True,
                       results_folder="importance_testing_results",
                       verbose=True
                       ):
    """
    A comprehensive testing loop over multiple models, datasets, and variable importance metrics

    Parameters:
    - models (dict-like): Dictionary of model names and their corresponding model objects.
    - param_grids (dict-like): Dictionary of model names and their 
        corresponding parameter grids for hyperparameter tuning.
    - datasets (dict-like): Dictionary of dataset names and their corresponding dataframes.
    - true_importances (dict-like): Dictionary of dataset names and their corresponding true importances.
    - score_functions (dict-like or callable, optional): Dictionary of scoring 
        functions or a single scoring function (default is {'model_importance': model_importance_score}).
    - importance_attrs (dict-like, optional): Dictionary of model names 
        and their corresponding importance attribute names.
    - trimming_steps (dict-like, optional): Dictionary of model names and their corresponding trimming steps.
    - final_predictors (dict-like, optional): Dictionary of final predictor names 
        and their corresponding model objects.
    - n_iters (dict-like or int, optional): Number of iterations for randomized search. 
        If an int, the same value is used for all models.
    - num_folds (int, optional): Number of folds for cross-validation (default is 3).
    - ranked (bool, optional): Whether to return ranked importances (default is False).
    - grid_search (bool, optional): Whether to use GridSearchCV instead of RandomizedSearchCV (default is False).
    - save_results (bool, optional): Whether to save the results to CSV files (default is True).
    - results_folder (str, optional): Folder to save the results (default is "importance_testing_results").
    - verbose (bool, optional): Whether to print verbose output (default is True).

    Returns:
    - dict: Aggregated scores for each dataset.
    """
     
    if verbose:
        print("Starting Importance Testing...")

    # Initialize default values for optional parameters
    if trimming_steps is None:
        trimming_steps = {}
    if final_predictors is None:
        final_predictors = {}
    if score_functions is None:
        score_functions = {'model_importance': model_importance_score}
    if importance_attrs is None:
        importance_attrs = {}
    if n_iters is None:
        n_iters = {}

    all_models = set()
    all_models.update(models.keys())
    all_models.update(trimming_steps.keys())
    all_models.update(final_predictors.keys())

    # If n_iters is an int, set all models to CV on the same number of iterations
    if isinstance(n_iters, int):
        n_iters = {model: n_iters for model in all_models}

    # Determine n_iters for each model and set importance_attrs to default None value
    for model in all_models:
        if model not in importance_attrs:
            importance_attrs[model] = None

        if model not in n_iters:
            total_iterations = 1
            for param_name, choices in param_grids[model].items():
                total_iterations *= len(choices)

            n_iters[model] = max(1, int(total_iterations * 0.1))

    models = models.copy()
    param_grids = param_grids.copy()

    model_names = []

    # Collect all model names and add trimming steps
    for model_name in models:
        model_names.append(model_name)

    for trimming_step in trimming_steps:
        if trimming_step not in model_names:
            model_names.append(trimming_step)
            models[trimming_step] = trimming_steps[trimming_step]

    # Prepare parameter grids, importance attributes, and n_iter fors final predictors (for pipelining)
    for predictor_name in final_predictors:
        new_grid = {}

        for param, values in param_grids[predictor_name].items():
            new_grid["prediction__" + param] = values

        param_grids[predictor_name + "_Pipeline"] = new_grid

        for trimming_name in trimming_steps:
            pipeline_name = trimming_name + " + " + predictor_name
            
            param_grids[pipeline_name] = param_grids[predictor_name + "_Pipeline"]
            importance_attrs[pipeline_name] = 'feature_importances_'
            n_iters[pipeline_name] = n_iters[predictor_name]

    print("Setup Done!")

    aggregated_scores = {}

    now = datetime.now() #Start time
    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    subfolder = results_folder + "/" + current_time

    # Create the folder if it doesn't exist
    if save_results:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

    print("Starting Testing...")

    for name, dataset in datasets.items():
        print(f"\n***###{name}:###***")
        if not isinstance(dataset, pd.DataFrame):
            dataset = pd.DataFrame(dataset)
            
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        model_scores = {}
        model_queue = deque([(model_name, models[model_name]()) for model_name in model_names])

        while len(model_queue) > 0:
            model_name, model = model_queue.popleft()

            print(f"\n***Scoring {model_name}...***")

            param_grid = param_grids[model_name]
            importance_attr = importance_attrs[model_name]

            # Perform CV
            if grid_search:
                cv = GridSearchCV(model, param_grid, cv=num_folds, scoring='r2', verbose=0, n_jobs=-1)
            else:
                cv = RandomizedSearchCV(model, param_grid, cv=num_folds, scoring='r2', verbose=0, n_iter=n_iters[model_name], n_jobs=-1)

            model_scores[model_name] = (importance_scores(cv, X, y, importance_attr=importance_attr, true_importances=true_importances[name], score_functions=score_functions, cross_validate=True, ranked=ranked, verbose=verbose))

            # Create new pipelines with cross-validated model to every final predictor if model in trimming
            if model_name in trimming_steps:
                if model_scores[model_name]["params"] is None:
                    continue

                for predictor_name in final_predictors:
                    new_selector = trimming_steps[model_name](**model_scores[model_name]["params"])
                    new_predictor = final_predictors[predictor_name]()

                    new_pipeline = VI_Pipeline(steps=[
                        ('feature_trimming', FeatureSelector(new_selector)),
                        ('prediction', new_predictor)
                    ], prediction_step=True, vi_step="prediction", vi_attr=importance_attrs[predictor_name])

                    pipeline_name = model_name + " + " + predictor_name
                    model_queue.append((pipeline_name, new_pipeline))

        # Save results to a csv in results folder
        if save_results:
            aggregated_scores[name] = pd.DataFrame(model_scores)

            filename = name + "_results_" + ".csv"
            file_path = os.path.join(subfolder, filename)

            aggregated_scores[name].to_csv(file_path, index=True)
            print(f"Results saved to {file_path}")

    print("All Done!")
    return aggregated_scores