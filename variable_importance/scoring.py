from collections import deque
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import random
import time
import warnings

from variable_importance.pipelining import VI_Pipeline, FeatureSelector

def importance_score(pred_importances, true_importances=None, 
                     score=spearmanr, scramble=True, num_scrambles=5, ranked=False):
    true_importances = true_importances if true_importances is not None else []

    if scramble:
        min_importance = min(true_importances)
        random_coeff = min(min_importance * -0.1, -0.00000001)

        #Scramble non-important variables in true importances and get score
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
    
    try:
        correlation, _ = score(true_importances, pred_importances)
        if scramble:
            correlation = correlation / scrambled_max

        if ranked:
            pred_linked = [(i, pred_importances[i]) for i in range(len(pred_importances))]
            pred_linked = sorted(pred_linked, key=lambda x: -x[1])
            ranks = [(pred_linked[i][0], i+1) for i in range(len(pred_linked))]
            ranks = sorted(ranks, key=lambda x: x[0])
            ranks = [val[1] for val in ranks]

    except Exception as e:
        print(e)
        correlation = 0
        ranks = []

    finally:
        if ranked:
            return correlation, ranks
        return correlation

def model_importance_score(model, true_importances, importance_attr, score=spearmanr, absolute_value=True, scramble=True, num_scrambles=5, ranked=False):
    pred_importances = list(getattr(model, importance_attr))

    if absolute_value:
        pred_importances = [abs(x) for x in pred_importances]

    return importance_score(pred_importances, true_importances=true_importances, score=score, scramble=scramble, num_scrambles=num_scrambles, ranked=ranked)

def importance_scores(model, 
                      X, 
                      y, 
                      true_importances, 
                      test_size=0.3, 
                      importance_attr=None, 
                      score_functions=None, 
                      cross_validate=False, 
                      ranked=False, 
                      include_results=False, 
                      verbose=False):
    '''
    Present an initialized cross-validator such as GridSearchCV or RandomizedSearchCV
    '''
    score_functions = score_functions if score_functions is not None else {"model importance": model_importance_score}
    
    # Handle if single true_importance and score_function
    if isinstance(true_importances, list):
        true_importances = {"standard": true_importances}
    if callable(score_functions):
        score_functions = {"standard": score_functions}

    scores = {}
    scores["times"] = {}

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if cross_validate:
        cv = model

        if verbose:
            print(f"Starting Cross-Validation...")

        failed = False
        start_time = time.time()
        try:
            warnings.simplefilter("ignore", DeprecationWarning)
            cv.fit(X_train, y_train)
        except Exception as e:
            print("something bad happened: " + str(e))
            scores['model'] = None
            scores['params'] = None
            scores['training_r2'] = None
            scores['test_r2'] = None
            for importance in true_importances:
                for score_func in score_functions:
                    scores[importance + "_" + score_func + "importance_score"] = None

            failed = True
        finally:
            if failed:
                return scores
        
        end_time = time.time()  # Stop tracking time

        cv_elapsed_time = end_time - start_time

        scores['cv_time'] = time.strftime("%H:%M:%S", time.gmtime(cv_elapsed_time))
        scores['times']['cv'] = scores['cv_time']

        if verbose:
            print(f"Finished Cross-Validating in {scores['cv_time']}")

        best_model = cv.best_estimator_
        scores['model'] = best_model
        scores['params'] = cv.best_params_
        if include_results:
            scores['cv_results'] = cv.cv_results_
    else:
        best_model = model
        best_model.fit(X_train, y_train)

    # Calculate predictions for the training set and the test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    if cross_validate:
        scores['cv_r2'] = cv.best_score_

    # Training and test R^2 score
    scores['training_r2'] = r2_score(y_train, y_train_pred)
    scores['test_r2'] = r2_score(y_test, y_test_pred)

    if importance_attr is None:
        if hasattr(best_model, 'feature_importances_'):
            importance_attr = 'feature_importances_'
        elif hasattr(best_model, 'coef_'):
            importance_attr = 'coef_'

    if ranked:
        scores["ranks"] = {}

        for importance in true_importances:
            scores["ranks"][importance] = {}

    for score_func in score_functions:
        if verbose:
            print(f"Starting {score_func} Scoring...")
            
        score_start_time = time.time()
        for importance in true_importances:
            #change this so it can accept all true importances at once to save time
            if ranked:
                scores[importance + "_" + score_func + "importance_score"], scores["ranks"][importance][score_func] = score_functions[score_func](model=best_model, X=X_train, y=y_train, true_importances=true_importances[importance], importance_attr=importance_attr, ranked=ranked)
            else:
                scores[importance + "_" + score_func + "importance_score"] = score_functions[score_func](model=best_model, X=X_train, y=y_train, true_importances=true_importances[importance], importance_attr=importance_attr, ranked=ranked)
        score_end_time = time.time()
        score_elapsed_time = score_end_time - score_start_time

        scores["times"][score_func] = time.strftime("%H:%M:%S", time.gmtime(score_elapsed_time))
        if verbose:
            print(f"Finished Scoring in {scores['times'][score_func]}")

    if verbose:
        print(f"Scores For {best_model.__class__}")
        print(f"Training R^2 Score: {scores['training_r2']}")
        if cross_validate:
            print(f"CV R^2 Score: {scores['cv_r2']}")

        print(f"Test R^2 Score: {scores['test_r2']}")
        for importance in true_importances:
            for score_func in score_functions:
                score = scores[importance + "_" + score_func + "_score"]
                print(f"{importance}_{score_func} Score: {score}")

    return scores


def importance_testing(models: dict, 
                       param_grids: dict, 
                       datasets: dict, 
                       true_importances: dict, 
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
    if verbose:
        print("Starting Importance Testing...")

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

    if isinstance(n_iters, int):
        n_iters = {model: n_iters for model in all_models}

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

    for model_name in models:
        model_names.append(model_name)

    for trimming_step in trimming_steps:
        if trimming_step not in model_names:
            model_names.append(trimming_step)
    
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

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    subfolder = results_folder + "/" + current_time

    # Create the folder if it doesn't exist
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    print("Starting Testing...")

    for name, dataset in datasets.items():
        print(f"\n***###{name}:###***")
        X = dataset.drop(["target"], axis=1)
        y = dataset["target"]

        model_scores = {}
        model_queue = deque([(model_name, models[model_name]()) for model_name in model_names])

        while len(model_queue) > 0:
            model_name, model = model_queue.popleft()

            print(f"\n***Scoring {model_name}...***")

            param_grid = param_grids[model_name]
            importance_attr = importance_attrs[model_name]

            if grid_search:
                cv = GridSearchCV(model, param_grid, cv=num_folds, scoring='r2', verbose=0, n_jobs=-1)
            else:
                cv = RandomizedSearchCV(model, param_grid, cv=num_folds, scoring='r2', verbose=0, n_iter=n_iters[model_name], n_jobs=-1)

            model_scores[model_name] = (importance_scores(cv, X, y, importance_attr=importance_attr, true_importances=true_importances[name], score_functions=score_functions, cross_validate=True, ranked=ranked, verbose=verbose))

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

        if save_results:
            aggregated_scores[name] = pd.DataFrame(model_scores)

            filename = name + "_results_" + ".csv"
            file_path = os.path.join(subfolder, filename)

            aggregated_scores[name].to_csv(file_path, index=True)
            print(f"Results saved to {file_path}")

    print("All Done!")