from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import random
import time
from variable_importance.cmr import CMR
from variable_importance.loco import LOCOImportance
from variable_importance.mr import MRImportance
import numpy as np
import shap
import warnings

def importance_score_estimator(estimator, X, y, true_importances=[], importance_attr='feature_importances_', score=spearmanr):
    warnings.filterwarnings("error")
    estimator.fit(X, y)

    try:
        pred_importances = getattr(estimator, importance_attr)
        correlation, _ = score(true_importances, pred_importances)
    except:
        correlation = 0
    finally:
        return correlation

def importance_score(pred_importances, true_importances=[], score=spearmanr, scramble=True, num_scrambles=5):
    warnings.filterwarnings("error")

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

        #Get average of scrambled scores as hypothetical "Max" score
        scrambled_average = sum(scrambled_correlations) / len(scrambled_correlations)
    
    try:
        correlation, _ = score(true_importances, pred_importances)

        if scramble:
            correlation = correlation / scrambled_average

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
        return correlation, ranks

def model_importance_score(model, true_importances, importance_attr, score=spearmanr, scramble=True, absolute_value=True):
    pred_importances = list(getattr(model, importance_attr))

    if absolute_value:
        pred_importances = [abs(x) for x in pred_importances]

    return importance_score(pred_importances, true_importances=true_importances, score=score, scramble=scramble)


def cross_validation_scores(cv, X, y, test_size=0.3, importance_attr='feature_importances_', true_importances={}, score_function=model_importance_score, verbose=False):
    '''
    Present an initialized cross-validator such as GridSearchCV or RandomizedSearchCV
    '''
    scores = {}
    names = X.columns
    ranks = {'features': names}
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

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
        #for name in score_functions:
        #    scores[name] = None

        failed = True
    finally:
        if failed:
            return scores
        
    end_time = time.time()  # Stop tracking time

    elapsed_time = end_time - start_time
    scores['cv_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    best_model = cv.best_estimator_
    scores['model'] = best_model
    scores['params'] = cv.best_params_
    scores['cv_results'] = cv.cv_results_

    # Calculate predictions for the training set and the test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Training and test R^2 score
    scores['training_r2'] = r2_score(y_train, y_train_pred)
    scores['cv_r2'] = cv.best_score_
    scores['test_r2'] = r2_score(y_test, y_test_pred)

    if isinstance(true_importances, list):
        scores['importance_score'] = score_function(best_model, true_importances, importance_attr)
    else:
        for importance in true_importances:
            scores[importance + "_importance_score"] = score_function(best_model, true_importances[importance], importance_attr)

    if verbose:
        print(f"Cross-Validation Completed in {scores['cv_time']}")
        print(f"Scores For {best_model.__class__}")
        print(f"Training R^2 Score: {scores['training_r2']}")
        print(f"CV R^2 Score: {scores['cv_r2']}")
        print(f"Test R^2 Score: {scores['test_r2']}")
        if isinstance(true_importances, list):
            print(f"Importance Score: {scores['importance_score']}")
        else:
            for importance in true_importances:
                score = scores[importance + "_importance_score"]
                print(f"{importance} Importance Score: {score}")

    return scores, ranks