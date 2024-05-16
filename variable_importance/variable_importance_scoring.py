from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import time
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

def importance_score(pred_importances, true_importances=[], score=spearmanr):
    warnings.filterwarnings("error")
    try:
        correlation, _ = score(true_importances, pred_importances)
    except Exception as e:
        print(e)
        correlation = 0
    finally:
        return correlation

def model_importance_score(model, true_importances, importance_attr, score=spearmanr):
    pred_importances = list(getattr(model, importance_attr))
    return importance_score(pred_importances, true_importances=true_importances, score=score)

def cross_validation_scores(cv, X, y, test_size=0.3, importance_attr='feature_importances_', true_importances={}, score_function=model_importance_score, verbose=False):
    '''
    Present an initialized cross-validator such as GridSearchCV or RandomizedSearchCV
    '''
    scores = {}
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
        scores['importance_score'] = None

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

    return scores