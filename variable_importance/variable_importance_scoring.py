from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
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

def cross_validation_scores(cv, X, y, test_size=0.2, importance_attr='feature_importances_', true_importances=[], score_function=model_importance_score, verbose=False):
    '''
    Present an initialized cross-validator such as GridSearchCV or RandomizedSearchCV
    '''
    scores = {}
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    cv.fit(X_train, y_train)

    best_model = cv.best_estimator_
    scores['model'] = best_model
    scores['params'] = cv.best_params_

    # Calculate predictions for the training set and the test set
    best_model.fit(X_train, y_train)
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Training and test R^2 score
    scores['training_r2'] = r2_score(y_train, y_train_pred)
    scores['test_r2'] = r2_score(y_test, y_test_pred)

    scores['importance_score'] = score_function(best_model, true_importances, importance_attr)

    if verbose:
        print(f"Scores For {best_model.__class__}")
        print(f"Training R^2 Score: {scores['training_r2']}")
        print(f"Test R^2 Score: {scores['test_r2']}")
        print(f"Importance Score: {scores['importance_score']}")

    return scores