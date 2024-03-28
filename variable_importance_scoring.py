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