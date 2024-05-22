import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from sklearn.utils import shuffle
import copy

# RID, Jon Donnelly
def perturb_divided(data, y, target_col):
    """Perturb column target_col of data using e_divide strategy

    Args:
        data (array): Data set
        target_col (int or list of ints): Index or indices of the column(s) to mess with
    Returns:
        data_perturbed: A copy of X with column target_col perturbed
    """
    n_samples = data.shape[0]

    # Create a copy of our given data
    data_copy = copy.deepcopy(data)
    y_copy = copy.deepcopy(y)
    # Shuffle along the first dimension to make sure
    # e_divide will be valid
    shuffle(data_copy, y_copy)

    # Grab our X and Y components
    x_copy = data_copy
    x_copy_src = copy.deepcopy(x_copy)

    if n_samples % 2 == 0:
        x_copy[:n_samples // 2, target_col] = x_copy_src[n_samples // 2:, target_col]
        x_copy[n_samples // 2:, target_col] = x_copy_src[:n_samples // 2, target_col]
    else:
        x_copy[n_samples // 2:, target_col] = x_copy_src[:n_samples // 2+1, target_col]
        x_copy[:n_samples // 2, target_col] = x_copy_src[n_samples // 2+1:, target_col]


    return x_copy, y_copy

class MRImportance:
    """
    Model Reliance (MR) permutation-based importance. Pass in 
    """
    def __init__(self, X, y, scoring, model, cv=5, loco_features=None, n_jobs=None):
        self.X = X.to_numpy() if not isinstance(X, np.ndarray) else X
        self.y = y
        self.scoring = scoring
        self.model = model
        self.cv = cv
        if loco_features is None:
            self.loco_features = X.columns.tolist()
        else:
            self.loco_features = loco_features
        self.n_jobs = n_jobs

    def _get_score(self, removed=None):
        if removed is not None:
            X, y = perturb_divided(self.X, self.y, removed)
        else:
            X = self.X
            y = self.y

        return get_scorer(self.scoring)(self.model, X, y) # regressor defaults to r2 scoring.

    def _get_score_parallel(self, removed, result_queue):
        test_score = self._get_score(removed=removed)
        result_queue.put((removed, test_score))
        return test_score
        

    def get_importance(self, verbose=False):
        base_cv_score = self._get_score()
        if verbose:
            print(f'base score={np.mean(base_cv_score)} {base_cv_score}')

        if self.n_jobs is not None and self.n_jobs > 1:
            pass
        else:
            mr_cv_scores = []
            for f in tqdm(self.loco_features):
                mr_cv_scores.append(self._get_score(f))
            mr_cv_scores_normalized = np.array([base_cv_score - mr_cv_score for mr_cv_score in mr_cv_scores])
        if verbose:
            print(mr_cv_scores_normalized)
        return mr_cv_scores_normalized
