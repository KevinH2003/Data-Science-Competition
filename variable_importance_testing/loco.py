# Modification of LOCO.py, Raymond Lin, 2024

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import cross_validate

def loco_to_df(loco_scores, feature_list):
    importance_df = pd.DataFrame()
    importance_df["feature"] = feature_list

    if loco_scores.ndim > 1:
        importance_df["importance_mean"] = loco_scores.mean(axis=1)
        importance_df["importance_std"] = loco_scores.std(axis=1)

        for val_score in range(loco_scores.shape[1]):
            importance_df["val_imp_{}".format(val_score)] = loco_scores[:, val_score]
    else:
        importance_df["importance_mean"] = loco_scores
    
    return importance_df.sort_values("importance_mean", ascending=False)

class LOCOImportance:
    """
    Leave One Covariate Out (LOCO) importance. Pass in 
    """
    def __init__(self, X, y, scoring, model, cv=5, loco_features=None, n_jobs=None):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.model = model
        self.cv = cv
        if loco_features is None:
            self.loco_features = X.columns.tolist()
        else:
            self.loco_features = loco_features
        self.n_jobs = n_jobs

    def _get_cv_score(self, removed=None):
        if removed is not None:
            X = self.X.drop(removed, axis=1)
        else:
            X = self.X

        if self.cv == 1:
            self.model.fit(X, self.y)
            return self.model.score(X, self.y) # regressor defaults to r2 scoring.
        else:
            cv_results = cross_validate(self.model, X, self.y, cv=self.cv, scoring=self.scoring)
            return np.average(cv_results['test_score'])

    def _get_cv_score_parallel(self, removed, result_queue):
        test_score = self._get_cv_score(removed=removed)
        result_queue.put((removed, test_score))
        return test_score
        

    def get_importance(self, verbose=False):
        base_cv_score = self._get_cv_score()
        if verbose:
            print(f'base score={np.mean(base_cv_score)} {base_cv_score}')

        if self.n_jobs is not None and self.n_jobs > 1:
            pass
        else:
            loco_cv_scores = []
            for f in tqdm(self.loco_features):
                loco_cv_scores.append(self._get_cv_score(f))
            loco_cv_scores_normalized = np.array([base_cv_score - loco_cv_score for loco_cv_score in loco_cv_scores])
        if verbose:
            print(loco_cv_scores_normalized)
        return loco_cv_scores_normalized
