from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import numpy as np


class FeatureSelector(SelectFromModel):
    def __init__(self, estimator, threshold=None, prefit=False, norm_order=1, max_features=None, importance_getter='auto'):
        super().__init__(estimator, threshold=threshold, prefit=prefit, norm_order=norm_order, max_features=max_features, importance_getter=importance_getter)

    def fit(self, X, y=None, **fit_params):
        super().fit(X=X, y=y, **fit_params)
        self.feature_names = X.columns
        self.feature_importances_ = self.get_support()
        return self
    
    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
    def transform(self, X):
        transformed_X = super().transform(X)
        if transformed_X.shape[1] == 0:
            if isinstance(X, np.ndarray):
                return X[:, [0]]
            elif hasattr(X, 'iloc'):  # Handling pandas DataFrame
                return X.iloc[:, [0]]
        return transformed_X
    
    def get_selected_features(self, feature_names=None):
        selected_features = self.get_support()
        if feature_names is None:
            feature_names = self.feature_names
        return [feature_names[i] for i, selected in enumerate(selected_features) if selected]

class VI_Pipeline(Pipeline):
    def __init__(self, steps, prediction_step=True, memory=None, verbose=False, vi_step="prediction", vi_attr="feature_importances_"):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.selection_steps = steps[:-1] if prediction_step else steps[:]
        self.prediction_step = prediction_step
        self.vi_step = vi_step
        self.vi_attr = vi_attr
        self.feature_importances_ = None

    def fit(self, X, y=None, **fit_params):
        self.features = X.columns
        super().fit(X, y, **fit_params)
        
        self.feature_importances_ = self.recover_features(X.columns)
        return self
        
    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y, fit_params=fit_params)
        return self.transform(X)
    
    def recover_features(self, all_features=None, selector_name="feature_trimming"):
        all_features = all_features if all_features is not None else self.features
        
        feature_selector = self.named_steps[selector_name]
        support_mask = feature_selector.get_support()
        
        full_importances = np.zeros(len(all_features))

        if support_mask.any():
            model_importances = getattr(self.named_steps[self.vi_step], self.vi_attr)
            full_importances[support_mask] = model_importances

        return full_importances