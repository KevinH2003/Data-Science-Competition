from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from variable_importance.variable_importance_scoring import cross_validation_scores
from variable_importance.dgp import DataGenerator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'feature_selection__model__alpha': [0.01, 0.1, 1.0],  # Lasso alpha parameter
    'xgboost__learning_rate': [0.05, 0.1, 0.2],  # XGBoost learning rate
    'xgboost__n_estimators': [100, 200, 300],  # Number of trees in XGBoost
    'xgboost__max_depth': [3, 5, 7],  # Maximum depth of each tree in XGBoost
}

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, max_features=10, feature_importance_attr='feature_importances_'):
        self.model = model
        self.num_features = None
        self.max_features = max_features
        self.selector = SelectFromModel(model, max_features=max_features)
        self.feature_importances_ = None
        self.feature_importance_attr = feature_importance_attr
        self.selected_features_ = None

    def fit(self, X, y=None):
        self.num_features = len(X.columns)
        self.selector.fit(X, y)
        if hasattr(self.model, self.feature_importance_attr):
            self.feature_importances_ = getattr(self.model, self.feature_importance_attr)
        self.selected_features_ = self.selector.get_support()
        return self

    def transform(self, X):
        return self.selector.transform(X)
    
    def get_selected_features(self, feature_names=None,):
        if feature_names is None:
            feature_names = [i for i in range(self.num_features)]
        return [feature_names[i] for i, selected in enumerate(self.selected_features_) if selected]
    
class VI_Pipeline(Pipeline):
    def __init__(self, steps, vi_step="prediction", vi_attr="feature_importances_"):
        super().__init__(steps)
        self.vi_step = vi_step
        self.vi_attr = vi_attr
        self.feature_importances_ = None

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        self.feature_importances_= getattr(self.named_steps[self.vi_step], self.vi_attr)

lasso = Lasso(alpha=0.001)
xgboost = XGBRegressor()

lasso_selector = FeatureSelector(model=lasso)
pipeline = VI_Pipeline([
    ('feature_trimming', lasso_selector),
    ('xgboost', xgboost)
], vi_step="xgboost")

dgp = DataGenerator(num_cols=100, num_rows=100, num_important=10, num_interaction_terms=0, effects='constant')
dataset = dgp.generate_data()

X = dataset.drop(["target"], axis=1)
y = dataset["target"]

rscv = RandomizedSearchCV(pipeline, param_grid, scoring='r2', verbose=0, n_iter=1, n_jobs=-1)
# Now you can use this pipeline in your cross-validation function
#cross_validation_scores(rscv, X, y, importance_attr='feature_importances_', true_importances=dgp.importances, verbose=True)

lasso_selector.fit(X, y)

#m = rscv.best_estimator_
print(lasso_selector.get_selected_features([i for i in range(100)]))
#print(m.named_steps["xgboost"].feature_importances_)
