from sklearn.base import BaseEstimator, TransformerMixin


class GuaranteeType(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return X.astype(float)


    def transform(self, X, y=None):
        return X.astype(float)
    

    def fit_transform(self, X, y=None):
        return X.astype(float)
    

    def predict(self, X, y=None):
        return X.astype(float)
    
    
    def predict_proba(self, X, y=None):
        return X.astype(float)