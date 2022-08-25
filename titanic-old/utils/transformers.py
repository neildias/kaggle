import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CabinExtraction(BaseEstimator, TransformerMixin):
    """Extracts Cabin initials"""

    def __init__(self, string_from_index=0):
        self.index = string_from_index

    def fit(self, X_train:"dataframe", y_train=None):
        return self

    def transform(self, X):
        return X.Cabin.apply(self.retrieve_str_by_index_wApply).values

    def retrieve_str_by_index_wApply(self,x):
            '''
            If not np.nan, returns 0th element of the str
            '''
            if not isinstance(x, float):
                # hardcoded to extract 0th element
                return x[self.index]
            return "Missing"


class FamilyPresence(BaseEstimator, TransformerMixin):
    '''Returns binary of whether passenger was travelling with family'''

    def fit(self, X: "dataframe", y=None):
        return self

    def transform(self, X):
        return self.has_family(X)

    def family_count(self, x):
        return x['SibSp'] + x["Parch"]

    def has_family(self, x):
        return np.where(self.family_count(x) != 0, True, False)


class CustomLabelEncoder(TransformerMixin, BaseEstimator):
    '''Similar to skleran's label encoder'''

    def fit(self, X: 'series', y=None):
        categories = X.value_counts().index.tolist()
        # get mappings
        self.dict_mapping = {}
        for index, cats in enumerate(categories):
            # these mappings will be used for transformation of test set
            self.dict_mapping[cats] = index

        # mapping for an unknown category
        self.dict_mapping["unknown"] = index + 1
        return self

    def transform(self, X: 'series'):
        X = X.copy()

        # test set for transformation based on the training set
        # change all cats not in categories to unknown
        X = X.apply(lambda x: x if x in self.dict_mapping else "unknown")
        return X.map(self.dict_mapping)