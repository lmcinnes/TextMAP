import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from vectorizers import TokenCooccurrenceVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize


class WordMAP(BaseEstimator, TransformerMixin):
    def __init__(self, sent_tokenizer):
        pass

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X, y=None):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        pass
