from collections.abc import Iterable
import itertools
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .transformers import (
    InformationWeightTransformer,
    RemoveEffectsTransformer,
)
from vectorizers import TokenCooccurrenceVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import normalize


_INFO_WEIGHT_TRANSFORERS = {
    "default": {"class": InformationWeightTransformer, "kwds": {}},
}

_REMOVE_EFFECT_TRANSFORMERS = {
    "default": {"class": RemoveEffectsTransformer, "kwds": {}},
}

_COOCCURRENCE_VECTORIZERS = {
    "symmetric": {"class": TokenCooccurrenceVectorizer, "kwds": {}},
    "before": {
        "class": TokenCooccurrenceVectorizer,
        "kwds": {"window_orientation": "before"},
    },
    "after": {
        "class": TokenCooccurrenceVectorizer,
        "kwds": {"window_orientation": "after"},
    },
}


def flatten(list_of_seq):
    assert isinstance(list_of_seq, Iterable)
    if len(list_of_seq) == 0:
        return list_of_seq
    if type(list_of_seq[0]) in (list, tuple, np.ndarray):
        return tuple(itertools.chain.from_iterable(list_of_seq))
    else:
        return list_of_seq


def flatten_list(list_of_seq):
    assert isinstance(list_of_seq, Iterable)
    if len(list_of_seq) == 0:
        return list_of_seq
    if type(list_of_seq[0]) in (list, tuple, np.ndarray):
        return list(itertools.chain.from_iterable(list_of_seq))
    else:
        return list_of_seq


def create_processing_pipeline_stage(class_to_create, class_dict, kwds, class_type):
    if class_to_create is None:
        return None
    if class_to_create in class_dict:
        _class = class_dict[class_to_create]["class"]
        _kwds = class_dict[class_to_create]["kwds"]
        if kwds is not None:
            _kwds.update(kwds)
        result = _class(**_kwds)
    elif callable(class_to_create):
        if kwds is not None:
            result = class_to_create(**kwds)
        else:
            result = class_to_create()
    else:
        raise ValueError(
            f"Unrecognized {class_type} {class_to_create}; should "
            f"be one of {tuple(class_dict.keys())} or a {class_type} class."
        )
    return result


# Takes a sequence of tokens.
# Takes a list of paramters to be passed to TokenCooccurence iteratively
# info weight and remove effects
# hstack them all and return
class MultiTokenCooccurrenceVectorizer(BaseEstimator, TransformerMixin):
    """
    Takes a sequence of token sequences, applies a set of TokenCooccurence views
    of that data and returns the sparse concatination of all these views.

    MultiTokenCooccurenceVectorizer(['before', 'after'],
    """

    def __init__(
        self,
        vectorizer_list,
        vectorizer_kwds_list=None,
        vectorizer_name_list=None,
        info_weight_transformer="default",
        info_weight_transformer_kwds=None,
        remove_effects_transformer="default",
        remove_effects_transformer_kwds=None,
    ):
        self.vectorizer_list = vectorizer_list
        self.vectorizer_kwds_list = vectorizer_kwds_list
        self.vectorizer_name_list = vectorizer_name_list
        self.info_weight_transformer = info_weight_transformer
        self.info_weight_transformer_kwds = info_weight_transformer_kwds
        self.remove_effects_transformer = remove_effects_transformer
        self.remove_effects_transformer_kwds = remove_effects_transformer_kwds

    def fit(self, X):
        """

        Parameters
        ----------
        X: sequence of token sequences

        Returns
        -------

        """
        if self.vectorizer_name_list is None:
            self.vectorizer_names_list_ = np.arange(len(self.vectorizer_list)).astype(
                str
            )
        else:
            self.vectorizer_names_list_ = self.vectorizer_name_list

        if self.vectorizer_kwds_list is None:
            self.vectorizer_kwds_list_ = [None] * len(self.vectorizer_list)
        else:
            self.vectorizer_kwds_list_ = self.vectorizer_kwds_list

        self.info_weight_transformer_ = create_processing_pipeline_stage(
            self.info_weight_transformer,
            _INFO_WEIGHT_TRANSFORERS,
            self.info_weight_transformer_kwds,
            f"info weight transformer",
        )

        self.remove_effects_transformer_ = create_processing_pipeline_stage(
            self.remove_effects_transformer,
            _REMOVE_EFFECT_TRANSFORMERS,
            self.remove_effects_transformer_kwds,
            f"remove effects transformer",
        )

        # TODO: Check to make sure all matrices have the same number of rows and throw
        # informative error.

        self.column_dictionary_ = {}

        for i, vectorizer in enumerate(self.vectorizer_list):
            vectorizer_ = create_processing_pipeline_stage(
                vectorizer,
                _COOCCURRENCE_VECTORIZERS,
                self.vectorizer_kwds_list_[i],
                f"vectorizer {self.vectorizer_names_list_[i]}",
            )
            token_cooccurence = vectorizer_.fit_transform(X)

            if self.info_weight_transformer_ is not None:
                token_cooccurence = self.info_weight_transformer_.fit_transform(
                    token_cooccurence
                )
            token_cooccurence = normalize(token_cooccurence, norm="l1")
            if self.remove_effects_transformer_ is not None:
                token_cooccurence = self.remove_effects_transformer_.fit_transform(
                    token_cooccurence
                )

            if i == 0:
                self.vocabulary_ = list(vectorizer_.token_dictionary_.keys())
                self.token_dictionary_ = vectorizer_.token_dictionary_
                self.inverse_token_dictionary_ = vectorizer_.inverse_token_dictionary_
                self.vocabulary_size_ = len(vectorizer_.token_dictionary_)
                self.representation_ = token_cooccurence
            else:
                self.representation_ = hstack([self.representation_, token_cooccurence])

            column_dictionary_ = {
                (item[0] + i * self.vocabulary_size_): self.vectorizer_names_list_[i]
                + "_"
                + item[1]
                for item in vectorizer_.inverse_token_dictionary_.items()
            }
            self.column_dictionary_.update(column_dictionary_)

        self.inverse_column_dictionary_ = {
            item[1]: item[0] for item in self.column_dictionary_.items()
        }
        self.representation_ = self.representation_.tocsr()
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.representation_
