from vectorizers import NgramVectorizer, TokenCooccurrenceVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from .transformers import (
    InformationWeightTransformer,
    RemoveEffectsTransformer,
    MultiTokenExpressionTransformer,
)
from .tokenizers import NLTKTokenizer, BaseTokenizer, NLTKTweetTokenizer, SpacyTokenizer
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np

# TODO: Should we wrap this in a try so we don't have a hard dependency?
# TruncatedSVD or a variety of other algorithms should also work.
from enstop import PLSA
from .utilities import flatten

_TOKENIZERS = {
    "nltk": {"class": NLTKTokenizer, "kwds": {}},
    "tweet": {"class": NLTKTweetTokenizer, "kwds": {}},
    "spacy": {"class": SpacyTokenizer, "kwds": {}},
}

_CONTRACTORS = {
    "aggressive": {
        "class": MultiTokenExpressionTransformer,
        "kwds": {"max_iterations": 6},
    },
    "conservative": {"class": MultiTokenExpressionTransformer, "kwds": {}},
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

_MULTITOKEN_COOCCURRENCE_VECTORIZERS = {
    "flat": {
        "class": MultiTokenExpressionTransformer,
        "kwds": {
            "vectorizer_list": ["before", "after"],
            "vectorizer_name_list": ["pre", "post"],
        },
    },
    "flat_1_5": {
        "class": MultiTokenExpressionTransformer,
        "kwds": {
            "vectorizer_list": ["before", "after", "before", "after"],
            "vectorizer_kwds_list": [
                {"window_args": (1,)},
                {"window_args": (1,)},
                {"window_args": (5,)},
                {"window_args": (5,)},
            ],
            "vectorizer_name_list": ["pre_1", "post_1", "pre_5", "post_5"],
        },
    },
}

_INFO_WEIGHT_TRANSFORERS = {
    "default": {"class": InformationWeightTransformer, "kwds": {}},
}

_REMOVE_EFFECT_TRANSFORMERS = {
    "defulat": {"class": RemoveEffectsTransformer, "kwds": {}},
}


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


class MultiTokenCooccurenceVectorizer(BaseEstimator, TransformerMixin):
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
            if self.remove_effects_transformer_ is not None:
                token_cooccurence = self.remove_effects_transformer.fit_transform(
                    token_cooccurence
                )

            if i == 0:
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
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.representation_


# Takes a sequence of tokens.
# Takes a list of paramters to be passed to TokenCooccurence iteratively
# info weight and remove effects
# hstack them all and return


class WordVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tokenizer="nltk",
        tokenizer_kwds=None,
        token_contractor="conservative",
        token_contractor_kwds=None,
        vectorizer="flat",
        vectorizer_kwds=None,
        info_weight_transformer="default",
        info_weight_transformer_kwds=None,
        remove_effects_transformer="default",
        remove_effects_transformer_kwds=None,
        normalize=True,
        ordered_cooccurrence=True,
        dedupe_sentences=True,
    ):
        # make sure we pass the before/after as a switch

        self.tokenizer = tokenizer
        self.tokenizer_kwds = tokenizer_kwds
        self.token_contractor = token_contractor
        self.token_contractor_kwds = token_contractor_kwds
        self.vectorizer = vectorizer
        self.vectorizer_kwds = vectorizer_kwds
        self.info_weight_transformer = info_weight_transformer
        self.info_weight_transformer_kwds = info_weight_transformer_kwds
        self.remove_effects_transformer = remove_effects_transformer
        self.remove_effects_transformer_kwds = remove_effects_transformer_kwds
        # Switches
        self.return_normalized = normalize
        self.ordered_cooccurence = ordered_cooccurrence
        self.dedupe_sentences = dedupe_sentences

    def fit(self, X, y=None, **fit_params):
        """
            Learns a good representation of a word as appropriately weighted count of the the
            words that it co-occurs with.  This representation also takes into account if the
            word appears before or after our work.

            Parameters
            ----------
            X = a sequence of strings
            This is typically a list of documents making up a corpus.

            Returns
            ------
            self
        """
        # TOKENIZATION
        # use tokenizer to build list of the sentences in the corpus
        # Word vectorizers are document agnostic.
        self.tokenizer_ = create_processing_pipeline_stage(
            self.tokenizer, _TOKENIZERS, self.tokenizer_kwds, "tokenizer"
        )
        if self.tokenizer_ is not None:
            tokens_by_sentence = self.tokenizer_.fit_transform(X)
        else:
            tokens_by_sentence = X

        # TOKEN CONTRACTOR
        self.token_contractor_ = create_processing_pipeline_stage(
            self.token_contractor,
            _CONTRACTORS,
            self.token_contractor_kwds,
            "contractor",
        )
        if self.token_contractor_ is not None:
            tokens_by_sentence = self.token_contractor_.fit_transform(tokens_by_sentence)

        # DEDUPE
        if self.dedupe_sentences:
            tokens_by_sentence = tuple(set(tokens_by_sentence))

        # VECTORIZE
        self.vectorizer_ = create_processing_pipeline_stage(
            self.vectorizer,
            _MULTITOKEN_COOCCURRENCE_VECTORIZERS,
            self.vectorizer_kwds,
            "MultiTokenCooccurrenceVectorizer",
        )
        self.representation_ = self.vectorizer_.fit_transform(tokens_by_sentence)

        # NORMALIZE
        if self.return_normalized:
            self.representation_ = normalize(self.representation_, norm="l1", axis=1)

        # For ease of finding we promote the token dictionary to be a full class property.
        self.token_dictonary_ = self.vectorizer_.token_dictionary_
        self.inverse_token_dictionary_ = self.vectorizer_.inverse_token_dictionary_
        self.column_dictionary_ = self.vectorizer_.column_dictionary_
        self.inverse_column_dictionary_ = self.vectorizer_.inverse_column_dictionary_
        self.vocabulary_ = self.vectorizer_.vocabulary_

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
            Learns a good representation of a word as appropriately weighted count of the the
            words that it co-occurs with.  This representation also takes into account if the
            word appears before or after our work.

            Parameters
            ----------
            X = a sequence of strings
            This is typically a list of documents making up a corpus.

            Returns
            -------
            sparse matrix
            of weighted counts of size number_of_tokens by vocabulary
            """
        self.fit(X)
        return self.representation_

    def lookup_words(self, words):
        """
        Query a model for the representations of a specific list of words.
        It ignores any words which are not contained in the model.
        Parameters
        ----------
        words=list, an iterable of the words to lookup within our model.

        Returns
        -------
        (vocabulary_present, scipy.sparse.matrix)
        A tuple with two elements.  The first is a list of the vocabulary in your words list that
        is also present in the model.
        The sparse matrix is the representations of those words
        """
        vocabulary_present = [w for w in words if w in self.vocabulary_]
        indices = [self.token_dictonary_[word] for word in vocabulary_present]
        return (vocabulary_present, self.representation_[indices, :])

    def to_DataFrame(self, max_entries=10000, words=None):
        """
        Converts the sparse matrix representation to a dense pandas DataFrame with
        one row per token and one column per token co-occurence.  This is either a
        vocabulary x vocabulary DataFrame or a vocabulary x 2*vocabulary DataFrame.
        Parameters
        ----------
        max_entries=int (10000): The maximum number of entries in a dense version of your reprsentation
            This will error if you attempt to cast to large a sparse matrix to a DataFrame
        words=iterable (None): An iterable of words to return.
            Useful for looking at a small subset of your rows.
        WARNING: this is expensive for large amounts of data since it requires the storing of zeros.
        Returns
        -------
        pandas.DataFrame
        """
        if words == None:
            words = self.vocabulary_
        vocab, submatrix = self.lookup_words(words)
        matrix_size = submatrix.shape[0] * submatrix.shape[1]
        if matrix_size > max_entries:
            return ValueError(
                f"Matrix size {matrix_size} > max_entries {max_entries}.  "
                f"Casting a sparse matrix to dense can consume large amounts of memory.  "
                f"Increase max_entries parameter in to_DataFrame() if you have enough ram "
                f"for this task. "
            )
        return pd.DataFrame(
            submatrix.todense(),
            columns=[
                self.column_dictionary_[x] for x in range(len(self.column_dictionary_))
            ],
            index=vocab,
        )


class DocVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tokenizer=NLTKTokenizer(),
        token_contractor=MultiTokenExpressionTransformer(),
        ngram_vectorizer=NgramVectorizer(),
        info_weight_transformer=InformationWeightTransformer(),
        remove_effects_transformer=RemoveEffectsTransformer(),
        dedupe_docs_for_fit=True,
    ):
        """
        A class for converting documents into a fixed width representation.  Useful for
        comparing documents with each other.
        This is done via:
        1) Tokenization defaults to NLTK but can use stanza, spacy or a custom tokenizer.
        2) Converts this sequence of tokens into counts of n-grams (default 1-grams).
        3) Re-weights counts based on how informative the presence of an n-gram is within a document.
        4) Build a low rank model for how often we'd expect a completely random n-gram to occur your text
            and correct for this effect.

        Parameters
        ----------
        tokenizer = textmap.tokenizers.BaseTokenizer (default NLTKTokenizer)
            Takes an instantiation of a class that inherits from BaseTokenizer.
            These are classes which take documents are parse them into individual tokens,
            then optionally contract frequently co-occuring tokens together into a single
            token.
            Examples of such tokenizers can be found in textmap.tokenizers and include:
            1) NLTKTokenizer
            2) NLTKTweetTokenizer
            3) SKLearnTokenizer
            4) StanzaTokenizer
            5) SpaCyTokenizer
            
        ngram_vectorizer = vectorizer.NgramVectorizer (default NgramVectorizer(ngram_size=1))
            Takes an instance of a class which turns sequences of sequences of tokens into
            fixed width representation through counting the occurence of n-grams.
            In the default case this simply counts the number of occurrences of each token.
            This class returns a documents by n-gram sparse matrix of counts.
            
        info_weight_transformer = textmap.transformers.InformationWeightTransformer (default InformationWeightTransformer())
            Takes an instance of a class which re-weights the counts in a sparse matrix.
            It does this by building a low rank model of the probability of a word being contained
            in any document, converting that into information by applying a log and scaling our
            counts by this value.
            If this is set to None this step is skipped in the pipeline.
            
        remove_effect_transformer = textmap.transformer.RemoveEffectsTranformer (default RemoveEffectsTransformer())
            Takes an instance of a class which builds a low rank model for how often we'd expect a completely random word to occur your text
            and correct for this effect.
            If this is set to None this step is skipped in the pipeline.
        """
        self.tokenizer = tokenizer
        self.ngram_vectorizer = ngram_vectorizer
        # These are more minor.  I'd be willing to default them to a string to clean
        # up the docstring help.
        self.info_weight_transformer = info_weight_transformer
        self.remove_effects_transformer = remove_effects_transformer
        self.dedupe_docs_for_fit = dedupe_docs_for_fit

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        if isinstance(value, BaseTokenizer):
            self._tokenizer = value
        else:
            raise TypeError(
                "Tokenizer is not an instance of textmap.tokenizers.BaseTokenizer. "
                "Did you forget to instantiate the tokenizer?"
            )

    def fit(self, X, y=None, **fit_params):
        """
        Learns the appropriately weighted n-gram representation of a corpus.

        Parameters
        ----------
        X = a sequence of strings
        This is typically a list of documents

        Returns
        -------
        self
        """
        self.tokenizer.fit(X)
        tokens_by_doc = self.tokenizer.tokenization_
        self.representation_ = self.ngram_vectorizer.fit_transform(tokens_by_doc)
        if self.info_weight_transformer is not None:
            self.representation_ = self.info_weight_transformer.fit_transform(
                self.representation_
            )
        if self.remove_effects_transformer is not None:
            self.representation_ = self.remove_effects_transformer.fit_transform(
                self.representation_
            )
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Learns the appropriately weighted n-gram representation of a corpus.

        Parameters
        ----------
        X = a sequence of strings
        This is typically a list of documents

        Returns
        -------
        sparse matrix
        of weighted counts of size number_of_sequences by number of n-grams (or tokens)
        """
        self.fit(X)
        return self.representation_

    def transform(self, X):
        """
        Converts a sequence of documents into a pre-learned re-weighted weighted matrix of
        n-gram counts.

        Parameters
        ----------
        X = a sequence of strings
        This is typically a list of documents

        Returns
        -------
        sparse matrix
        of weighted counts of size number_of_sequences by number of n-grams (or tokens)

        """
        self.tokenizer.fit(X)
        tokens_by_doc = self.tokenizer.tokenization_
        token_counts = self.ngram_vectorizer.transform(tokens_by_doc)
        if self.info_weight_transformer is not None:
            token_counts = self.info_weight_transformer.transform(token_counts)
        if self.remove_effects_transformer is not None:
            token_counts = self.remove_effects_transformer.transform(token_counts)
        return token_counts


class JointVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tokenizer=NLTKTokenizer(),
        token_contractor=MultiTokenExpressionTransformer(),
        ngram_vectorizer=NgramVectorizer(),
        info_weight_transformer=InformationWeightTransformer(),
        remove_effects_transformer=RemoveEffectsTransformer(),
        dedupe_docs_for_fit=True,
    ):
        """
        A class for converting documents into a fixed width representation.  Useful for
        comparing documents with each other.
        This is done via:
        1) Tokenization defaults to NLTK but can use stanza, spacy or a custom tokenizer.
        2) Converts this sequence of tokens into counts of n-grams (default 1-grams).
        3) Re-weights counts based on how informative the presence of an n-gram is within a document.
        4) Build a low rank model for how often we'd expect a completely random n-gram to occur your text
            and correct for this effect.

        Parameters
        ----------
        tokenizer = textmap.tokenizers.BaseTokenizer (default NLTKTokenizer)
            Takes an instantiation of a class that inherits from BaseTokenizer.
            These are classes which take documents are parse them into individual tokens,
            then optionally contract frequently co-occuring tokens together into a single
            token.
            Examples of such tokenizers can be found in textmap.tokenizers and include:
            1) NLTKTokenizer
            2) NLTKTweetTokenizer
            3) SKLearnTokenizer
            4) StanzaTokenizer
            5) SpaCyTokenizer

        ngram_vectorizer = vectorizer.NgramVectorizer (default NgramVectorizer(ngram_size=1))
            Takes an instance of a class which turns sequences of sequences of tokens into
            fixed width representation through counting the occurence of n-grams.
            In the default case this simply counts the number of occurrences of each token.
            This class returns a documents by n-gram sparse matrix of counts.

        info_weight_transformer = textmap.transformers.InformationWeightTransformer (default InformationWeightTransformer())
            Takes an instance of a class which re-weights the counts in a sparse matrix.
            It does this by building a low rank model of the probability of a word being contained
            in any document, converting that into information by applying a log and scaling our
            counts by this value.
            If this is set to None this step is skipped in the pipeline.

        remove_effect_transformer = textmap.transformer.RemoveEffectsTranformer (default RemoveEffectsTransformer())
            Takes an instance of a class which builds a low rank model for how often we'd expect a completely random word to occur your text
            and correct for this effect.
            If this is set to None this step is skipped in the pipeline.
        """

        self.tokenizer = tokenizer
        self.ngram_vectorizer = ngram_vectorizer
        # These are more minor.  I'd be willing to default them to a string to clean
        # up the docstring help.
        self.info_weight_transformer = info_weight_transformer
        self.remove_effects_transformer = remove_effects_transformer
        self.dedupe_docs_for_fit = dedupe_docs_for_fit
