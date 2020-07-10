from vectorizers import NgramVectorizer, TokenCooccurrenceVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from .transformers import MultiTokenExpressionTransformer
from .utilities import (
    MultiTokenCooccurrenceVectorizer,
    create_processing_pipeline_stage,
    _INFO_WEIGHT_TRANSFORERS,
    _REMOVE_EFFECT_TRANSFORMERS,
    _COOCCURRENCE_VECTORIZERS,
    initialize_vocabulary,
    add_kwds,
    flatten,
)
from .tokenizers import (
    NLTKTokenizer,
    NLTKTweetTokenizer,
    SpacyTokenizer,
    StanzaTokenizer,
    SKLearnTokenizer,
)

import scipy.sparse as sparse
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.decomposition import TruncatedSVD

# TruncatedSVD or a variety of other algorithms should also work.
# TODO: should we wrap PLSA in a try and fall back on TruncatedSVD to remove the hard dependency?
from enstop import PLSA, EnsembleTopics

_DOCUMENT_TOKENIZERS = {
    "nltk": {"class": NLTKTokenizer, "kwds": {"tokenize_by": "document"}},
    "tweet": {"class": NLTKTweetTokenizer, "kwds": {"tokenize_by": "document"}},
    "spacy": {"class": SpacyTokenizer, "kwds": {"tokenize_by": "document"}},
    "stanza": {"class": StanzaTokenizer, "kwds": {"tokenize_by": "document"}},
    "sklearn": {"class": SKLearnTokenizer, "kwds": {"tokenize_by": "document"}},
}

_SENTENCE_TOKENIZERS = {
    "nltk": {"class": NLTKTokenizer, "kwds": {"tokenize_by": "sentence"}},
    "tweet": {"class": NLTKTweetTokenizer, "kwds": {"tokenize_by": "sentence"}},
    "spacy": {"class": SpacyTokenizer, "kwds": {"tokenize_by": "sentence"}},
    "stanza": {"class": StanzaTokenizer, "kwds": {"tokenize_by": "sentence"}},
    "sklearn": {"class": SKLearnTokenizer, "kwds": {"tokenize_by": "sentence"}},
}

_CONTRACTORS = {
    "aggressive": {
        "class": MultiTokenExpressionTransformer,
        "kwds": {"max_iterations": 6},
    },
    # "max_token_frequency": 1e-4
    "conservative": {
        "class": MultiTokenExpressionTransformer,
        "kwds": {"max_iterations": 2},
    },
}

_TOKEN_VECTORIZERS = {
    "bow": {"class": NgramVectorizer, "kwds": {"min_frequency": 1e-5},},
    "bigram": {
        "class": NgramVectorizer,
        "kwds": {"ngram_size": 2, "min_frequency": 1e-5},
    },
    "bow_words": {
        "class": NgramVectorizer,
        "kwds": {"min_frequency": 1e-5, "excluded_token_regex": "\W+"},
    },
    "bigram_words": {
        "class": NgramVectorizer,
        "kwds": {"ngram_size": 2, "min_frequency": 1e-5, "excluded_token_regex": "\W+"},
    },
}

# We need a few aggressive vocabulary pruning tokenizer defaults
# It's a bit unfortunate that they are buried deeply in this class.
# Maybe expose parameters at the top layer and push them down if they don't conflict.
_MULTITOKEN_COOCCURRENCE_VECTORIZERS = {
    "flat": {
        "class": MultiTokenCooccurrenceVectorizer,
        "kwds": {
            "vectorizer_list": ["before", "after"],
            "vectorizer_name_list": ["pre", "post"],
        },
    },
    "flat_1_5": {
        "class": MultiTokenCooccurrenceVectorizer,
        "kwds": {
            "vectorizer_list": ["before", "after", "before", "after"],
            "vectorizer_kwds_list": [
                {"window_radius": 1},
                {"window_radius": 1},
                {"window_radius": 5},
                {"window_radius": 5},
            ],
            "vectorizer_name_list": ["pre_1", "post_1", "pre_5", "post_5"],
        },
    },
}

############ DEBUG ###############
import time
def ts():
    return time.ctime(time.time())
##################################


class WordVectorizer(BaseEstimator, TransformerMixin):
    """
    Take a corpus of of documents and embed the words into a vector space in such a way that
    words used in similar contexts are close together.

    Parameters
    ----------
    tokenizer: string or callable (default='nltk')
        The method to be used to turn your sequence of documents into a sequence of sequences of tokens.
        If a string the options are ['nltk', 'tweet', 'spacy', 'stanza','sklearn']
        Set this to None if you have already tokenized your data.
    tokenizer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the tokenizer.
    token_contractor: string or callable (default='conservative')
        The method to be used to contract frequently co-occurring tokens into a single token.
        If a string the options are ['conservative', 'aggressive']
        Set this to None if you would prefer to not contract your tokens.
    token_contractor_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the contractor.
    vectorizer: string or callable (default flat)
        The method to be used to convert the list of lists of tokens into a fixed width numeric representation.
        If a string the options are ['before', 'after', 'symmetric', 'directional']
            before: TokenCooccurrenceVectorizer representing a word by counts of the words that occurred before it
                in your corpus.  The window width is the default of 5.
            after: TokenCooccurrenceVectorizer representing a word by counts of the words that occurred after it
                in your corpus.  The window width is the default of 5
            symmetric: TokenCooccurrenceVectorizer representing a word by counts of the words that occurred within a
                a window of a word.  The window width is the default of 5
            directional: TokenCooccurrenceVectorizer representing a word by counts of the words that occurred before
                and after each word treating each as different tokens.  The window width is the default of 5
        A useful class to create callables out of here would be TokenCooccurrenceVectorizer.
    vectorizer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the vectorizer.
    normalize: bool (default=True)
        Should the output be L1 normalized?
    dedupe_sentences: bool (default=True)
        Should you remove duplicate sentences.  Repeated sentences (such as signature blocks) often
        don't provide any extra linguistic information about word usage.
    token_dictionary: dict or sequence (default=None)
        A fixed dictionary mapping tokens to indices, or None if the dictionary should be learned from the training data.
        If a sequence of tokens is passed in we will treat it is a dictionary from the sequence to a sequence of one up
        numbers.
    """

    def __init__(
        self,
        tokenizer="nltk",
        tokenizer_kwds=None,
        token_contractor="conservative",
        token_contractor_kwds=None,
        vectorizer="directional",
        vectorizer_kwds=None,
        normalize=True,
        dedupe_sentences=True,
        token_dictionary=None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwds = tokenizer_kwds
        self.token_contractor = token_contractor
        self.token_contractor_kwds = token_contractor_kwds
        self.vectorizer = vectorizer
        self.vectorizer_kwds = vectorizer_kwds
        # Switches
        self.return_normalized = normalize
        self.dedupe_sentences = dedupe_sentences
        self.token_dictionary = token_dictionary

    def fit(self, X, y=None, **fit_params):
        """
        Learns a good representation of a word as appropriately weighted count of the the
        words that it co-occurs with.  This representation also takes into account if the
        word appears before or after our work.

        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.
        This is used when you have pre-tokenized your corpus.

        Returns
        ------
        self
        """
        # TOKENIZATION
        # use tokenizer to build list of the sentences in the corpus
        # Word vectorizers are document agnostic.
        self.tokenizer_ = create_processing_pipeline_stage(
            self.tokenizer, _SENTENCE_TOKENIZERS, self.tokenizer_kwds, "tokenizer"
        )
        if self.tokenizer_ is not None:
            tokens_by_sentence = self.tokenizer_.fit_transform(X)
        else:
            tokens_by_sentence = X

        # TOKEN CONTRACTOR
        # Takes a sequence of token sequences and contracts surprisingly frequent adjacent tokens
        # into single tokens.
        self.token_contractor_ = create_processing_pipeline_stage(
            self.token_contractor,
            _CONTRACTORS,
            self.token_contractor_kwds,
            "contractor",
        )
        if self.token_contractor_ is not None:
            tokens_by_sentence = self.token_contractor_.fit_transform(
                tokens_by_sentence
            )

        # DEDUPE
        # Remove duplicate sentences.  Repeated sentences (such as signature blocks) often
        # don't provide any extra linguistic information about word usage.
        if self.dedupe_sentences:
            tokens_by_sentence = tuple(set(tokens_by_sentence))

        # VECTORIZE
        # Convert from a sequence of sequences of tokens to a sequence of fixed width numeric
        # representation.
        self.token_label_dictionary_ = initialize_vocabulary(
            self.token_dictionary
        )  # returns none or a dict of tokens to indices
        self.vectorizer_kwds_ = add_kwds(
            self.vectorizer_kwds, "token_dictionary", self.token_label_dictionary_
        )
        self.vectorizer_ = create_processing_pipeline_stage(
            self.vectorizer,
            _COOCCURRENCE_VECTORIZERS,
            self.vectorizer_kwds_,
            "CooccurrenceVectorizer",
        )
        if self.vectorizer_ is not None:
            self.representation_ = self.vectorizer_.fit_transform(tokens_by_sentence)
        else:
            # This should only be the case where all the tokenizers are also set to None
            # and the user passed in a csr matrix.
            self.representation_ = tokens_by_sentence

        # NORMALIZE
        if self.return_normalized:
            self.representation_ = normalize(self.representation_, norm="l1", axis=1)

        # For ease of finding we promote the token dictionary to be a full class property.
        self.token_label_dictionary_ = self.vectorizer_.token_label_dictionary_
        self.token_index_dictionary_ = self.vectorizer_.token_index_dictionary_
        self.column_label_dictionary_ = self.vectorizer_.column_label_dictionary_
        self.column_index_dictionary_ = self.vectorizer_.column_index_dictionary_

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Learns a good representation of a word as appropriately weighted count of the the
        words that it co-occurs with.  This representation also takes into account if the
        word appears before or after our work.

        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.

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
        vocabulary_present = [w for w in words if w in self.token_label_dictionary_]
        indices = [self.token_label_dictionary_[word] for word in vocabulary_present]
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
            words = self.token_label_dictionary_
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
            columns=list(self.column_index_dictionary_.values()),
            index=vocab,
        )


class DocVectorizer(BaseEstimator, TransformerMixin):
    """
    Takes a corpus of of documents and embeds them into a vector space in such a way that
    documents that use similar words are close together.
    This is done via a sequence of steps some of which can omitted by setting that step to None
    in the constructor:
    1) tokenizer: Tokenization defaults to NLTK but can use stanza, spacy or a custom tokenizer.
        This cuts a sequence of strings (i.e. documents) into a sequence of sequences of tokens.
    2) token_contractor: contracts frequenctly co-occuring tokens together into a new single token.
        e.g. ('ice' 'cream') might be contracted into one token ('ice cream').
    2) vectorizer: Converts this sequence of tokens into counts of n-grams (default 1-grams).
    3) info_weight_transformer: Re-weights counts based on how informative the presence of an n-gram is within a document.
        This is a parameter for advanced users.  See documention for more details.
    4) remove_effects_transformer: Build a low rank model for how often we'd expect a completely random n-gram to occur your text
        and correct for this effect.
    5) Normalize

    Parameters
    ----------
    tokenizer: string or callable (default='nltk')
        The method to be used to turn your sequence of documents into a sequence of sequences of tokens.
        If a string the options are ['nltk', 'tweet', 'spacy', 'stanza','sklearn']
        Set this to None if you have already tokenized your data.
    tokenizer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the tokenizer.
    token_contractor: string or callable (default='conservative')
        The method to be used to contract frequently co-occurring tokens into a single token.
        If a string the options are ['conservative', 'aggressive']
        Set this to None if you would prefer to not contract your tokens.
    token_contractor_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the contractor.
    vectorizer: string or callable (default='bow')
        The method to be used to convert the list of lists of tokens into a fixed width
        numeric representation.
        If a string the options are ['bow', 'bigram', 'bow_words', 'bigram_words']
            bow: Standard bag of words model.  A document is the counts of of the words or tokens it contains
            bigram: Standard bigram model.  A document is the counts of adjacent pairs of words or tokens.
            _words: the two options with _words prefilter the token set to match the regular expression '\W+'
        If you would like to pass a callable I suggest using an instance of vectorizers.NgramVectorizer().
    vectorizer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the vectorizer.
    info_weight_transformer: string or callable (default='default')
        A transformer which re-weights the counts be their fisher information.
    info_weight_transformer_kwds: dict (optional, default=None)
            A dictionary of parameter names and values to pass to the info_weight_transformer.
    remove_effects_transformer: string or callable (optional, default=None)
        A transformer which removes the effect of a background text model.  This can be though of as adjusting
        for for the random document.
    remove_effects_transformer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the remove_effects_transformer.
    normalize: bool (default=True)
        Should the output be L1 normalized?
    fit_unique: bool (default=True)
        Should you remove duplicate documents before fitting.  The presence of large numbers of duplicate documents
        in a corpus can skew the models learned by info_weight_transformer and remove_effects_transformer.
    token_dictionary: dict or sequence (default=None)
        A fixed dictionary mapping tokens to indices, or None if the dictionary should be learned from the training data.
        If a sequence of tokens is passed in we will treat it is a dictionary from the sequence to a sequence of one up
        numbers.
    """

    def __init__(
        self,
        tokenizer="nltk",
        tokenizer_kwds=None,
        token_contractor="conservative",
        token_contractor_kwds=None,
        vectorizer="bow",
        vectorizer_kwds=None,
        info_weight_transformer="default",
        info_weight_transformer_kwds=None,
        remove_effects_transformer="default",
        remove_effects_transformer_kwds=None,
        normalize=True,
        fit_unique=False,
        token_dictionary=None,
    ):
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
        self.normalize = normalize
        self.fit_unique = fit_unique
        self.token_dictionary = token_dictionary

    def fit(self, X, y=None, **fit_params):
        """
        Learns the corrected information weight token count representation of a corpus.

        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.
        This is used when you have pre-tokenized your corpus.

        Returns
        -------
        self
        """
        # VALIDATE INPUT
        if self.vectorizer is None:
            raise ValueError(
                f"Sorry vectorizer must be a valid Vectorizer you passed in {self.vectorizer}"
            )
            # TODO: validate parameter strings

        if self.fit_unique:
            raise ValueError(
                f"Sorry fit_unique=True is not currently working.\n"
                f"Please unique your corpus yourself for now."
            )

        # UNIQUE
        # This needs to be a sequence of strings (documents)
        # or a sequence of token sequences (tokenized documents)
        # Fortunately both cases can be handled via ndarrays.
        X_ = np.array(X, copy=False)
        if self.fit_unique:
            _, self._indices, self._inverse = np.unique(
                X, return_index=True, return_inverse=True
            )
        else:
            self._indices = np.arange(len(X))
            self._inverse = self._indices

        # TOKENIZATION
        # use tokenizer to build list of the sentences in the corpus
        # Word vectorizers are document agnostic.
        self.tokenizer_ = create_processing_pipeline_stage(
            self.tokenizer, _DOCUMENT_TOKENIZERS, self.tokenizer_kwds, "tokenizer"
        )
        if self.tokenizer_ is not None:
            tokens_by_document = self.tokenizer_.fit_transform(X_[self._indices])
        else:
            tokens_by_document = X_[self._indices]

        # TOKEN CONTRACTOR
        self.token_contractor_ = create_processing_pipeline_stage(
            self.token_contractor,
            _CONTRACTORS,
            self.token_contractor_kwds,
            "contractor",
        )
        if self.token_contractor_ is not None:
            tokens_by_document = self.token_contractor_.fit_transform(
                tokens_by_document
            )

        # VECTORIZE
        self.token_label_dictionary_ = initialize_vocabulary(
            self.token_dictionary
        )  # returns none or a dict of tokens to indices
        self.vectorizer_kwds_ = add_kwds(
            self.vectorizer_kwds, "ngram_dictionary", self.token_label_dictionary_
        )
        self.vectorizer_ = create_processing_pipeline_stage(
            self.vectorizer,
            _TOKEN_VECTORIZERS,
            self.vectorizer_kwds_,
            "DocumentVectorizer",
        )
        self.representation_ = self.vectorizer_.fit_transform(tokens_by_document)

        # INFO WEIGHT TRANSFORMER
        self.info_weight_transformer_ = create_processing_pipeline_stage(
            self.info_weight_transformer,
            _INFO_WEIGHT_TRANSFORERS,
            self.info_weight_transformer_kwds,
            "InformationWeightTransformer",
        )
        if self.info_weight_transformer_:
            self.representation_ = self.info_weight_transformer_.fit_transform(
                self.representation_
            )
        # REMOVE EFFECTS TRANSFORMER
        self.remove_effects_transformer_ = create_processing_pipeline_stage(
            self.remove_effects_transformer,
            _REMOVE_EFFECT_TRANSFORMERS,
            self.remove_effects_transformer_kwds,
            "RemoveEffectsTransformer",
        )
        if self.remove_effects_transformer_:
            self.representation_ = self.remove_effects_transformer_.fit_transform(
                self.representation_
            )

        # NORMALIZE
        if self.normalize:
            self.representation_ = normalize(self.representation_, norm="l1", axis=1)

        # Undo any unique transform that was performed at the beginning of fit.
        self.representation_ = self.representation_[self._inverse]

        # For ease of finding we promote the token dictionary to be a full class property.
        self.column_label_dictionary_ = self.vectorizer_.column_label_dictionary_
        self.column_index_dictionary_ = self.vectorizer_.column_index_dictionary_
        self.token_label_dictionary_ = self.vectorizer_.column_label_dictionary_

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Learns the corrected information weight token count representation of a corpus.

        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.
        This is used when you have pre-tokenized your corpus.


        Returns
        -------
        sparse matrix
        The weighted counts of size (number_of_documents X number_of_n-grams) (or tokens)
        """
        self.fit(X)
        return self.representation_

    def transform(self, X):
        """
        Converts a sequence of documents into a pre-learned representation

        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.
        This is used when you have pre-tokenized your corpus.

        Returns
        -------
        sparse matrix
        The weighted counts of size (number_of_documents X number_of_n-grams) (or tokens)

        """
        check_is_fitted(self, ["vectorizer_"])
        # TOKENIZATION
        if self.tokenizer_ is not None:
            tokens_by_doc = self.tokenizer_.fit_transform(X)
        else:
            tokens_by_doc = X

        # TOKEN CONTRACTOR
        if self.token_contractor_ is not None:
            tokens_by_doc = self.token_contractor_.transform(tokens_by_doc)

        # VECTORIZE
        representation = self.vectorizer_.transform(tokens_by_doc)

        # INFO WEIGHT TRANSFORMER
        if self.info_weight_transformer_ is not None:
            representation = self.info_weight_transformer_.transform(representation)

        # REMOVE EFFECTS TRANSFORMER
        if self.remove_effects_transformer_ is not None:
            representation = self.remove_effects_transformer_.transform(representation)

        # NORMALIZE
        if self.normalize:
            representation = normalize(representation, norm="l1", axis=1)
        return representation

    def to_DataFrame(self, max_entries=10000, documents=None):
        """
        Converts the sparse matrix representation to a dense pandas DataFrame with
        one row per token and one column per n-gram.  This is typically a
        |document| x |vocabulary| sized DataFrame for a standard bag of words model.
        Parameters
        ----------
        max_entries: int (default=10000): The maximum number of entries in a dense version of your reprsentation
            This will error if you attempt to cast to large a sparse matrix to a DataFrame
        documents: list (optional, default=None)
            A sequence of document indices to return.
            Useful for looking at a small subset of your documents.
        WARNING: this is expensive for large amounts of data since it requires the storing of zeros.
        Returns
        -------
        pandas.DataFrame
        """
        if documents == None:
            documents = np.arange(self.representation_.shape[0])
        submatrix = self.representation_[documents, :]
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
                self.column_index_dictionary_[x] for x in np.arange(submatrix.shape[1])
            ],
            index=documents,
        )


#####################################################################
# Might cut the vectorizers.py module here and call this something else
#####################################################################

# Parameter dictionaries for FeatureBasisTransformer
_WORD_VECTORIZERS = {
    "default": {"class": WordVectorizer, "kwds": {"vectorizer": "directional"}},
    "tokenized": {
        "class": WordVectorizer,
        "kwds": {
            "tokenizer": None,
            "token_contractor": None,
            "vectorizer": "directional",
        },
    },
}

_TRANSFORMERS = {
    "tsvd": {"class": TruncatedSVD, "kwds": {}},
    "plsa": {"class": PLSA, "kwds": {}},
    "ensemble": {"class": EnsembleTopics, "kwds": {}},
}

_DOCUMENT_VECTORIZERS = {
    "default": {"class": DocVectorizer, "kwds": {}},
    "tokenized": {
        "class": DocVectorizer,
        "kwds": {"tokenizer": None, "token_contractor": None},
    },
}


class FeatureBasisConverter(BaseEstimator, TransformerMixin):
    """
    Applies a metric over your features and learns a change of basis that will approximate this distance.
    This is typically used to incorporate the knowledge contained in a word embedding into a document embedding.

    Parameters
    ----------
    word_vectorizer: string or callable (default flat)
    The method to be used to convert the list of lists of tokens into a fixed width numeric representation.
    If a string the options are ['before', 'after', 'symmetric', 'directional']
        before: TokenCooccurrenceVectorizer representing a word by counts of the words that occurred before it
            in your corpus.  The window width is the default of 5.
        after: TokenCooccurrenceVectorizer representing a word by counts of the words that occurred after it
            in your corpus.  The window width is the default of 5
        symmetric: TokenCooccurrenceVectorizer representing a word by counts of the words that occurred within a
            a window of a word.  The window width is the default of 5
        directional: TokenCooccurrenceVectorizer representing a word by counts of the words that occurred before
            and after each word treating each as different tokens.  The window width is the default of 5
    A useful class to create callables out of here would be TokenCooccurrenceVectorizer.
    word_vectorizer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the vectorizer.
    transformer: A transformer to learn a low dimensional representation of the basis learned by the word_vectorizer.
        This has the effect of collapsing similar words to reduce your vocabulary.
    transformer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the transformer.
    n_components: A parameter to pass to the transformer.  The number of basis vectors to represent the vocabulary.
        The transformer should take n_components as a parameter.
    token_dictionary: dict or sequence (default=None)
        A fixed dictionary mapping tokens to indices, or None if the dictionary should be learned from the training data.
        If a sequence of tokens is passed in we will treat it is a dictionary from the sequence to a sequence of one up
        numbers.
    """

    def __init__(
        self,
        word_vectorizer="tokenized",
        word_vectorizer_kwds=None,
        transformer="plsa",
        transformer_kwds=None,
        n_components=10,
        token_dictionary=None,
    ):

        self.word_vectorizer = word_vectorizer
        self.word_vectorizer_kwds = word_vectorizer_kwds
        self.transformer = transformer
        self.transformer_kwds = transformer_kwds
        self.n_components = n_components
        self.token_dictionary = token_dictionary

    def fit(self, X, y=None, **fit_params):
        """
        Applies a metric over your features and learns a change of basis that will approximate this distance.

        Parameters
        ----------
        X: a sequence of sequences of tokens
            This can be any input acceptable by the vectorizer.
            That is you can pass a sequence of documents with the word_vectorizer='default' instead of 'tokenized'
        Returns
        -------
        self
        """
        # Induce a similarity over your Features
        self.token_label_dictionary_ = initialize_vocabulary(
            self.token_dictionary
        )  # returns none or a dict of tokens to indices
        self.word_vectorizer_kwds_ = add_kwds(
            self.word_vectorizer_kwds, "token_dictionary", self.token_label_dictionary_
        )
        self.vectorizer_ = create_processing_pipeline_stage(
            self.word_vectorizer,
            _WORD_VECTORIZERS,
            self.word_vectorizer_kwds_,
            "WordVectorizer",
        )
        self.basis_transformer_ = self.vectorizer_.fit_transform(X)
        # Find a low dimensional (ideally linear) representation of this
        # for easy application to your features.

        # n_components as set in the init has precedence over any other.
        # This is a bit inelegant.  Suggestions?
        self.transformer_kwds_ = add_kwds(
            self.transformer_kwds, "n_components", self.n_components
        )
        self.transformer_ = create_processing_pipeline_stage(
            self.transformer, _TRANSFORMERS, self.transformer_kwds_, "Transformer"
        )
        # Your transformer must have a fit_transform, transform and n_components
        self.original_n_features = self.basis_transformer_.shape[1]
        if self.transformer_ is not None:
            if self.transformer_.n_components > self.original_n_features:
                raise ValueError(
                    f"Number of components must be less than or equal to the "
                    f"number of features;  Got {self.n_components} > {self.original_n_features}."
                )
            self.basis_transformer_ = self.transformer_.fit_transform(
                self.basis_transformer_
            )

        self.token_label_dictionary_ = self.vectorizer_.token_label_dictionary_
        self.token_index_dictionary_ = self.vectorizer_.token_index_dictionary_
        self.tokens_ = list(self.token_label_dictionary_.keys())
        self.vocabulary_ = self.token_label_dictionary_
        return self

    def change_basis(self, X, column_index_dictionary):
        """
        Applies a learned vocabulary change of basis to a new matrix.
        This is well defined if you are using a subset of the vocabulary learned by the basis converter.
        If you are using new tokens we currently throw an error to prevent you from unintentional errors.

        Parameters
        ----------
        X: scipy.sparse.matrix
            This matrix has a column space which matches the row space learned in .fit
        column_index_dictionary: dict
            This is a dictionary mapping from column indices in X to tokens representations.
            If there are columns that were not present in the .fit then they will be dropped.

        Returns
        -------
        self
        """
        # Align the columns of X with our transformer
        # It would likely be cheaper to permute the rows of our transformer rather than the
        # Columns of our data X.  It's not good practice though.
        # Copying their data is expensive.
        # Modifying the model or the data in a transform is wrong.

        X_ = check_array(X, accept_sparse="csr")
        if not isinstance(column_index_dictionary, dict):
            raise ValueError(
                f"column_index_dictionary should be a dictionary mapping from column indices to column labels."
                f"column_index_dictionary is currently a {type(column_index_dictionary)}"
            )
        first_key = next(iter(column_index_dictionary))
        if not isinstance(first_key, int):
            raise ValueError(
                f"column_index_dictionary should be a dictionary mapping from column indices to column labels."
                f"The first key of the dictionary is currently {type(first_key)}.\n"
                f"Did you accidentally pass a dictionary from labels to indices?"
            )

        # To guarantee sorted order
        column_names = [
            column_index_dictionary[row] for row in range(len(column_index_dictionary))
        ]
        difference = set(column_names).difference(self.tokens_)
        # Maybe in future we'll drop the unseen tokens with a warning.
        if len(difference) > 0:
            raise ValueError(
                f"Sorry your feature space contained tokens unseen by your FeatureBasisConverter."
                f"Unrecognized tokens: {difference}"
                f"In future releases we may simply drop these with a warning"
            )

        # Only select the rows from our basis_transformer that correspond to features in our data
        permutation = [self.token_label_dictionary_[x] for x in column_names]
        basis_transformer = self.basis_transformer_[permutation, :]

        return X_.dot(basis_transformer)


# n_basis_vectors should be set by a first class parameter
_FEATURE_BASIS_CONVERTERS = {
    "tokenized": {
        "class": FeatureBasisConverter,
        "kwds": {"word_vectorizer": "tokenized",},
    },
}


class JointWordDocVectorizer(BaseEstimator, TransformerMixin):
    """
    Takes a corpus of of documents and embeds both them and the words they contain into a vector space in such a way that
    documents that use similar words are close together, words that appear most relevent to
    groups of documents apear both closer together and close to the documents themselves.

    This is done via a sequence of steps some of which can omitted by setting that step to None
    in the constructor:
    1) tokenizer: Tokenization defaults to NLTK but can use stanza, spacy or a custom tokenizer.
        This cuts a sequence of strings (i.e. documents) into a sequence of sequences of tokens.
    2) token_contractor: contracts frequenctly co-occuring tokens together into a new single token.
        e.g. ('ice' 'cream') might be contracted into one token ('ice cream').
    2) vectorizer: Converts this sequence of tokens into counts of n-grams (default 1-grams).
    3) info_weight_transformer: Re-weights counts based on how informative the presence of an n-gram is within a document.
        This is a parameter for advanced users.  See documention for more details.
    4) remove_effects_transformer: Build a low rank model for how often we'd expect a completely random n-gram to occur your text
        and correct for this effect.
    5) Normalize

    Parameters
    ----------
    tokenizer: string or callable (default='nltk')
        The method to be used to turn your sequence of documents into a sequence of sequences of tokens.
        If a string the options are ['nltk', 'tweet', 'spacy', 'stanza','sklearn']
        Set this to None if you have already tokenized your data.
    tokenizer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the tokenizer.
    token_contractor: string or callable (default='conservative')
        The method to be used to contract frequently co-occurring tokens into a single token.
        If a string the options are ['conservative', 'aggressive']
        Set this to None if you would prefer to not contract your tokens.
    token_contractor_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the contractor.
    feature_basis_converter: string or callable (default='tokenized')
        A converter for inducing a distance measure over your feature space then changing the basis
        of your feature space via a linear transformation into a lower dimensional representation of that
        space which preserves much of the similarity structure amongst your features.
        If a string the options are ['tokenized']
            tokenized: Use a word embedding to induce a similarity between words and then pLSA
            to find a low dimensional multinomial representation of that vocabulary space.  The dimensionality
            of this multinomial can be set with the n_components parameter.
        If you would like to pass a callable I suggest using an instance of FeatureBasisConverter().
    feature_basis_converter_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the feature_basis_converter.
    word_cooccurrence_vectorizer: string or callable (default='symmetric')
        A word vectorizer which represents your words in the same feature space as your documents.
        By default this treats a word as a document of the sentences which contain that word.
        If a string the options are ['symmetric', 'before', 'after']
            before: Only words that occurred before a given word in a sentence are counted as tokens.
            after: Only words that occurred after a given word in a sentence are counted as tokens.
            symmetric: Both words the occur before and after a given word are considered tokens though they
            are distinct tokens.
        If you would like to pass a callable I suggest using instances of TokenCooccurrenceVectorizer().
    word_cooccurrence_vectorizer_kwds: dict (optional, default=None)
        A dictionary of parameter names and values to pass to the word_cooccurrence_vectorizer.
    doc_vectorizer: string or callable (default='tokenized')
        The method to be used to convert the list of lists of tokens into a fixed width
        numeric representation.
        If a string the options are ['tokenized', 'default']
            tokenized: Standard bag of words model with tokenizer and contractor set to None.
            default: Standard bag of words model.  Should only be used if you are doing no tokenization
            in JointWordDocVectorizers.
        If you would like to pass a callable I suggest using an instance of
        DocVectorizers(tokenizer=None, token_contractor=None).
    fit_unique: bool (default=True)
        Should you remove duplicate documents before fitting.  The presence of large numbers of duplicate documents
        in a corpus can skew the models learned by info_weight_transformer and remove_effects_transformer.
    token_dictionary: sequence or dict (default=None)
        The valid tokens to be included in your model.  If you would like a specific column order then you should
        pass a dict of tokens to desired column index.
    """

    def __init__(
        self,
        n_components=20,
        tokenizer="nltk",
        tokenizer_kwds=None,
        token_contractor="conservative",
        token_contractor_kwds=None,
        feature_basis_converter="tokenized",
        feature_basis_converter_kwds=None,
        word_cooccurrence_vectorizer="symmetric",
        word_cooccurrence_vectorizer_kwds=None,
        doc_vectorizer="tokenized",
        doc_vectorizer_kwds=None,
        fit_unique=False,
        exclude_token_regex=None,
        token_dictionary=None,
    ):
        self.n_components = n_components
        self.tokenizer = tokenizer
        self.tokenizer_kwds = tokenizer_kwds
        self.token_contractor = token_contractor
        self.token_contractor_kwds = token_contractor_kwds
        self.feature_basis_converter = feature_basis_converter
        self.feature_basis_converter_kwds = feature_basis_converter_kwds
        self.word_cooccurrence_vectorizer = word_cooccurrence_vectorizer
        self.word_cooccurrence_vectorizer_kwds = word_cooccurrence_vectorizer_kwds
        self.doc_vectorizer = doc_vectorizer
        self.doc_vectorizer_kwds = doc_vectorizer_kwds
        self.normalize = normalize
        self.fit_unique = fit_unique
        self.exclude_token_regex = exclude_token_regex
        self.token_dictionary = token_dictionary

    def fit(self, X):
        """
        Learns a joint word and document representation.
        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.
        This is used when you have pre-tokenized your corpus.

        Returns
        -------
        self
        """
        # TOKENIZATION
        # use tokenizer to build list of the sentences in the corpus
        # Force the tokenizer to tokenize into a sentence_by_document representation.
        self.tokenizer_kwds_ = add_kwds(
            self.tokenizer_kwds, "tokenize_by", "sentence_by_document"
        )
        self.tokenizer_ = create_processing_pipeline_stage(
            self.tokenizer, _DOCUMENT_TOKENIZERS, self.tokenizer_kwds_, "Tokenizer"
        )
        if self.tokenizer_ is not None:
            tokens_by_sentence_by_document = self.tokenizer_.fit_transform(X)
        else:
            tokens_by_sentence_by_document = X

        # TOKEN CONTRACTION
        self.token_contractor_ = create_processing_pipeline_stage(
            self.token_contractor,
            _CONTRACTORS,
            self.token_contractor_kwds,
            "Contractor",
        )
        # We need to learn the vocabulary on the tokens_by_sentence contracted.
        tokens_by_sentence = flatten(tokens_by_sentence_by_document)
        if self.token_contractor_ is not None:
            tokens_by_sentence = self.token_contractor_.fit_transform(
                tokens_by_sentence
            )
            tokens_by_sentence_by_document = [
                self.token_contractor_.transform(doc)
                for doc in tokens_by_sentence_by_document
            ]
        tokens_by_document = [flatten(doc) for doc in tokens_by_sentence_by_document]

        # tokens_by_sentence_by_document should be a document by sentence by tokens nested sequence.
        # This will be flattened in two different ways to save on work.
        # If you set tokenizer to be None then X should match this format.

        # WORD EMBEDDING (and change of basis transformer)
        self.token_label_dictionary_ = initialize_vocabulary(self.token_dictionary)
        self.feature_basis_converter_kwds_ = add_kwds(
            self.feature_basis_converter_kwds,
            "token_dictionary",
            self.token_label_dictionary_,
        )
        self.feature_basis_converter_kwds_ = add_kwds(
            self.feature_basis_converter_kwds_, "n_components", self.n_components
        )
        self.feature_basis_converter_ = create_processing_pipeline_stage(
            self.feature_basis_converter,
            _FEATURE_BASIS_CONVERTERS,
            self.feature_basis_converter_kwds_,
            "FeatureBasisTransformer",
        )
        if self.feature_basis_converter_ is not None:
            self.feature_basis_converter_.fit(tokens_by_sentence)

        # WORD COOCCURRENCE VECTORIZER
        # By default this essentially treats a word a document made up of all the sentences containing that word then
        # creates a bag of words representation of that document.
        self.word_cooccurrence_vectorizer_kwds_ = add_kwds(
            self.word_cooccurrence_vectorizer_kwds,
            "token_dictionary",
            self.token_label_dictionary_,
        )
        self.word_cooccurrence_vectorizer_ = create_processing_pipeline_stage(
            self.word_cooccurrence_vectorizer,
            _COOCCURRENCE_VECTORIZERS,
            self.word_cooccurrence_vectorizer_kwds_,
            "Word CooccurrenceVectorizer",
        )
        if self.word_cooccurrence_vectorizer_ is not None:
            self.representation_words_ = self.word_cooccurrence_vectorizer_.fit_transform(
                tokens_by_sentence
            )
            if self.feature_basis_converter_ is not None:
                self.representation_words_ = self.feature_basis_converter_.change_basis(
                    self.representation_words_,
                    self.word_cooccurrence_vectorizer_.column_index_dictionary_,
                )

        # DOCUMENT VECTORIZER
        # This should be the same column representation oas the word cooccurrence vectorizer above.
        # By default this is a bag of words representation.
        self.doc_vectorizer_kwds_ = add_kwds(
            self.doc_vectorizer_kwds, "fit_unique", self.fit_unique
        )
        self.doc_vectorizer_kwds_ = add_kwds(
            self.doc_vectorizer_kwds_, "token_dictionary", self.token_label_dictionary_
        )
        self.doc_vectorizer_ = create_processing_pipeline_stage(
            self.doc_vectorizer,
            _DOCUMENT_VECTORIZERS,
            self.doc_vectorizer_kwds_,
            "DocVectorizer",
        )
        if self.doc_vectorizer_ is not None:
            self.representation_docs_ = self.doc_vectorizer_.fit_transform(
                tokens_by_document
            )
            if self.feature_basis_converter_ is not None:
                self.representation_docs_ = self.feature_basis_converter_.change_basis(
                    self.representation_docs_,
                    self.doc_vectorizer_.column_index_dictionary_,
                )

        # Ensure column spaces of doc and word representations were the same.
        if (
            self.doc_vectorizer_.column_label_dictionary_
            != self.word_cooccurrence_vectorizer_.column_label_dictionary_
        ):
            raise ValueError(
                f"doc_vectorizer_.column_label_dictionary didn't match  word_cooccurence_vectorizer_.column_label_dictionary."
                f"These must be identicial."
                f"doc_vectorizer_.column_label_dictionary={self.doc_vectorizer_.column_label_dictionary}\n"
                f"word_cooccurence_vectorizer_.column_label_dictionary = {self.word_cooccurence_vectorizer_.column_label_dictionary}"
            )

        self.representation_docs_ = check_array(
            self.representation_docs_, accept_sparse="csr"
        )
        self.representation_words_ = check_array(
            self.representation_words_, accept_sparse="csr"
        )

        if isinstance(self.representation_docs_, sparse.csr.csr_matrix):
            vstack = sparse.vstack
        elif isinstance(self.representation_docs_, np.ndarray):
            vstack = np.vstack
        else:
            raise ValueError(
                f"Your representation must be a numpy array or sparse matrix;"
                f"representation_docs_ is of type {type(self.representation_docs_)}"
            )
        # Putting docs above words.  Docs are often referenced solely by their index while words have a better
        # token label.  This way docs keep their index.
        self.representation_ = vstack(
            [self.representation_docs_, self.representation_words_]
        )

        # Promote a bunch of dictionaries
        # Ugh, this will currently break horribly if you set lots of steps to None.
        # We can either prevent them from doing that for the feature_basis_transformer and doc_vectorizer or...
        # This is returned in the order that the tokens occur as rows in the represenation_words_
        self.token_label_dictionary_ = (
            self.doc_vectorizer_.token_label_dictionary_
            if self.doc_vectorizer_
            else None
        )
        self.n_words_ = (
            len(self.token_label_dictionary_) if self.token_label_dictionary_ else 0
        )
        self.n_documents_ = self.representation_docs_.shape[0]
        self.word_or_doc_ = ["word"] * self.n_words_ + ["doc"] * self.n_documents_
        self.doc_label_dictionary_ = {f"d_{i}": i for i in range(self.n_documents_)}
        self.doc_index_dictionary_ = {
            index: label for label, index in self.doc_label_dictionary_.items()
        }
        self.word_label_dictionary_ = {
            f"w_{self.word_cooccurrence_vectorizer_.column_index_dictionary_[i]}": i
            + self.n_documents_
            for i in range(self.n_words_)
        }
        self.word_index_dictionary_ = {
            index: label for label, index in self.word_label_dictionary_.items()
        }
        self.row_label_dictionary_ = {
            **self.doc_label_dictionary_,
            **self.word_label_dictionary_,
        }
        self.row_index_dictionary_ = {
            index: label for label, index in self.row_label_dictionary_.items()
        }

        return self

    def fit_transform(self, X):
        """
        Learns and returns a joint word and document representation.
        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.
        This is used when you have pre-tokenized your corpus.

        Returns
        -------
        sparse matrix
        with n_components columns.
        The first number of documents (.n_documents_) rows will be representations of the documents
        The last number of words (.n_words_) rows will be representations of the words.
        """
        self.fit(X)
        return self.representation_

    def transform(self, X):
        """
        Filters a corpus for a pre-learned vocabulary then embeds the documents with respect to this vocabulary
        and pre-learned change of basis.  Also embeds the subset of words from the existing vocabulary to
        new points in our representation.  Since these words likely have different co-occurrences in this
        new corpus they may move to different positions then the occupied in your original embedding.

        If you would like to hold your vocabulary fixed and just embed new documents please use
        .transform_documents() instead.
        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.
        This is used when you have pre-tokenized your corpus.

        Returns
        -------
        np.array with n_components columns
        (or scipy.sparse.csr_matrix with len(.vocabulary_) columns if feature_basis_converter=None)
        The first number of documents (.n_documents_) rows will be representations of the documents
        The last number of words (.n_words_) rows will be representations of the words.
        """
        check_is_fitted(self, ["doc_vectorizer_"])
        # TOKENIZATION
        if self.tokenizer_ is not None:
            tokens_by_sentence_by_document = self.tokenizer_.fit_transform(X)
        else:
            tokens_by_sentence_by_document = X

        # TOKEN CONTRACTION
        if self.token_contractor_ is not None:
            tokens_by_sentence_by_document = [
                self.token_contractor_.transform(doc)
                for doc in tokens_by_sentence_by_document
            ]
        tokens_by_sentence = flatten(tokens_by_sentence_by_document)
        tokens_by_document = tuple(
            [flatten(doc) for doc in tokens_by_sentence_by_document]
        )
        # This restricts to a fixed vocabulary
        if self.word_cooccurrence_vectorizer_ is not None:
            representation_words = self.word_cooccurrence_vectorizer_.transform(
                tokens_by_sentence
            )

        if self.doc_vectorizer_ is not None:
            representation_docs = self.doc_vectorizer_.transform(tokens_by_document)

        if self.feature_basis_converter_ is not None:
            representation_docs = self.feature_basis_converter_.change_basis(
                representation_docs, self.doc_vectorizer_.column_index_dictionary_,
            )
            representation_words = self.feature_basis_converter_.change_basis(
                representation_words,
                self.word_cooccurrence_vectorizer_.column_index_dictionary_,
            )

        representation_docs = check_array(representation_docs, accept_sparse="csr")
        representation_words = check_array(representation_words, accept_sparse="csr")
        if isinstance(representation_docs, sparse.csr.csr_matrix):
            vstack = sparse.vstack
        elif isinstance(representation_docs, np.ndarray):
            vstack = np.vstack
        else:
            raise ValueError(
                f"Your representation must be a numpy array or sparse matrix;"
                f"representation_docs is of type {type(representation_docs)}"
            )

        representation = vstack([representation_docs, representation_words])
        return representation

    def transform_document(self, X):
        """
        Vectorized a corpus of documents X into the joint space learned via fit while holding the vocabulary constant.
        Parameters
        ----------
        X: a sequence of strings or a sequence of sequences of tokens
        This is typically a list of documents making up a corpus.
        if tokenizer=None then this should be passed a sequence of sequences of tokens.
        This is used when you have pre-tokenized your corpus.

        Returns
        -------
        np.array with n_components columns
        (or scipy.sparse.csr_matrix with len(.vocabulary_) columns if feature_basis_converter=None)
        The first number of documents (.n_documents_) rows will be representations of the documents
        The last number of words (.n_words_) rows will be representations of the words.
        """
        check_is_fitted(self, ["doc_vectorizer_"])
        if self.tokenizer_ is not None:
            tokens_by_sentence_by_document = self.tokenizer_.fit_transform(X)
        else:
            tokens_by_sentence_by_document = X
        if self.token_contractor_ is not None:
            tokens_by_sentence_by_document = [
                self.token_contractor_.transform(doc)
                for doc in tokens_by_sentence_by_document
            ]
        tokens_by_document = tuple(
            [flatten(doc) for doc in tokens_by_sentence_by_document]
        )

        # This restricts vocabulary
        if self.doc_vectorizer_ is not None:
            representation_docs = self.doc_vectorizer_.transform(tokens_by_document)

        if self.feature_basis_converter_ is not None:
            representation_docs = self.feature_basis_converter_.change_basis(
                representation_docs, self.doc_vectorizer_.column_index_dictionary_,
            )

        return representation_docs
