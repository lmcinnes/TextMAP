from vectorizers import NgramVectorizer, TokenCooccurrenceVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from .tranformers import InformationWeightTransformer, RemoveEffectsTransformer
from .tokenizers import NLTKTokenizer, BaseTokenizer
from scipy.sparse import hstack
from sklearn.preprocessing import normalize
import pandas as pd
from .utilities import flatten


class WordVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tokenizer=NLTKTokenizer(tokenize_by="sentence"),
        vectorizer=TokenCooccurrenceVectorizer(),
        info_weight_transformer=InformationWeightTransformer(),
        remove_effects_transformer=RemoveEffectsTransformer(),
        normalize=True,
        ordered_cooccurrence=True,
    ):
        # make sure we pass the before/after as a switch
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.info_weight_transformer = info_weight_transformer
        self.remove_effects_transformer = remove_effects_transformer
        self.return_normalized = normalize
        self.ordered_cooccurence = ordered_cooccurrence

    def fit(self, X, y=None, **fit_params):
        # use tokenizer to build list of the sentences in the corpus
        # Word vectorizers are document agnostic.
        tokens_by_sentence = flatten(self.tokenizer.fit_transform(X))
        # Sparse matrix of tokens occuring BEFORE our token in the sequence
        # This requires TokenCooccurrenceVectorize to have symmetrize = False
        # TODO: Should we check for that?
        token_cooccurence = self.vectorizer.fit_transform(tokens_by_sentence)
        tokens_after = token_cooccurence.copy()
        if self.info_weight_transformer is not None:
            tokens_after = self.info_weight_transformer.fit_transform(tokens_after)
        if self.remove_effects_transformer is not None:
            tokens_after = self.remove_effects_transformer.fit_transform(tokens_after)

        # Sparse matrix of tokens occuring AFTER our token in the sequence
        # Don't need a copy here
        tokens_before = token_cooccurence.T
        if self.info_weight_transformer is not None:
            tokens_before = self.info_weight_transformer.fit_transform(tokens_before)
        if self.remove_effects_transformer is not None:
            tokens_before = self.remove_effects_transformer.fit_transform(tokens_before)

        # Take a word to be to concatination of it's before and after co-occurences.
        self.vocabulary_size_ = len(self.vectorizer.token_dictionary_)
        # Python>3.6 guarantees key order.
        self.vocabulary_ = self.vectorizer.token_dictionary_.keys()
        if self.ordered_cooccurence:
            self.representation_ = hstack([tokens_before, tokens_after])
            self.column_dictionary_ = {
                item[0]: "pre_" + item[1]
                for item in self.vectorizer.inverse_token_dictionary_.items()
            }
            self.column_dictionary_.update(
                {
                    item[0] + self.vocabulary_size_: "post_" + item[1]
                    for item in self.vectorizer.inverse_token_dictionary_.items()
                }
            )
            self.inverse_column_dictionary_ = {
                item[1]: [0] for item in self.column_dictionary_.items()
            }
        else:
            # Add the two together
            self.representation_ = tokens_before + tokens_after
            self.column_dictionary_ = self.vectorizer.token_dictionary_
            self.inverse_column_dictionary_ = self.vectorizer.inverse_token_dictionary_

        if self.return_normalized:
            self.representation_ = normalize(self.representation_, norm="l1", axis=1)

        # For ease of finding we promote the token dictionary to be a full class property.
        self.token_dictonary_ = self.vectorizer.token_dictionary_
        self.inverse_token_dictionary_ = self.vectorizer.inverse_token_dictionary_

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
                self.inverse_column_dictionary_[x]
                for x in range(len(self.column_dictionary_))
            ],
            index=vocab,
        )


class DocVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tokenizer=NLTKTokenizer(),
        ngram_vectorizer=NgramVectorizer(),
        info_weight_transformer=InformationWeightTransformer(),
        remove_effects_transformer=RemoveEffectsTransformer(),
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
