from vectorizers import NgramVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from vectorizers._vectorizers import preprocess_token_sequences
from .tranformers import InformationWeightTransformer, RemoveEffectsTransformer
from .tokenizers import NLTKTokenizer


class WordVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # make sure we pass the before/after as a switch
        pass

    def fit(self):
        # use tokenizers to build list of lists
        # bigram contraction
        # co-occurence vectorizer
        # information transformer
        # em transformer
        pass

    pass


class DocVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tokenizer=NLTKTokenizer(),
        ngram_vectorizer=NgramVectorizer(ngram_size=1),
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
        tokens_by_doc = self.tokenizer.tokens_by_doc()
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
        tokens_by_doc = self.tokenizer.tokens_by_doc()
        token_counts = self.ngram_vectorizer.transform(tokens_by_doc)
        if self.info_weight_transformer is not None:
            token_counts = self.info_weight_transformer.transform(token_counts)
        if self.remove_effects_transformer is not None:
            token_counts = self.remove_effects_transformer.transform(token_counts)
        return token_counts

