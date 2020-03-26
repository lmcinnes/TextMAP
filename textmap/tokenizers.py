from .utilities import flatten
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import CountVectorizer

import spacy.lang.en
import stanza


class BaseTokenizer(BaseEstimator, TransformerMixin):
    """
    Base class for all textmap tokenizers inherit from.
    
      Parameters
      ----------
      collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
        The function to score bigrams

      max_collocation_iterations = int (default = 2)
        The maximal number of recursive bigram contractions

      min_collocation_score: int (default = 12)
        The minimal PMI value to contract a bigram per iteration

    """

    def __init__(
        self,
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):
        self.collocation_score_function = collocation_score_function
        self.max_collocation_iterations = max_collocation_iterations
        self.min_collocation_score = min_collocation_score

    def fit(self, X, **fit_params):
        """

        Parameters
        ----------
        X :  collection
            The collection of documents

        Returns
        -------
        self
        """
        self.tokens_by_sent_by_doc_ = None
        return self

    def fit_transform(self, X, **fit_params):
        """

        Parameters
        ----------
        X : collection
            The collection of documents

        Returns
        -------
        The list of lists of tokenized sentences per document
        """
        self.fit(X, **fit_params)
        return self.tokens_by_sent_by_doc_

    def tokens_by_sent_by_doc(self):
        """

        Returns
        -------
        The list of lists of tokenized sentences per document
        """
        return self.tokens_by_sent_by_doc_

    def tokens_by_sent(self):
        """

        Returns
        -------
        The complete list of tokenized sentences
        """
        return flatten(self.tokens_by_sent_by_doc())

    def tokens_by_doc(self):
        """

        Returns
        -------
        The list of tokenized documents
        """

        return [flatten(doc) for doc in self.tokens_by_sent_by_doc()]

    def iteratively_contract_bigrams(self):
        """
        Procedure to iteratively contract bigrams (up to max_collocation_iterations times)
        that score higher on the collocation_function than the min_collocation_score
        """
        for i in range(self.max_collocation_iterations):
            bigramer = BigramCollocationFinder.from_documents(self.tokens_by_sent())
            mwes = list(
                bigramer.above_score(
                    self.collocation_score_function, self.min_collocation_score
                )
            )
            if len(mwes) == 0:
                break
            contracter = MWETokenizer(mwes)
            self.tokens_by_sent_by_doc_ = [
                contracter.tokenize_sents(doc) for doc in self.tokens_by_sent_by_doc()
            ]


class NLTKTokenizer(BaseTokenizer):
    """
    Tokenizes via NLTK sentence and word tokenizers, together with iterations of BaseTokenizer bigram contraction

      Parameters
      ----------
      lower_case = bool (default = False)
        Lower-case the sentences before tokenization.

      collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
        The function to score bigrams

      max_collocation_iterations = int (default = 2)
        The maximal number of recursive bigram contractions

      min_collocation_score: int (default = 12)
        The minimal PMI value to contract a bigram per iteration

    """

    def __init__(
        self,
        lower_case=False,
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):
        BaseTokenizer.__init__(
            self,
            collocation_score_function=collocation_score_function,
            max_collocation_iterations=max_collocation_iterations,
            min_collocation_score=min_collocation_score,
        )

        self.lower_case = lower_case

    def fit(self, X, **fit_params):
        """

        Parameters
        ----------
        X: collection
            The list of documents

        Returns
        -------
        self
        """
        if self.lower_case:
            self.tokens_by_sent_by_doc_ = [
                [word_tokenize(sent.lower()) for sent in sent_tokenize(doc)]
                for doc in X
            ]
        else:
            self.tokens_by_sent_by_doc_ = [
                [word_tokenize(sent) for sent in sent_tokenize(doc)] for doc in X
            ]

        self.iteratively_contract_bigrams()

        return self


class CountVectorizerTokenizer(BaseTokenizer):
    """
    Uses CountVectorizers document preprocessing and word tokenizer (but NLTK sentence tokenizer)

    Parameters
    ----------
    collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
      The function to score bigrams

    max_collocation_iterations = int (default = 2)
      The maximal number of recursive bigram contractions

    min_collocation_score: int (default = 12)
      The minimal PMI value to contract a bigram per iteration
    """

    def fit(self, X, **fit_params):
        """

        Parameters
        ----------
        X: collection
            The list of documents

        Returns
        -------
        self
        """
        cv_word_tokenize = CountVectorizer().build_tokenizer()
        cv_preprocesser = CountVectorizer().build_preprocessor()
        self.tokens_by_sent_by_doc_ = [
            [cv_word_tokenize(sent) for sent in sent_tokenize(cv_preprocesser(doc))]
            for doc in X
        ]

        self.iteratively_contract_bigrams()

        return self


class StanzaTokenizer(BaseTokenizer):
    """

    Parameters
    ----------
    nlp = "DEFAULT" or a stanza.Pipeline
      The stanza tokenizer

    collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
      The function to score bigrams

    max_collocation_iterations = int (default = 2)
      The maximal number of recursive bigram contractions

    min_collocation_score: int (default = 12)
      The minimal PMI value to contract a bigram per iteration
    """

    def __init__(
        self,
        nlp="DEFAULT",
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):

        BaseTokenizer.__init__(
            self,
            collocation_score_function=collocation_score_function,
            max_collocation_iterations=max_collocation_iterations,
            min_collocation_score=min_collocation_score,
        )

        if nlp == "DEFAULT":
            # A default Stanza NLP pipeline
            stanza.download(lang="en", processors="tokenize")
            BASIC_STANZA_PIPELINE = stanza.Pipeline(processors="tokenize")
            self.nlp = BASIC_STANZA_PIPELINE
        else:
            self.nlp == nlp

    def fit(self, X, **fit_params):
        """

        Parameters
        ----------
        X: collection
            The list of documents

        Returns
        -------
        self
        """

        self.tokens_by_sent_by_doc_ = tokens = [
            [[token.text for token in sent.tokens] for sent in self.nlp(doc).sentences]
            for doc in X
        ]

        self.iteratively_contract_bigrams()

        return self


class SpaCyTokenizer(BaseTokenizer):
    """

    Parameters
    ----------
    nlp = "DEFAULT" or a spaCy pipeline
      The spaCy tokenizer

    collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
      The function to score bigrams

    max_collocation_iterations = int (default = 2)
      The maximal number of recursive bigram contractions

    min_collocation_score: int (default = 12)
      The minimal PMI value to contract a bigram per iteration
    """

    def __init__(
        self,
        nlp="DEFAULT",
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):
        BaseTokenizer.__init__(
            self,
            collocation_score_function=collocation_score_function,
            max_collocation_iterations=max_collocation_iterations,
            min_collocation_score=min_collocation_score,
        )

        if nlp == "DEFAULT":
            # A default spaCy NLP pipeline
            BASIC_SPACY_PIPELINE = spacy.lang.en.English()
            BASIC_SPACY_PIPELINE.add_pipe(
                BASIC_SPACY_PIPELINE.create_pipe("sentencizer")
            )
            self.nlp = BASIC_SPACY_PIPELINE
        else:
            self.nlp == nlp

    def fit(self, X, **fit_params):
        """

        Parameters
        ----------
        X: collection
            The list of documents

        Returns
        -------
        self
        """

        self.tokens_by_sent_by_doc_ = [
            [[token.text for token in sent] for sent in doc.sents]
            for doc in self.nlp.pipe(X)
        ]

        self.iteratively_contract_bigrams()

        return self
