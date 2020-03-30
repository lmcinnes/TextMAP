from .utilities import flatten
from warnings import warn

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer, TweetTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import CountVectorizer

# Optional tokenization packages
try:
    import stanza
except ImportError:
    warn(
        "The stanza library could not be imported StanzaTokenizer will not be available."
    )
try:
    import spacy
    from spacy.lang.en import English
except ImportError:
    warn(
        "The SpaCy library could not be imported SpaCyTokenizer will not be available."
    )


class BaseTokenizer(BaseEstimator, TransformerMixin):
    """
    Base class for all textmap tokenizers to inherit from.
    
      Parameters
      ----------
      tokenize_by = 'document' (default) or 'sentence'
        Return a list of lists of tokens per document or a list of lists of lists of tokens per sentence per document

      collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
        The function to score bigrams

      max_collocation_iterations = int (default = 2)
        The maximal number of recursive bigram contractions

      min_collocation_score: int (default = 12)
        The minimal score value to contract a bigram per iteration

    """

    def __init__(
        self,
        tokenize_by="document",
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):
        try:
            assert tokenize_by in ["document", "sentence"]
            self.tokenize_by = tokenize_by
        except AssertionError:
            raise ValueError("tokenize_by parameter must be 'document' or 'sentence'")

        self.collocation_score_function = collocation_score_function
        self.max_collocation_iterations = max_collocation_iterations
        self.min_collocation_score = min_collocation_score
        self.tokenization_ = None

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
        self.tokenization_ = None
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
        return self.tokenization_

    def iteratively_contract_bigrams(self):
        """
        Procedure to iteratively contract bigrams (up to max_collocation_iterations times)
        that score higher on the collocation_function than the min_collocation_score
        """
        if self.tokenize_by == "document":
            for i in range(self.max_collocation_iterations):
                bigramer = BigramCollocationFinder.from_documents(self.tokenization_)
                mwes = list(
                    bigramer.above_score(
                        self.collocation_score_function, self.min_collocation_score
                    )
                )
                if len(mwes) == 0:
                    break
                contracter = MWETokenizer(mwes)
                self.tokenization_ = [
                    contracter.tokenize(doc) for doc in self.tokenization_
                ]
        else:
            for i in range(self.max_collocation_iterations):
                bigramer = BigramCollocationFinder.from_documents(
                    flatten(self.tokenization_)
                )
                mwes = list(
                    bigramer.above_score(
                        self.collocation_score_function, self.min_collocation_score
                    )
                )
                if len(mwes) == 0:
                    break
                contracter = MWETokenizer(mwes)
                self.tokenization_ = [
                    [contracter.tokenize(sent) for sent in doc]
                    for doc in self.tokenization_
                ]


class NLTKTokenizer(BaseTokenizer):
    """
    Tokenizes via NLTK sentence and word tokenizers, together with iterations of BaseTokenizer bigram contraction

      Parameters
      ----------
      tokenize_by = 'document' (default) or 'sentence'
        Return a list of lists of tokens per document or a list of lists of lists of tokens per sentence per document

      lower_case = bool (default = False)
        Lower-case the sentences before tokenization.

      collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
        The function to score bigrams

      max_collocation_iterations = int (default = 2)
        The maximal number of recursive bigram contractions

      min_collocation_score: int (default = 12)
        The minimal score value to contract a bigram per iteration

    """

    def __init__(
        self,
        tokenize_by="document",
        lower_case=False,
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):
        BaseTokenizer.__init__(
            self,
            tokenize_by=tokenize_by,
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
        if self.tokenize_by == "sentence":
            if self.lower_case:
                self.tokenization_ = [
                    [word_tokenize(sent.lower()) for sent in sent_tokenize(doc)]
                    for doc in X
                ]
            else:
                self.tokenization_ = [
                    [word_tokenize(sent) for sent in sent_tokenize(doc)] for doc in X
                ]
        else:
            if self.lower_case:
                self.tokenization_ = [word_tokenize(doc.lower()) for doc in X]
            else:
                self.tokenization_ = [word_tokenize(doc) for doc in X]

        self.iteratively_contract_bigrams()

        return self


class NLTKTweetTokenizer(BaseTokenizer):
    """
    Tokenizes via NLTK TweetTokenizer and sent_tokenize, together with iterations of BaseTokenizer bigram contraction

      Parameters
      ----------
      tokenize_by = 'document' (default) or 'sentence'
        Return a list of lists of tokens per document or a list of lists of lists of tokens per sentence per document

      lower_case = bool (default = False)
        Lower-case the sentences after tokenization.

      reduce_length = bool (default = False)
        Replaces repeated characters of length greater than 3 with the sequence of length 3.

      strip_handles = bool (defaul = False)
        Removes twitter username handles from the text

      collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
        The function to score bigrams

      max_collocation_iterations = int (default = 2)
        The maximal number of recursive bigram contractions

      min_collocation_score: int (default = 12)
        The minimal score value to contract a bigram per iteration

    """

    def __init__(
        self,
        tokenize_by="document",
        lower_case=False,
        reduce_length=False,
        strip_handles=False,
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):
        BaseTokenizer.__init__(
            self,
            tokenize_by=tokenize_by,
            collocation_score_function=collocation_score_function,
            max_collocation_iterations=max_collocation_iterations,
            min_collocation_score=min_collocation_score,
        )

        self.lower_case = lower_case
        self.reduce_length = reduce_length
        self.strip_handles = strip_handles

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
        tweet_tokenize = TweetTokenizer(
            preserve_case=not self.lower_case,
            reduce_len=self.reduce_length,
            strip_handles=self.strip_handles,
        ).tokenize

        if self.tokenize_by == "sentence":
            self.tokenization_ = [
                [tweet_tokenize(sent) for sent in sent_tokenize(doc)] for doc in X
            ]
        else:
            self.tokenization_ = [tweet_tokenize(doc) for doc in X]

        self.iteratively_contract_bigrams()

        return self


class SKLearnTokenizer(BaseTokenizer):
    """
    Uses CountVectorizers' document preprocessing and word tokenizer (NLTK sentence tokenizer if tokenizing by sentence)

    Note: This will lower case the text.

    Parameters
    ----------
    tokenize_by = 'document' (default) or 'sentence'
        Return a list of lists of tokens per document or a list of lists of lists of tokens per sentence per document

    collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
      The function to score bigrams

    max_collocation_iterations = int (default = 2)
      The maximal number of recursive bigram contractions

    min_collocation_score: int (default = 12)
      The minimal score value to contract a bigram per iteration
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
        sk_word_tokenize = CountVectorizer().build_tokenizer()
        sk_preprocesser = CountVectorizer().build_preprocessor()
        if self.tokenize_by == "sentence":
            self.tokenization_ = [
                [sk_word_tokenize(sent) for sent in sent_tokenize(sk_preprocesser(doc))]
                for doc in X
            ]
        else:
            self.tokenization_ = [sk_word_tokenize(sk_preprocesser(doc)) for doc in X]

        self.iteratively_contract_bigrams()

        return self


class StanzaTokenizer(BaseTokenizer):
    """

    Parameters
    ----------
    tokenize_by = 'document' (default) or 'sentence'
        Return a list of lists of tokens per document or a list of lists of lists of tokens per sentence per document

    nlp = "DEFAULT" or a stanza.Pipeline
      The stanza tokenizer

    collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
      The function to score bigrams

    max_collocation_iterations = int (default = 2)
      The maximal number of recursive bigram contractions

    min_collocation_score: int (default = 12)
      The minimal score value to contract a bigram per iteration
    """

    def __init__(
        self,
        nlp="DEFAULT",
        tokenize_by="document",
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):

        BaseTokenizer.__init__(
            self,
            tokenize_by=tokenize_by,
            collocation_score_function=collocation_score_function,
            max_collocation_iterations=max_collocation_iterations,
            min_collocation_score=min_collocation_score,
        )
        self.nlp = nlp

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, model):
        if model == "DEFAULT":
            # A default Stanza NLP pipeline
            stanza.download(lang="en", processors="tokenize")
            if self.tokenize_by == "sentence":
                BASIC_STANZA_PIPELINE = stanza.Pipeline(processors="tokenize")
            else:
                BASIC_STANZA_PIPELINE = stanza.Pipeline(
                    processors="tokenize", tokenize_no_ssplit=True
                )
            self._nlp = BASIC_STANZA_PIPELINE
        else:
            if self.tokenize_by == "sentence":
                if model.config["tokenize_no_ssplit"]:
                    model.processors["tokenize"].config["no_ssplit"] = False
                    model.config["tokenize_no_ssplit"] = False
                    warn(
                        "NLP does not have a sentencizer pipe; one has been added to tokenize by sentence."
                    )
            else:
                if not model.config["tokenize_no_ssplit"]:
                    model.processors["tokenize"].config["no_ssplit"] = True
                    model.config["tokenize_no_ssplit"] = True
                    warn(
                        "NLP contains a sentencizer pipe which has been removed to tokenize by document."
                    )

            self._nlp = model

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
        if self.tokenize_by == "sentence":
            self.tokenization_ = [
                [
                    [token.text for token in sent.tokens]
                    for sent in self.nlp(doc).sentences
                ]
                for doc in X
            ]
        else:
            self.tokenization_ = [
                [token.text for token in self.nlp(doc).iter_tokens()] for doc in X
            ]

        self.iteratively_contract_bigrams()

        return self


class SpaCyTokenizer(BaseTokenizer):
    """

    Parameters
    ----------
    tokenize_by = 'document' (default) or 'sentence'
        Return a list of lists of tokens per document or a list of lists of lists of tokens per sentence per document

    nlp = "DEFAULT" or a spaCy pipeline
      The spaCy tokenizer

    collocation_score_function = nltk.metrics.BigramAssocMeasures (default likelihood_ratio)
      The function to score bigrams

    max_collocation_iterations = int (default = 2)
      The maximal number of recursive bigram contractions

    min_collocation_score: int (default = 12)
      The minimal score value to contract a bigram per iteration
    """

    def __init__(
        self,
        tokenize_by="document",
        nlp="DEFAULT",
        collocation_score_function=BigramAssocMeasures.likelihood_ratio,
        max_collocation_iterations=2,
        min_collocation_score=12,
    ):
        BaseTokenizer.__init__(
            self,
            tokenize_by=tokenize_by,
            collocation_score_function=collocation_score_function,
            max_collocation_iterations=max_collocation_iterations,
            min_collocation_score=min_collocation_score,
        )

        self.nlp = nlp

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, model):
        if model == "DEFAULT":
            # A default spaCy NLP pipeline
            BASIC_SPACY_PIPELINE = spacy.lang.en.English()
            if self.tokenize_by == "sentence":
                BASIC_SPACY_PIPELINE.add_pipe(
                    BASIC_SPACY_PIPELINE.create_pipe("sentencizer"), first=True
                )
            self._nlp = BASIC_SPACY_PIPELINE
        else:
            # Check that the required components are there
            if self.tokenize_by == "sentence":
                if "sentencizer" not in model.pipe_names:
                    try:
                        model.add_pipe(model.create_pipe("sentencizer"), first=True)
                        warn(
                            "NLP does not have a sentencizer pipe; one has been added to tokenize by sentence."
                        )
                    except KeyError:
                        raise ValueError("NLP model does not have a sentencizer pipe.")
            else:
                if "sentencizer" in model.pipe_names:
                    model.remove_pipe("sentencizer")
                    warn(
                        "NLP contains a sentencizer pipe which has been removed to tokenize by document."
                    )
            self._nlp = model

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
        if self.tokenize_by == "sentence":
            self.tokenization_ = [
                [[token.text for token in sent] for sent in doc.sents]
                for doc in self.nlp.pipe(X)
            ]
        else:
            self.tokenization_ = [
                [token.text for token in doc] for doc in self.nlp.pipe(X)
            ]

        self.iteratively_contract_bigrams()

        return self
