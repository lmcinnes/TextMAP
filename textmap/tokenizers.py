from .utilities import flatten
from warnings import warn

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
import nltk.tokenize.api
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
      tokenize_by = 'document' (default), 'sentence' or 'sentence by document'
        Return a tuple of tuples of tokens per document or a tuple of tuples of tokens per sentence, or a tuple of
        tuples of tuples of tokens per sentence per document.

      nlp = None or a tokenizer class
        If nlp = None then a default tokenizer will be constructed.  Otherwise the one provided will be used.

      lower_case = bool (default = True)
        Apply str.lower() to the tokens upon tokenization

    """

    def __init__(self, tokenize_by="document", nlp="default", lower_case=True):
        try:
            assert tokenize_by in ["document", "sentence", "sentence by document"]
            self.tokenize_by = tokenize_by
        except AssertionError:
            raise ValueError(
                'The tokenize_by parameter must be "document",  "sentence", or "sentence by document".'
            )
        if self.tokenize_by == "sentence by document":
            self._flatten = lambda x: tuple(x)
        else:
            self._flatten = flatten
        self.tokenization_ = None
        self.lower_case = lower_case
        self.nlp = nlp

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, model):
        self._nlp = model

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

        self.tokenization_ = X
        return self

    def fit_transform(self, X, **fit_params):
        """

        Parameters
        ----------
        X : collection
            The collection of documents

        Returns
        -------
        The list of tokenized documents or sentences
        """

        self.fit(X, **fit_params)
        return self.tokenization_


class NLTKTokenizer(BaseTokenizer):
    """
    Tokenizes via any NLTKTokenizer like class, using sent_tokenize and word_tokenize by default,

      Parameters
      ----------
      tokenize_by = 'document' (default), 'sentence' or 'sentence by document'
        Return a tuple of tuples of tokens per document or a tuple of tuples of tokens per sentence, or a tuple of
        tuples of tuples of tokens per sentence per document.

      nlp = 'default' or an NLTK style tokenizer
        The default will tokenize via an NLTK tokenizer using NLTK's word_tokenize function.

      lower_case = bool (default = True)
        Apply str.lower() to the tokens upon tokenization
    """

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, model):
        if model == "default":
            model = nltk.tokenize.api.StringTokenizer()
            model.tokenize = word_tokenize
            self._nlp = model
        else:
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

        if self.lower_case:
            tokenize = lambda d: (w.lower() for w in self.nlp.tokenize(d))
        else:
            tokenize = lambda d: tuple(self.nlp.tokenize(d))

        if self.tokenize_by in ["sentence", "sentence by document"]:
            self.tokenization_ = self._flatten(
                [
                    tuple([tuple(tokenize(sent)) for sent in sent_tokenize(doc)])
                    for doc in X
                ]
            )

        elif self.tokenize_by == "document":
            self.tokenization_ = tuple([tuple(tokenize(doc)) for doc in X])
        else:
            raise ValueError(
                'The tokenize_by parameter must be "document",  "sentence", or "sentence by document".'
            )
        return self


class NLTKTweetTokenizer(NLTKTokenizer):
    """
    A class to create an NLTKTokenizer using nltk.tokenize.TweetTokenizer

      Parameters
      ----------
      tokenize_by = 'document' (default), 'sentence' or 'sentence by document'
        Return a tuple of tuples of tokens per document or a tuple of tuples of tokens per sentence, or a tuple of
        tuples of tuples of tokens per sentence per document.

      nlp = 'default' or an NLTKTokenizer
        The default it will be the NLTK's TweetTokenizer with default settings

      lower_case = bool (default = True)
        Apply str.lower() to the tokens upon tokenization
    """

    def __init__(self, tokenize_by="document", nlp="default", lower_case=True):
        self.nlp = nlp
        NLTKTokenizer.__init__(
            self, tokenize_by=tokenize_by, nlp=self.nlp, lower_case=lower_case
        )

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, model):
        if model == "default":
            model = TweetTokenizer(
                preserve_case=True, reduce_len=False, strip_handles=False,
            )
            self._nlp = model
        else:
            self._nlp = model


class SKLearnTokenizer(BaseTokenizer):
    """
    A generic class that tokenizes via any tokenization function accepted in scikit-learn. By default it uses
    CountVectorizer's default document preprocessing and word tokenizer.

    Note: It will use NLTK sentence tokenizer if tokenizing by sentence as there is no scikit-learn default.

    Parameters
    ----------
    tokenize_by = 'document' (default), 'sentence' or 'sentence by document'
        Return a tuple of tuples of tokens per document or a tuple of tuples of tokens per sentence, or a tuple of
        tuples of tuples of tokens per sentence per document.

    nlp = 'default' or a function
        This can be any function which takes in strings and returns a list of strings.  The default is the
        scikit-learn's tokenization function used in CountVectorizer().

    lower_case = bool (default = True)
        Apply str.lower() to the tokens upon tokenization
    """

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, model):
        if model == "default":
            cv = CountVectorizer(lowercase=self.lower_case)
            sk_word_tokenize = cv.build_tokenizer()
            sk_preprocesser = cv.build_preprocessor()
            self._nlp = lambda doc: sk_word_tokenize(sk_preprocesser(doc))
        else:
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

        if self.tokenize_by in ["sentence", "sentence by document"]:
            self.tokenization_ = self._flatten(
                [
                    tuple([tuple(self.nlp(sent)) for sent in sent_tokenize(doc)])
                    for doc in X
                ]
            )

        elif self.tokenize_by == "document":
            self.tokenization_ = tuple([tuple(self.nlp(doc)) for doc in X])

        else:
            raise ValueError(
                'The tokenize_by parameter must be "document",  "sentence", or "sentence by document".'
            )

        return self


class StanzaTokenizer(BaseTokenizer):
    """

    Parameters
    ----------
    tokenize_by = 'document' (default), 'sentence' or 'sentence by document'
        Return a tuple of tuples of tokens per document or a tuple of tuples of tokens per sentence, or a tuple of
        tuples of tuples of tokens per sentence per document.

    nlp = 'default' (default) or a stanza.Pipeline
        The stanza tokenizer.  If None then a default one will be created.

    lower_case = bool (default = True)
        Apply str.lower() to the tokens upon tokenization
    """

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, model):
        if model == "default":
            # A default Stanza NLP pipeline
            stanza.download(lang="en", processors="tokenize")
            if self.tokenize_by in ["sentence", "sentence by document"]:
                BASIC_STANZA_PIPELINE = stanza.Pipeline(processors="tokenize")
            else:
                BASIC_STANZA_PIPELINE = stanza.Pipeline(
                    processors="tokenize", tokenize_no_ssplit=True
                )
            self._nlp = BASIC_STANZA_PIPELINE
        else:
            if self.tokenize_by in ["sentence", "sentence by document"]:
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

        if self.lower_case:
            token_text = lambda t: (t.text).lower()
        else:
            token_text = lambda t: t.text

        if self.tokenize_by in ["sentence", "sentence by document"]:
            self.tokenization_ = self._flatten(
                [
                    tuple(
                        [
                            tuple([token_text(token) for token in sent.tokens])
                            for sent in self.nlp(doc).sentences
                        ]
                    )
                    for doc in X
                ]
            )
        elif self.tokenize_by == "document":
            self.tokenization_ = tuple(
                [
                    tuple([token_text(token) for token in self.nlp(doc).iter_tokens()])
                    for doc in X
                ]
            )
        else:
            raise ValueError(
                'The tokenize_by parameter must be "document",  "sentence", or "sentence by document".'
            )
        return self


class SpacyTokenizer(BaseTokenizer):
    """

    Parameters
    ----------
    tokenize_by = 'document' (default), 'sentence' or 'sentence by document'
        Return a tuple of tuples of tokens per document or a tuple of tuples of tokens per sentence, or a tuple of
        tuples of tuples of tokens per sentence per document.

    nlp = 'default' or a spaCy pipeline
        The spaCy tokenizer.  If None, a default one will be created.

    lower_case = bool (default = True)
        Tokenizes as the token.text (if False) or token.lower (if True)
    """

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, model):
        if model == "default":
            # A default spaCy NLP pipeline
            BASIC_SPACY_PIPELINE = spacy.lang.en.English()
            if self.tokenize_by in ["sentence", "sentence by document"]:
                BASIC_SPACY_PIPELINE.add_pipe(
                    BASIC_SPACY_PIPELINE.create_pipe("sentencizer"), first=True
                )
            self._nlp = BASIC_SPACY_PIPELINE
        else:
            # Check that the required components are there
            if self.tokenize_by in ["sentence", "sentence by document"]:
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

        if self.lower_case:
            token_text = lambda t: t.lower_
        else:
            token_text = lambda t: t.text

        if self.tokenize_by in ["sentence", "sentence by document"]:
            self.tokenization_ = self._flatten(
                [
                    tuple(
                        [
                            tuple([token_text(token) for token in sent])
                            for sent in doc.sents
                        ]
                    )
                    for doc in self.nlp.pipe(X)
                ]
            )

        elif self.tokenize_by == "document":
            self.tokenization_ = tuple(
                [
                    tuple([token_text(token) for token in doc])
                    for doc in self.nlp.pipe(X)
                ]
            )
        else:
            raise ValueError(
                'The tokenize_by parameter must be "document",  "sentence", or "sentence by document".'
            )

        return self
