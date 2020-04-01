import pytest

import spacy
from spacy.lang.en import English

# import stanza

from textmap.tokenizers import (
    SpaCyTokenizer,
    NLTKTokenizer,
    NLTKTweetTokenizer,
    SKLearnTokenizer,
    StanzaTokenizer,
)
from .test_common import test_text
from textmap.tranformers import MWETransformer


@pytest.mark.parametrize("tokens_by", ["document", "sentence"])
@pytest.mark.parametrize("lower_case", [True, False])
def test_sklearn_tokenizer(tokens_by, lower_case):
    tokenizer = SKLearnTokenizer(tokenize_by=tokens_by, lower_case=lower_case).fit(
        test_text
    )


@pytest.mark.parametrize("tokens_by", ["document", "sentence"])
@pytest.mark.parametrize("lower_case", [True, False])
def test_nltk_tokenizer(tokens_by, lower_case):
    tokenizer = NLTKTokenizer(tokenize_by=tokens_by, lower_case=lower_case).fit(
        test_text
    )


@pytest.mark.parametrize("tokens_by", ["document", "sentence"])
@pytest.mark.parametrize("lower_case", [True, False])
def test_tweet_tokenizer(tokens_by, lower_case):
    tokenizer = NLTKTweetTokenizer(tokenize_by=tokens_by, lower_case=lower_case).fit(
        test_text
    )


@pytest.mark.parametrize("tokens_by", ["document", "sentence"])
@pytest.mark.parametrize("lower_case", [True, False])
def test_spacy_tokenizer(tokens_by, lower_case):
    tokenizer = SpaCyTokenizer(tokenize_by=tokens_by, lower_case=lower_case).fit(
        test_text
    )


def test_spacy_add_sentencizer():
    nlp = English()
    # Remove all of the components
    for p in nlp.pipe_names:
        nlp.remove_pipe(p)
    tokenizer = SpaCyTokenizer(tokenize_by="sentence", nlp=nlp)
    assert "sentencizer" in tokenizer.nlp.pipe_names


def test_spacy_remove_sentencizer():
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"), first=True)
    tokenizer = SpaCyTokenizer(tokenize_by="document", nlp=nlp)
    assert not ("sentencizer" in tokenizer.nlp.pipe_names)


"""

Stanza requires PyTorch which isn't behaving well in the github testing (the tests did pass locally though)  

def test_stanza_tokenizer():
    for tokens_by in ["document", "sentence"]:
        tokenizer = StanzaTokenizer(tokenize_by=tokens_by).fit(test_text)


def test_stanza_add_sentencizer():
    stanza.download(lang="en", processors="tokenize")
    nlp = stanza.Pipeline(processors="tokenize", tokenize_no_ssplit=True)
    tokenizer = StanzaTokenizer(tokenize_by="sentence", nlp=nlp)
    assert not tokenizer.nlp.config["tokenize_no_ssplit"]


def test_stanza_remove_sentencizer():
    stanza.download(lang="en", processors="tokenize")
    nlp = stanza.Pipeline(processors="tokenize", tokenize_no_ssplit=False)
    tokenizer = StanzaTokenizer(tokenize_by="document", nlp=nlp)
    assert tokenizer.nlp.config["tokenize_no_ssplit"]
"""


@pytest.mark.parametrize("min_score", [0, 3])
@pytest.mark.parametrize("min_token_occurrences", [None, 3])
@pytest.mark.parametrize("max_token_occurrences", [None, 6])
@pytest.mark.parametrize("min_ngram_occurrences", [None, 3])
def test_mwe_transformer(
    min_score, min_token_occurrences, max_token_occurrences, min_ngram_occurrences
):
    tokens = SKLearnTokenizer().fit_transform(test_text)
    test = MWETransformer(
        min_score=min_score,
        min_token_occurrences=min_token_occurrences,
        max_token_occurrences=max_token_occurrences,
        min_ngram_occurrences=min_ngram_occurrences,
    ).fit_transform(tokens)
