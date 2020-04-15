import pytest
import numpy as np
import spacy
from spacy.lang.en import English

# import stanza

from textmap.tokenizers import (
    SpacyTokenizer,
    NLTKTokenizer,
    NLTKTweetTokenizer,
    SKLearnTokenizer,
    StanzaTokenizer,
)
from .test_common import (
    test_text,
    test_matrix,
    test_matrix_zero_column,
    test_matrix_zero_row,
)

from textmap.transformers import (
    MultiTokenExpressionTransformer,
    RemoveEffectsTransformer,
    InformationWeightTransformer,
)


@pytest.mark.parametrize("tokens_by", ["document", "sentence", "sentence by document"])
@pytest.mark.parametrize("lower_case", [True, False])
def test_sklearn_tokenizer(tokens_by, lower_case):
    tokenizer = SKLearnTokenizer(tokenize_by=tokens_by, lower_case=lower_case).fit(
        test_text
    )


@pytest.mark.parametrize("tokens_by", ["document", "sentence", "sentence by document"])
@pytest.mark.parametrize("lower_case", [True, False])
def test_nltk_tokenizer(tokens_by, lower_case):
    tokenizer = NLTKTokenizer(tokenize_by=tokens_by, lower_case=lower_case).fit(
        test_text
    )


@pytest.mark.parametrize("tokens_by", ["document", "sentence", "sentence by document"])
@pytest.mark.parametrize("lower_case", [True, False])
def test_tweet_tokenizer(tokens_by, lower_case):
    tokenizer = NLTKTweetTokenizer(tokenize_by=tokens_by, lower_case=lower_case).fit(
        test_text
    )


@pytest.mark.parametrize("tokens_by", ["document", "sentence", "sentence by document"])
@pytest.mark.parametrize("lower_case", [True, False])
def test_spacy_tokenizer(tokens_by, lower_case):
    tokenizer = SpacyTokenizer(tokenize_by=tokens_by, lower_case=lower_case).fit(
        test_text
    )


def test_spacy_add_sentencizer():
    nlp = English()
    # Remove all of the components
    for p in nlp.pipe_names:
        nlp.remove_pipe(p)
    tokenizer = SpacyTokenizer(tokenize_by="sentence", nlp=nlp)
    assert "sentencizer" in tokenizer.nlp.pipe_names


def test_spacy_remove_sentencizer():
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"), first=True)
    tokenizer = SpacyTokenizer(tokenize_by="document", nlp=nlp)
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
@pytest.mark.parametrize("min_token_frequency", [None, 0.1])
@pytest.mark.parametrize("max_token_frequency", [None, 0.1])
@pytest.mark.parametrize("ignored_tokens", [None, {"foo", "bar"}])
@pytest.mark.parametrize("excluded_token_regex", [None, r"b\w+"])
def test_mwe_transformer(
    min_score,
    min_token_occurrences,
    max_token_occurrences,
    min_token_frequency,
    max_token_frequency,
    min_ngram_occurrences,
    ignored_tokens,
    excluded_token_regex,
):
    tokens = SKLearnTokenizer().fit_transform(test_text)
    mte = MultiTokenExpressionTransformer(
        min_score=min_score,
        min_token_occurrences=min_token_occurrences,
        max_token_occurrences=max_token_occurrences,
        min_token_frequency=min_token_frequency,
        max_token_frequency=max_token_frequency,
        min_ngram_occurrences=min_ngram_occurrences,
        ignored_tokens=ignored_tokens,
        excluded_token_regex=excluded_token_regex,
    )
    out_fit_transform = mte.fit_transform(tokens)
    out_transform = mte.transform(tokens)

    assert out_fit_transform == out_transform


@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("model_type", ["EnsTop"])
@pytest.mark.parametrize("em_precision", [1e-4])
@pytest.mark.parametrize("em_background_prior", [0.1])
@pytest.mark.parametrize("em_threshold", [1e-4])
@pytest.mark.parametrize("em_prior_strength", [0.05])
@pytest.mark.parametrize("normalize", [False])
def test_enstop_re_transformer(
    n_components,
    model_type,
    em_precision,
    em_background_prior,
    em_threshold,
    em_prior_strength,
    normalize,
):
    RET = RemoveEffectsTransformer(
        n_components=n_components,
        model_type=model_type,
        em_precision=em_precision,
        em_background_prior=em_background_prior,
        em_threshold=em_threshold,
        em_prior_strength=em_prior_strength,
        normalize=normalize,
    )

    RET.fit(test_matrix)
    RET.transform(test_matrix)
    RET.fit_transform(test_matrix)
    RET.fit(test_matrix_zero_column)
    RET.transform(test_matrix_zero_column)
    RET.fit_transform(test_matrix_zero_column)
    RET.fit(test_matrix_zero_row, bootstrap=False)
    RET.transform(test_matrix_zero_row)
    RET.fit_transform(test_matrix_zero_row, bootstrap=False)


@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("model_type", ["pLSA"])
@pytest.mark.parametrize("em_precision", [1e-3, 1e-4])
@pytest.mark.parametrize("em_background_prior", [0.1, 10.0])
@pytest.mark.parametrize("em_threshold", [1e-4, 1e-5])
@pytest.mark.parametrize("em_prior_strength", [1.0, 10.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_re_transformer(
    n_components,
    model_type,
    em_precision,
    em_background_prior,
    em_threshold,
    em_prior_strength,
    normalize,
):
    RET = RemoveEffectsTransformer(
        n_components=n_components,
        model_type=model_type,
        em_precision=em_precision,
        em_background_prior=em_background_prior,
        em_threshold=em_threshold,
        em_prior_strength=em_prior_strength,
        normalize=normalize,
    )

    RET.fit(test_matrix)
    RET.transform(test_matrix)
    RET.fit_transform(test_matrix)
    RET.fit(test_matrix_zero_column)
    RET.transform(test_matrix_zero_column)
    RET.fit_transform(test_matrix_zero_column)
    RET.fit(test_matrix_zero_row)
    RET.transform(test_matrix_zero_row)
    RET.fit_transform(test_matrix_zero_row)


@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("model_type", ["pLSA", "EnsTop"])
def test_iw_transformer(
    n_components, model_type,
):
    IWT = InformationWeightTransformer(
        n_components=n_components, model_type=model_type,
    )

    IWT.fit(test_matrix)
    IWT.transform(test_matrix)
    IWT.fit_transform(test_matrix)
    IWT.fit(test_matrix_zero_column)
    IWT.transform(test_matrix_zero_column)
    IWT.fit_transform(test_matrix_zero_column)
    if model_type == "pLSA":
        IWT.fit(test_matrix_zero_row)
        IWT.transform(test_matrix_zero_row)
        IWT.fit_transform(test_matrix_zero_row)
    else:
        IWT.fit(test_matrix_zero_row, bootstrap=False)
        IWT.transform(test_matrix_zero_row)
        IWT.fit_transform(test_matrix_zero_row, bootstrap=False)
