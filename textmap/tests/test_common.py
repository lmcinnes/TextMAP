import pytest
from hypothesis import given, example, settings, note, HealthCheck
import hypothesis.strategies as st
from hypothesis.strategies import composite

import os

from sklearn.utils.estimator_checks import check_estimator
import scipy.sparse
import numpy as np

from textmap import WordMAP
from textmap import DocMAP
from textmap import TopicMAP

from textmap.vectorizers import (
    DocVectorizer,
    WordVectorizer,
    FeatureBasisConverter,
    JointWordDocVectorizer,
    _MULTITOKEN_COOCCURRENCE_VECTORIZERS,
)

from textmap.utilities import MultiTokenCooccurrenceVectorizer


import nltk

nltk.download("punkt")


# @pytest.mark.parametrize(
#     "Estimator", [WordMAP, DocMAP, TopicMAP]
# )
# def test_all_estimators(Estimator):
#     return check_estimator(Estimator)

test_text_example = [
    "foo bar pok wer pok pok foo bar wer qwe pok asd fgh",
    "foo bar pok wer pok pok foo bar wer qwe pok asd fgh",
    "",
    "fgh asd foo pok qwe pok wer pok foo bar pok pok wer",
    "pok wer pok qwe foo asd foo bar pok wer asd wer pok",
]

test_text_token_data = (
    ("foo", "pok", "foo", "wer", "bar"),
    (),
    ("bar", "foo", "bar", "pok", "wer", "foo", "bar", "foo", "pok", "bar", "wer"),
    ("wer", "foo", "foo", "pok", "bar", "wer", "bar"),
    ("foo", "bar", "bar", "foo", "bar", "foo", "pok", "wer", "pok", "bar", "wer"),
    ("pok", "wer", "bar", "foo", "pok", "foo", "wer", "wer", "foo", "pok", "bar"),
    (
        "bar",
        "foo",
        "pok",
        "foo",
        "wer",
        "wer",
        "foo",
        "wer",
        "foo",
        "pok",
        "bar",
        "wer",
    ),
)

test_matrix = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
test_matrix_zero_row = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
test_matrix_zero_row.eliminate_zeros()
test_matrix_zero_column = scipy.sparse.csr_matrix([[1, 2, 0], [4, 5, 0], [7, 8, 0]])
test_matrix_zero_column.eliminate_zeros()

## Setup for randomized text generation
from string import ascii_letters

## don't run multiple examples when doing a coverage run
if os.environ.get("COVERAGE") == "true":
    DEFAULT_MAX_EXAMPLES = 2
else:
    DEFAULT_MAX_EXAMPLES = 100  # standard hypothesis default

VOCAB_SIZE = 50
ALPHABET = ascii_letters  # st.characters(blacklist_characters=' ')

VocabularyStrategy = st.lists(
    st.text(alphabet=ALPHABET, min_size=2, max_size=15),
    min_size=VOCAB_SIZE,
    max_size=VOCAB_SIZE,
    unique=True,
)


def indices_to_sentence(indices, vocabulary):
    """
    Turn a list of indices of a vocabulary into a sentence.
    """
    y = indices
    words = [vocabulary[idx] for idx in y]
    return " ".join(words)


@composite
def generate_test_text_info(draw):
    """
    Generates a list of test text, where one of the elements is duplicated, one of the elements is the
    empty string, and one of the elements is duplicated with an extra word added.

    Returns
    -------
    (test_text, vocabulary)
    """
    vocabulary = draw(VocabularyStrategy)
    vocab_size = len(vocabulary) - 1
    x = draw(
        st.lists(
            st.lists(
                st.integers(min_value=0, max_value=vocab_size),
                min_size=5,
                max_size=20,
                unique=True,
            ),
            min_size=10,
            max_size=30,
        )
    )
    text = [indices_to_sentence(y, vocabulary) for y in x]

    text.append("")

    index_to_duplicate = draw(st.integers(min_value=0, max_value=len(text) - 1))
    text.append(text[index_to_duplicate])

    index_to_add_word = draw(st.integers(min_value=0, max_value=len(text) - 1))
    new_word = vocabulary[draw(st.integers(min_value=0, max_value=vocab_size))]
    text.append(text[index_to_add_word] + " " + new_word)

    return (text, vocabulary)


# TODO: Add a set of tests for passing in instantiated classes

# TODO: Test that DocVectorizer transform preserves column order and size on new data


@given(generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=DEFAULT_MAX_EXAMPLES,
)
@example((test_text_example, None))
def test_joint_nobasistransformer(test_text_info):
    test_text = test_text_info[0]
    model = JointWordDocVectorizer(
        feature_basis_converter=None, token_contractor_kwds={"min_score": 8}
    )
    result = model.fit_transform(test_text)
    assert isinstance(result, scipy.sparse.csr_matrix)
    if test_text == test_text_example:
        assert result.shape == (12, 7)
    else:
        assert result.shape[0] == model.n_words_ + len(test_text)
        assert result.shape[1] == model.n_words_


@given(generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=DEFAULT_MAX_EXAMPLES,
)
@example((test_text_example, None))
def test_jointworddocvectorizer_vocabulary(test_text_info):
    test_text, vocabulary = test_text_info
    vocabulary_size = 3
    if test_text == test_text_example:
        vocab = (["foo", "bar", "pok"],)
    else:
        vocab = list(set(test_text[0].split()))[:vocabulary_size]
    model = JointWordDocVectorizer(feature_basis_converter=None, token_dictionary=vocab)
    result = model.fit_transform(test_text)
    assert isinstance(result, scipy.sparse.csr_matrix)
    # assert result.shape == (8, 3)
    assert model.n_words_ == vocabulary_size
    assert result.shape[0] == vocabulary_size + len(test_text)
    assert result.shape[1] == vocabulary_size


@given(generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=DEFAULT_MAX_EXAMPLES,
)
@example((test_text_example, None))
def test_jointworddocvectorizer(test_text_info):
    test_text, vocabulary = test_text_info
    num_components = 3
    model = JointWordDocVectorizer(n_components=num_components)
    result = model.fit_transform(test_text)
    transform = model.transform(test_text)
    assert np.allclose(result, transform)
    assert isinstance(result, np.ndarray)
    if test_text == test_text_example:
        assert result.shape == (12, num_components)
        assert model.n_words_ == 7
    else:
        assert model.n_words_ <= len(vocabulary)
        assert result.shape[1] == num_components
        assert result.shape[0] == len(test_text) + model.n_words_


def test_featurebasisconverter_tokenized():
    converter = FeatureBasisConverter(word_vectorizer="tokenized", n_components=3)
    converter.fit(test_text_token_data)
    doc_vectorizer = DocVectorizer(tokenizer=None, token_contractor=None)
    doc_rep = doc_vectorizer.fit_transform(test_text_token_data)
    new_rep = converter.change_basis(doc_rep, doc_vectorizer.column_index_dictionary_)
    assert new_rep.shape == (7, 3)


@given(generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=DEFAULT_MAX_EXAMPLES,
)
@example((test_text_example, None))
def test_wordvectorizer_todataframe(test_text_info):
    test_text, vocabulary = test_text_info
    model = WordVectorizer().fit(test_text)
    df = model.to_DataFrame()
    if test_text == test_text_example:
        assert df.shape == (7, 14)
    else:
        assert df.shape[0] <= len(vocabulary)
        assert df.shape[1] == df.shape[0] * 2


@given(generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=DEFAULT_MAX_EXAMPLES,
)
@example((test_text_example, None))
def test_wordvectorizer_vocabulary(test_text_info):
    test_text, vocabulary = test_text_info
    if test_text == test_text_example:
        vocab = ["foo", "bar"]
    else:
        vocab = list(set(test_text[0].split()))[:2]
    model = WordVectorizer(token_dictionary=vocab).fit(test_text)
    assert model.representation_.shape == (2, 4)
    assert model.token_dictionary == vocab


@given(generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=DEFAULT_MAX_EXAMPLES,
)
@example((test_text_example, None))
def test_docvectorizer_todataframe(test_text_info):
    test_text, vocabulary = test_text_info
    model = DocVectorizer().fit(test_text)
    df = model.to_DataFrame()
    if test_text == test_text_example:
        assert df.shape == (5, 7)
    else:
        assert df.shape[0] == len(test_text)
        assert df.shape[1] <= len(vocabulary)


def test_docvectorizer_unique():
    with pytest.raises(ValueError):
        model_unique = DocVectorizer(
            token_contractor_kwds={"min_score": 25}, fit_unique=True
        ).fit(test_text_example)
        assert "foo_bar" not in model_unique.column_label_dictionary_
        model_duplicates = DocVectorizer(
            token_contractor_kwds={"min_score": 25}, fit_unique=False
        ).fit(test_text_example)
        assert "foo_bar" in model_duplicates.column_label_dictionary_


@given(generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=DEFAULT_MAX_EXAMPLES,
)
@example((test_text_example, None))
def test_docvectorizer_vocabulary(test_text_info):
    test_text, vocabulary = test_text_info
    if test_text == test_text_example:
        vocab = ["foo", "bar"]
    else:
        vocab = list(set(test_text[0].split()))[:2]
        note(vocab)
    model = DocVectorizer(token_dictionary=vocab)
    results = model.fit_transform(test_text)
    assert results.shape == (len(test_text), 2)
    assert model.token_dictionary == vocab


@given(test_text_info=generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=DEFAULT_MAX_EXAMPLES,
)
@example(test_text_info=(test_text_example, None))
@pytest.mark.parametrize("tokenizer", ["nltk", "tweet", "sklearn"])
@pytest.mark.parametrize("token_contractor", ["aggressive", "conservative", None])
@pytest.mark.parametrize("vectorizer", ["bow", "bigram"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("fit_unique", [False])  # TODO: add True once code is fixed.
def test_docvectorizer_basic(
    tokenizer, token_contractor, vectorizer, normalize, fit_unique, test_text_info
):
    test_text, vocabulary = test_text_info
    model = DocVectorizer(
        tokenizer=tokenizer,
        token_contractor=token_contractor,
        vectorizer=vectorizer,
        normalize=normalize,
        fit_unique=fit_unique,
    )

    result = model.fit_transform(test_text)
    assert model.tokenizer_.tokenize_by == "document"
    transform = model.transform(test_text)
    assert np.allclose(result.toarray(), transform.toarray())
    if test_text == test_text_example:
        if vectorizer == "bow":
            assert result.shape == (5, 7)
        if vectorizer == "bigram":
            assert result.shape == (5, 19)
    else:
        assert result.shape[0] == len(test_text)
        if (token_contractor is None) and (vectorizer == "bow"):
            output_vocab = set(model.column_label_dictionary_.keys())
            lower_vocabulary = set([x.lower() for x in vocabulary] + [" "])
            note(output_vocab.difference(lower_vocabulary))
            assert output_vocab.issubset(lower_vocabulary)


# Should we also test for stanza?  Stanza's pytorch dependency makes this hard.
@given(test_text_info=generate_test_text_info())
@settings(
    deadline=None,
    suppress_health_check=[HealthCheck(3)],
    max_examples=min(50, DEFAULT_MAX_EXAMPLES),
)
@example(test_text_info=(test_text_example, None))
@pytest.mark.parametrize("tokenizer", ["nltk", "tweet", "sklearn"])
@pytest.mark.parametrize("token_contractor", ["aggressive", "conservative", None])
@pytest.mark.parametrize("vectorizer", ["before", "after", "symmetric", "directional"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dedupe_sentences", [True, False])
def test_wordvectorizer_basic(
    tokenizer, token_contractor, vectorizer, normalize, dedupe_sentences, test_text_info
):
    test_text, vocabulary = test_text_info
    model = WordVectorizer(
        tokenizer=tokenizer,
        token_contractor=token_contractor,
        vectorizer=vectorizer,
        normalize=normalize,
        dedupe_sentences=dedupe_sentences,
    )
    result = model.fit_transform(test_text)

    if test_text == test_text_example:
        if vectorizer in ["before", "after", "symmetric"]:
            assert result.shape == (7, 7)
        if vectorizer == "directional":
            assert result.shape == (7, 14)
    else:
        if token_contractor is None:
            output_vocab = set(
                [
                    x.lstrip("pre_").lstrip("post_")
                    for x in model.column_label_dictionary_.keys()
                ]
            )
            lower_vocabulary = set([x.lower() for x in vocabulary] + [" "])
            note(output_vocab.difference(lower_vocabulary))
            assert result.shape[0] <= len(lower_vocabulary)
            # assert output_vocab.issubset(lower_vocabulary)
    assert type(result) == scipy.sparse.csr.csr_matrix


def test_multitokencooccurrencevectorizer():
    model = WordVectorizer(
        vectorizer=MultiTokenCooccurrenceVectorizer,
        vectorizer_kwds=_MULTITOKEN_COOCCURRENCE_VECTORIZERS["flat_1_5"]["kwds"],
    ).fit(test_text_example)
    assert model.representation_.shape == (7, 28)
