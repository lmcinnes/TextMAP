import pytest

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
)

import nltk

nltk.download("punkt")

# @pytest.mark.parametrize(
#     "Estimator", [WordMAP, DocMAP, TopicMAP]
# )
# def test_all_estimators(Estimator):
#     return check_estimator(Estimator)

test_text = [
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


# TODO: Add a set of tests for passing in instantiated classes

# TODO: Test that DocVectorizer transform preserves column order and size on new data

def test_joint_nobasistransformer():
    model = JointWordDocVectorizer(
        feature_basis_converter=None, token_contractor_kwds={"min_score": 8}
    )
    result = model.fit_transform(test_text)
    assert isinstance(result, scipy.sparse.csr_matrix)
    assert result.shape == (12, 7)


def test_joinworddocvectorizer_vocabulary():
    model = JointWordDocVectorizer(
        feature_basis_converter=None, token_dictionary=['foo', 'bar', 'pok'],
    )
    result = model.fit_transform(test_text)
    print(result)
    assert isinstance(result, scipy.sparse.csr_matrix)
    assert result.shape == (8, 3)

def test_jointworddocvectorizer():
    model = JointWordDocVectorizer(n_components=3)
    result = model.fit_transform(test_text)
    transform = model.transform(test_text)
    assert np.allclose(result, transform)
    assert result.shape == (12, 3)
    assert model.n_words_ == 7
    assert isinstance(result, np.ndarray)


def test_featurebasisconverter_tokenized():
    converter = FeatureBasisConverter(word_vectorizer="tokenized", n_components=3)
    converter.fit(test_text_token_data)
    doc_vectorizer = DocVectorizer(tokenizer=None, token_contractor=None)
    doc_rep = doc_vectorizer.fit_transform(test_text_token_data)
    new_rep = converter.change_basis(doc_rep, doc_vectorizer.column_index_dictionary_)
    assert new_rep.shape == (7, 3)

def test_wordvectorizer_todataframe():
    model = WordVectorizer().fit(test_text)
    df = model.to_DataFrame()
    assert df.shape == (7, 14)

def test_wordvectorizer_vocabulary():
    model = WordVectorizer(token_dictionary=['foo', 'bar']).fit(test_text)
    assert model.representation_.shape == (2, 4)

def test_docvectorizer_todataframe():
    model = DocVectorizer().fit(test_text)
    df = model.to_DataFrame()
    assert df.shape == (5, 7)

def test_docvectorizer_unique():
    with pytest.raises(ValueError):
        model_unique = DocVectorizer(token_contractor_kwds={"min_score": 25}, fit_unique=True).fit(test_text)
        assert 'foo_bar' not in model_unique.column_label_dictionary_
        model_duplicates = DocVectorizer(token_contractor_kwds={"min_score": 25}, fit_unique=False).fit(test_text)
        assert 'foo_bar' in model_duplicates.column_label_dictionary_

def test_docvectorizer_vocabulary():
    model = DocVectorizer(token_dictionary=['foo', 'bar'])
    results = model.fit_transform(test_text)
    assert results.shape == (5, 2)

@pytest.mark.parametrize("tokenizer", ["nltk", "tweet", "spacy", "sklearn"])
@pytest.mark.parametrize("token_contractor", ["aggressive", "conservative"])
@pytest.mark.parametrize("vectorizer", ["bow", "bigram"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("fit_unique", [False]) #TODO: add True once code is fixed.
def test_docvectorizer_basic(tokenizer, token_contractor, vectorizer, normalize, fit_unique):
    model = DocVectorizer(
        tokenizer=tokenizer,
        token_contractor=token_contractor,
        vectorizer=vectorizer,
        normalize=normalize,
        fit_unique=fit_unique
    )

    result = model.fit_transform(test_text)
    assert model.tokenizer_.tokenize_by == "document"
    transform = model.transform(test_text)
    assert np.allclose(result.toarray(), transform.toarray())
    if vectorizer == "bow":
        assert result.shape == (5, 7)
    if vectorizer == "bigram":
        assert result.shape == (5, 19)


# Should we also test for stanza?  Stanza's pytorch dependency makes this hard.
@pytest.mark.parametrize("tokenizer", ["nltk", "tweet", "spacy", "sklearn"])
@pytest.mark.parametrize("token_contractor", ["aggressive", "conservative"])
@pytest.mark.parametrize("vectorizer", ["before", "after", "symmetric", "directional"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dedupe_sentences", [True, False])
def test_wordvectorizer_basic(
        tokenizer, token_contractor, vectorizer, normalize, dedupe_sentences
):
    model = WordVectorizer(
        tokenizer=tokenizer,
        token_contractor=token_contractor,
        vectorizer=vectorizer,
        normalize=normalize,
        dedupe_sentences=dedupe_sentences,
    )
    result = model.fit_transform(test_text)

    if vectorizer in ["before", "after", "symmetric"]:
        assert result.shape == (7, 7)
    if vectorizer == "directional":
        assert result.shape == (7, 14)
    assert type(result) == scipy.sparse.csr.csr_matrix
