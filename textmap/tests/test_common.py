import pytest

from sklearn.utils.estimator_checks import check_estimator
import scipy.sparse

from textmap import WordMAP
from textmap import DocMAP
from textmap import TopicMAP

from textmap.vectorizers import DocVectorizer, WordVectorizer

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

@pytest.mark.parametrize("tokenizer", ["nltk", "tweet", "spacy", "sklearn"])
@pytest.mark.parametrize("token_contractor", ["aggressive", "conservative"])
@pytest.mark.parametrize("vectorizer", ["bow", "bigram"])
@pytest.mark.parametrize("normalize", [True, False])
def test_docvectorizer_basic(
        tokenizer, token_contractor, vectorizer, normalize
):
    model = DocVectorizer(
        tokenizer=tokenizer,
        token_contractor=token_contractor,
        vectorizer=vectorizer,
        normalize=normalize,
    )
    result = model.fit_transform(test_text)
    transform = model.transform(test_text)
    assert (result != transform).nnz == 0
    if vectorizer == 'bow':
        assert result.shape == (5, 7)
    if vectorizer == 'bigram':
        assert result.shape == (5, 19)


# Should we also test for stanza?  Stanza's pytorch dependency makes this hard.
@pytest.mark.parametrize("tokenizer", ["nltk", "tweet", "spacy", "sklearn"])
@pytest.mark.parametrize("token_contractor", ["aggressive", "conservative"])
@pytest.mark.parametrize("vectorizer", ["flat", "flat_1_5"])
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
    if vectorizer == "flat":
        assert result.shape == (7, 14)
    if vectorizer == "flat_1_5":
        assert result.shape == (7, 28)
    assert type(result) == scipy.sparse.csr.csr_matrix
