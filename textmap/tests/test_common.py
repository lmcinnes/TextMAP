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


def test_docvectorizer_basic():
    vectorizer = DocVectorizer()
    result = vectorizer.fit(test_text)


# Should we also test for stanza?  It's failing in Travis.
@pytest.mark.parametrize("tokenizer", ["nltk", "tweet", "spacy", "sklearn"])
@pytest.mark.parametrize("token_contractor", ["aggressive", "conservative"])
@pytest.mark.parametrize("vectorizer", ["flat", "flat_1_5"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dedupe_sentences", [True, False])
def test_wordvectorizer_basic(
    tokenizer, token_contractor, vectorizer, normalize, dedupe_sentences
):
    vectorizer = WordVectorizer(
        tokenizer=tokenizer,
        token_contractor=token_contractor,
        vectorizer=vectorizer,
        normalize=normalize,
        dedupe_sentences=dedupe_sentences,
    )
    result = vectorizer.fit_transform(test_text)
    if vectorizer == "flat":
        assert result.shape == (7, 14)
    if vectorizer == "flat_1_5":
        assert result.shape == (7, 28)
    assert type(result) == scipy.sparse.csr.csr_matrix
