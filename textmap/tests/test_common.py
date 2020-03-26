import pytest

from sklearn.utils.estimator_checks import check_estimator
import scipy.sparse

from textmap import WordMAP
from textmap import DocMAP
from textmap import TopicMAP

from textmap.vectorizers import DocVectorizer

import nltk
nltk.download('punkt')


# @pytest.mark.parametrize(
#     "Estimator", [WordMAP, DocMAP, TopicMAP]
# )
# def test_all_estimators(Estimator):
#     return check_estimator(Estimator)

test_text = [
    "foo bar pok wer pok pok foo bar wer qwe pok asd fgh",
    "fgh asd foo pok qwe pok wer pok foo bar pok pok wer",
    "pok wer pok qwe foo asd foo bar pok wer asd wer pok",
]

def test_docvectorizer_basic():
    vectorizer = DocVectorizer()
    result = vectorizer.fit(test_text)
    assert scipy.sparse.issparse(result)
    assert result.shape == (7, 14)