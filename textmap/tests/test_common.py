import pytest

from sklearn.utils.estimator_checks import check_estimator

from textmap import WordMAP
from textmap import DocMAP
from textmap import TopicMAP


@pytest.mark.parametrize(
    "Estimator", [WordMAP, DocMAP, TopicMAP]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
