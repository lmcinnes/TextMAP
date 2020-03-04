import pytest

from sklearn.utils.estimator_checks import check_estimator

from textmap import TemplateEstimator
from textmap import TemplateClassifier
from textmap import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
