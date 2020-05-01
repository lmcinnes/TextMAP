import pytest
from .test_common import (
    test_matrix,
    test_matrix_zero_column,
    test_matrix_zero_row,
)

from textmap.transformers import (
    RemoveEffectsTransformer,
    InformationWeightTransformer,
)
import numpy as np


# @pytest.mark.parametrize("n_components", [1, 2])
# @pytest.mark.parametrize("model_type", ["EnsTop"])
# @pytest.mark.parametrize("em_precision", [1e-4])
# @pytest.mark.parametrize("em_background_prior", [0.1])
# @pytest.mark.parametrize("em_threshold", [1e-4])
# @pytest.mark.parametrize("em_prior_strength", [0.05])
# @pytest.mark.parametrize("normalize", [False])
# def test_enstop_re_transformer(
#     n_components,
#     model_type,
#     em_precision,
#     em_background_prior,
#     em_threshold,
#     em_prior_strength,
#     normalize,
# ):
#     RET = RemoveEffectsTransformer(
#         n_components=n_components,
#         model_type=model_type,
#         em_precision=em_precision,
#         em_background_prior=em_background_prior,
#         em_threshold=em_threshold,
#         em_prior_strength=em_prior_strength,
#         normalize=normalize,
#     )
#     #result = RET.fit_transform(test_matrix, bootstrap=False)
#     result = RET.fit_transform(test_matrix)
#     transform = RET.transform(test_matrix)
#     assert np.allclose(result.toarray(), transform.toarray())
#
#     # RET.fit(test_matrix_zero_column)
#     # RET.transform(test_matrix_zero_column)
#     # RET.fit_transform(test_matrix_zero_column)
#     # RET.fit(test_matrix_zero_row, bootstrap=False)
#     # RET.transform(test_matrix_zero_row)
#     # RET.fit_transform(test_matrix_zero_row, bootstrap=False)

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
    result = RET.fit_transform(test_matrix)
    transform = RET.transform(test_matrix)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("model_type", ["pLSA"])
@pytest.mark.parametrize("em_precision", [1e-3, 1e-4])
@pytest.mark.parametrize("em_background_prior", [0.1, 10.0])
@pytest.mark.parametrize("em_threshold", [1e-4, 1e-5])
@pytest.mark.parametrize("em_prior_strength", [1.0, 10.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_re_transformer_zero_column(
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
    result = RET.fit_transform(test_matrix_zero_column)
    transform = RET.transform(test_matrix_zero_column)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("n_components", [1, 2])
@pytest.mark.parametrize("model_type", ["pLSA"])
@pytest.mark.parametrize("em_precision", [1e-3, 1e-4])
@pytest.mark.parametrize("em_background_prior", [0.1, 10.0])
@pytest.mark.parametrize("em_threshold", [1e-4, 1e-5])
@pytest.mark.parametrize("em_prior_strength", [1.0, 10.0])
@pytest.mark.parametrize("normalize", [True, False])
def test_re_transformer_zero_row(
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
    result = RET.fit_transform(test_matrix_zero_row)
    transform = RET.transform(test_matrix_zero_row)
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("n_components", [1, ])
@pytest.mark.parametrize("model_type", ["pLSA"])
@pytest.mark.parametrize("information_function", ['idf', 'average_idf', 'column_kl', 'bernoulli_kl'])
def test_iw_transformer(
        n_components, model_type, information_function
):
    IWT = InformationWeightTransformer(
        n_components=n_components, model_type=model_type, information_function=information_function
    )
    result = IWT.fit_transform(test_matrix)
    transform = IWT.transform(test_matrix)
    print(transform.toarray())
    print(result.toarray())
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("n_components", [1, ])
@pytest.mark.parametrize("model_type", ["pLSA"])
@pytest.mark.parametrize("information_function", ['idf', 'average_idf', 'column_kl', 'bernoulli_kl'])
def test_iw_transformer_zer_column(
        n_components, model_type, information_function
):
    IWT = InformationWeightTransformer(
        n_components=n_components, model_type=model_type, information_function=information_function
    )
    result = IWT.fit_transform(test_matrix_zero_column)
    transform = IWT.transform(test_matrix_zero_column)
    print(transform.toarray())
    print(result.toarray())
    assert np.allclose(result.toarray(), transform.toarray())


@pytest.mark.parametrize("n_components", [1, ])
@pytest.mark.parametrize("model_type", ["pLSA"])
@pytest.mark.parametrize("information_function", ['idf', 'average_idf', 'column_kl', 'bernoulli_kl'])
def test_iw_transformer_zer_row_plsa(
        n_components, model_type, information_function
):
    IWT = InformationWeightTransformer(
        n_components=n_components, model_type=model_type, information_function=information_function
    )
    result = IWT.fit_transform(test_matrix_zero_row)
    transform = IWT.transform(test_matrix_zero_row)
    print(transform.toarray())
    print(result.toarray())
    assert np.allclose(result.toarray(), transform.toarray())

# @pytest.mark.parametrize("n_components", [1, 2])
# @pytest.mark.parametrize("model_type", ["pLSA"])
# @pytest.mark.parametrize("information_function", ['idf', 'average_idf', 'column_kl', 'bernoulli_kl'])
# def test_iw_transformer_zer_row_enstop(
#         n_components, model_type, information_function
# ):
#     IWT = InformationWeightTransformer(
#         n_components=n_components, model_type=model_type, information_function=information_function
#     )
#     result = IWT.fit_transform(test_matrix_zero_row, bootstrap=False)
#     transform = IWT.transform(test_matrix_zero_row)
#     print(transform.toarray())
#     print(result.toarray())
#     assert np.allclose(result.toarray(), transform.toarray())
