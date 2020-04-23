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
@pytest.mark.parametrize("information_function", ['idf', 'average idf', 'column KL', 'Bernoulli KL'])
def test_iw_transformer(
    n_components, model_type, information_function
):
    IWT = InformationWeightTransformer(
        n_components=n_components, model_type=model_type, information_function= information_function
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

