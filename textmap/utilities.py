import numpy as np
import numba
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import enstop


@numba.njit()
def numba_info_weight_matrix(row, col, val, frequencies_i, frequencies_j):
    n_rows = np.max(row)
    nnz = val.shape[0]

    expected_entropy = -np.log(n_rows / nnz)

    for idx in range(row.shape[0]):
        i = row[idx]
        j = col[idx]
        prob = 0.0
        for k in range(frequencies_i.shape[1]):
            prob += frequencies_i[i, k] * frequencies_j[k, j]

        if prob > 0.0:
            # I'm suspect of this... it does weird things to small examples
            # info_weight = max(0.0, -np.log(prob) - expected_entropy)  # * nnz / n_rows)

            info_weight = max(0.0, -np.log(prob))  # * nnz / n_rows)
        else:
            info_weight = -np.log(
                1.0 / n_rows
            )  # it's unclear to me why this is a good default value in this case

        val[idx] = val[idx] * info_weight

    return val


def info_weight_matrix(matrix, frequencies_i, frequencies_j):
    result = matrix.tocoo().copy()

    new_data = numba_info_weight_matrix(
        result.row, result.col, result.data, frequencies_i, frequencies_j
    )
    result.data = new_data

    return result.tocsr()


class InformationWeightTransformer(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    n_components: int
        The target dimension for the low rank model (will be exact under pLSA or approximate under EnsTop).

    model_type: string
        The model used for the low-rank approximation.  To options are
        * 'pLSA'
        * 'EnsTop'

    """

    def __init__(self, n_components=1, model_type="pLSA"):

        self.n_components = n_components
        self.model_type = model_type

    def fit(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: sparse matrix of shape (n_docs, n_words)
            The data matrix that get's binarized that the model is attempting to fit to.

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        self

        """
        if self.model_type == "pLSA":
            self.model_ = enstop.PLSA(n_components=self.n_components, **fit_params).fit(
                (X != 0).astype(np.float32)
            )
        elif self.model_type == "EnsTop":
            self.model_ = enstop.EnsembleTopics(
                n_components=self.n_components, **fit_params
            ).fit((X != 0).astype(np.float32))
        else:
            raise ValueError("model_type is not supported")

        self.model_.components_ /= self.model_.components_.sum(axis=1)[
            :, np.newaxis
        ]  # Hack to fix a bug in enstop

        return self

    def transform(self, X, y=None, **fit_params):
        """

        X: sparse matrix of shape (n_docs, n_words)
            The data matrix that gets rescaled by the information weighting

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        X: scipy.sparse csr_matrix
            The matrix X scaled by the relative log likelyhood of the entry being non-zero under the
            model vs the uniformly random distribution of values.


        """

        check_is_fitted(self, ["model_"])
        embedding_ = self.model_.transform((X != 0).astype(np.float32))
        result = info_weight_matrix(X, embedding_, self.model_.components_)

        return result

    def fit_transform(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: sparse matrix of shape (n_docs, n_words)
            The data matrix that get's rescaled by the information weighting and the matrix that gets
             binarized that the model fits to

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        X: scipy.sparse csr_matrix
            The matrix X scaled by the relative log likelyhood of the entry being non-zero under the
            model vs the uniformly random distribution of values.

        """

        self.fit(X)
        result = info_weight_matrix(X, self.model_.embedding_, self.model_.components_)

        return result


class RemoveEffectsTransformer(BaseEstimator, TransformerMixin):

    pass
