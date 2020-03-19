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

        return self

    def transform(self, X, y=None):
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

        self.fit(X, **fit_params)
        result = info_weight_matrix(X, self.model_.embedding_, self.model_.components_)

        return result


@numba.njit()
def numba_multinomial_em_sparse(
        indptr,
        inds,
        data,
        background_i,
        background_j,
        precision=1e-4,
        low_thresh=1e-5,
        bg_prior=5.0,
):
    result = np.zeros(data.shape[0], dtype=np.float32)
    mix_weights = np.zeros(indptr.shape[0] - 1, dtype=np.float32)

    prior = np.array([1.0, bg_prior])
    mp = 1.0 + 1.0 * np.sum(prior)

    for i in range(indptr.shape[0] - 1):
        indices = inds[indptr[i]: indptr[i + 1]]
        row_data = data[indptr[i]: indptr[i + 1]]

        row_background = np.empty_like(row_data)
        for idx in range(indices.shape[0]):
            j = indices[idx]
            bg_val = 0.0
            for k in range(background_i.shape[1]):
                bg_val += background_i[i, k] * background_j[k, j]
            row_background[idx] = bg_val

        row_background = row_background / row_background.sum()

        mix_param = 0.5
        current_dist = mix_param * row_data + (1.0 - mix_param) * row_background

        last_estimated_dist = mix_param * current_dist + (1.0 - mix_param)

        change_vec = last_estimated_dist
        change_magnitude = 1.0 + precision

        while (
                change_magnitude > precision and mix_param > 1e-2 and mix_param < 1.0 - 1e-2
        ):

            posterior_dist = (current_dist * mix_param) / (
                    current_dist * mix_param + row_background * (1.0 - mix_param)
            )

            current_dist = posterior_dist * row_data
            mix_param = (current_dist.sum() + prior[0]) / mp
            current_dist = current_dist / current_dist.sum()

            estimated_dist = mix_param * current_dist + (1.0 - mix_param)
            change_vec = np.abs(estimated_dist - last_estimated_dist)
            change_vec /= estimated_dist
            change_magnitude = np.sum(change_vec)

            last_estimated_dist = estimated_dist

            # zero out any small values
            norm = 0.0
            for n in range(current_dist.shape[0]):
                if current_dist[n] < low_thresh:
                    current_dist[n] = 0.0
                else:
                    norm += current_dist[n]
            current_dist /= norm

        result[indptr[i]: indptr[i + 1]] = current_dist
        mix_weights[i] = mix_param

    return result, mix_weights


def multinomial_em_sparse(
        matrix, background_i, background_j, precision=1e-4, low_thresh=1e-5, bg_prior=5.0
):
    result = matrix.tocsr().copy().astype(np.float32)
    new_data, mix_weights = numba_multinomial_em_sparse(
        result.indptr,
        result.indices,
        result.data,
        background_i,
        background_j,
        precision,
        low_thresh,
        bg_prior,
    )
    result.data = new_data

    return result, mix_weights


class RemoveEffectsTransformer(BaseEstimator, TransformerMixin):
    """

       Parameters
       ----------
       n_components: int
           The target dimension for the low rank model (will be exact under pLSA or approximate under EnsTop).

       model_type: string
           The model used for the low-rank approximation.  To options are
           * 'pLSA'
           * 'EnsTop'

        optional EM params:
        * precision = 1e-4,
        * low_thresh = 1e-5,
        * bg_prior = 5.0,

       """

    def __init__(self, n_components=1, model_type="pLSA", **em_params):

        self.n_components = n_components
        self.model_type = model_type
        self.em_threshold = em_threshold
        self.em_background_prior = em_background_prior
        self.em_precision = em_precision

    def fit(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: sparse matrix of shape (n_docs, n_words)
            The data matrix to used to find the low-rank effects

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        self

        """
        if self.model_type == "pLSA":
            self.model_ = enstop.PLSA(n_components=self.n_components, **fit_params).fit(X)
        elif self.model_type == "EnsTop":
            self.model_ = enstop.EnsembleTopics(
                n_components=self.n_components, **fit_params).fit(X)
        else:
            raise ValueError("model_type is not supported")

        return self

    def transform(self, X, y=None):
        """

        X: sparse matrix of shape (n_docs, n_words)
            The data matrix that has the effects removed

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        X: scipy.sparse csr_matrix
            The matrix X with the low-rank effects removed.

        """

        check_is_fitted(self, ["model_"])
        embedding_ = self.model_.transform(X)
        result, weights = multinomial_em_sparse(X, embedding_, self.model_.components_, **self.em_params)
        self.mix_weights_ = weights

        return result

    def fit_transform(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: sparse matrix of shape (n_docs, n_words)
            The data matrix that is used to deduce the low-rank effects and then has them removed

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        X: scipy.sparse csr_matrix
            The matrix X with the low-rank effects removed.

        """

        self.fit(X, **fit_params)
        result, weights = multinomial_em_sparse(X, self.model_.embedding_, self.model_.components_, **self.em_params)
        self.mix_weights_ = weights
        return result
