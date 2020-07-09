import numpy as np
import numba
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import normalize
import scipy.sparse
import enstop
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import MWETokenizer
import re
from warnings import warn

EPS = 1e-11


@numba.njit()
def fuzz01(val):
    if val >= 1.0:
        return 1.0 - EPS
    elif val <= 0.0:
        return EPS
    return val


@numba.njit()
def idf_avg_weight(
    row, col, val, frequencies_i, frequencies_j, document_lengths, token_counts
):
    """

    For a given rank 1 model frequencies_j[k], the P(token_j in document) = P(token_j) * E(tokens per document[k]).
    The idf information weight Info_k(token_j) = -log_2(P(token_j in document)). For a given document_i described as a
    distribution of frequencies_i[i] over k latent models frequencies_j[k], the information weight of the
    token_j in document_i is the information of the weighted sum of probabilities

    Info_k(token_j) = -log_2(\sum_k frequencies[i,k] P(token_j in document|k))
                    = -log_2(\sum_k frequencies[i,k] P(token_j|k) * E(tokens per document|k)))

    In the case k=1 and frequencies_j is the distribution of unique tokens in documents, this is the
    idf weight -log_2(P(token_j in document)).

    The function returns the vals of a coo matrix (row, col, val) scaled by the information weight as calculated above.

    """

    expected_tokens_per_doc = (
        np.dot(document_lengths, frequencies_i) / frequencies_i.shape[0]
    )

    for idx in range(row.shape[0]):
        i = row[idx]
        j = col[idx]

        info_weight = 0.0
        for k in range(frequencies_i.shape[1]):
            col_prob = frequencies_j[k, j] * expected_tokens_per_doc[k]
            info_weight += frequencies_i[i, k] * col_prob
        val[idx] = val[idx] * -np.log2(fuzz01(info_weight))

    return val


@numba.njit()
def avg_idf_weight(
    row, col, val, frequencies_i, frequencies_j, document_lengths, token_counts
):
    """

    For a given rank 1 model frequencies_j[k], the P(token_j in document) = P(token_j) * #(tokens per document[k]).
    The information weight Info_k(token_j) = -log_2(P(token_j in document)). For a given document_i described as a
    distribution of frequencies_i[i] over k latent rank 1 models frequencies_j, the information weight of the token_j in
    document_i is the expected information weight \sum_k frequencies[i,k] Info_k(token_j). In the case k=1 and
    frequencies_j is the distribution of unique tokens in documents, this is the idf weight -log_2(P(token_j in doc)).

    The function returns the vals of a coo matrix (row, col, val) scaled by the information weight as calculated above.

    """

    expected_tokens_per_doc = (
        np.dot(document_lengths, frequencies_i) / frequencies_i.shape[0]
    )

    for idx in range(row.shape[0]):
        i = row[idx]
        j = col[idx]

        info_weight = EPS
        for k in range(frequencies_i.shape[1]):
            col_prob = fuzz01(frequencies_j[k, j] * expected_tokens_per_doc[k])
            info_weight += fuzz01(frequencies_i[i, k]) * -np.log2(col_prob)

        val[idx] = val[idx] * info_weight

    return val


@numba.njit()
def column_kl_divergence_weight(
    row, col, val, frequencies_i, frequencies_j, document_lengths, token_counts
):
    """

    For a given latent topic model as a prior, we compute the matrix reconstruction
    (token_counts*frequencies_i).dot(frequencies_j) as a null.  For a given column j, this function computes the
    KL-divergence between the null model (from the latent topic model reconstruction) and the actual column
    distribution and records this as the information weight.

    The function returns the vals of a coo matrix (row, col, val) scaled by the information weight as calculated above.

    """

    model_token_sum = (document_lengths.dot(frequencies_i)).dot(frequencies_j)

    kl = np.zeros(frequencies_j.shape[1]).astype(np.float32)
    for idx in range(row.shape[0]):
        i = row[idx]
        j = col[idx]
        model_prob = 0.0
        for k in range(frequencies_i.shape[1]):
            model_prob += (
                frequencies_j[k, j] * frequencies_i[i, k] * document_lengths[i]
            )
        model_prob = fuzz01(model_prob / model_token_sum[j])
        actual_prob = fuzz01(val[idx] / token_counts[j])

        kl[j] += actual_prob * np.log2(actual_prob / model_prob)

    for idx in range(row.shape[0]):
        i = row[idx]
        j = col[idx]
        val[idx] = val[idx] * kl[j]

    return val


@numba.njit()
def bernoulli_kl_divergence_weight(
    row, col, val, frequencies_i, frequencies_j, document_lengths, token_counts
):
    """

    For a given latent topic model as a prior, we compute the matrix reconstruction (frequencies_i).dot(frequencies_j)
    as a null for each document as multinomial distribution. For a given entry i, j, this function computes the
    KL-divergence between  the null model for that entry (from the latent topic model reconstruction) and the actual
    probability in the row-normalized data, when viewed as a Bernoulli trial, i.e.

       KL(p,q) = p log_2(p/q) + (1-p) log ((1-p)/(1-q))

    and records this as the information weight for that entry.

    The function returns the vals of a coo matrix (row, col, val) scaled by the information weight as calculated above.

    """

    for idx in range(row.shape[0]):
        i = row[idx]
        j = col[idx]

        model_prob = 0.0
        for k in range(frequencies_i.shape[1]):
            model_prob += frequencies_j[k, j] * frequencies_i[i, k]

        actual_prob = fuzz01(val[idx] / document_lengths[i])
        model_prob = fuzz01(model_prob)

        kl = actual_prob * np.log2(actual_prob / model_prob) + (
            1 - actual_prob
        ) * np.log2((1 - actual_prob) / (1 - model_prob))

        val[idx] = val[idx] * kl

    return val


_INFORMATION_FUNCTIONS = {
    "average_idf": avg_idf_weight,
    "idf": idf_avg_weight,
    "column_kl": column_kl_divergence_weight,
    "bernoulli_kl": bernoulli_kl_divergence_weight,
}


def info_weight_matrix(
    info_function, matrix, frequencies_i, frequencies_j, document_lengths, token_counts
):
    if scipy.sparse.isspmatrix_coo(matrix):
        matrix = matrix.copy().astype(np.float32)
    else:
        matrix = matrix.tocoo().astype(np.float32)

    new_data = info_function(
        matrix.row,
        matrix.col,
        matrix.data,
        frequencies_i,
        frequencies_j,
        document_lengths,
        token_counts,
    )
    matrix.data = new_data
    matrix.eliminate_zeros()
    return matrix.tocsr()

class SingleComponentModel(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        self.embedding_ = np.ones((X.shape[0], 1), dtype=np.float32)
        if scipy.sparse.issparse(X):
            self.components_ = np.array(X.sum(axis=0), dtype=np.float32)
        else:
            self.components_ = X.sum(axis=0)[:, np.newaxis]
        self.components_ /= self.components_.sum()

        return self

    def transform(self, X, y=None):
        return np.ones((X.shape[0], 1), dtype=np.float32)

class InformationWeightTransformer(BaseEstimator, TransformerMixin):
    """

    Parameters
    ----------
    n_components: int
        The target dimension for the low rank model (will be exact under pLSA or approximate under EnsTop).

    model_type: string
        The model used for the low-rank approximation.  To options are
        * 'pLSA' (default)
        * 'EnsTop'

    information_function: callable or str
        Either a numba.jit function that takes in coo data, model frequencies, and row_sums or a string that calls
        a predefined option.  The string options are
        * 'column_kl' (default)
        * 'idf'
        * 'average_idf'
        * 'bernoulli_kl'

    binarize_matrix: bool (optional)
        If the information function is callable, this can be set to fit the model on the binarized matrix or the count
        matrix.  If the information function is a string, this is set internally depending on the function choice.
    """

    def __init__(
        self,
        n_components=1,
        model_type="pLSA",
        information_function="column_kl",
        binarize_matrix=None,
    ):

        self.n_components = n_components
        self.model_type = model_type
        self.information_function = information_function
        self.binarize_matrix = binarize_matrix

    def fit(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: sparse matrix of shape (n_docs, n_words)
            The data matrix (that potentially get's binarized) that the model is attempting to fit to.

        y: Ignored

        fit_params:
            optional model params

        Returns
        -------
        self

        """

        if callable(self.information_function):
            self._information_function = self.information_function
        elif self.information_function in _INFORMATION_FUNCTIONS:
            self._information_function = _INFORMATION_FUNCTIONS[
                self.information_function
            ]
        else:
            raise ValueError(
                f"Unrecognized kernel_function; should be callable or one of {_INFORMATION_FUNCTIONS.keys()}"
            )

        if self.binarize_matrix == None:
            if self.information_function in ["idf", "average idf"]:
                self.binarize_matrix = True
            elif self.information_function in ["column KL", "Bernoulli KL"]:
                self.binarize_matrix = False

        if self.binarize_matrix:
            binary_indicator_matrix = (X != 0).astype(np.float32)
            if self.n_components == 1 and self.model_type == "pLSA":
                self.model_ = SingleComponentModel().fit(binary_indicator_matrix)
            elif self.model_type == "pLSA":
                self.model_ = enstop.StreamedPLSA(
                    n_components=self.n_components, **fit_params
                ).fit(binary_indicator_matrix)
            elif self.model_type == "EnsTop":
                self.model_ = enstop.EnsembleTopics(
                    n_components=self.n_components, **fit_params
                ).fit(binary_indicator_matrix)
            else:
                raise ValueError("model_type is not supported")

        else:
            if self.n_components == 1 and self.model_type == "pLSA":
                self.model_ = SingleComponentModel().fit(X)
            elif self.model_type == "pLSA":
                self.model_ = enstop.PLSA(
                    n_components=self.n_components, **fit_params
                ).fit(X.astype(np.float32))
            elif self.model_type == "EnsTop":
                self.model_ = enstop.EnsembleTopics(
                    n_components=self.n_components, **fit_params
                ).fit(X.astype(np.float32))
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

        if self.binarize_matrix:
            for_transform = (X != 0).astype(np.float32)
        else:
            for_transform = X.astype(np.float32)

        document_lengths = np.array(for_transform.sum(axis=1, dtype=np.float32)).T[0]
        token_counts = np.array(for_transform.sum(axis=0), dtype=np.float32)[0]

        result = info_weight_matrix(
            self._information_function,
            X,
            self.model_.transform(for_transform),
            self.model_.components_,
            document_lengths,
            token_counts,
        )
        result.eliminate_zeros()

        return result

    def fit_transform(self, X, y=None, **fit_params):
        """

        Parameters
        ----------
        X: sparse matrix of shape (n_docs, n_words)
            The data matrix that get's rescaled by the information weighting and the matrix that (potentially gets
             binarized and) the model fits to

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

        if self.binarize_matrix:
            for_transform = (X != 0).astype(np.float32)
        else:
            for_transform = X.astype(np.float32)

        document_lengths = np.array(for_transform.sum(axis=1), dtype=np.float32).T[0]
        token_counts = np.array(for_transform.sum(axis=0), dtype=np.float32)[0]

        result = info_weight_matrix(
            self._information_function,
            X,
            self.model_.transform(for_transform),
            self.model_.components_,
            document_lengths,
            token_counts,
        )
        result.eliminate_zeros()

        return result


@numba.njit()
def numba_multinomial_em_sparse(
    indptr,
    inds,
    data,
    background_i,
    background_j,
    precision=1e-7,
    low_thresh=1e-5,
    bg_prior=5.0,
    prior_strength=0.3,
):
    result = np.zeros(data.shape[0], dtype=np.float32)
    mix_weights = np.zeros(indptr.shape[0] - 1, dtype=np.float32)

    prior = np.array([1.0, bg_prior]) * prior_strength
    mp = 1.0 + 1.0 * np.sum(prior)

    for i in range(indptr.shape[0] - 1):
        indices = inds[indptr[i] : indptr[i + 1]]
        row_data = data[indptr[i] : indptr[i + 1]]

        row_background = np.zeros_like(row_data)
        for idx in range(indices.shape[0]):
            j = indices[idx]
            bg_val = 0.0
            for k in range(background_i.shape[1]):
                bg_val += background_i[i, k] * background_j[k, j]
            row_background[idx] = bg_val

        row_background = row_background / row_background.sum()

        mix_param = 0.5
        current_dist = mix_param * row_data + (1.0 - mix_param) * row_background

        last_mix_param = mix_param
        change_magnitude = 1.0

        while (
            change_magnitude > precision
            and mix_param > precision
            and mix_param < 1.0 - precision
        ):

            posterior_dist = current_dist * mix_param
            posterior_dist /= current_dist * mix_param + row_background * (
                1.0 - mix_param
            )

            current_dist = posterior_dist * row_data
            mix_param = (current_dist.sum() + prior[0]) / mp
            current_dist = current_dist / current_dist.sum()

            change_magnitude = np.abs(mix_param - last_mix_param)
            last_mix_param = mix_param

        # zero out any small values
        norm = 0.0
        for n in range(current_dist.shape[0]):
            if current_dist[n] < low_thresh:
                current_dist[n] = 0.0
            else:
                norm += current_dist[n]
        current_dist /= norm

        result[indptr[i] : indptr[i + 1]] = current_dist
        mix_weights[i] = mix_param

    return result, mix_weights


def multinomial_em_sparse(
    matrix,
    background_i,
    background_j,
    precision=1e-7,
    low_thresh=1e-5,
    bg_prior=5.0,
    prior_strength=0.3,
):
    if scipy.sparse.isspmatrix_csr(matrix):
        result = matrix.copy().astype(np.float32)
    else:
        result = matrix.tocsr().astype(np.float32)
    new_data, mix_weights = numba_multinomial_em_sparse(
        result.indptr,
        result.indices,
        result.data,
        background_i,
        background_j,
        precision,
        low_thresh,
        bg_prior,
        prior_strength,
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

        normalize = False
            Return the modified count matrix (default) or the L_1 normalization of each row.

        optional EM params:
        * em_precision = 1e-7, (halt EM when the mix_param changes less than this)
        * em_threshold = 1e-5, (set to zero any values below this)
        * em_background_prior = 5.0, (a non-negative number)
        * em_prior_strength = 0.3 (a non-negative number)

       """

    def __init__(
        self,
        n_components=1,
        model_type="pLSA",
        em_precision=1.0e-7,
        em_background_prior=1.0,
        em_threshold=1.0e-8,
        em_prior_strength=0.5,
        normalize=False,
    ):

        self.n_components = n_components
        self.model_type = model_type
        self.em_threshold = em_threshold
        self.em_background_prior = em_background_prior
        self.em_precision = em_precision
        self.em_prior_strength = em_prior_strength
        self.normalize = normalize

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
        if self.n_components == 1 and self.model_type == "pLSA":
            self.model_ = SingleComponentModel().fit(X)
        elif self.model_type == "pLSA":
            self.model_ = enstop.StreamedPLSA(n_components=self.n_components, **fit_params).fit(
                X
            )
        elif self.model_type == "EnsTop":
            self.model_ = enstop.EnsembleTopics(
                n_components=self.n_components, **fit_params
            ).fit(X)
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
        if self.model_type == "enstop" and self.model_.n_components_ == 0:
            if self.normalize:
                return normalize(X, norm="l1")
            else:
                return X
        row_sums = np.array(X.sum(axis=1)).T[0]
        embedding_ = self.model_.transform(X.astype(np.float32))

        result, weights = multinomial_em_sparse(
            normalize(X, norm="l1"),
            embedding_,
            self.model_.components_,
            low_thresh=self.em_threshold,
            bg_prior=self.em_background_prior,
            precision=self.em_precision,
            prior_strength=self.em_prior_strength,
        )
        self.mix_weights_ = weights
        if not self.normalize:
            result = scipy.sparse.diags(row_sums * weights) * result

        result.eliminate_zeros()

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
        return self.transform(X)


class MultiTokenExpressionTransformer(BaseEstimator, TransformerMixin):
    """
    The transformer takes sequences of tokens and contracts bigrams meeting certain criteria set out by the parameters.
    This is repeated max_iterations times on the previously contracted text to (potentially) contract higher ngrams.

    Parameters
    ----------
    score_function = nltk.BigramAssocMeasures function (default = likelihood ratio)
        The function to score the bigrams of tokens.

    max_iterations = int (default = 2)
        The maximum number of times to iteratively contact bigrams or tokens.

    min_score = int (default = 128)
        The minimal score to contract an ngram.

    min_word_occurrences = int (default = None)
        If not None, the minimal number of occurrences of a token to be in a contracted ngram.

    max_word_occurrences = int (default = None)
        If not None, the minimal number of occurrences of a token to be in a contracted ngram.

    min_ngram_occurrences = int (default = None)
        If not None, the minimal number of occurrences of an ngram to be contracted.

    ignored_tokens = set (default = None)
        Only contracts bigrams where both tokens are not in the ignored_tokens

    excluded_token_regex = str (default = r\"\W+\")
        Do not contract bigrams when either of the tokens fully matches the regular expression via re.fullmatch

    """

    def __init__(
        self,
        score_function=BigramAssocMeasures.likelihood_ratio,
        max_iterations=2,
        min_score=2 ** 7,
        min_token_occurrences=None,
        max_token_occurrences=None,
        min_token_frequency=None,
        max_token_frequency=None,
        min_ngram_occurrences=None,
        ignored_tokens=None,
        excluded_token_regex=r"\W+",
    ):

        self.score_function = score_function
        self.max_iterations = max_iterations
        self.min_score = min_score
        self.min_token_occurrences = min_token_occurrences
        self.max_token_occurrences = max_token_occurrences
        self.min_token_frequency = min_token_frequency
        self.max_token_frequency = max_token_frequency
        self.min_ngram_occurrences = min_ngram_occurrences
        self.ignored_tokens = ignored_tokens
        self.excluded_token_regex = excluded_token_regex
        self.mtes_ = list([])

    def fit(self, X, **fit_params):
        """
        Procedure to iteratively contract bigrams (up to max_collocation_iterations times)
        that score higher on the collocation_function than the min_collocation_score (and satisfy other
        criteria set out by the optional parameters).
        """
        self.tokenization_ = X
        n_tokens = sum([len(x) for x in X])
        for i in range(self.max_iterations):
            bigramer = BigramCollocationFinder.from_documents(self.tokenization_)

            if not self.ignored_tokens == None:
                ignore_fn = lambda w: w in self.ignored_tokens
                bigramer.apply_word_filter(ignore_fn)

            if not self.excluded_token_regex == None:
                exclude_fn = (
                    lambda w: re.fullmatch(self.excluded_token_regex, w) is not None
                )
                bigramer.apply_word_filter(exclude_fn)

            if not self.min_token_occurrences == None:
                minocc_fn = lambda w: bigramer.word_fd[w] < self.min_token_occurrences
                bigramer.apply_word_filter(minocc_fn)

            if not self.max_token_occurrences == None:
                maxocc_fn = lambda w: bigramer.word_fd[w] > self.max_token_occurrences
                bigramer.apply_word_filter(maxocc_fn)

            if not self.min_token_frequency == None:
                minfreq_fn = (
                    lambda w: bigramer.word_fd[w] < self.min_token_frequency * n_tokens
                )
                bigramer.apply_word_filter(minfreq_fn)

            if not self.max_token_frequency == None:
                maxfreq_fn = (
                    lambda w: bigramer.word_fd[w] > self.max_token_frequency * n_tokens
                )
                bigramer.apply_word_filter(maxfreq_fn)

            if not self.min_ngram_occurrences == None:
                bigramer.apply_freq_filter(self.min_ngram_occurrences)

            new_grams = list(bigramer.above_score(self.score_function, self.min_score))

            if len(new_grams) == 0:
                break

            self.mtes_.append(new_grams)

            contracter = MWETokenizer(new_grams)
            self.tokenization_ = tuple(
                [tuple(contracter.tokenize(doc)) for doc in self.tokenization_]
            )

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.tokenization_

    def transform(self, X, y=None):
        result = X
        for i in range(len(self.mtes_)):
            contracter = MWETokenizer(self.mtes_[i])
            result = tuple([tuple(contracter.tokenize(doc)) for doc in result])
        return result
