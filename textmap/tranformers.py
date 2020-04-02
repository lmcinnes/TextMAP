import numpy as np
import numba
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import enstop
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import MWETokenizer
import re

EPS = 1e-11


@numba.njit()
def numba_info_weight_matrix(
    row, col, val, frequencies_i, frequencies_j, tokens_per_doc
):
    """

    For a given rank 1 model frequencies_j[k], the P(token_j in document) = P(token_j) * #(tokens per document[k]).
    The information weight Info_k(token_j) = -log_2(P(token_j in document)). For a given document_i described as a
    distribution of frequencies_i[i] over k latent rank 1 models frequencies_j, the information weight of the token_j in
    document_i is the expected information weight \sum_k frequencies[i,k] Info_k(token_j). In the case k=1 and
    frequencies_j is the distribution of unique tokens in documents, this is the idf weight -log_2(P(token_j in doc)).

    The function returns the coo matrix (row, col, val) scaled by the information weight as calculated above.

    """

    for idx in range(row.shape[0]):
        i = row[idx]
        j = col[idx]

        info_weight = EPS
        for k in range(frequencies_i.shape[1]):
            col_prob = frequencies_j[k, j] * tokens_per_doc[k]
            if col_prob > 0.0:
                info_weight += -frequencies_i[i, k] * np.log2(col_prob)

        val[idx] = val[idx] * info_weight

    return val


def info_weight_matrix(matrix, frequencies_i, frequencies_j, tokens_per_doc):
    result = matrix.tocoo().copy()

    new_data = numba_info_weight_matrix(
        result.row,
        result.col,
        result.data,
        frequencies_i,
        frequencies_j,
        tokens_per_doc,
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
        binary_indicator_matrix = (X != 0).astype(np.float32)
        if self.model_type == "pLSA":
            self.model_ = enstop.PLSA(n_components=self.n_components, **fit_params).fit(
                binary_indicator_matrix
            )
        elif self.model_type == "EnsTop":
            self.model_ = enstop.EnsembleTopics(
                n_components=self.n_components, **fit_params
            ).fit(binary_indicator_matrix)
        else:
            raise ValueError("model_type is not supported")
        token_counts = np.array(binary_indicator_matrix.sum(axis=1))
        self.tokens_per_doc_ = (
            token_counts.T.dot(self.model_.embedding_)[0]
            / self.model_.embedding_.shape[0]
        )

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
        result = info_weight_matrix(
            X, embedding_, self.model_.components_, self.tokens_per_doc_
        )

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
        result = info_weight_matrix(
            X, self.model_.embedding_, self.model_.components_, self.tokens_per_doc_
        )

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
        indices = inds[indptr[i] : indptr[i + 1]]
        row_data = data[indptr[i] : indptr[i + 1]]

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

        result[indptr[i] : indptr[i + 1]] = current_dist
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
        * em_precision = 1e-4,
        * em_threshold = 1e-5,
        * em_background_prior = 5.0,

       """

    def __init__(
        self,
        n_components=1,
        model_type="pLSA",
        em_precision=1.0e-4,
        em_background_prior=5.0,
        em_threshold=1.0e-5,
    ):

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
            self.model_ = enstop.PLSA(n_components=self.n_components, **fit_params).fit(
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
        embedding_ = self.model_.transform(X)

        result, weights = multinomial_em_sparse(
            X,
            embedding_,
            self.model_.components_,
            low_thresh=self.em_threshold,
            bg_prior=self.em_background_prior,
            precision=self.em_precision,
        )
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
        result, weights = multinomial_em_sparse(
            X,
            self.model_.embedding_,
            self.model_.components_,
            low_thresh=self.em_threshold,
            bg_prior=self.em_background_prior,
            precision=self.em_precision,
        )
        self.mix_weights_ = weights
        return result


class MWETransformer(BaseEstimator, TransformerMixin):
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

    include_regex = str (default = None)
        Only contra bigrams where both of the tokens fully match the regular expression

    exclude_regex = str (default = r\"\W+\")
        Do not contract bigrams when either of the tokens fully matches the regular expression

    """

    def __init__(
        self,
        score_function=BigramAssocMeasures.likelihood_ratio,
        max_iterations=2,
        min_score=2 ** 7,
        min_token_occurrences=None,
        max_token_occurrences=None,
        min_ngram_occurrences=None,
        include_regex = None,
        exclude_regex = r"\W+"
    ):

        self.score_function = score_function
        self.max_iterations = max_iterations
        self.min_score = min_score
        self.min_token_occurrences = min_token_occurrences
        self.max_token_occurrences = max_token_occurrences
        self.min_ngram_occurrences = min_ngram_occurrences
        self.include_regex = include_regex
        self.exclude_regex = exclude_regex
        self.mwes_ = list([])

    def fit(self, X, **fit_params):
        """
        Procedure to iteratively contract bigrams (up to max_collocation_iterations times)
        that score higher on the collocation_function than the min_collocation_score (and satisfy other
        criteria set out by the optional parameters).
        """
        for i in range(self.max_iterations):
            self.tokenization_ = X
            bigramer = BigramCollocationFinder.from_documents(self.tokenization_)
            if not self.include_regex == None:
                include_re_fn = lambda w: re.fullmatch(self.include_regex, w) == None
                bigramer.apply_word_filter(include_re_fn)
            if not self.exclude_regex == None:
                exclude_re_fn = lambda w: re.fullmatch(self.exclude_regex, w) != None
                bigramer.apply_word_filter(exclude_re_fn)
            if not self.min_token_occurrences == None:
                minfreq_fn = lambda w: bigramer.word_fd[w] < self.min_token_occurrences
                bigramer.apply_word_filter(minfreq_fn)
            if not self.max_token_occurrences == None:
                maxfreq_fn = lambda w: bigramer.word_fd[w] > self.max_token_occurrences
                bigramer.apply_word_filter(maxfreq_fn)
            if not self.min_ngram_occurrences == None:
                bigramer.apply_freq_filter(self.min_ngram_occurrences)
            self.mwes_.extend(
                list(bigramer.above_score(self.score_function, self.min_score))
            )
            if len(self.mwes_) == 0:
                break
            contracter = MWETokenizer(self.mwes_)
            self.tokenization_ = [
                contracter.tokenize(doc) for doc in self.tokenization_
            ]
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.tokenization_
