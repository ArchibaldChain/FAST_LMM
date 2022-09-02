import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from numpy.linalg import det
from numpy.linalg import LinAlgError
import scipy.optimize as opt
import warnings
import time

from numpy.linalg import inv as inv_
from numpy.linalg import pinv

# import utils as u
try:
    import utils as u
except ImportError:
    print("import uitls from the calling time")


class FASTLMM:
    def __init__(self, lowRank=False, REML=False):
        print('------------- FAST-LMM------------------')
        if REML:
            print('LowRank is set as {}, using REML'.format(lowRank))
        else:
            print('LowRank is set as {}, not using REML'.format(lowRank))

        self.lowRank = lowRank
        self.REML = REML
        self.delta_temp = None

    def fit(self, X, y, W=None):

        start_time = time.time()
        self.X = np.array(X).astype('float64')
        self.y = np.array(y).reshape([-1, 1])
        self.W = W

        p = X.shape[1]

        # normally we calculate K =  W @ W.T
        # or K = 1/n *  X @ X.T
        # so we can set W = 1/ sqrt(d) * X
        if W is None:
            W = 1 / np.sqrt(p) * X.copy()

        n, sc = W.shape
        if (n != X.shape[0]) or (n !=
                                 self.y.shape[0]) or (self.y.shape[1]) > 1:
            raise ValueError(
                'Incompatible shape of X(shape of {}), y(shape of {}) and w(shape of {}).'
                .format(X.shape, self.y.shape, W.shape))

        K = W @ W.T
        self.K = K.copy()
        self.rank = matrix_rank(K)
        print('Rank of W is {}, shape of W is {}.'.format(self.rank, W.shape))

        if self.lowRank:
            if self.rank == max(n, sc):
                warnings.warn("W is set as lowRank, but actually not lowRank.")
                self.lowRank = False

            U, S, _ = svd(W, overwrite_a=True)
            U = U[:, :self.rank]
            S = S[:self.rank]
        else:
            if self.rank < max(n, sc):
                warnings.warn('W is set lowRank False, but actually lowRank.')

            # incase that set lowRank is False
            U, S, _ = svd(W, overwrite_a=True)
            # setting the last n - sc eigenvalues as 0
            S[self.rank:] = 0

        # check if S is a matrix
        if S.ndim > 1:
            print('Get a 2 dimensional S')
            print(S)
            S = np.diag(S)

        # in case that length of S is smaller than the columns of U
        # This only will happen when W is lowRank but set as not LowRank
        if len(S) < U.shape[1]:
            S = np.concatenate([S, np.zeros(U.shape[1] - len(S))])

        self.U = U
        self.S = S**2
        self._buffer_preCalculation()
        self._set_parameter()
        self.summary()
        print("------ %s seconds ------" % (time.time() - start_time))

    def summary(self):
        print('---------------Summary------------------')
        if self.REML:
            print('LowRank is set as {}, using REML'.format(self.lowRank))
        else:
            print('LowRank is set as {}, not using REML'.format(self.lowRank))
        print('Heritability h2:', 1 / (1 + self.delta))
        print('Sigma_g2:', self.sigma_g2)
        print('Sigma_e2:', self.sigma_e2)

    def predict(self, X_predict, W_predict=None):
        if W_predict is None:
            if self.W is not None:
                raise Exception('W must be the same form as training data.')

            p = X_predict.shape[1]
            K_te_tr = 1 / p * X_predict @ self.X
        else:
            K_te_tr = W_predict @ self.W.T

        V_inv = self.U / (self.S + self.delta) @ self.U.T
        if self.lowRank:
            V_inv += self.I_minus_UUT / self.delta
        V_inv = V_inv / self.sigma_g2

        u = self.sigma_g2 * K_te_tr @ V_inv @ (self.y - self.X @ self.beta)
        y_pred = X_predict @ self.beta + u
        return y_pred

    def V(self, W=None, sigma_g2=None, sigma_e2=None):
        '''
        get the Variance of Y-Xbeta
        parameters are set for new parameters calculation
        '''
        if sigma_g2 is None:
            sigma_g2 = self.sigma_g2
        if sigma_e2 is None:
            sigma_e2 = self.sigma_e2

        n = self.X.shape[0]
        if W is None:
            V = self.U * (sigma_g2 * self.S + sigma_e2) @ self.U.T
            if self.lowRank:
                V += self.I_minus_UUT * sigma_e2
        else:
            V = sigma_g2 * W @ W.T + sigma_e2 * np.identity(n)
        return V

    def V_inv(self, W=None, sigma_g2=None, sigma_e2=None):
        '''
        get the inverse of Variance of Y-Xbeta
        parameters are set for new parameters calculation
        '''
        if sigma_g2 is None:
            sigma_g2 = self.sigma_g2
        if sigma_e2 is None:
            sigma_e2 = self.sigma_e2
        delta = sigma_e2 / sigma_g2

        if W is None:
            V_inv = self.U / (self.S + delta) @ self.U.T
            if self.lowRank:
                V_inv += self.I_minus_UUT / delta
            V_inv = V_inv / sigma_g2
        else:
            V_inv = utils.inv(sigma_g2 * W @ W.T +
                              sigma_e2 * np.identity(W.shape[0]))
        return V_inv

    def _set_parameter(self):
        neg_LL = self._neg_cover()
        delta, funs = self._optimization(neg_LL)
        self.delta = delta
        print('Optimization Results:')
        print('Delta is calculated as: ', delta)
        if not self.REML:
            print('Maximum Likelihood is calculated as: ',
                  self._log_likelhood_delta(delta))
        else:
            print('Maximum REML is calculated as: ',
                  self._restricted_log_likelihood(delta))

        self.beta = self._beta(delta)
        self.sigma_g2 = self._sigma_g2(delta)
        self.sigma_e2 = self.sigma_g2 * delta

    def _buffer_preCalculation(self):
        n, _ = self.X.shape
        self.UTX = self.U.T @ self.X
        self.UTy = self.U.T @ self.y

        if self.lowRank:
            self.I_minus_UUT = np.identity(n) - self.U @ self.U.T
            self.I_minus_UUT_X = self.I_minus_UUT @ self.X
            self.I_minus_UUT_y = self.I_minus_UUT @ self.y
            self.I_UUTX_sq = self.I_minus_UUT_X.T @ self.I_minus_UUT_X
            self.I_UUTX_I_UUTy = self.I_minus_UUT_X.T @ self.I_minus_UUT_y

        if self.REML:
            self.log_XTX = np.log(det(self.X.T @ self.X))

    # delta_temp = None

    def _buffer_preCalculation_with_delta(self, delta):
        '''
        It is a pre-calculation of some matrix calculatons.
        When delta is given, some matrix calculations take place several time.
        This function is meant to calculate these pieces in advance to save some time.
        '''
        self.delta_temp = delta
        self.beta_delta = self._beta(delta)

        self.UTy_minus_UTXbeta = self.UTy - self.UTX @ self.beta_delta

        # already calculated in _beta
        # self.UTXT_inv_S_delta_UTX = \
        #     (self.UTX).T / (self.S + delta) @ (self.UTX)
        # self.UTXT_inv_S_delta_UTy = \
        #     (self.UTX).T / (self.S + delta) @ (self.UTy)

        if self.lowRank:
            self.I_UUTy_minus_I_UUTXbeta = self.I_minus_UUT_y - \
                self.I_minus_UUT_X @ self.beta_delta

    def _beta(self, delta):
        '''
        beta_function of delta
        '''
        self.UTXT_inv_S_delta_UTX = (self.UTX).T / (self.S +
                                                    delta) @ (self.UTX)
        self.UTXT_inv_S_delta_UTy = (self.UTX).T / (self.S +
                                                    delta) @ (self.UTy)

        if self.lowRank:
            inversepart = self.UTXT_inv_S_delta_UTX +\
                1/delta * self.I_minus_UUT_X.T @ self.I_minus_UUT_X
            beta = utils.inv(inversepart) @\
                (self.UTXT_inv_S_delta_UTy + 1/delta * self.I_UUTX_I_UUTy)

        else:
            inversepart = self.UTXT_inv_S_delta_UTX
            beta = utils.inv(inversepart) @ \
                self.UTXT_inv_S_delta_UTy

        return beta

    def _sigma_g2(self, delta):
        '''
        Sigma_g2 function of delta
        '''
        # Update buffer
        if delta != self.delta_temp:
            self._buffer_preCalculation_with_delta(delta)

        n, d = self.X.shape

        # the squeeze is making shape from (n,1) to (n,)
        sigma_g2 = 1/n * \
            np.sum(self.UTy_minus_UTXbeta.squeeze() ** 2/(self.S + delta))

        if self.lowRank:

            sigma_g2 += 1/n * 1/delta * \
                np.sum(self.I_UUTy_minus_I_UUTXbeta ** 2)

        # from formula in page 10, the sigma_g2 of REML is given by
        if self.REML:
            sigma_g2 = sigma_g2 * n / (n - d)

        return sigma_g2.squeeze()

    def _log_likelhood_delta(self, delta):
        '''
        log likehood function of delta
        '''

        # Update buffer
        if delta != self.delta_temp:
            self._buffer_preCalculation_with_delta(delta)

        n = self.X.shape[0]

        if self.lowRank:
            k = self.rank
            LL = -1 / 2 * (
                n * np.log(2 * np.pi) + np.sum(np.log(self.S + delta)) +
                (n - k) * np.log(delta) + n + n * np.log(
                    1 / n *
                    (np.sum(self.UTy_minus_UTXbeta.squeeze()**2 /
                            (self.S + delta)) + np.sum(
                                (self.I_UUTy_minus_I_UUTXbeta)**2) / delta)))
        else:
            LL = -1 / 2 * (n * np.log(2 * np.pi) + np.sum(
                np.log(self.S + delta)) + n + n * np.log(1 / n * np.sum(
                    (self.UTy_minus_UTXbeta.squeeze()**2) / (self.S + delta))))
        return LL.squeeze()

    def _restricted_log_likelihood(self, delta):
        '''
        restricted log likelihood function
        '''
        # Update buffer
        if delta != self.delta_temp:
            self._buffer_preCalculation_with_delta(delta)

        n, d = self.X.shape

        if self.lowRank:
            REMLL = self._log_likelhood_delta(delta) + \
                1/2 * (
                d * np.log(2*np.pi * self._sigma_g2(delta))  -
                np.log(
                    det(self.UTXT_inv_S_delta_UTX + self.I_UUTX_sq/delta)
                )
            )
        else:
            REMLL = self._log_likelhood_delta(delta) + \
                1/2 * (
                d * np.log(2*np.pi * self._sigma_g2(delta)) -
                np.log(
                    det(self.UTXT_inv_S_delta_UTX)
                )
            )

        if REMLL.shape == (1, 1):
            REMLL = REMLL.reshape((1, ))

        return REMLL

    def plot_likelihood(self, REML=True):
        deltas = np.logspace(-10, 10, 21)
        if REML and self.REML:
            LL = [self._restricted_log_likelihood(d) for d in deltas]
            yLabel = 'Restricted LL'
        else:
            LL = [self._log_likelhood_delta(d) for d in deltas]
            yLabel = 'Log-likelihood'

        x_ = np.log10(deltas)
        plt.plot(x_, LL)
        plt.xlabel('log(delta)')
        plt.ylabel(yLabel)
        plt.title('Lod-Likelihood(Restricted) of Delta')
        plt.show()

    def _neg_cover(self):
        if self.REML:

            def neg_LL(d):
                self._buffer_preCalculation_with_delta(d)
                return -self._restricted_log_likelihood(d)
        else:

            def neg_LL(d):
                self._buffer_preCalculation_with_delta(d)
                return -self._log_likelhood_delta(d)

        return neg_LL

    def _optimization(self, fun):
        # Using - 'brent' method for optimization

        deltas = np.logspace(-10, 10, 21)

        local_minimums = []
        minimum_values = []
        for i in range(len(deltas) - 1):
            # bracket = opt.bracket(fun, xa = deltas[i], xb = deltas[i+1])
            bounds = (deltas[i], deltas[i + 1])
            minimize_result = opt.minimize_scalar(fun,
                                                  bounds=bounds,
                                                  method='bounded')
            x = minimize_result.x
            funs = minimize_result.fun

            if (type(x) != np.ndarray):
                local_minimums.append(x)
            else:
                local_minimums += x.tolist()
            if (type(fun) != np.ndarray):
                minimum_values.append(funs)
            else:
                minimum_values += funs.tolist()

        min_value = min(minimum_values)
        # minmums = [local_minimums[i] for i, v in enumerate(minimum_values) if v == min_value]
        minmum = local_minimums[minimum_values.index(min_value)]
        return minmum, min_value


class utils:
    @staticmethod
    def issparse(m):
        return np.sum(m == 0) > (m.shape[0] * m.shape[1] / 2)

    @staticmethod
    def inv(matrix):
        try:
            inv_mat = inv_(matrix)
        except LinAlgError as lae:
            if str(lae) != "Singular matrix":
                print('shape is {}'.format(matrix.shape))
                raise lae

            print('Singluar Matrix')
            inv_mat = pinv(matrix)
        finally:
            return inv_mat
