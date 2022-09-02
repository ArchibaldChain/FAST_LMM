from numpy.linalg import inv as inv_
from numpy.linalg import pinv
from numpy.linalg import LinAlgError
import numpy as np


def issparse(m):
    return np.sum(m == 0) > (m.shape[0] * m.shape[1] / 2)


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
