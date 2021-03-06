from typing import Tuple

import numpy as np


def LU_decomposition(A: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs LU decomposition on matrix on A with pivoting, saving results in
    a new matrix.
    :param A: square matrix of shape (n, n)
    :return: matrix with L and U matrices
    """
    A = A.astype(float)
    n = A.shape[0]
    L = np.zeros(A.shape)
    U = np.zeros(A.shape)
    for k in range(n):
        L[k, k] = 1
        L[k + 1:, k] = A[k + 1:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        A[k + 1:, k] = 0
        for j in range(k + 1, n):
            A[k + 1:, j] -= L[k + 1:, k] * U[k, j]
    return L, U


if __name__ == '__main__':
    A = np.array([[4, 3, 5],
                  [6, 3, 8],
                  [12, 3, 15]])

    L, U = LU_decomposition(A)

    print("LU decomposition")
    print("Generated:")
    print("L\n", L, "\nU\n", U)
