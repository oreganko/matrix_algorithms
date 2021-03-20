import numpy as np


def pivoting_A(A: np.ndarray,
               k: int) -> np.ndarray:
    """
    Performs pivoting on matrix A for k-th step. For row version pivot comes
    from k-th column (from k-th row downwards).
    :param A: matrix to find pivot for
    :param k: algorithm step, row / column for pivoting
    :return: matrix A after pivoting, i.e. exchanging rows for optimal
    (largest) pivot
    """
    A = A.copy()
    n = A.shape[0]

    max_i = k
    for i in range(k, n):
        if abs(A[i, k]) > abs(A[max_i, k]):
            max_i = i

    if max_i != k:
        A[[k, max_i], k:] = A[[max_i, k], k:]
    return A


def pivoting_Ab(A: np.ndarray,
                b: np.ndarray,
                k: int) -> np.ndarray:
    """
    Performs pivoting on matrix A for k-th step. For row version pivot comes
    from k-th column (from k-th row downwards).
    :param A: matrix to find pivot for
    :param b: right-hand vector
    :param k: algorithm step, row / column for pivoting
    :return: matrix A after pivoting, i.e. exchanging rows for optimal
    (largest) pivot and vector b also after pivoting
    """
    A = A.copy()
    b = b.copy()
    n = A.shape[0]

    max_i = k
    for i in range(k, n):
        if abs(A[i, k]) > abs(A[max_i, k]):
            max_i = i

    if max_i != k:
        A[[k, max_i], k:] = A[[max_i, k], k:]
        b[[k, max_i]] = b[[max_i, k]]
    return A, b
