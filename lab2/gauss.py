from typing import Tuple

import numpy as np

from lab2.pivoting import pivoting_Ab


def gaussian_elimination_gen_U_1(
        A: np.ndarray,
        b: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Gaussian elimination on matrices A and b row-wise.
    This version creates 1 on the diagonal of the U matrix generated
    by the elimination.
    :param A: square matrix of shape (n, n)
    :param b: vector of shape (n,)
    :return: matrix A and vector b after Gaussian elimination
    """
    A = A.astype(float)
    b = b.astype(float)
    n = A.shape[0]

    for k in range(n):
        Akk = A[k, k]
        A[k, k:] /= Akk
        b[k] /= Akk
        for j in range(k + 1, n):
            Ajk = A[j, k]
            A[j] -= A[k] * Ajk
            b[j] -= b[k] * Ajk

    return A, b


def gaussian_elimination_gen_U_det(
        A: np.ndarray,
        b: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Gaussian elimination on matrices A and b row-wise.
    This version creates elements on the diagonal of the U matrix that
    equal the determinant when multiplied.
    :param A: square matrix of shape (n, n)
    :param b: vector of shape (n,)
    :return: matrix A after Gaussian elimination
    """
    A = A.astype(float)
    b = b.astype(float)
    n = A.shape[0]

    for k in range(n):
        Akk = A[k, k]
        for j in range(k + 1, n):
            multiplier = A[j, k] / Akk
            A[j] -= A[k] * multiplier
            b[j] -= b[j] * multiplier

    return A, b


def gaussian_elimination_with_pivoting(
        A: np.ndarray,
        b: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Gaussian elimination on matrices A and b row-wise.
    This version creates elements on the diagonal of the U matrix that
    equal the determinant when multiplied.
    :param A: square matrix of shape (n, n)
    :param b: vector of shape (n,)
    :return: matrix A after Gaussian elimination
    """
    A = A.astype(float)
    b = b.astype(float)
    n = A.shape[0]

    for k in range(n):
        A, b = pivoting_Ab(A, b, k)
        Akk = A[k, k]
        for j in range(k + 1, n):
            multiplier = A[j, k]/Akk
            A[j] -= A[k] * multiplier
            b[j] -= b[k] * multiplier

    return A, b


if __name__ == '__main__':
    # Gaussian elimination, generating 1 on diagonal of U
    A = np.array([[3, 2, 1],
                  [2, 3, 1],
                  [1, 2, 3]])
    b = np.array([39, 34, 26])

    A, b = gaussian_elimination_gen_U_1(A, b)
    print("Gaussian elimination, generating 1 on diagonal")
    print("Generated:")
    print(A, "\n", b)
    print("Answer:")
    print(np.array([[1, 2/3, 1/3],
                    [0, 1, 0.2],
                    [0, 0, 1]]))

    print("\n\n")

    # Gaussian elimination, generating determinant on diagonal of U
    A = np.array([[2, -2, 1],
                  [0, 4, 1],
                  [1, 1, 3]])
    b = np.array([1, 1, 1])  # not used in the answer

    A, b = gaussian_elimination_gen_U_det(A, b)
    print("Gaussian elimination, generating determinant on diagonal")
    print("Generated:")
    print(A, "\n", b)
    print("Answer:")
    print(np.array([[2, -2, 1],
                    [0, 4, 1],
                    [0, 0, 2]]))

    print("\n\n")

    # Gaussian elimination with pivoting

    A = np.array([[6, 2, 2],
                  [6, 2, 1],
                  [1, 2, -1]])
    b = np.array([0, 5, 0])

    A, b = gaussian_elimination_with_pivoting(A, b)
    print("Gaussian elimination with pivoting")
    print("Generated:")
    print(A, "\n", b)
    print("Answer:")
    print(np.array([[6, 2, 2],
                    [0, 5/3, -4/3],
                    [0, 0, -1]]))
    print(np.array([0, 0, 5]))

    print("\n\n")