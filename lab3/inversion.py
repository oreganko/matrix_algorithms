import numpy as np


def get_determinant(A: np.ndarray) -> float:
    """
    Calculates determinant of matrix A.
    :param A: matrix to calculate the determinant for
    :return: determinant of matrix A
    """
    if A.shape[0] == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]

    determinant = 0.0
    for i in range(A.shape[0]):
        determinant += ((-1) ** i) * A[0][i] * get_minor(A, 0, i)
    return determinant


def get_minor(A: np.ndarray, row: int, column: int) -> float:
    """
    Calculates minor of matrix A after removing given row and column.
    :param A: matrix to calculate the minor for
    :param column: index of column to remove from matrix
    :param row: index of row to remove from matrix
    :return: minor of matrix A
    """
    return get_determinant(get_minor_matrix(A, row, column))


def get_minor_matrix(A: np.ndarray, row: int, column: int) -> np.ndarray:
    """
    Calculates smaller matrix of matrix A after removing given row and column.
    :param A: matrix to remove row and columns from
    :param column: index of column to remove from matrix
    :param row: index of row to remove from matrix
    :return: smaller matrix from matrix A
    """
    B = A.copy()
    B = np.delete(B, row, axis=0)
    B = np.delete(B, column, axis=1)
    return B


def get_cofactors(A: np.ndarray) -> np.ndarray:
    """
    Calculates minor of matrix A.
    :param A: matrix to calculate the cofactors matrix for
    :return: cofactors matrix of matrix A
    """
    n = A.shape[0]
    cofactors = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cofactors[i][j] = ((-1) ** (i + j)) * get_minor(A, i, j)
    return cofactors


def transpose(A: np.ndarray) -> np.ndarray:
    """
    Transposes matrix A.
    :param A: matrix to transpose
    :return: transposed matrix A
    """
    n = A.shape[0]
    At = np.zeros((n, n))
    for i in range(n):
        At[i, ] = A[:, i]
    return At


def inverse_matrix(A: np.ndarray) -> np.ndarray:
    """
    Inverses matrix A.
    :param A: matrix to inverse
    :return: copied matrix A after inversion
    """
    A.astype(float)
    detA = get_determinant(A)
    return (1/detA) * transpose(get_cofactors(A))


if __name__ == '__main__':
    A = np.array([[4, 9, 2],
                  [3, 5, 7],
                  [8, 1, 6]])

    print("After my inversion:")
    print(inverse_matrix(A))
    print("After lib inversion:")
    print(np.linalg.inv(A))
