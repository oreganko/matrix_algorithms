from statistics import mean
from time import process_time
import numpy as np


def matmul_3_loops(A: np.ndarray,
                   B: np.ndarray,
                   order="ijk") \
        -> np.ndarray:
    """
    IJK matrix multiplication, using dot product between rows of A and
    columns of B.
    :param A: matrix of shape (m, l)
    :param B: matrix of shape (l, n)
    :param order: order of multiplication loops
    :return: matrix of shape (m, n)
    """
    m = A.shape[0]
    l = A.shape[1]  # B.shape[0]
    n = B.shape[1]

    C = np.zeros((m, n))

    # i - rows of C
    # j - columns of C
    # k - vectors

    if order == "ijk":
        for i in range(m):
            for j in range(n):
                for k in range(l):
                    C[i, j] += A[i, k] * B[k, j]
    elif order == "ikj":
        for i in range(m):
            for k in range(l):
                # for j in range(n):
                #     C[i, j] += A[i, k] * B[k, j]
                C[i, :n] += A[i, k] * B[k, :n]
    elif order == "jik":
        for j in range(n):
            for i in range(m):
                for k in range(l):
                    C[i, j] += A[i, k] * B[k, j]
    elif order == "jki":
        for j in range(n):
            for k in range(l):
                # for i in range(m):
                #     C[i, j] += A[i, k] * B[k, j]
                C[:m, j] += A[:m, k] * B[k, j]
    elif order == "kij":
        for k in range(l):
            for i in range(m):
                # for j in range(n):
                #     C[i, j] += A[i, k] * B[k, j]
                C[i, :n] += A[i, k] * B[k, :n]
    elif order == "kji":
        for k in range(l):
            for j in range(n):
                # for i in range(m):
                #     C[i, j] += A[i, k] * B[k, j]
                C[:m, j] += B[:m, k] * B[k, j]

    return C


if __name__ == "__main__":
    for size in [10, 100, 1000]:
        print(size)
        times = {key: [] for key in ["ijk", "ikj", "jik", "jki", "kij", "kji"]}
        for order in ["ijk", "ikj", "jik", "jki", "kij", "kji"]:
            for _ in range(10):
                A = np.random.rand(size, size)
                B = np.random.rand(size, size)

                start = process_time()
                C = matmul_3_loops(A, B, order)
                end = process_time()

                ms = (end - start) * 1000
                times[order].append(ms)
        for order in ["ijk", "ikj", "jik", "jki", "kij", "kji"]:
            print("\t", order, mean(times[order]), "ms")


# loops case
# 10
# 	 ijk 0.5566166999999997 ms
# 	 ikj 0.5645484000000047 ms
# 	 jik 0.5697270000000004 ms
# 	 jki 0.5631753999999961 ms
# 	 kij 0.5576417000000056 ms
# 	 kji 0.5719293999999958 ms
# 100
# 	 ijk 557.2288085 ms
# 	 ikj 544.6575708000001 ms
# 	 jik 532.1933099 ms
# 	 jki 531.4345627999994 ms
# 	 kij 604.5244750999998 ms
# 	 kji 648.9184055999996 ms

# vector dot case
# 10
# 	 ijk 0.5665842000000004 ms
# 	 ikj 0.235041299999994 ms
# 	 jik 0.567125299999996 ms
# 	 jki 0.26234060000000226 ms
# 	 kij 0.22911290000000029 ms
# 	 kji 0.26269659999999806 ms
# 100
# 	 ijk 530.6047142000001 ms
# 	 ikj 23.281358499999882 ms
# 	 jik 531.2797496000003 ms
# 	 jki 28.210963800000144 ms
# 	 kij 23.23250909999981 ms
# 	 kji 27.843350999999927 ms
# 1000
# 	 ijk 615531.6932260001 ms
# 	 ikj 3314.9499971000296 ms
# 	 jik 566973.9197850002 ms
# 	 jki 5256.862550700316 ms
# 	 kij 3217.0629045998794 ms
# 	 kji 5250.734282600206 ms
