import numpy as np


def gram_schmidt_classic(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        a = A[:, i]
        v = A[:, i].copy()
        for j in range(i):
            R[j, i] = a @ Q[:, j]
            v -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    return Q, R


def gram_schmidt_modified(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        v = A[:, i].copy()
        for j in range(i):
            R[j, i] = v @ Q[:, j]
            v -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]
    return Q, R


def cholesky(A):
    L = np.zeros_like(A, dtype=float)
    for i in range(A.shape[0]):
        for j in range(i + 1):
            if i == j:
                temp = A[i, i] - L[i,:i] @ L[i,:i]
                if temp <= 0:
                    raise np.linalg.LinAlgError("A no es definida positiva")
                L[i, i] = np.sqrt(temp)
            else:
                L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L
