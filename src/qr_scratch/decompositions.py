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
    """Cholesky de una matriz simétrica definida positiva. Devuelve L con A = L @ L.T."""
    raise NotImplementedError
