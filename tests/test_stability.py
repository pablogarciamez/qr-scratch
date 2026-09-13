from qr_scratch import gram_schmidt_classic, gram_schmidt_modified
import numpy as np

def lauchli(eps=1e-8):
    return np.array([
        [1.0, 1.0, 1.0],
        [eps, 0.0, 0.0],
        [0.0, eps, 0.0],
        [0.0, 0.0, eps],
    ])

def loss_of_orthogonality(Q):
    n = Q.shape[1]
    return np.linalg.norm(Q.T @ Q - np.eye(n))

def test_modified_is_more_stable_than_classic():
    Q_c, _ = gram_schmidt_classic(lauchli())
    Q_m, _ = gram_schmidt_modified(lauchli())

    err_c = loss_of_orthogonality(Q_c)
    err_m = loss_of_orthogonality(Q_m)

    assert err_c > 0.4
    assert err_m < 1e-7
    assert err_m > 1e-10