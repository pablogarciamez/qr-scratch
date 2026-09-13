import numpy as np
import pytest

from qr_scratch import gram_schmidt_classic, gram_schmidt_modified, cholesky


@pytest.mark.parametrize("qr", [gram_schmidt_classic, gram_schmidt_modified])
def test_qr_reconstruye_A(qr):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((6, 4))
    Q, R = qr(A)
    assert np.allclose(Q @ R, A)


@pytest.mark.parametrize("qr", [gram_schmidt_classic, gram_schmidt_modified])
def test_R_es_triangular_superior(qr):
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 4))
    _, R = qr(A)
    assert np.allclose(R, np.triu(R))


@pytest.mark.parametrize("qr", [gram_schmidt_classic, gram_schmidt_modified])
def test_Q_es_ortonormal_caso_facil(qr):
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 4))
    Q, _ = qr(A)
    assert np.allclose(Q.T @ Q, np.eye(4))


def test_cholesky_reconstruye_A():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((5, 5))
    A = M @ M.T + 5 * np.eye(5)   # simétrica y definida positiva por construcción
    L = cholesky(A)
    assert np.allclose(L @ L.T, A)
    assert np.allclose(L, np.tril(L))
