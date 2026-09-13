# qr-scratch

Implementaciones desde cero en NumPy de Gram-Schmidt clásico, Gram-Schmidt modificado y Cholesky.
QR descompone una matriz cualquiera en Q, una matriz ortonormal y R una matriz triangular superior que permite reconstruir la matriz original. Este paquete permite comparar dos métodos (Gram-Schmidt clásico y modificado) que aunque en aritmética exacta son equivalentes, el método clásico pierde ortogonalidad con matrices mal condicionadas mientras que el modificado la conserva mejor.
## Instalación

```bash
git clone https://github.com/pablogarciamez/qr-scratch.git
cd qr-scratch
pip install -e .
```

## Uso

```python
import numpy as np
from qr_scratch import gram_schmidt_modified

A = np.array([[3.0, 5.0],
              [4.0, 0.0]])

Q, R = gram_schmidt_modified(A)

print(Q @ R)      # reconstruye A
print(Q.T @ Q)    # ≈ identidad
```

## Tests

```bash
pytest
```
