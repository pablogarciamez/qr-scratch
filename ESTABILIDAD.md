ESTABILIDAD NUMÉRICA: SVD, QR Y CHOLESKY

Pérdida de ortogonalidad en Gram-Schmidt modificado

El test test_stability.py aplica Gram-Schmidt modificado a la matriz de Läuchli, una matriz construida deliberadamente mal condicionada: sus columnas son casi paralelas, diferenciadas únicamente por un parámetro epsilon = 1e-8 introducido en su construcción. Analíticamente, las columnas de Q resultantes deberían ser exactamente ortogonales. Numéricamente, Q^T*Q se desvía de la identidad en 1.1547e-8.

Esa desviación no se explica por el error de redondeo de máquina por sí solo (epsilon_maq ≈ 1.1e-16 en doble precisión), ocho órdenes de magnitud menor. El factor de amplificación es el número de condición de la matriz, kappa(A) = sigma_max / sigma_min, que para Läuchli es del orden de 1e8. La cota teórica de pérdida de ortogonalidad para Gram-Schmidt modificado es del orden de epsilon_maq * kappa(A):

1.1e-16 * 1e8 ≈ 1.1e-8

consistente con el valor medido.

Comparación con Gram-Schmidt clásico

La cota equivalente para Gram-Schmidt clásico es del orden de sqrt(epsilon_maq) * kappa(A):

sqrt(1e-16) * 1e8 = 1e-8 * 1e8 = 1

Un resultado de orden 1 significa pérdida total de ortogonalidad: el método no es utilizable en este régimen de condicionamiento. La diferencia de ocho órdenes de magnitud frente al modificado es la razón práctica por la que el modificado es el estándar en producción y el clásico se considera solo de valor pedagógico.

El parámetro epsilon y el cuadrado en A^T*A

El valor epsilon = 1e-8 de Läuchli no es arbitrario: es aproximadamente sqrt(epsilon_maq). Esto importa en cualquier método que forme A^TA explícitamente (por ejemplo, ecuaciones normales en mínimos cuadrados): al elevar al cuadrado, epsilon^2 ≈ epsilon_maq, un valor que la aritmética de máquina ya no distingue de cero. La información que diferenciaba las columnas se pierde en la formación de A^TA, antes incluso de empezar a resolver el sistema. Es una razón adicional, más allá del coste, para preferir descomposiciones tipo QR sobre las ecuaciones normales en problemas mal condicionados.

Fallo de Cholesky en matrices no definidas positivas

Cholesky descompone A = L*L^T mediante una secuencia de raíces cuadradas de los pivotes internos. Si A no es definida positiva (autovalor negativo), algún pivote resulta negativo y la raíz no es real. Verificado con cholesky.py sobre una matriz simétrica no definida positiva: lanza numpy.linalg.LinAlgError("A no es definida positiva"), una excepción explícita definida en el propio código (no un fallo silencioso ni un error genérico de dominio matemático).