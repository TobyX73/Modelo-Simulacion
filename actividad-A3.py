# EJERCICIO A.3 — Test Kolmogorov-Smirnov
import math

def lcg(a, c, m, seed, n):
    x = seed
    uniformes = []
    for _ in range(n):
        x = (a * x + c) % m
        uniformes.append(x / m)
    return uniformes

# Parámetros del grupo (los mismos de A.1)
a, c, m, seed = 1_738_019_701, 2_531_011, 2**32, 2026
numeros = lcg(a, c, m, seed, 1000)
N = len(numeros)

# Paso 1: ordenar de menor a mayor
ordenados = sorted(numeros)

# Paso 2: calcular D máximo punto a punto
d_max = 0
idx_dmax = 0

for i, xi in enumerate(ordenados, 1):
    fn      = i / N           # FDA empírica: fracción de datos hasta xi
    fn_prev = (i - 1) / N    # FDA empírica antes del salto
    f0      = xi              # FDA teórica U(0,1): igual al valor xi
    dp      = abs(fn - f0)   # diferencia por derecha
    dm      = abs(f0 - fn_prev)  # diferencia por izquierda
    d       = max(dp, dm)

    if d > d_max:
        d_max = d
        idx_dmax = i

# Paso 3: valor crítico y decisión
d_critico = 1.36 / math.sqrt(N)

print("=" * 50)
print("EJERCICIO A.3 — TEST KOLMOGOROV-SMIRNOV")
print("=" * 50)
print(f"N              = {N}")
print(f"α              = 0.05")
print(f"D calculado    = {d_max:.6f}  (en posición i={idx_dmax})")
print(f"D crítico      = 1.36/√{N} = {d_critico:.6f}")
print()
print("DECISIÓN")
print("-" * 50)
print(f"{d_max:.6f} {'>' if d_max > d_critico else '≤'} {d_critico:.6f}")
if d_max > d_critico:
    print("→ Se rechaza H₀")
    print("→ El generador FALLA el test K-S ✗")
    print(f"→ Margen: {abs(d_max - d_critico):.6f} ({abs(d_max-d_critico)/d_critico*100:.2f}% del valor crítico)")
else:
    print("→ No se rechaza H₀")
    print("→ El generador PASA el test K-S ✓")