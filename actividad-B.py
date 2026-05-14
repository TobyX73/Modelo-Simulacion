"""
PARTE B — Ejercicio B.1
Generador de Variables Aleatorias Exponencial(lambda=0.8)
por el Método de Transformada Inversa.

Fórmula: X = -ln(U) / lambda
donde U ~ U(0,1) proviene del LCG implementado en A.1.

Parámetros del grupo:
  a = 1_738_019_701
  c = 2_531_011
  m = 2**32
  semilla = 2026
"""

import math

# ─────────────────────────────────────────
# 1. LCG (igual que en A.1)
# ─────────────────────────────────────────
def lcg(a, c, m, seed, n):
    x = seed
    uniformes = []
    for _ in range(n):
        x = (a * x + c) % m
        uniformes.append(x / m)
    return uniformes

# ─────────────────────────────────────────
# 2. Transformada Inversa → Exponencial
# ─────────────────────────────────────────
def generar_exponencial(lam, uniformes):
    """
    Aplica la Transformada Inversa:
      X = -ln(U) / lambda
    para cada U en la lista de uniformes.
    """
    muestras = []
    for u in uniformes:
        # Protección ante u=0 (evita log(0))
        if u == 0:
            u = 1e-10
        x = -math.log(u) / lam
        muestras.append(x)
    return muestras

# ─────────────────────────────────────────
# 3. Parámetros y generación
# ─────────────────────────────────────────
a, c, m, seed = 1_738_019_701, 2_531_011, 2**32, 2026
LAM = 0.8
N   = 500

uniformes = lcg(a, c, m, seed, N)
muestras  = generar_exponencial(LAM, uniformes)

# ─────────────────────────────────────────
# 4. Estadísticos muestrales
# ─────────────────────────────────────────
n        = len(muestras)
media    = sum(muestras) / n
varianza = sum((x - media)**2 for x in muestras) / (n - 1)
minimo   = min(muestras)
maximo   = max(muestras)

# Valores teóricos  Exp(lambda):
#   media = 1/lambda    varianza = 1/lambda^2
media_teo    = 1 / LAM
varianza_teo = 1 / LAM**2

# ─────────────────────────────────────────
# 5. Test K-S (implementación manual)
# ─────────────────────────────────────────
ordenados = sorted(muestras)
D_max = 0.0
for i, xi in enumerate(ordenados):
    fn_sup = (i + 1) / n          # F_empírica por la derecha
    fn_inf = i / n                # F_empírica por la izquierda
    f0     = 1 - math.exp(-LAM * xi)   # CDF exponencial teórica
    d_sup  = abs(fn_sup - f0)
    d_inf  = abs(fn_inf - f0)
    D_max  = max(D_max, d_sup, d_inf)

D_critico = 1.36 / math.sqrt(n)

# ─────────────────────────────────────────
# 6. Impresión de resultados
# ─────────────────────────────────────────
print("=" * 58)
print("PARTE B — Ejercicio B.1: Generador Exponencial(λ=0.8)")
print("=" * 58)
print(f"Método: Transformada Inversa  →  X = -ln(U) / λ")
print(f"Muestra: N = {N}  |  λ = {LAM}")
print()
print(f"{'Estadístico':<22} {'Muestral':>12} {'Teórico':>12}")
print("-" * 48)
print(f"{'Media':<22} {media:>12.4f} {media_teo:>12.4f}")
print(f"{'Varianza':<22} {varianza:>12.4f} {varianza_teo:>12.4f}")
print(f"{'Mínimo':<22} {minimo:>12.4f} {'—':>12}")
print(f"{'Máximo':<22} {maximo:>12.4f} {'—':>12}")
print()
print("─" * 48)
print("TEST KOLMOGOROV-SMIRNOV")
print("─" * 48)
print(f"  D calculado : {D_max:.6f}")
print(f"  D crítico   : {D_critico:.6f}  (α = 0.05, N={N})")
print()
if D_max <= D_critico:
    print("  → No se rechaza H₀")
    print("  → La muestra sigue una distribución Exp(0.8) ✓")
else:
    print("  → Se rechaza H₀")
    print("  → La muestra NO sigue una distribución Exp(0.8) ✗")
    
print("=" * 58)
