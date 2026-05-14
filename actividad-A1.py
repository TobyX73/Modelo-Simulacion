def lcg(a, c, m, seed, n):
    """
    Generador Congruencial Lineal (LCG).
    
    Parámetros originales del grupo:
      a = 1_738_019_701  →  a mod 4 = 1  (condiciones 2 y 3 de Hull-Dobell)
      c = 2_531_011      →  primo impar, mcd(c, m) = 1  (condición 1)
      m = 2**32          
      X0 = 2026          →  semilla inicial
    """
    x = seed
    uniformes = []
    for _ in range(n):
        x = (a * x + c) % m
        uniformes.append(x / m)
    return uniformes

# Parámetros
a    = 1_738_019_701
c    = 2_531_011
m    = 2**32
seed = 2026

numeros = lcg(a, c, m, seed, 1000)