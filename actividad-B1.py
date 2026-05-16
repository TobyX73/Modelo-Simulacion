import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────
# FUNCIÓN 1: Generador Congruencial Lineal (LCG)
# Mismo generador implementado en A.1
# ─────────────────────────────────────────────────────────────
def lcg(a, c, m, seed, n):

    x = seed
    uniformes = []
    for _ in range(n):
        x = (a * x + c) % m
        uniformes.append(x / m)
    return uniformes


# ─────────────────────────────────────────────────────────────
# FUNCIÓN 2: Transformada Inversa → Exponencial
# ─────────────────────────────────────────────────────────────
def generar_exponencial(lam, uniformes):

    muestras = []
    for u in uniformes:
        # Protección ante u=0 para evitar log(0) = -inf
        if u == 0:
            u = 1e-10
        x = -math.log(u) / lam
        muestras.append(x)
    return muestras


# ─────────────────────────────────────────────────────────────
# FUNCIÓN 3: Estadísticos muestrales
# ─────────────────────────────────────────────────────────────
def calcular_estadisticos(muestras, lam):

    n = len(muestras)
    media = sum(muestras) / n
    varianza = sum((x - media)**2 for x in muestras) / (n - 1)
    minimo = min(muestras)
    maximo = max(muestras)
    desvio = math.sqrt(varianza)

    # Valores teóricos Exp(λ): E[X] = 1/λ,  Var[X] = 1/λ²
    media_teo = 1 / lam
    varianza_teo = 1 / lam**2
    desvio_teo = 1 / lam

    return {
        "n": n,
        "media": media,
        "varianza": varianza,
        "desvio": desvio,
        "minimo": minimo,
        "maximo": maximo,
        "media_teo": media_teo,
        "varianza_teo": varianza_teo,
        "desvio_teo": desvio_teo,
    }


# ─────────────────────────────────────────────────────────────
# FUNCIÓN 4: Test Kolmogorov-Smirnov (implementación manual)
# ─────────────────────────────────────────────────────────────
def test_ks_exponencial(muestras, lam):

    n = len(muestras)
    ordenados = sorted(muestras)
    D_max = 0.0
    pos_max = 0

    for i, xi in enumerate(ordenados):
        fn_sup = (i + 1) / n      # FDA empírica por la derecha
        fn_inf = i / n            # FDA empírica por la izquierda
        f0 = 1 - math.exp(-lam * xi)  # CDF teórica Exp(λ)

        d_sup = abs(fn_sup - f0)
        d_inf = abs(fn_inf - f0)
        d_local = max(d_sup, d_inf)

        if d_local > D_max:
            D_max = d_local
            pos_max = i

    D_critico = 1.36 / math.sqrt(n)
    return D_max, D_critico, pos_max, ordenados


# ─────────────────────────────────────────────────────────────
# FUNCIÓN 5: Gráficos
# ─────────────────────────────────────────────────────────────
def graficar_resultados(muestras, lam, stats, D_max, D_critico):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Parte B — Ejercicio B.1: Distribución Exponencial(λ={lam})\n"
        f"Transformada Inversa con LCG propio | N={stats['n']}",
        fontsize=13, fontweight='bold'
    )

    # ── Panel izquierdo: Histograma + PDF teórica ──────────────
    ax1 = axes[0]
    ax1.set_title("(a) Histograma muestral + PDF teórica", fontsize=11)

    # Histograma normalizado (densidad)
    n_bins = 25
    counts, edges, patches = ax1.hist(
        muestras, bins=n_bins, density=True,
        color='steelblue', alpha=0.65, edgecolor='white', linewidth=0.5,
        label=f"Muestras LCG (n={stats['n']})"
    )

    # Curva teórica Exp(λ)
    x_vals = np.linspace(0, max(muestras) * 1.1, 300)
    pdf_teo = lam * np.exp(-lam * x_vals)
    ax1.plot(x_vals, pdf_teo, 'r-', linewidth=2.5, label=f"Teórica Exp(λ={lam})")

    # Líneas de medias
    ax1.axvline(stats['media'], color='darkorange', linestyle='-', linewidth=1.8,
                label=f"Media muestral = {stats['media']:.4f} min")
    ax1.axvline(stats['media_teo'], color='darkgreen', linestyle=':', linewidth=1.8,
                label=f"Media teórica = {stats['media_teo']:.4f} min")

    ax1.set_xlabel("Tiempo entre llegadas (min)", fontsize=11)
    ax1.set_ylabel("Densidad", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Cuadro de estadísticos en el gráfico
    texto_stats = (
        f"Media muestral:  {stats['media']:.4f}\n"
        f"Media teórica:   {stats['media_teo']:.4f}\n"
        f"Varianza muestral: {stats['varianza']:.4f}\n"
        f"Varianza teórica:  {stats['varianza_teo']:.4f}\n"
        f"Mínimo: {stats['minimo']:.4f}\n"
        f"Máximo: {stats['maximo']:.4f}"
    )
    ax1.text(0.62, 0.97, texto_stats, transform=ax1.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Panel derecho: FDA empírica vs teórica (K-S) ───────────
    ax2 = axes[1]
    ax2.set_title("(b) FDA empírica vs FDA teórica — Test K-S", fontsize=11)

    n = len(muestras)
    ordenados = sorted(muestras)
    fn_vals = [(i + 1) / n for i in range(n)]
    f0_vals = [1 - math.exp(-lam * xi) for xi in ordenados]

    ax2.step(ordenados, fn_vals, color='steelblue', linewidth=1.5,
             label="FDA empírica F_n(x)", where='post')
    ax2.plot(ordenados, f0_vals, 'r-', linewidth=2,
             label=f"FDA teórica F₀(x) = 1−e^(−{lam}x)")

    # Marcar el punto de D máximo
    xi_max = ordenados[int(D_max * n)]  # aproximación visual
    ax2.annotate(
        f"D_max = {D_max:.6f}",
        xy=(ordenados[n//2], abs(fn_vals[n//2] - f0_vals[n//2])),
        xytext=(max(ordenados) * 0.5, 0.15),
        fontsize=9, color='darkred',
        arrowprops=dict(arrowstyle='->', color='darkred')
    )

    color_resultado = 'darkgreen' if D_max <= D_critico else 'red'
    resultado_texto = "PASA ✓" if D_max <= D_critico else "FALLA ✗"
    ax2.set_xlabel("x (minutos)", fontsize=11)
    ax2.set_ylabel("F(x)", fontsize=11)

    # Texto con resultado del test
    texto_ks = (
        f"D calculado = {D_max:.6f}\n"
        f"D crítico   = {D_critico:.6f} (α=0.05)\n"
        f"Resultado: {resultado_texto}"
    )
    ax2.text(0.55, 0.25, texto_ks, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', color=color_resultado,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/claude/grafico_parteB.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[Gráfico guardado como 'grafico_parteB.png']")


# ─────────────────────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────────────────────
def main():
    # Parámetros del LCG del grupo
    a     = 1_738_019_701
    c     = 2_531_011
    m     = 2**32
    seed  = 2026
    LAM   = 0.8    # λ = 0.8 llamadas/min → E[X] = 1.25 min
    N     = 500    # Tamaño de muestra

    print("=" * 60)
    print("PARTE B — Ejercicio B.1")
    print("Generador de Variables Aleatorias Exponencial(λ=0.8)")
    print("Método: Transformada Inversa → X = -ln(U) / λ")
    print("=" * 60)
    print(f"\nParámetros LCG: a={a}, c={c}, m=2^32, semilla={seed}")
    print(f"Parámetro distribución: λ = {LAM} llamadas/min")
    print(f"Tamaño de muestra: N = {N}\n")

    # ── Paso 1: Generar N números U(0,1) con el LCG ────────────
    uniformes = lcg(a, c, m, seed, N)
    print(f"Primeros 5 uniformes generados:")
    for i, u in enumerate(uniformes[:5], 1):
        print(f"  U_{i} = {u:.6f}")

    # ── Paso 2: Aplicar Transformada Inversa ───────────────────
    muestras = generar_exponencial(LAM, uniformes)
    print(f"\nPrimeras 5 muestras Exponencial generadas:")
    for i, x in enumerate(muestras[:5], 1):
        print(f"  X_{i} = -ln({uniformes[i-1]:.6f}) / {LAM} = {x:.6f} min")

    # ── Paso 3: Estadísticos ───────────────────────────────────
    stats = calcular_estadisticos(muestras, LAM)

    print("\n" + "─" * 60)
    print("TABLA COMPARATIVA DE ESTADÍSTICOS")
    print("─" * 60)
    print(f"{'Estadístico':<28} {'Muestral':>12} {'Teórico':>12} {'Error %':>10}")
    print("─" * 60)

    error_media = abs(stats['media'] - stats['media_teo']) / stats['media_teo'] * 100
    error_var   = abs(stats['varianza'] - stats['varianza_teo']) / stats['varianza_teo'] * 100
    error_desvio = abs(stats['desvio'] - stats['desvio_teo']) / stats['desvio_teo'] * 100

    print(f"{'Media (min)':<28} {stats['media']:>12.4f} {stats['media_teo']:>12.4f} {error_media:>9.2f}%")
    print(f"{'Varianza (min²)':<28} {stats['varianza']:>12.4f} {stats['varianza_teo']:>12.4f} {error_var:>9.2f}%")
    print(f"{'Desvío estándar (min)':<28} {stats['desvio']:>12.4f} {stats['desvio_teo']:>12.4f} {error_desvio:>9.2f}%")
    print(f"{'Mínimo (min)':<28} {stats['minimo']:>12.4f} {'0 (límite)':>12}")
    print(f"{'Máximo (min)':<28} {stats['maximo']:>12.4f} {'∞ (cola)':>12}")
    print("─" * 60)

    # ── Paso 4: Test K-S ───────────────────────────────────────
    D_max, D_critico, pos_max, ordenados = test_ks_exponencial(muestras, LAM)

    print("\n" + "─" * 60)
    print("TEST KOLMOGOROV-SMIRNOV — Verificación distribucional")
    print("─" * 60)
    print(f"  Hipótesis nula H₀: la muestra sigue Exp(λ={LAM})")
    print(f"  Nivel de significancia: α = 0.05")
    print(f"  N = {N} muestras")
    print()
    print(f"  D calculado  : {D_max:.6f}")
    print(f"  D crítico    : {D_critico:.6f}  (1.36 / √{N})")
    print(f"  Posición D_max: índice {pos_max} → x = {ordenados[pos_max]:.4f} min")
    print()

    if D_max <= D_critico:
        margen = D_critico - D_max
        print(f"  → D_max ({D_max:.6f}) ≤ D_crítico ({D_critico:.6f})")
        print(f"  → Margen de seguridad: +{margen:.6f}")
        print(f"  → No se rechaza H₀ ✓")
        print(f"  → La muestra sigue una distribución Exp({LAM})")
        conclusion = "PASA"
    else:
        margen = D_max - D_critico
        print(f"  → D_max ({D_max:.6f}) > D_crítico ({D_critico:.6f})")
        print(f"  → Margen de falla: +{margen:.6f}")
        print(f"  → Se rechaza H₀ ✗")
        print(f"  → La muestra NO sigue una distribución Exp({LAM})")
        conclusion = "FALLA"

    print("─" * 60)

    # ── Paso 5: Interpretación del contexto ───────────────────
    print("\n" + "─" * 60)
    print("INTERPRETACIÓN EN EL CONTEXTO DEL CALL CENTER")
    print("─" * 60)
    print(f"  Distribución modelada: Exp(λ={LAM} llamadas/min)")
    print(f"  Tiempo medio entre llamadas: {stats['media_teo']:.2f} min = {stats['media_teo']*60:.0f} seg")
    print(f"  Tiempo muestral medio:       {stats['media']:.4f} min")
    print(f"  Variabilidad observada:      σ = {stats['desvio']:.4f} min")
    print(f"  Tiempo mínimo registrado:    {stats['minimo']:.4f} min ({stats['minimo']*60:.1f} seg)")
    print(f"  Tiempo máximo registrado:    {stats['maximo']:.4f} min ({stats['maximo']*60:.1f} seg)")
    print()
    print(f"  Test K-S: {conclusion} → El generador es {'válido' if conclusion=='PASA' else 'cuestionable'}")
    print(f"  para modelar llegadas del turno mañana del call center.")
    print("=" * 60)

    # ── Paso 6: Gráficos ───────────────────────────────────────
    graficar_resultados(muestras, LAM, stats, D_max, D_critico)

    return muestras, stats, D_max, D_critico


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    muestras, stats, D_max, D_critico = main()
