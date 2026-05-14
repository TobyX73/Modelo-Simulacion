# EJERCICIO A.4 — Comparación Chi-cuadrado: LCG propio vs NumPy

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1. GENERACIÓN DE NÚMEROS


def lcg(a, c, m, seed, n):
    """Generador LCG propio — Ejercicio A.1"""
    x = seed
    vals = []
    for _ in range(n):
        x = (a * x + c) % m
        vals.append(x / m)
    return vals

# Parámetros del grupo (A.1)
a, c, m, seed_val = 1_738_019_701, 2_531_011, 2**32, 2026

# 1000 números con LCG propio
lcg_nums = lcg(a, c, m, seed_val, 1000)

# 1000 números con NumPy (misma semilla para comparación justa)
np.random.seed(seed_val)
np_nums = list(np.random.uniform(0, 1, 1000))


# 2. TEST CHI-CUADRADO


N         = 1000
K         = 10
E         = N / K      # = 100
CHI2_CRIT = 16.919     # χ²(9, 0.05) de tabla

def aplicar_chi2(nums, nombre):
    """Calcula frecuencias e imprime tabla Chi-cuadrado"""
    obs = [0] * K
    for u in nums:
        i = min(int(u * K), K - 1)
        obs[i] += 1

    print(f"\n{'='*62}")
    print(f"TEST CHI-CUADRADO — {nombre}")
    print(f"{'='*62}")
    print(f"N={N}  |  k={K} clases  |  E_i=N/k={E:.0f}  |  α=0.05")
    print(f"\n{'Intervalo':<14} {'O_i':>6} {'E_i':>6} {'(O-E)²':>10} {'(O-E)²/E':>10}")
    print("-" * 50)

    chi2 = 0
    for i in range(K):
        Oi    = obs[i]
        diff2 = (Oi - E) ** 2
        term  = diff2 / E
        chi2 += term
        print(f"[{i/K:.1f} , {(i+1)/K:.1f})   {Oi:>6} {E:>6.0f} {diff2:>10.1f} {term:>10.4f}")

    print("-" * 50)
    print(f"{'TOTAL':<14} {sum(obs):>6} {N:>6}                χ² = {chi2:.4f}")
    print(f"\nGrados de libertad : gl = k-1 = {K-1}")
    print(f"Valor crítico      : χ²({K-1}, 0.05) = {CHI2_CRIT}")
    print(f"\nDECISIÓN: {chi2:.4f} {'≤' if chi2 <= CHI2_CRIT else '>'} {CHI2_CRIT}")
    if chi2 <= CHI2_CRIT:
        print("→ No se rechaza H₀ — El generador PASA ✓")
    else:
        print("→ Se rechaza H₀   — El generador FALLA ✗")

    return chi2, obs

chi2_lcg, obs_lcg = aplicar_chi2(lcg_nums, "LCG PROPIO")
chi2_np,  obs_np  = aplicar_chi2(np_nums,  "NUMPY")

# 3. GRÁFICO COMPARATIVO — HISTOGRAMA CHI-CUADRADO


intervalos_lbl = [f"[{i/K:.1f},{(i+1)/K:.1f})" for i in range(K)]
x_pos          = list(range(K))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle(
    "Ejercicio A.4 — Test Chi-Cuadrado: LCG propio vs NumPy\n"
    "N=1000 | k=10 clases | E=100 | χ²crítico=16.919 (gl=9, α=0.05)",
    fontsize=12, fontweight="bold"
)
fig.patch.set_facecolor("#F5F5F5")

def dibujar_histograma(ax, obs, chi2, nombre, color_bajo, color_alto):
    colores = [color_bajo if o <= E else color_alto for o in obs]
    barras  = ax.bar(x_pos, obs, color=colores, edgecolor="white",
                     linewidth=1, width=0.75, zorder=3)

    # Línea esperada E=100
    ax.axhline(y=E, color="#E24B4A", linewidth=2.2,
               linestyle="--", zorder=4, label="E = 100")

    # Zona de variación normal
    ax.axhspan(80, 120, alpha=0.07, color="green", zorder=1)

    # Valores encima de cada barra
    for b, o in zip(barras, obs):
        diff  = o - int(E)
        signo = f"+{diff}" if diff > 0 else str(diff)
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"{o}\n({signo})", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#1a1a1a")

    # Formato
    ax.set_ylim(55, 148)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(intervalos_lbl, rotation=45, ha="right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("#FAFAFA")
    ax.set_xlabel("Intervalo", fontsize=10)
    ax.set_ylabel("Frecuencia observada (Oᵢ)", fontsize=10)

    pasa = chi2 <= CHI2_CRIT
    ax.set_title(
        f"{nombre}\n"
        f"χ² = {chi2:.4f}  |  "
        f"{'PASA ✓' if pasa else 'FALLA ✗'}  "
        f"(margen: {CHI2_CRIT - chi2:+.4f})",
        fontsize=10,
        color="#375623" if pasa else "#C00000"
    )
    ax.legend(fontsize=8)

dibujar_histograma(ax1, obs_lcg, chi2_lcg, "LCG propio",
                   "#2E75B6", "#1F4E79")
dibujar_histograma(ax2, obs_np,  chi2_np,  "NumPy",
                   "#70AD47", "#375623")

# Leyenda global
p1 = mpatches.Patch(color="#2E75B6", label="LCG: Oᵢ ≤ 100")
p2 = mpatches.Patch(color="#1F4E79", label="LCG: Oᵢ > 100")
p3 = mpatches.Patch(color="#70AD47", label="NumPy: Oᵢ ≤ 100")
p4 = mpatches.Patch(color="#375623", label="NumPy: Oᵢ > 100")
p5 = mpatches.Patch(color="#E24B4A", label="E = 100 (esperado)")
fig.legend(handles=[p1, p2, p3, p4, p5],
           loc="lower center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.8)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("A4_chi2_comparacion.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("\nGráfico guardado: A4_chi2_comparacion.png")