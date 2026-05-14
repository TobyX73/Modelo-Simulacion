# EJERCICIO A.4 — Comparación K-S: LCG propio vs NumPy
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

# 1000 números con NumPy (misma semilla)
np.random.seed(seed_val)
np_nums = list(np.random.uniform(0, 1, 1000))

# 2. TEST KOLMOGOROV-SMIRNOV


N      = 1000
D_CRIT = 1.36 / math.sqrt(N)   # = 0.043007  (α=0.05)

def aplicar_ks(nums, nombre):
    """Calcula D máximo e imprime resultado K-S"""
    ordered = sorted(nums)
    d_max, idx_max, xi_max = 0, 0, 0

    for i, xi in enumerate(ordered, 1):
        fn      = i / N
        fn_prev = (i - 1) / N
        f0      = xi
        dp      = abs(fn - f0)
        dm      = abs(f0 - fn_prev)
        d       = max(dp, dm)
        if d > d_max:
            d_max, idx_max, xi_max = d, i, xi

    print(f"\n{'='*62}")
    print(f"TEST KOLMOGOROV-SMIRNOV — {nombre}")
    print(f"{'='*62}")
    print(f"N={N}  |  α=0.05")
    print(f"\nPunto de D máximo:")
    print(f"  Posición i    = {idx_max}")
    print(f"  Valor x_i     = {xi_max:.6f}")
    print(f"  Fn(x_i) = i/N = {idx_max}/{N} = {idx_max/N:.6f}")
    print(f"  F₀(x_i) = x_i = {xi_max:.6f}")
    print(f"  D máximo      = {d_max:.6f}")
    print(f"\nD crítico = 1.36/√{N} = {D_CRIT:.6f}")
    print(f"\nDECISIÓN: {d_max:.6f} {'≤' if d_max <= D_CRIT else '>'} {D_CRIT:.6f}")
    if d_max <= D_CRIT:
        print("→ No se rechaza H₀ — El generador PASA ✓")
    else:
        print("→ Se rechaza H₀   — El generador FALLA ✗")
        print(f"   Margen de falla: {d_max - D_CRIT:.6f} ({(d_max-D_CRIT)/D_CRIT*100:.2f}% del crítico)")

    return d_max, ordered

d_lcg, ord_lcg = aplicar_ks(lcg_nums, "LCG PROPIO")
d_np,  ord_np  = aplicar_ks(np_nums,  "NUMPY")


# 3. GRÁFICO COMPARATIVO — CURVAS FDA EMPÍRICA vs TEÓRICA

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Ejercicio A.4 — Test K-S: FDA empírica vs FDA teórica U(0,1)\n"
    f"N=1000  |  D crítico = {D_CRIT:.6f}  (α=0.05)",
    fontsize=12, fontweight="bold"
)
fig.patch.set_facecolor("#F5F5F5")

def dibujar_ks(ax, ordered, d_max, nombre, color_emp):
    n    = len(ordered)
    xs   = [0] + ordered + [1]
    fn_y = [i/n for i in range(n+1)]   # FDA empírica (por derecha)

    # FDA teórica — diagonal perfecta
    ax.plot([0, 1], [0, 1], color="#E24B4A", linewidth=2,
            linestyle="--", zorder=4, label="F₀(x) = x  (U(0,1) perfecta)")

    # FDA empírica — función escalón
    ax.step(xs, fn_y + [1], where="post", color=color_emp,
            linewidth=1.5, alpha=0.85, zorder=3,
            label=f"Fn(x) empírica  ({nombre})")

    # Encontrar punto de D máximo para marcarlo
    d_i, xi_d, fn_d, f0_d = 0, 0, 0, 0
    for i, xi in enumerate(ordered, 1):
        fn   = i / n
        fnp  = (i-1) / n
        d    = max(abs(fn - xi), abs(xi - fnp))
        if abs(d - d_max) < 1e-9:
            xi_d, fn_d, f0_d = xi, fn, xi
            break

    # Marcar la distancia D máxima con flecha
    ax.annotate("",
        xy=(xi_d, f0_d), xytext=(xi_d, fn_d),
        arrowprops=dict(arrowstyle="<->", color="#FF6B00",
                        lw=2, mutation_scale=12))
    ax.text(xi_d + 0.02, (fn_d + f0_d)/2,
            f"D = {d_max:.4f}", color="#FF6B00",
            fontsize=9, fontweight="bold", va="center")

    # Formato
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("F(x)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_facecolor("#FAFAFA")

    pasa  = d_max <= D_CRIT
    margen = D_CRIT - d_max
    ax.set_title(
        f"{nombre}\n"
        f"D = {d_max:.6f}  |  "
        f"{'PASA ✓' if pasa else 'FALLA ✗'}  "
        f"(margen: {margen:+.6f})",
        fontsize=10,
        color="#375623" if pasa else "#C00000"
    )
    ax.legend(fontsize=8, loc="upper left")

dibujar_ks(ax1, ord_lcg, d_lcg, "LCG propio", "#2E75B6")
dibujar_ks(ax2, ord_np,  d_np,  "NumPy",      "#70AD47")

# Leyenda D
p_d = mpatches.Patch(color="#FF6B00",
                     label=f"↕ D máximo (distancia entre curvas)")
fig.legend(handles=[p_d], loc="lower center", fontsize=9,
           bbox_to_anchor=(0.5, -0.01), framealpha=0.8)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("A4_ks_comparacion.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("\nGráfico guardado: A4_ks_comparacion.png")