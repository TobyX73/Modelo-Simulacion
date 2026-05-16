import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sin_cache = [
    14.58, 5.70, 2.17, 2.90,  1.30, 12.16,  2.99, 13.94,  6.91,  4.14,
     3.01, 7.72, 1.94, 5.19,  2.04,  4.45,  2.83,  5.27,  7.91,  4.41,
    17.63,10.90, 7.70, 3.70,  3.70, 14.00,  4.28,  6.40,  6.60,  2.94,
     4.43, 1.81, 1.51, 4.24,  2.70,  3.90,  7.49,  3.49,  4.16, 10.50,
    11.78, 3.94, 7.25, 6.91, 10.35,  6.25,  2.99,  4.57,  5.40,  3.16,
     2.27, 6.44, 4.31, 0.89,  3.65,  7.36,  2.77,  4.33, 15.16,  3.07,
     2.93, 1.76, 3.26, 3.41,  0.66,  1.45,  1.44,  2.77,  0.73,  0.66,
     1.97, 1.62, 3.53, 1.82,  4.37,  1.24,  7.22, 12.83,  2.63,  3.16,
     1.80,
]


con_cache = [
    3.48, 5.18, 0.573, 1.90, 1.15, 7.24, 2.11, 10.29, 6.08, 3.29, 
    2.87, 2.73, 1.82, 6.90, 1.00, 2.79, 2.57, 1.89, 5.34, 3.26, 
    1.09, 7.17, 6.40, 3.50, 2.50, 7.00, 1.04, 4.70, 3.50, 2.92, 
    2.24, 1.00, 7.50, 3.28, 3.22, 4.90, 3.59, 3.50, 2.70, 6.50,  
    3.80, 3.80, 1.45, 3.73, 3.60, 4.36, 2.94, 2.19, 4.16, 2.78,  
    0.94, 3.35, 1.24, 0.44, 2.88, 1.25, 1.80, 4.03, 1.80, 2.55, 
    2.17, 1.18, 2.39, 1.50, 0.44, 0.99, 1.17, 1.59, 0.54, 0.37, 
    0.68, 0.329, 2.33, 1.82, 1.10, 0.43, 1.06, 0.90, 2.43, 2.03,
    1.06  
];

def calcular_stats(d):
    arr = np.array(d)
    n   = len(d)
    return {
        "n":        n,
        "media":    statistics.mean(d),
        "mediana":  statistics.median(d),
        "var":      statistics.variance(d),
        "std":      statistics.stdev(d),
        "min":      min(d),
        "max":      max(d),
        "q1":       float(np.percentile(arr, 25)),
        "q3":       float(np.percentile(arr, 75)),
        "cv":       statistics.stdev(d) / statistics.mean(d) * 100,
        "asim":     float(stats.skew(arr)),
        "kurt":     float(stats.kurtosis(arr)),
        "k":        math.ceil(1 + 3.322 * math.log10(n)),
        "arr":      arr,
    }

sc = calcular_stats(sin_cache)
cc = calcular_stats(con_cache)

for nombre, s in [("SIN CACHÉ", sc), ("CON CACHÉ", cc)]:
    print(f"\n{'='*50}")
    print(f"  {nombre}  (n={s['n']})")
    print(f"{'='*50}")
    print(f"  Media          = {s['media']:.4f} s")
    print(f"  Mediana        = {s['mediana']:.4f} s")
    print(f"  Varianza       = {s['var']:.4f} s²")
    print(f"  Desvío Est.    = {s['std']:.4f} s")
    print(f"  Mínimo         = {s['min']:.4f} s")
    print(f"  Máximo         = {s['max']:.4f} s")
    print(f"  Q1             = {s['q1']:.4f} s")
    print(f"  Q3             = {s['q3']:.4f} s")
    print(f"  IQR            = {s['q3']-s['q1']:.4f} s")
    print(f"  CV             = {s['cv']:.2f} %")
    print(f"  Asimetría      = {s['asim']:.4f}")
    print(f"  Curtosis exc.  = {s['kurt']:.4f}")
    print(f"  k (Sturges)    = {s['k']}")

# ── Paleta ────────────────────────────────────────────────────────────────────
C_SC     = "#2e6da4"   # azul — sin caché
C_CC     = "#e07b39"   # naranja — con caché
C_SC_L   = "#7fb3d3"
C_CC_L   = "#f2b48a"
C_DARK   = "#1a3a5c"
C_BG     = "#f7fafd"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    C_BG,
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#dde6ef",
    "grid.linewidth":    0.7,
})

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Histograma SIN CACHÉ
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
counts, edges, patches = ax.hist(sin_cache, bins=sc["k"],
    color=C_SC, edgecolor="white", linewidth=0.8, alpha=0.85, zorder=3)
for i, (l, r) in enumerate(zip(edges[:-1], edges[1:])):
    if l <= sc["media"] < r:
        patches[i].set_facecolor(C_DARK); patches[i].set_alpha(1.0)
ax.axvline(sc["media"],   color=C_DARK,   lw=1.8, ls="-",  label=f"Media = {sc['media']:.2f} s")
ax.axvline(sc["mediana"], color=C_CC,     lw=1.8, ls="--", label=f"Mediana = {sc['mediana']:.2f} s")
ax.set_xlabel("Tiempo de carga sin caché (s)", fontsize=10)
ax.set_ylabel("Frecuencia absoluta", fontsize=10)
ax.set_title(f"Histograma — Sin caché  (k = {sc['k']} clases, Sturges; n = {sc['n']})",
             fontsize=11, color=C_DARK, pad=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("d2_histograma_sin.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("\nd2_histograma_sin.png guardado")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Histograma CON CACHÉ
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
counts, edges, patches = ax.hist(con_cache, bins=cc["k"],
    color=C_CC, edgecolor="white", linewidth=0.8, alpha=0.85, zorder=3)
for i, (l, r) in enumerate(zip(edges[:-1], edges[1:])):
    if l <= cc["media"] < r:
        patches[i].set_facecolor(C_DARK); patches[i].set_alpha(1.0)
ax.axvline(cc["media"],   color=C_DARK, lw=1.8, ls="-",  label=f"Media = {cc['media']:.2f} s")
ax.axvline(cc["mediana"], color=C_SC,   lw=1.8, ls="--", label=f"Mediana = {cc['mediana']:.2f} s")
ax.set_xlabel("Tiempo de carga con caché (s)", fontsize=10)
ax.set_ylabel("Frecuencia absoluta", fontsize=10)
ax.set_title(f"Histograma — Con caché  (k = {cc['k']} clases, Sturges; n = {cc['n']})",
             fontsize=11, color=C_DARK, pad=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("d2_histograma_con.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("d2_histograma_con.png guardado")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Boxplot comparativo
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.2))
bp = ax.boxplot(
    [sin_cache, con_cache], vert=True, patch_artist=True,
    widths=0.45, labels=["Sin caché", "Con caché"],
    boxprops=dict(linewidth=1.4),
    medianprops=dict(linewidth=2.2, color="white"),
    whiskerprops=dict(linewidth=1.3, linestyle="--"),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=5, alpha=0.7),
)
colores = [C_SC, C_CC]
for patch, color in zip(bp["boxes"], colores):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for flier, color in zip(bp["fliers"], colores):
    flier.set_markerfacecolor(color)
    flier.set_markeredgecolor(color)
# Medias
for i, (s, color) in enumerate([(sc, C_DARK), (cc, C_DARK)], start=1):
    ax.plot(i, s["media"], marker="D", color=color, markersize=7,
            zorder=5, label=f"Media {'SC' if i==1 else 'CC'} = {s['media']:.2f} s")
ax.set_ylabel("Tiempo de carga (s)", fontsize=10)
ax.set_title("Boxplot comparativo — Sin caché vs Con caché", fontsize=11, color=C_DARK, pad=10)
ax.legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig("d2_boxplot_comparativo.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("d2_boxplot_comparativo.png guardado")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — KDE comparativo
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.linspace(0, 20, 600)
for d, s, color, lcolor, label in [
    (sin_cache, sc, C_SC, C_SC_L, f"Sin caché (μ={sc['media']:.2f} s)"),
    (con_cache, cc, C_CC, C_CC_L, f"Con caché (μ={cc['media']:.2f} s)"),
]:
    kde = stats.gaussian_kde(d, bw_method="scott")
    y   = kde(x)
    ax.fill_between(x, y, alpha=0.20, color=color)
    ax.plot(x, y, color=color, lw=2.2, label=label)
    ax.axvline(s["media"],   color=color, lw=1.4, ls="-",  alpha=0.8)
    ax.axvline(s["mediana"], color=color, lw=1.4, ls="--", alpha=0.8)
    ax.plot(d, np.zeros_like(d) - 0.003, "|", color=color, alpha=0.4, markersize=7)

ax.set_xlabel("Tiempo de carga (s)", fontsize=10)
ax.set_ylabel("Densidad", fontsize=10)
ax.set_title("Gráfico de densidad (KDE) — Sin caché vs Con caché\n"
             "(línea sólida = media; línea discontinua = mediana)",
             fontsize=11, color=C_DARK, pad=10)
ax.legend(fontsize=9)
ax.set_ylim(bottom=-0.012)
fig.tight_layout()
fig.savefig("d2_kde_comparativo.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("d2_kde_comparativo.png guardado")

print("\nTodos los gráficos generados exitosamente.")
