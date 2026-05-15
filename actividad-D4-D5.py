"""
TP2 — Modelo y Simulación
Ejercicio D.4: Test de Bondad de Ajuste (K-S formal)
Ejercicio D.5: Generación de Datos Sintéticos y Comparación
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import weibull_min, lognorm

# ── Configuración de estilo ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ── Carga de datos ───────────────────────────────────────────────────────────
df = pd.read_excel(
    "Recoleccion de Datos - Modelo y Simulacion TP2.xlsx",
    header=1,
)

sin_cache = df["Tiempo de carga sin cache (s)"].dropna().values   # n = 81
con_cache = df["Tiempo de carga con cache (s)"].dropna().values   # n = 81

# ── Parámetros MLE (estimados en D.3) ────────────────────────────────────────
# Weibull
k_sin,  _, lam_sin  = weibull_min.fit(sin_cache, floc=0)
k_con,  _, lam_con  = weibull_min.fit(con_cache, floc=0)

# Log-normal (candidata secundaria)
sig_sin = np.std(np.log(sin_cache), ddof=1);  mu_sin = np.mean(np.log(sin_cache))
sig_con = np.std(np.log(con_cache), ddof=1);  mu_con = np.mean(np.log(con_cache))

print("=" * 60)
print("PARÁMETROS MLE UTILIZADOS")
print("=" * 60)
print(f"  Sin caché — Weibull : k = {k_sin:.4f},  λ = {lam_sin:.4f} s")
print(f"  Con caché — Weibull : k = {k_con:.4f},  λ = {lam_con:.4f} s")
print(f"  Sin caché — LogNorm : μ = {mu_sin:.4f},  σ = {sig_sin:.4f}")
print(f"  Con caché — LogNorm : μ = {mu_con:.4f},  σ = {sig_con:.4f}")


# EJERCICIO D.4 — Test K-S (una muestra)
print("\n" + "=" * 60)
print("D.4 — TEST K-S FORMAL  (α = 0,05)")
print("=" * 60)

# H0: los datos siguen la distribución teórica especificada
# H1: los datos NO siguen dicha distribución

ks_wb_sin = stats.kstest(sin_cache, "weibull_min", args=(k_sin, 0, lam_sin))
ks_wb_con = stats.kstest(con_cache, "weibull_min", args=(k_con, 0, lam_con))
ks_ln_sin = stats.kstest(sin_cache, "lognorm",     args=(sig_sin, 0, np.exp(mu_sin)))
ks_ln_con = stats.kstest(con_cache, "lognorm",     args=(sig_con, 0, np.exp(mu_con)))

resultados_d4 = pd.DataFrame({
    "Condición":    ["Sin caché", "Sin caché", "Con caché", "Con caché"],
    "Distribución": ["Weibull", "Log-normal", "Weibull", "Log-normal"],
    "D calculado":  [ks_wb_sin.statistic, ks_ln_sin.statistic,
                     ks_wb_con.statistic, ks_ln_con.statistic],
    "p-valor":      [ks_wb_sin.pvalue, ks_ln_sin.pvalue,
                     ks_wb_con.pvalue, ks_ln_con.pvalue],
})
resultados_d4["Resultado"] = resultados_d4["p-valor"].apply(
    lambda p: "NO se rechaza H₀ ✓" if p > 0.05 else "Se rechaza H₀ ✗"
)
resultados_d4["D calculado"] = resultados_d4["D calculado"].round(4)
resultados_d4["p-valor"]     = resultados_d4["p-valor"].round(4)

print("\nTabla D4.1 — Resultados K-S")
print(resultados_d4.to_string(index=False))


# ── Figura D4.1 — CDF empírica vs. Weibull teórica ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Figura D4.1 — CDF empírica vs. Weibull teórica",
             fontsize=13, fontweight="bold")

for ax, data, k, lam, ks_result, label, n in [
    (axes[0], sin_cache, k_sin, lam_sin, ks_wb_sin, "Sin caché", 81),
    (axes[1], con_cache, k_con, lam_con, ks_wb_con, "Con caché", 81),
]:
    x_sorted = np.sort(data)
    ecdf     = np.arange(1, n + 1) / n
    x_range  = np.linspace(0, x_sorted[-1] * 1.08, 500)

    # Punto de máxima diferencia D
    cdf_at_pts = weibull_min.cdf(x_sorted, k, 0, lam)
    idx_max    = np.argmax(np.abs(ecdf - cdf_at_pts))
    x_d        = x_sorted[idx_max]
    y_emp      = ecdf[idx_max]
    y_teo      = cdf_at_pts[idx_max]

    ax.step(x_sorted, ecdf, where="post", color="#2563EB", lw=2,
            label="CDF empírica (FDAE)")
    ax.plot(x_range, weibull_min.cdf(x_range, k, 0, lam),
            color="#EA580C", lw=2, ls="--",
            label=f"Weibull(k={k:.4f}, λ={lam:.4f} s)")

    # Segmento que representa D
    ax.plot([x_d, x_d], [y_emp, y_teo], color="#16A34A", lw=2.5,
            solid_capstyle="round", zorder=5)
    ax.annotate(
        f"  D = {ks_result.statistic:.4f}\np = {ks_result.pvalue:.4f}",
        xy=(x_d, (y_emp + y_teo) / 2), fontsize=9, color="#16A34A",
        va="center",
    )

    ax.set_title(f"{label} (n = {n})", fontweight="bold", fontsize=11)
    ax.set_xlabel("Tiempo de carga (s)")
    ax.set_ylabel("Probabilidad acumulada")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("Figura_D4_1_KS.png", bbox_inches="tight")
plt.close()
print("\n✓ Figura D4.1 guardada → Figura_D4_1_KS.png")


# EJERCICIO D.5 — Generación de datos sintéticos
print("\n" + "=" * 60)
print("D.5 — DATOS SINTÉTICOS (Weibull, seed = 42)")
print("=" * 60)

np.random.seed(42)
syn_sin = weibull_min.rvs(k_sin, loc=0, scale=lam_sin,
                           size=len(sin_cache), random_state=42)
syn_con = weibull_min.rvs(k_con, loc=0, scale=lam_con,
                           size=len(con_cache), random_state=42)


# ── Tabla de estadísticos descriptivos ──────────────────────────────────────
def tabla_estadisticos(real, synth, label):
    filas = {
        "Media (s)":       (np.mean(real),                    np.mean(synth)),
        "Mediana (s)":     (np.median(real),                  np.median(synth)),
        "Desvío std. (s)": (np.std(real, ddof=1),             np.std(synth, ddof=1)),
        "Varianza (s²)":   (np.var(real, ddof=1),             np.var(synth, ddof=1)),
        "Mínimo (s)":      (np.min(real),                     np.min(synth)),
        "Máximo (s)":      (np.max(real),                     np.max(synth)),
        "IQR (s)":         (np.percentile(real, 75) - np.percentile(real, 25),
                            np.percentile(synth,75) - np.percentile(synth,25)),
        "CV (%)":          (np.std(real, ddof=1)  / np.mean(real)  * 100,
                            np.std(synth, ddof=1) / np.mean(synth) * 100),
        "Asimetría":       (stats.skew(real),                 stats.skew(synth)),
    }
    tbl = pd.DataFrame(filas, index=["Real", "Sintético"]).T
    # Diferencia porcentual solo para métricas con sentido relativo
    mask = ~tbl.index.isin(["Mínimo (s)", "Máximo (s)", "IQR (s)", "Asimetría"])
    tbl["Dif. (%)"] = np.nan
    tbl.loc[mask, "Dif. (%)"] = (
        (tbl.loc[mask, "Sintético"] - tbl.loc[mask, "Real"]).abs()
        / tbl.loc[mask, "Real"].abs() * 100
    ).round(2)
    print(f"\nTabla D5 — {label}")
    print(tbl.round(4).to_string())
    return tbl

tabla_estadisticos(sin_cache, syn_sin, "Sin caché")
tabla_estadisticos(con_cache, syn_con, "Con caché")


# ── Figura D5.1-D5.2 — Histogramas superpuestos ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figuras D5.1-D5.2 — Histogramas: datos reales vs. sintéticos",
             fontsize=13, fontweight="bold")

for ax, real, synth, k, lam, label, n in [
    (axes[0], sin_cache, syn_sin, k_sin, lam_sin, "Sin caché", len(sin_cache)),
    (axes[1], con_cache, syn_con, k_con, lam_con, "Con caché", len(con_cache)),
]:
    x_max = max(real.max(), synth.max()) * 1.05
    bins  = np.linspace(0, x_max, 16)
    x_pdf = np.linspace(0, x_max, 400)

    ax.hist(real,  bins=bins, density=True, alpha=0.55, color="#2563EB",
            label="Datos reales",     zorder=2)
    ax.hist(synth, bins=bins, density=True, alpha=0.50, color="#EA580C",
            label="Datos sintéticos", zorder=2)
    ax.plot(x_pdf, weibull_min.pdf(x_pdf, k, 0, lam),
            color="#EA580C", lw=2.2, ls="--", label="PDF Weibull teórica")

    ax.set_title(f"{label} (n = {n})", fontweight="bold", fontsize=11)
    ax.set_xlabel("Tiempo de carga (s)")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("Figura_D5_1_2_Histogramas.png", bbox_inches="tight")
plt.close()
print("\n✓ Figuras D5.1-D5.2 guardadas → Figura_D5_1_2_Histogramas.png")


# ── Test K-S de dos muestras ─────────────────────────────────────────────────
print("\nTabla D5.3 — Test K-S de dos muestras (real vs. sintético)")
print("-" * 62)

for real, synth, label in [
    (sin_cache, syn_sin, "Sin caché"),
    (con_cache, syn_con, "Con caché"),
]:
    res = stats.ks_2samp(real, synth)
    conclusion = "No se rechaza H₀ ✓" if res.pvalue > 0.05 else "Se rechaza H₀ ✗"
    print(f"  {label:12s} | D = {res.statistic:.4f} | p = {res.pvalue:.4f} | {conclusion}")

print("-" * 62)
print("\n✅ D.4 y D.5 completados.")
