
# ═══════════════════════════════════════════════════════════════
# PARTE C — Análisis Estadístico
# ═══════════════════════════════════════════════════════════════
import math
import matplotlib.pyplot as plt
import numpy as np

# ── LCG (mismo de Parte A) ──────────────────────────────────────
def lcg(a, c, m, seed, n):
    x = seed
    out = []
    for _ in range(n):
        x = (a * x + c) % m
        out.append(x / m)
    return out

a, c, m, seed = 1_738_019_701, 2_531_011, 2**32, 2026

# ══════════════════════════════════════════════════════════════
# EJERCICIO C.1 — Tabla estadísticos muestrales vs. teóricos
# ══════════════════════════════════════════════════════════════
lam = 0.8
U = lcg(a, c, m, seed, 500)
X = np.array([-math.log(u) / lam for u in U])

n_muestra = len(X)
media_m  = X.mean()
var_m    = X.var(ddof=1)
desvio_m = math.sqrt(var_m)
x_min, x_max = X.min(), X.max()

mu_teo   = 1 / lam       # 1.2500
var_teo  = 1 / lam**2    # 1.5625

print("═══ C.1 — ESTADÍSTICOS MUESTRALES VS. TEÓRICOS ═══")
print(f"{'Estadístico':<22} {'Muestral':>10} {'Teórico':>10} {'Dif. %':>8}")
print("-" * 54)
for nombre, muestral, teorico in [
    ("Media (min)",       media_m,  mu_teo),
    ("Varianza (min²)",   var_m,    var_teo),
    ("Desvío estándar",   desvio_m, math.sqrt(var_teo)),
]:
    dif_pct = abs(muestral - teorico) / teorico * 100
    print(f"{nombre:<22} {muestral:>10.4f} {teorico:>10.4f} {dif_pct:>7.2f}%")
print(f"{'Mínimo':<22} {x_min:>10.4f} {'0.0000':>10}")
print(f"{'Máximo':<22} {x_max:>10.4f} {'∞':>10}")

# Histograma con curva teórica
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(X, bins=20, density=True, color='#4472C4', alpha=0.75,
        edgecolor='white', label='Muestras LCG (n=500)')
x_teo = np.linspace(0, X.max() + 0.5, 500)
ax.plot(x_teo, lam * np.exp(-lam * x_teo), 'r-', lw=2.2,
        label=f'Teórica Exp(λ={lam})')
ax.axvline(media_m, color='orange', ls='--', lw=1.8, label=f'Media muestral={media_m:.4f}')
ax.axvline(mu_teo,  color='green',  ls=':',  lw=1.8, label=f'Media teórica={mu_teo:.4f}')
ax.set_xlabel('Tiempo entre llegadas (min)')
ax.set_ylabel('Densidad')
ax.set_title('C.1 — Exponencial(λ=0.8): muestral vs. teórica')
ax.legend(); plt.tight_layout()
plt.savefig('c1_histograma_exponencial.png', dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════
# EJERCICIO C.2 — Simulación 100 días, 3 escenarios
# ══════════════════════════════════════════════════════════════
TURNO_MIN = 480  # 8 horas

def simular_dias(lam_sim, n_dias=100):
    """Simula n_dias días con 3 turnos cada uno."""
    dias = []
    for d in range(n_dias):
        U_dia = lcg(a, c, m, seed + d * 1000, 5000)
        idx, total = 0, 0
        for _ in range(3):
            t = 0
            while idx < len(U_dia):
                dt = -math.log(U_dia[idx]) / lam_sim
                idx += 1
                t += dt
                if t > TURNO_MIN:
                    break
                total += 1
        dias.append(total)
    return np.array(dias)

base  = simular_dias(0.80)
menos = simular_dias(0.72)
mas   = simular_dias(0.88)

umbral = base.mean() + 1.5 * base.std()
print(f"\n═══ C.2 — RESUMEN POR ESCENARIO (umbral crítico: {umbral:.0f}) ═══")
for lbl, d in [("Base (λ=0.80)", base), ("−10% (λ=0.72)", menos), ("+10% (λ=0.88)", mas)]:
    criticos = int((d > umbral).sum())
    print(f"{lbl:20s}  media={d.mean():.1f}  σ={d.std():.1f}  "
          f"min={d.min()}  max={d.max()}  críticos={criticos}/100")

# Gráfico comparativo
colores = ['#4472C4', '#70AD47', '#FF0000']
etiquetas = ['Base (λ=0.80)', '−10% (λ=0.72)', '+10% (λ=0.88)']
datos = [base, menos, mas]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
bp = axes[0].boxplot(datos, patch_artist=True,
                     medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], colores):
    patch.set_facecolor(color); patch.set_alpha(0.75)
axes[0].axhline(umbral, color='crimson', ls='--', lw=1.5,
                label=f'Umbral crítico ({umbral:.0f})')
axes[0].set_xticks([1, 2, 3]); axes[0].set_xticklabels(etiquetas, fontsize=9)
axes[0].set_ylabel('Llamadas / día'); axes[0].legend(fontsize=9)
axes[0].set_title('Boxplot comparativo'); axes[0].grid(axis='y', alpha=0.3)

for d, col, lbl in zip(datos, colores, etiquetas):
    axes[1].hist(d, bins=18, density=True, alpha=0.45,
                 color=col, label=lbl, edgecolor='white', lw=0.4)
axes[1].axvline(umbral, color='crimson', ls='--', lw=1.5)
axes[1].set_xlabel('Llamadas / día'); axes[1].set_ylabel('Densidad')
axes[1].set_title('Histograma superpuesto'); axes[1].legend(fontsize=9)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('C.2 — Análisis de sensibilidad: Base vs ±10% en λ',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('c2_escenarios_comparativo.png', dpi=150, bbox_inches='tight')
plt.close()
