import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# EJERCICIO A.2 — Test Chi-cuadrado + Histograma

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

# Parámetros del test
N, k = 1000, 10
E = N / k  # frecuencia esperada por clase = 100

# Contar frecuencias por intervalo
observados = [0] * k
for u in numeros:
    i = min(int(u * k), k - 1)
    observados[i] += 1

# Imprimir tabla completa
print("=" * 62)
print("EJERCICIO A.2 — TEST CHI-CUADRADO")
print("=" * 62)
print(f"N = {N} números generados con LCG (semilla = {seed})")
print(f"k = {k} clases    |    E_i = N/k = {E:.0f} por clase")
print()
print(f"{'Intervalo':<14} {'O_i':>6} {'E_i':>6} {'(O-E)²':>10} {'(O-E)²/E':>10}")
print("-" * 50)

chi2 = 0
for i in range(k):
    inf = i / k
    sup = (i + 1) / k
    Oi = observados[i]
    diff2 = (Oi - E) ** 2
    termino = diff2 / E
    chi2 += termino
    print(f"[{inf:.1f} , {sup:.1f})   {Oi:>6} {E:>6.0f} {diff2:>10.1f} {termino:>10.4f}")

print("-" * 50)
print(f"{'TOTAL':<14} {sum(observados):>6} {N:>6}                χ² = {chi2:.4f}")
print()
print(f"Grados de libertad : gl = k - 1 = {k} - 1 = {k-1}")
print(f"Nivel significancia : α = 0.05")
print(f"Valor crítico       : χ²(9, 0.05) = 16.919  (tabla)")
print()
print("=" * 62)
print("DECISIÓN")
print("=" * 62)
print(f"χ² calculado = {chi2:.4f}")
print(f"χ² crítico   = 16.919")
print(f"{chi2:.4f} {'≤' if chi2 <= 16.919 else '>'} 16.919")
if chi2 <= 16.919:
    print("→ No se rechaza H₀")
    print("→ El generador PASA el test de uniformidad ✓")
else:
    print("→ Se rechaza H₀")
    print("→ El generador FALLA el test ✗")


# HISTOGRAMA GRÁFICO
intervalos = [f"[{i/k:.1f},{(i+1)/k:.1f})" for i in range(k)]
colores = ["#2E75B6" if o <= E else "#1F4E79" for o in observados]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#F8F8F8")
ax.set_facecolor("#F8F8F8")

barras = ax.bar(intervalos, observados, color=colores,
                edgecolor="white", linewidth=1.2, width=0.7, zorder=3)

# Línea de frecuencia esperada
ax.axhline(y=E, color="#E24B4A", linewidth=2.5,
           linestyle="--", zorder=4)

# Valor encima de cada barra
for barra, obs in zip(barras, observados):
    diff = obs - int(E)
    signo = f"+{diff}" if diff > 0 else str(diff)
    ax.text(barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 1.5,
            f"{obs}\n({signo})",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold",
            color="#1F4E79")

# Zona de variación normal
ax.axhspan(80, 120, alpha=0.08, color="green", zorder=1)

# Ejes y grilla
ax.set_ylim(60, 140)
ax.set_yticks(range(60, 145, 10))
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# Etiquetas
ax.set_xlabel("Intervalo", fontsize=12, labelpad=8)
ax.set_ylabel("Frecuencia observada (Oᵢ)", fontsize=12, labelpad=8)
ax.set_title(
    "Ejercicio A.2 — Test Chi-Cuadrado: Frecuencias observadas vs esperadas\n"
    f"LCG: a=1.738.019.701, c=2.531.011, m=2³², semilla=2026  |  "
    f"χ²={chi2:.2f}  |  χ²crítico=16.919  →  PASA ✓",
    fontsize=11, pad=14
)

# Leyenda
p1 = mpatches.Patch(color="#2E75B6", label="Oᵢ ≤ 100 (debajo del esperado)")
p2 = mpatches.Patch(color="#1F4E79", label="Oᵢ > 100 (encima del esperado)")
p3 = mpatches.Patch(color="#E24B4A", label="Frecuencia esperada E = 100")
p4 = mpatches.Patch(color="green",   alpha=0.2, label="Zona de variación normal")
ax.legend(handles=[p1, p2, p3, p4], loc="upper left", fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.savefig("A2_histograma_chi2.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
