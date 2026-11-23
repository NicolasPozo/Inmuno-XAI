# Gráfico de la evaluación final (lee resultados_banco_pruebas/evaluacion_final.csv)
import os
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_banco_pruebas")
df = pd.read_csv(os.path.join(carpeta, "evaluacion_final.csv"))

TINTA, TINTA2, MUTED, GRID, SUP = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
AZUL = "#2a78d6"

df = df.iloc[::-1].reset_index(drop=True)  # base arriba
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=120)
fig.patch.set_facecolor(SUP); ax.set_facecolor(SUP)

for i, r in df.iterrows():
    es_base = r["receta"].startswith("R0")
    ax.plot([r["IC95_bajo"], r["IC95_alto"]], [i, i], color=AZUL, lw=2, solid_capstyle="round", alpha=0.0 if es_base else 1)
    ax.plot(r["dif_vs_base"], i, "o", color=MUTED if es_base else AZUL, markersize=9,
            markeredgecolor=SUP, markeredgewidth=2, zorder=3)
    texto = "referencia" if es_base else f"{r['dif_vs_base']:+.2f} años  (MAE {r['MAE_CV_473']:.2f})"
    ax.annotate(texto, xy=(max(r["IC95_alto"], r["dif_vs_base"], 0), i), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=TINTA2)

ax.axvline(0, color="#c3c2b7", lw=1)
ax.set_yticks(range(len(df)))
ax.set_yticklabels([r.split(" ", 1)[1] if not r.startswith("R0") else "Base: LightGBM | 45 originales" for r in df["receta"]],
                   fontsize=9, color=TINTA)
ax.set_xlabel("Cambio en el error medio vs. la base (años; negativo = mejor)", color=TINTA2, fontsize=10)
fig.suptitle("Evaluación final de recetas: cambio del error vs. la base", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=TINTA)
fig.text(0.01, 0.905, "CV agrupada por paciente 5×5 sobre 473 pacientes · punto = cambio medio · línea = IC 95%",
         ha="left", fontsize=9, color=TINTA2)
ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
for lado in ["top", "right", "left"]:
    ax.spines[lado].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(axis="x", colors=MUTED, labelsize=9); ax.tick_params(axis="y", length=0)
ax.set_xlim(df["IC95_bajo"].min() - 0.15, 0.9)
fig.tight_layout(rect=(0, 0, 1, 0.88))
ruta = os.path.join(carpeta, "evaluacion_final.png")
fig.savefig(ruta, dpi=300, facecolor=SUP)
print("Gráfico:", ruta)
