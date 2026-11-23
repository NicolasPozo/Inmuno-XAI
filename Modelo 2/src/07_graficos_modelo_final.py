# ==============================================================================
# GRÁFICOS DEL MODELO FINAL (TabPFN un modelo) Y DE LA EVOLUCIÓN DEL PROYECTO
# Requiere haber ejecutado src/06_modelo_final_tabpfn.py
# Salida: resultados/modelo_final/*.png
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CARPETA_RES_MODELO_FINAL  # rutas del proyecto: src/config.py
carpeta = CARPETA_RES_MODELO_FINAL
AZUL, GRIS, TINTA, TINTA2, FONDO = "#2a78d6", "#b9b8b1", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
                     "axes.spines.top": False, "axes.spines.right": False})

met = pd.read_csv(os.path.join(carpeta, "metricas_modelo_final.csv")).set_index("evaluacion")
pred = pd.read_csv(os.path.join(carpeta, "predicciones_test.csv"))
t = met.loc["Test (119 pacientes)"]

# ------------------------------------------------------------------------------
# 1. Edad real vs predicha en el test
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 6))
lim = [0, 100]
ax.fill_between(lim, [l - 5 for l in lim], [l + 5 for l in lim], color=AZUL, alpha=0.08, lw=0, label="±5 años")
ax.plot(lim, lim, color=TINTA2, lw=1.2, ls="--", label="Predicción perfecta")
ax.scatter(pred.edad_real, pred.edad_predicha, s=36, color=AZUL, edgecolor=FONDO, linewidth=1, label="Paciente de test", zorder=3)
ax.set(xlim=lim, ylim=lim, xlabel="Edad real (años)", ylabel="Edad predicha (años)")
ax.grid(color="#e8e7e3", lw=0.6)
ax.set_title("TabPFN un modelo: test de 119 pacientes", color=TINTA, loc="left", fontsize=12)
ax.text(3, 97, f"MAE {t.MAE:.2f} años\nR² {t.R2:.3f}   r {t.r:.3f}\nAcierto ±5 años {t['acierto_5a_%']:.1f} %",
        va="top", color=TINTA, fontsize=10)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(carpeta, "test_real_vs_predicha.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------------------
# 2. Evolución del error a lo largo del proyecto (dos paneles, una sola escala cada uno)
# Fuentes (en experimentos/): 01_modelo_inicial/resultados/graficos_entrenamiento_multimodal*/
# comparativa_modelos_multimodal.csv, 03_banco_pruebas/resultados/evaluacion_final.csv,
# 05_modelo_dos_ramas/resultados/modelo_final*/metricas_modelo_final.csv, 06_tabpfn/resultados/
# tabpfn_un_modelo/*.csv; y resultados/modelo_final/metricas_modelo_final.csv
# ------------------------------------------------------------------------------
lu = met.loc["Estudios no vistos (promedio de 5)"]
etapas_test = [
    ("1. LightGBM, train aumentado (inicio)", 8.94),
    ("2. LightGBM, solo pacientes reales", 8.35),
    ("3. Ensamble + corrección de sesgo (R8)", 8.34),
    ("4. Dos ramas + fusión (modelo anterior)", 8.69),
    ("5. TabPFN dos ramas", 7.27),
    ("6. TabPFN un modelo (FINAL)", t.MAE),
]
etapas_loso = [
    ("Dos ramas (modelo anterior)", 17.24),
    ("Dos ramas + ImmPort", 14.98),
    ("Dos ramas + ELISA extra", 14.29),
    ("TabPFN dos ramas", 16.82),
    ("TabPFN un modelo (FINAL)", lu.MAE),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
for ax, datos, titulo in [(axes[0], etapas_test, "Pacientes nuevos del 10K (test de 119)"),
                          (axes[1], etapas_loso, "Estudios no vistos (promedio de 5)")]:
    nombres = [d[0] for d in datos][::-1]
    valores = [d[1] for d in datos][::-1]
    colores = [AZUL if "FINAL" in n else GRIS for n in nombres]
    ax.barh(nombres, valores, color=colores, height=0.62)
    for i, v in enumerate(valores):
        ax.text(v + 0.2, i, f"{v:.2f}", va="center", color=TINTA, fontsize=9.5,
                fontweight="bold" if "FINAL" in nombres[i] else "normal")
    ax.set_xlim(0, max(valores) * 1.15)
    ax.set_xlabel("MAE (años, menor es mejor)")
    ax.set_title(titulo, color=TINTA, loc="left", fontsize=11.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
fig.suptitle("Evolución del error del modelo", color=TINTA, x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(carpeta, "evolucion_modelos.png"), dpi=200)
plt.close(fig)
print("Gráficos guardados en", carpeta)
