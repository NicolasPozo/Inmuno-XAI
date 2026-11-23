# ==============================================================================
# PASO 10: FIGURAS Y CIFRAS DEL INFORME (EXPERIMENTO A)
# Proyecto: Immuno-XAI - UTEM
#
# Genera las figuras del Experimento A que usa el informe de título
# (Modelo 1/Informe) y las cifras que se citan en el texto:
#   1. Edad por sexo (Mann-Whitney U)                       -> Ilustración 11
#   2. Curva de soporte: biomarcadores según su completitud -> Ilustración 13
#   3. Porcentaje de datos faltantes por biomarcador         -> Ilustración 14
#   4. Asimetría de los 45 biomarcadores                     -> Ilustración 15
#   5. Correlación cruzada células x citoquinas              -> Ilustración 16
#   6. Cinética de los 4 biomarcadores más ligados a la edad -> Ilustración 17
#   7. Edad según el estudio de origen (firma del estudio)
#   8. Edad real vs predicha en el test para los 7 modelos  -> Ilustración 26
#   9. Curva de aprendizaje con y sin aumento de datos      -> Ilustración 19
# Además mide el tiempo de ajuste de cada modelo (búsqueda de hiperparámetros +
# ajuste final con los 354 de train), con la misma configuración del paso 05.
#
# Requiere: Datasets/pareados.csv (paso 04) y resultados/comparacion_modelos/ (paso 05).
# Uso:  python -X utf8 src/10_figuras_informe.py              (~5 min en CPU)
#       python -X utf8 src/10_figuras_informe.py --sin-tiempos   (solo figuras, ~10 s)
# Salida: resultados/informe/ (PNG, cifras_informe.json, tiempos_ajuste.csv)
# ==============================================================================

import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu, skew, spearmanr

warnings.filterwarnings("ignore")
from config import (
    ARCHIVO_CYTOF_10K,
    ARCHIVO_ELISA_10K,
    ARCHIVO_PAREADOS,
    ARCHIVO_TEST_K45,
    EXP_MAS_PACIENTES,
    CARPETA_RES_COMPARACION,
    CARPETA_RES_INFORME,
)  # rutas del proyecto: src/config.py

carpeta = CARPETA_RES_INFORME
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

AZUL, NARANJA, GRIS, TINTA, TINTA2, FONDO, REJILLA = (
    "#2a78d6", "#eb6834", "#b9b8b1", "#0b0b0b", "#52514e", "#fcfcfb", "#e8e7e3")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
DIVERGENTE = LinearSegmentedColormap.from_list("div", [AZUL, "#f2f1ee", NARANJA])


def rejilla(ax, eje="both"):
    ax.grid(axis=eje, color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(carpeta / nombre)
    plt.close(fig)
    log(f"  {nombre}")


# ------------------------------------------------------------------------------
# Datos (los mismos de los pasos 05 y 06)
# ------------------------------------------------------------------------------
par = pd.read_csv(ARCHIVO_PAREADOS)
el = pd.read_csv(ARCHIVO_ELISA_10K)
cy = pd.read_csv(ARCHIVO_CYTOF_10K)
k = el.assign(edad=el.age.round(2))[["subject_accession", "edad", "study_accession"]].drop_duplicates(["subject_accession", "edad"])
par = par.assign(edad=par.age.round(2)).merge(k, on=["subject_accession", "edad"], how="left")
y = par["age"].values.astype(float)
estudio = par["study_accession"].fillna("otro")
hombre = (par["gender"] == "Male").values

BIOMARCADORES = list(par.columns[4:par.columns.get_loc("edad")])
CELULAS_TODAS = [c for c in BIOMARCADORES if c in cy.columns]
test_k45 = pd.read_csv(ARCHIVO_TEST_K45)
VARS45 = [c for c in test_k45.columns if c not in ["age", "gender", "source_type", "subject_accession", "race"]]
CEL45 = [c for c in VARS45 if c in CELULAS_TODAS]
CIT45 = [c for c in VARS45 if c not in CELULAS_TODAS]
nombre_corto = lambda c: c.replace("_T_cells", " T").replace("_cells", "").replace("_", " ").replace("IFN1.", "IFN1")
cifras = {}

# ------------------------------------------------------------------------------
# Cifras de volumen (archivos normalizados del 10K) y demografía
# ------------------------------------------------------------------------------
cifras["volumen"] = {
    "cytof_filas": int(len(cy)), "cytof_sujetos": int(cy.subject_accession.nunique()),
    "elisa_filas": int(len(el)), "elisa_sujetos": int(el.subject_accession.nunique()),
    "pareados": int(len(par)), "train": int((par.conjunto == "train").sum()), "test": int((par.conjunto == "test").sum()),
    "biomarcadores": len(BIOMARCADORES), "celulas": len(CELULAS_TODAS), "citoquinas": len(BIOMARCADORES) - len(CELULAS_TODAS),
    "modelo_celulas": len(CEL45), "modelo_citoquinas": len(CIT45),
}
ed_h, ed_m = y[hombre], y[~hombre]
mw = mannwhitneyu(ed_h, ed_m)
q1, q3 = np.percentile(y, [25, 75])
cifras["demografia"] = {
    "N": int(len(y)), "mediana": float(np.median(y)), "iqr": float(q3 - q1), "media": float(y.mean()), "de": float(y.std(ddof=1)),
    "min": float(y.min()), "max": float(y.max()), "mujeres": int((~hombre).sum()), "hombres": int(hombre.sum()),
    "mediana_hombres": float(np.median(ed_h)), "mediana_mujeres": float(np.median(ed_m)),
    "U": float(mw.statistic), "p": float(mw.pvalue),
    "pct_menor30": float(100 * np.mean(y < 30)), "pct_80mas": float(100 * np.mean(y >= 80)),
    "pct_30_42": float(100 * np.mean((y >= 30) & (y < 42))), "n_edad_90": int((y >= 90).sum()),
    "mediana_train": float(np.median(y[par.conjunto == "train"])), "mediana_test": float(np.median(y[par.conjunto == "test"])),
}
log(f"Pacientes {len(y)} | biomarcadores {len(BIOMARCADORES)} | modelo {len(VARS45)} ({len(CEL45)} células, {len(CIT45)} citoquinas)")

# 1. Edad por sexo
fig, ax = plt.subplots(figsize=(7.2, 4.6))
rng = np.random.default_rng(0)
for i, (lab, v, col) in enumerate([("Mujeres", ed_m, NARANJA), ("Hombres", ed_h, AZUL)]):
    ax.boxplot(v, positions=[i], widths=0.45, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="none", edgecolor=TINTA2, lw=1.2), medianprops=dict(color=TINTA, lw=2),
               whiskerprops=dict(color=TINTA2), capprops=dict(color=TINTA2))
    ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=14, color=col, alpha=0.55, lw=0, zorder=3)
    ax.text(i + 0.3, np.median(v), f"mediana {np.median(v):.1f}", va="center", color=TINTA, fontsize=9)
ax.set_xticks([0, 1], [f"Mujeres (n = {len(ed_m)})", f"Hombres (n = {len(ed_h)})"])
ax.set_ylabel("Edad cronológica (años)")
ax.set_xlim(-0.6, 1.9)
rejilla(ax, "y")
u_txt = f"{mw.statistic:,.0f}".replace(",", ".")
ax.set_title(f"Edad por sexo biológico, {len(y)} pacientes (Mann-Whitney U = {u_txt}; p = {mw.pvalue:.3f})",
             color=TINTA, loc="left", fontsize=11)
guardar(fig, "01_edad_por_sexo.png")

# ------------------------------------------------------------------------------
# 2-3. Datos faltantes
# ------------------------------------------------------------------------------
completo = par[BIOMARCADORES].notna().mean()
umbrales = np.round(np.arange(0.50, 1.0001, 0.01), 2)
retenidos = [int((completo >= u).sum()) for u in umbrales]
cifras["faltantes"] = {
    "completos_100": int((completo == 1).sum()), "con_80": int((completo >= 0.8).sum()), "con_70": int((completo >= 0.7).sum()),
    "vacios_95": int((completo < 0.05).sum()), "entre_5_y_80": int(((completo >= 0.05) & (completo < 0.8)).sum()),
    "faltante_max_45": float(100 * (1 - completo[VARS45]).max()), "var_faltante_max_45": nombre_corto(completo[VARS45].idxmin()),
    "faltante_medio_45": float(100 * (1 - completo[VARS45]).mean()),
    "valores_faltantes_45": int(par[VARS45].isna().sum().sum()), "valores_45": int(par[VARS45].size),
    "celulas_con_faltantes": int((completo[CELULAS_TODAS] < 1).sum()),
    "faltante_IL6": float(100 * (1 - completo["IL6"])), "faltante_TNF": float(100 * (1 - completo["TNF"])),
}
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.plot(umbrales * 100, retenidos, color=AZUL, lw=2)
ax.axvline(80, color=TINTA2, ls="--", lw=1)
ax.text(80.8, cifras["faltantes"]["con_80"] + 8, f"80 %: {cifras['faltantes']['con_80']} biomarcadores", color=TINTA, fontsize=9)
ax.set(xlabel="Completitud mínima exigida por biomarcador (%)", ylabel="Biomarcadores que la cumplen", ylim=(0, len(BIOMARCADORES) + 5))
ax.axhline(len(BIOMARCADORES), color=GRIS, lw=0.8)
ax.text(50.5, len(BIOMARCADORES) + 1, f"{len(BIOMARCADORES)} biomarcadores en total", color=TINTA2, fontsize=8.5)
rejilla(ax)
ax.set_title(f"Curva de soporte: biomarcadores retenidos según su completitud ({len(y)} pacientes)", color=TINTA, loc="left", fontsize=11)
guardar(fig, "02_curva_soporte.png")

inc = (1 - completo[completo < 1]).sort_values() * 100
fig, ax = plt.subplots(figsize=(8.2, max(4.5, 0.16 * len(inc) + 1.2)))
colores = [AZUL if c in VARS45 else GRIS for c in inc.index]
ax.barh([nombre_corto(c) for c in inc.index], inc.values, color=colores, height=0.72)
ax.axvline(20, color=TINTA2, ls="--", lw=1)
ax.set_xlabel("Datos faltantes (% de los pacientes)")
ax.tick_params(axis="y", labelsize=7, length=0)
ax.set_ylim(-0.7, len(inc) - 0.3)
rejilla(ax, "x")
ax.scatter([], [], marker="s", color=AZUL, label="Usado por el modelo (45)")
ax.scatter([], [], marker="s", color=GRIS, label="No usado")
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
ax.set_title(f"Biomarcadores con datos faltantes ({len(inc)} de {len(BIOMARCADORES)}); línea: 20 % de faltantes",
             color=TINTA, loc="left", fontsize=10.5)
guardar(fig, "03_faltantes_por_biomarcador.png")

# ------------------------------------------------------------------------------
# 4. Asimetría de los 45
# ------------------------------------------------------------------------------
asim = pd.Series({c: skew(par[c].dropna()) for c in VARS45}).sort_values(ascending=False)
cifras["asimetria"] = {"severa": int((asim > 1).sum()), "moderada": int(((asim > 0.5) & (asim <= 1)).sum()),
                       "simetrica": int((asim <= 0.5).sum()), "top": [[nombre_corto(c), float(v)] for c, v in asim.head(3).items()],
                       "severa_cel": int((asim[CEL45] > 1).sum()), "severa_cit": int((asim[CIT45] > 1).sum())}
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.bar(range(len(asim)), asim.values, color=[AZUL if c in CEL45 else NARANJA for c in asim.index], width=0.75)
for v, ls in [(1, "--"), (0.5, ":")]:
    ax.axhline(v, color=TINTA2, ls=ls, lw=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_xticks(range(len(asim)), [nombre_corto(c) for c in asim.index], rotation=90, fontsize=7)
ax.set_ylabel("Coeficiente de asimetría (escala symlog)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_yticks([-1, 0, 0.5, 1, 2, 5, 10, 20])
ax.tick_params(axis="x", length=0)
rejilla(ax, "y")
ax.bar([0], [0], color=AZUL, label="Población celular")
ax.bar([0], [0], color=NARANJA, label="Citoquina")
ax.legend(frameon=False, loc="upper right")
ax.set_title(f"Asimetría de Fisher-Pearson de los 45 biomarcadores ({cifras['asimetria']['severa']} con asimetría > 1)",
             color=TINTA, loc="left", fontsize=11)
guardar(fig, "04_asimetria.png")

# ------------------------------------------------------------------------------
# 5-6. Correlaciones con la edad
# ------------------------------------------------------------------------------
rho = par[VARS45].corrwith(pd.Series(y, index=par.index), method="spearman")
top_cel = rho[CEL45].abs().sort_values(ascending=False).index[:8].tolist()
top_cit = rho[CIT45].abs().sort_values(ascending=False).index[:8].tolist()
cruz = pd.DataFrame([[spearmanr(par[a], par[b], nan_policy="omit")[0] for b in top_cit] for a in top_cel],
                    index=top_cel, columns=top_cit)
corr45 = par[VARS45].corr(method="spearman")
pares = [(a, b, corr45.loc[a, b]) for i, a in enumerate(VARS45) for b in VARS45[i + 1:]]
cifras["correlaciones"] = {
    "rho_edad": {nombre_corto(c): float(v) for c, v in rho.sort_values().items()},
    "pares_colineales_070": int(sum(abs(v) >= 0.7 for _, _, v in pares)),
    "pares_colineales_lista": [[nombre_corto(a), nombre_corto(b), float(v)] for a, b, v in sorted(pares, key=lambda t: -abs(t[2])) if abs(v) >= 0.7],
    "cruz_max_abs": float(cruz.abs().values.max()), "cruz_media_abs": float(cruz.abs().values.mean()),
    "cruz": {nombre_corto(a): {nombre_corto(b): float(cruz.loc[a, b]) for b in top_cit} for a in top_cel},
}
fig, ax = plt.subplots(figsize=(8.4, 6.2))
lim = max(0.3, np.ceil(cruz.abs().values.max() * 10) / 10)
im = ax.imshow(cruz.values, cmap=DIVERGENTE, vmin=-lim, vmax=lim)
ax.set_xticks(range(len(top_cit)), [nombre_corto(c) for c in top_cit], rotation=45, ha="right")
ax.set_yticks(range(len(top_cel)), [nombre_corto(c) for c in top_cel])
for i in range(len(top_cel)):
    for j in range(len(top_cit)):
        ax.text(j, i, f"{cruz.values[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=TINTA)
ax.spines[:].set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("ρ de Spearman")
cb.outline.set_visible(False)
ax.set_title("Correlación entre las 8 poblaciones celulares y las 8 citoquinas\nmás ligadas a la edad (473 pacientes)",
             color=TINTA, loc="left", fontsize=11)
guardar(fig, "05_correlacion_celulas_citoquinas.png")

cardinales = rho[CEL45].abs().sort_values(ascending=False).index[:2].tolist() + rho[CIT45].abs().sort_values(ascending=False).index[:2].tolist()
cifras["cardinales"] = [[nombre_corto(c), float(rho[c])] for c in cardinales]
fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
for ax, c in zip(axes.ravel(), cardinales):
    ok = par[c].notna().values
    xv, yv = y[ok], par[c].values[ok]
    col = AZUL if c in CEL45 else NARANJA
    if c in CIT45:  # colas largas: escala log10(1 + valor)
        yv = np.log10(1 + np.clip(yv, 0, None))
    ax.scatter(xv, yv, s=12, color=col, alpha=0.5, lw=0)
    b, a = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 100)
    res = yv - (a + b * xv)
    s = np.sqrt((res ** 2).sum() / (len(xv) - 2))
    se = s * np.sqrt(1 / len(xv) + (xs - xv.mean()) ** 2 / ((xv - xv.mean()) ** 2).sum())
    ax.plot(xs, a + b * xs, color=TINTA, lw=1.6)
    ax.fill_between(xs, a + b * xs - 1.96 * se, a + b * xs + 1.96 * se, color=TINTA, alpha=0.12, lw=0)
    ax.set_title(f"{nombre_corto(c)}  (ρ = {rho[c]:+.3f})", color=TINTA, loc="left", fontsize=10.5)
    ax.set_xlabel("Edad cronológica (años)")
    ax.set_ylabel("log10(1 + valor normalizado)" if c in CIT45 else "Valor normalizado del 10K")
    rejilla(ax)
fig.suptitle("Biomarcadores con mayor correlación con la edad: 2 poblaciones celulares (arriba) y 2 citoquinas (abajo)",
             color=TINTA, x=0.01, ha="left", fontsize=11.5)
guardar(fig, "06_cinetica_biomarcadores.png")

# ------------------------------------------------------------------------------
# 7. Edad según el estudio de origen
# ------------------------------------------------------------------------------
g = pd.DataFrame({"estudio": estudio, "edad": y})
cuenta = g.estudio.value_counts()
grandes = cuenta[cuenta >= 10].index
med = g[g.estudio.isin(grandes)].groupby("estudio").edad.median().sort_values()
cifras["estudios"] = {"n_estudios": int(g.estudio.nunique()), "n_grandes": int(len(grandes)),
                      "detalle": {e: {"n": int(cuenta[e]), "min": float(g[g.estudio == e].edad.min()),
                                      "max": float(g[g.estudio == e].edad.max()), "mediana": float(np.median(g[g.estudio == e].edad))}
                                  for e in cuenta.index}}
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for i, e in enumerate(med.index):
    v = g[g.estudio == e].edad.values
    ax.boxplot(v, positions=[i], widths=0.55, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="none", edgecolor=TINTA2), medianprops=dict(color=TINTA, lw=1.8),
               whiskerprops=dict(color=TINTA2), capprops=dict(color=TINTA2))
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=AZUL, alpha=0.5, lw=0, zorder=3)
ax.set_xticks(range(len(med)), [f"{e}\n(n = {cuenta[e]})" for e in med.index], fontsize=8)
ax.set_ylabel("Edad cronológica (años)")
rejilla(ax, "y")
ax.set_title(f"Edad de los pacientes según su estudio de origen ({len(grandes)} estudios con 10 o más pacientes)",
             color=TINTA, loc="left", fontsize=11)
guardar(fig, "07_edad_por_estudio.png")

# ------------------------------------------------------------------------------
# 8. Edad real vs predicha en el test (7 modelos, paso 05)
# ------------------------------------------------------------------------------
pt = pd.read_csv(CARPETA_RES_COMPARACION / "predicciones_test.csv")
modelos = [c for c in pt.columns if c not in ["subject_accession", "edad_real"]]
yt = pt["edad_real"].values
met = {m: (1 - ((pt[m] - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum(), np.abs(pt[m] - yt).mean()) for m in modelos}
orden = sorted(modelos, key=lambda m: met[m][1])
cifras["test_7_modelos"] = {m: {"R2": float(met[m][0]), "MAE": float(met[m][1])} for m in orden}
lo, hi = 0, 100
fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
for ax, m in zip(axes.ravel(), orden):
    final = m.startswith("TabPFN")
    ax.plot([lo, hi], [lo, hi], color=TINTA2, ls="--", lw=1, label="Ideal (y = x)")
    ax.scatter(yt, pt[m], s=18, color=AZUL if final else TINTA2, alpha=0.75 if final else 0.55, lw=0.4, edgecolor=FONDO)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    ax.set_title(f"{m}{' (final)' if final else ''}\nR² = {met[m][0] * 100:.1f} % | MAE = {met[m][1]:.2f} años",
                 color=TINTA, loc="left", fontsize=10, fontweight="bold" if final else "normal")
    rejilla(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("Edad inmune predicha (años)")
for ax in axes[1, :3]:
    ax.set_xlabel("Edad cronológica real (años)")
axes[0, 3].set_xlabel("Edad cronológica real (años)")
ax = axes[1, 3]
vals = [met[m][1] for m in orden][::-1]
nombres = [m for m in orden][::-1]
ax.barh(nombres, vals, color=[AZUL if n.startswith("TabPFN") else GRIS for n in nombres], height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=8.5, color=TINTA)
ax.set_xlim(0, max(vals) * 1.2)
ax.tick_params(axis="y", labelsize=8.5, length=0)
ax.set_xlabel("MAE en el test (años)")
ax.set_title("Resumen: MAE en el test", color=TINTA, loc="left", fontsize=10)
rejilla(ax, "x")
fig.suptitle(f"Evaluación en pacientes reales de test (N = {len(yt)}; 45 biomarcadores + sexo)",
             color=TINTA, x=0.01, ha="left", fontsize=13)
guardar(fig, "08_evaluacion_test_7_modelos.png")

# ------------------------------------------------------------------------------
# 9. Curva de aprendizaje con y sin aumento (experimentos/02_mas_pacientes)
# ------------------------------------------------------------------------------
ca = pd.read_csv(EXP_MAS_PACIENTES / "resultados" / "curva_aprendizaje_resumen.csv")
fig, ax = plt.subplots(figsize=(8.4, 4.8))
for var, col in [("Con aumento", AZUL), ("Sin aumento", NARANJA)]:
    d = ca[ca.variante == var].sort_values("pacientes_reales")
    ax.fill_between(d.pacientes_reales, d.MAE_media - d.MAE_sd, d.MAE_media + d.MAE_sd, color=col, alpha=0.12, lw=0)
    ax.plot(d.pacientes_reales, d.MAE_media, color=col, lw=2, marker="o", ms=7, mec=FONDO, mew=1.5,
            label=f"{var} de datos sintéticos")
    u = d.iloc[-1]
    ax.text(u.pacientes_reales + 6, u.MAE_media, f"{u.MAE_media:.2f}", va="center", color=TINTA, fontsize=9.5)
ax.set(xlabel="Pacientes reales en entrenamiento", ylabel="MAE en el test (años)", xlim=(70, 385))
rejilla(ax)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Curva de aprendizaje de LightGBM con y sin aumento de datos (test de 119; banda = ±1 DE)",
             color=TINTA, loc="left", fontsize=11)
guardar(fig, "09_curva_aprendizaje.png")
cifras["curva_aprendizaje"] = ca.round(3).to_dict(orient="records")

# ------------------------------------------------------------------------------
# Tiempos de ajuste (misma configuración del paso 05)
# ------------------------------------------------------------------------------
if "--sin-tiempos" not in sys.argv:
    log("Midiendo tiempos de ajuste (búsqueda de hiperparámetros + ajuste final con los 354 de train)...")
    import importlib.util
    import torch
    from sklearn.base import clone
    spec = importlib.util.spec_from_file_location("comparar", os.path.join(os.path.dirname(__file__), "05_comparar_modelos.py"))
    codigo = open(spec.origin, encoding="utf-8").read()
    corte = codigo.index("for nombre, (est, grilla) in CLASICOS.items():")
    ns = {"__file__": spec.origin, "__name__": "comparar"}
    exec(compile(codigo[:corte], spec.origin, "exec"), ns)  # datos, CLASICOS y afinar() del paso 05
    X, yy, itr, ite = ns["X"], ns["y"], ns["idx_tr"], ns["idx_te"]
    tiempos = []
    for nombre, (est, grilla) in ns["CLASICOS"].items():
        t = time.time()
        modelo, _ = ns["afinar"](est, grilla)
        clone(modelo).fit(X[itr], yy[itr]).predict(X[ite])
        tiempos.append({"modelo": nombre, "segundos": time.time() - t, "ajustes_cv": ns["N_ITER"] * 5})
        log(f"  {nombre:<14s} {tiempos[-1]['segundos']:.1f} s")
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(os.cpu_count())
    t = time.time()
    TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0).fit(X[itr], yy[itr]).predict(X[ite])
    tiempos.append({"modelo": "TabPFN v2", "segundos": time.time() - t, "ajustes_cv": 0})
    log(f"  TabPFN v2      {tiempos[-1]['segundos']:.1f} s")
    pd.DataFrame(tiempos).round(2).to_csv(carpeta / "tiempos_ajuste.csv", index=False)

json.dump(cifras, open(carpeta / "cifras_informe.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("Listo.")
