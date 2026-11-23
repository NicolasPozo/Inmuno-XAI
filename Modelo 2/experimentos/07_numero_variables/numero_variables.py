# ==============================================================================
# ¿CUÁNTAS VARIABLES NECESITA TabPFN? (etapa 12)
# Proyecto: Immuno-XAI - UTEM
#
# Las 45 variables del modelo final vienen de los archivos k45, seleccionadas al
# inicio del proyecto con otros modelos y con datos aumentados. Aquí se revisa
# si 45 es el número adecuado para TabPFN.
#
# Candidatas: los 71 biomarcadores de pareados.csv con >= 80 % de datos en el
# train (las 45 actuales + 8 poblaciones celulares + 18 citoquinas). El sexo
# siempre se incluye.
# Orden de las variables: |correlación de Spearman con la edad|, calculada SOLO
# con los pacientes de entrenamiento de cada partición (sin fuga).
#
# Fase 1 - elegir (solo train): CV agrupada 5x3 sobre los 354 pacientes de train.
#   a) k mejores de las 71 candidatas (k = 10 ... 71)
#   b) k mejores dentro de las 45 actuales (k = 15 ... 45; k = 45 es el modelo actual)
# Fase 2 - confirmar: el mejor k de cada curva (si no es el actual) se evalúa
#   igual que el modelo final: CV 5x5 sobre 473, test de 119 y estudios no vistos,
#   y se compara con el modelo actual (bootstrap pareado en el test).
#
# Requiere: Datasets/pareados.csv y resultados/modelo_final/ (src/06).
# Uso:  python -X utf8 experimentos/07_numero_variables/numero_variables.py   (~1 h en CPU)
# Salida: experimentos/07_numero_variables/resultados/
# ==============================================================================

import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import (  # noqa: E402  (rutas del proyecto: src/config.py)
    ARCHIVO_ELISA_10K, ARCHIVO_PAREADOS, ARCHIVO_TEST_K45, CARPETA_RES_MODELO_FINAL, EXP_NUMERO_VARIABLES)
carpeta = EXP_NUMERO_VARIABLES / "resultados"
os.makedirs(carpeta, exist_ok=True)
torch.set_num_threads(os.cpu_count())
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)

from tabpfn import TabPFNRegressor  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

def ajustar_predecir(Xt, yt, Xv):
    m = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)
    return m.fit(Xt, yt).predict(Xv)

def metricas(yv, p):
    return {"MAE": mean_absolute_error(yv, p), "RMSE": np.sqrt(mean_squared_error(yv, p)), "R2": r2_score(yv, p),
            "r": np.corrcoef(yv, p)[0, 1], "acierto_5a_%": np.mean(np.abs(yv - p) <= 5) * 100}

# ------------------------------------------------------------------------------
# Datos (los mismos del modelo final)
# ------------------------------------------------------------------------------
par = pd.read_csv(ARCHIVO_PAREADOS)
el = pd.read_csv(ARCHIVO_ELISA_10K)
k_ = el.assign(edad=el.age.round(2))[["subject_accession", "edad", "study_accession"]].drop_duplicates(["subject_accession", "edad"])
par = par.assign(edad=par.age.round(2)).merge(k_, on=["subject_accession", "edad"], how="left")
y = par["age"].values.astype(float)
grupos = par["subject_accession"].values
estudios = par["study_accession"].fillna("otro").values
idx_tr = np.where(par["conjunto"] == "train")[0]
idx_te = np.where(par["conjunto"] == "test")[0]
sexo = (par["gender"] == "Male").astype(float).values.reshape(-1, 1)

test_k45 = pd.read_csv(ARCHIVO_TEST_K45)
VARS45 = [c for c in test_k45.columns if c not in ["age", "gender", "source_type", "subject_accession", "race"]]
meta = ["subject_accession", "gender", "age", "conjunto", "edad", "study_accession"]
extra = [c for c in par.columns if c not in meta + VARS45 and par.loc[idx_tr, c].notna().mean() >= 0.8]
POOL = VARS45 + extra
log(f"Candidatas: {len(POOL)} (45 actuales + {len(extra)} nuevas: {', '.join(extra)})")

def ranking(idx, pool):
    """Variables ordenadas por |Spearman| con la edad, usando solo los pacientes idx."""
    rho = par.iloc[idx][pool].corrwith(pd.Series(y[idx], index=par.index[idx]), method="spearman").abs()
    return rho.sort_values(ascending=False).index.tolist()

def matriz(cols, idx):
    return np.hstack([par.iloc[idx][cols].astype(float).values, sexo[idx]]).astype(np.float32)

def predecir(pool, k, tr, va):
    cols = ranking(tr, pool)[:k]
    return ajustar_predecir(matriz(cols, tr), y[tr], matriz(cols, va))

# ------------------------------------------------------------------------------
# Fase 1: curvas con CV 5x3 solo sobre el train
# ------------------------------------------------------------------------------
REP1 = 3
SPLITS1 = [[(idx_tr[a], idx_tr[b]) for a, b in
            GroupKFold(n_splits=5, shuffle=True, random_state=r).split(idx_tr, groups=grupos[idx_tr])]
           for r in range(REP1)]
CONFIGS = ([("Las 45 actuales", VARS45, k) for k in [45, 40, 35, 30, 25, 20, 15]]
           + [("Candidatas (71)", POOL, k) for k in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, len(POOL)]])

cache = carpeta / "fase1_predicciones.npz"
F1 = dict(np.load(cache, allow_pickle=True))["F1"].item() if cache.exists() else {}
for fam, pool, k in CONFIGS:
    clave = f"{fam}|{k}"
    if clave in F1:
        continue
    t = time.time()
    oof = np.full((REP1, len(y)), np.nan)
    for r in range(REP1):
        for tr, va in SPLITS1[r]:
            oof[r, va] = predecir(pool, k, tr, va)
    F1[clave] = oof
    np.savez(cache, F1=np.array(F1, dtype=object))
    mae = [mean_absolute_error(y[idx_tr], oof[r, idx_tr]) for r in range(REP1)]
    log(f"{fam:<16s} k={k:<3d} | MAE CV-train {np.mean(mae):.3f} ± {np.std(mae):.3f} | {(time.time()-t)/60:.1f} min")

ref = F1["Las 45 actuales|45"]
filas = []
for fam, pool, k in CONFIGS:
    oof = F1[f"{fam}|{k}"]
    mae = np.array([mean_absolute_error(y[idx_tr], oof[r, idx_tr]) for r in range(REP1)])
    dif = mae - np.array([mean_absolute_error(y[idx_tr], ref[r, idx_tr]) for r in range(REP1)])
    filas.append({"candidatas": fam, "k": k, "variables": k + 1, "MAE_CV_train": mae.mean(), "MAE_sd": mae.std(),
                  "R2_CV_train": np.mean([r2_score(y[idx_tr], oof[r, idx_tr]) for r in range(REP1)]),
                  "dif_vs_actual": dif.mean(), "dif_vs_actual_min": dif.min(), "dif_vs_actual_max": dif.max()})
curva = pd.DataFrame(filas).round(3)
curva.to_csv(carpeta / "curva_numero_variables.csv", index=False)

# Orden de las 71 con todo el train (solo para mostrar; las curvas ordenan dentro de cada partición)
rho = par.iloc[idx_tr][POOL].corrwith(pd.Series(y[idx_tr], index=par.index[idx_tr]), method="spearman")
pd.DataFrame({"variable": rho.abs().sort_values(ascending=False).index,
              "spearman_edad": rho[rho.abs().sort_values(ascending=False).index].values,
              "en_las_45": [v in VARS45 for v in rho.abs().sort_values(ascending=False).index]}).round(3).to_csv(
    carpeta / "orden_variables_train.csv", index=False)

# ------------------------------------------------------------------------------
# Fase 2: confirmar el mejor k de cada curva frente al modelo actual
# ------------------------------------------------------------------------------
REP2 = 5
SPLITS2 = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REP2)]
LOSO = ["SDY420", "SDY515", "SDY519", "SDY113", "SDY311"]
act_met = pd.read_csv(CARPETA_RES_MODELO_FINAL / "metricas_modelo_final.csv").set_index("evaluacion")
act_cv = np.load(CARPETA_RES_MODELO_FINAL / "predicciones_cv.npz", allow_pickle=True)["oof"]
act_te = pd.read_csv(CARPETA_RES_MODELO_FINAL / "predicciones_test.csv")["edad_predicha"].values
act_lo = pd.read_csv(CARPETA_RES_MODELO_FINAL / "metricas_estudios_no_vistos.csv").set_index("estudio_no_visto")

conf = [{"configuracion": "Actual: las 45", "variables": 46,
         "MAE_CV_473": act_met.loc["CV 5x5 (473 pacientes)", "MAE"], "dif_CV_vs_actual": 0.0,
         "MAE_test": act_met.loc["Test (119 pacientes)", "MAE"], "R2_test": act_met.loc["Test (119 pacientes)", "R2"],
         "MAE_estudios_no_vistos": act_met.loc["Estudios no vistos (promedio de 5)", "MAE"],
         "r_estudios_no_vistos": act_met.loc["Estudios no vistos (promedio de 5)", "r"],
         **{f"MAE_{e}": act_lo.loc[e, "MAE"] for e in LOSO}}]
mejores = curva.loc[curva.groupby("candidatas")["MAE_CV_train"].idxmin()]
for _, fila in mejores.iterrows():
    if fila.candidatas == "Las 45 actuales" and fila.k == 45:
        log("En las 45 actuales, el mejor k es 45: no hay nada que confirmar.")
        continue
    pool = VARS45 if fila.candidatas == "Las 45 actuales" else POOL
    k, nombre = int(fila.k), f"Top {int(fila.k)} de {fila.candidatas.lower()}"
    log(f"Confirmando: {nombre}")
    oof = np.zeros((REP2, len(y)))
    for r in range(REP2):
        for tr, va in SPLITS2[r]:
            oof[r, va] = predecir(pool, k, tr, va)
    p_te = predecir(pool, k, idx_tr, idx_te)
    lo = {e: metricas(y[estudios == e], predecir(pool, k, np.where(estudios != e)[0], np.where(estudios == e)[0]))
          for e in LOSO}
    dif_cv = np.array([mean_absolute_error(y, oof[r]) - mean_absolute_error(y, act_cv[r]) for r in range(REP2)])
    e_new, e_act = np.abs(p_te - y[idx_te]), np.abs(act_te - y[idx_te])
    rng = np.random.default_rng(0)
    dif = np.array([(e_new[i] - e_act[i]).mean() for i in (rng.integers(0, len(e_new), len(e_new)) for _ in range(10000))])
    m_te = metricas(y[idx_te], p_te)
    conf.append({"configuracion": nombre, "variables": k + 1, "MAE_CV_473": np.mean([mean_absolute_error(y, oof[r]) for r in range(REP2)]),
                 "dif_CV_vs_actual": dif_cv.mean(), "MAE_test": m_te["MAE"], "R2_test": m_te["R2"],
                 "dif_test_vs_actual": dif.mean(), "IC95_bajo": np.percentile(dif, 2.5), "IC95_alto": np.percentile(dif, 97.5),
                 "MAE_estudios_no_vistos": np.mean([v["MAE"] for v in lo.values()]),
                 "r_estudios_no_vistos": np.mean([v["r"] for v in lo.values()]),
                 **{f"MAE_{e}": lo[e]["MAE"] for e in LOSO}})
    log(f"  CV 473 {conf[-1]['MAE_CV_473']:.3f} (dif {dif_cv.mean():+.3f}) | test {m_te['MAE']:.3f} "
        f"(dif {dif.mean():+.2f} [{np.percentile(dif, 2.5):+.2f}, {np.percentile(dif, 97.5):+.2f}]) | "
        f"estudios no vistos {conf[-1]['MAE_estudios_no_vistos']:.2f}")
confirmacion = pd.DataFrame(conf).round(3)
confirmacion.to_csv(carpeta / "confirmacion.csv", index=False)

# ------------------------------------------------------------------------------
# Gráfico: MAE de la CV sobre el train según el número de variables
# ------------------------------------------------------------------------------
AZUL, NARANJA, GRIS, TINTA, TINTA2, FONDO = "#2a78d6", "#eb6834", "#b9b8b1", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for fam, color in [("Candidatas (71)", AZUL), ("Las 45 actuales", NARANJA)]:
    d = curva[curva.candidatas == fam].sort_values("k")
    etiqueta = "k mejores de las 71 candidatas" if fam == "Candidatas (71)" else "k mejores dentro de las 45 actuales"
    ax.fill_between(d.k, d.MAE_CV_train - d.MAE_sd, d.MAE_CV_train + d.MAE_sd, color=color, alpha=0.12, lw=0)
    ax.plot(d.k, d.MAE_CV_train, color=color, lw=2, marker="o", ms=6, mec=FONDO, mew=1.5, label=etiqueta)
a = curva[(curva.candidatas == "Las 45 actuales") & (curva.k == 45)].iloc[0]
ax.scatter([45], [a.MAE_CV_train], s=120, facecolor="none", edgecolor=TINTA, lw=1.6, zorder=5)
ax.annotate(f"Modelo actual (45): {a.MAE_CV_train:.2f}", (45, a.MAE_CV_train), xytext=(-165, -26),
            textcoords="offset points", color=TINTA, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.8))
ax.set(xlabel="Número de biomarcadores (k, más el sexo)", ylabel="MAE CV 5×3 en el train (años)")
ax.grid(color="#e8e7e3", lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
ax.set_title("TabPFN: error según el número de variables (solo train, banda = ±1 DE)", color=TINTA, loc="left", fontsize=11.5)
fig.tight_layout()
fig.savefig(carpeta / "curva_numero_variables.png", dpi=200)
plt.close(fig)

pd.set_option("display.width", 250)
print("\n" + curva.to_string(index=False))
print("\n" + confirmacion.to_string(index=False))
log("Listo.")
