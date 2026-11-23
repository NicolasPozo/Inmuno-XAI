# ==============================================================================
# MODELOS PESADOS EN EL PLAN B: TabPFN v2, TabM y RealMLP
# Proyecto: Immuno-XAI - UTEM
#
# TabPFN v2 : transformer pre-entrenado para tablas pequeñas (modelo fundacional)
# TabM      : ensamble eficiente de redes neuronales para tablas (pytabkit)
# RealMLP   : red neuronal con preprocesamiento y defaults afinados (pytabkit)
# Se comparan, en cada rama, contra los modelos del modelo final:
#   células = ensamble de 4 (afinado) | citoquinas = LightGBM (afinado)
#
# Mismos datos que el modelo final (solo 10K): rama células con el CyTOF extra
# del 10K; rama citoquinas solo con pareados. Los modelos pesados usan sus
# valores por defecto (están diseñados para eso).
#
# Evaluación: CV agrupada por paciente 3x5 sobre los 473 pareados + test de 119.
# Fusión: regresión lineal sobre las predicciones fuera de muestra de cada rama
# (pesos estimados con los pacientes de entrenamiento de cada partición).
# ==============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
REPETICIONES = 3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_modelos_pesados")
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)

# Reutiliza datos, variables y modelos afinados del modelo final
import sys
sys.argv = [sys.argv[0]]  # modelo final por defecto (sin ELISA extra)
# Se ejecuta modelo_final_dos_ramas.py solo hasta el afinado (datos, variables y modelos actuales)
codigo = open(ruta("modelo_final_dos_ramas.py"), encoding="utf-8").read()
corte = codigo.index("# ------------------------------------------------------------------------------\n# Dos ramas + fusión")
ns = {"__file__": ruta("modelo_final_dos_ramas.py"), "__name__": "mf"}
exec(compile(codigo[:corte], "modelo_final_dos_ramas.py", "exec"), ns)
y, grupos, es_train = ns["y"], ns["grupos"], ns["es_train"]
XC, XK, EC, yEC = ns["XC"], ns["XK"], ns["EC"], ns["yEC"]
idx_tr, idx_te = ns["idx_tr"], ns["idx_te"]
MODELO_CEL, MODELO_CIT = ns["MODELO_CEL"], ns["MODELO_CIT"]
log("Datos y modelos actuales cargados.")

# ------------------------------------------------------------------------------
# Modelos pesados
# ------------------------------------------------------------------------------
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from pytabkit import RealMLP_TD_Regressor, TabM_D_Regressor

def log1p_positivo(Z):
    return np.log1p(np.clip(Z, 0, None))

def tabpfn():
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)

def con_prepro(m):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(log1p_positivo)), ("m", m)])

CANDIDATOS = {
    "Actual": None,  # ensamble (células) / LightGBM (citoquinas)
    "TabPFN v2": lambda: tabpfn(),
    "TabM": lambda: con_prepro(TabM_D_Regressor(device="cpu", random_state=0, verbosity=0)),
    "RealMLP": lambda: con_prepro(RealMLP_TD_Regressor(device="cpu", random_state=0, verbosity=0)),
}

def nuevo(nombre, rama):
    if nombre == "Actual":
        return clone(MODELO_CEL if rama == "células" else MODELO_CIT)
    return CANDIDATOS[nombre]()

def ajustar_predecir(nombre, rama, tr, va):
    X = XC if rama == "células" else XK
    Xt, yt = X[tr], y[tr]
    if rama == "células":  # CyTOF extra del 10K (personas no pareadas)
        Xt, yt = np.vstack([Xt, EC]), np.concatenate([yt, yEC])
    return nuevo(nombre, rama).fit(Xt.astype(np.float32), yt).predict(X[va].astype(np.float32))

SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]
SPLITS_TRAIN = list(GroupKFold(n_splits=5, shuffle=True, random_state=7).split(idx_tr, groups=grupos[idx_tr]))

# ------------------------------------------------------------------------------
# Predicciones de cada modelo en cada rama
# ------------------------------------------------------------------------------
PRED = {}
ruta_cache = os.path.join(carpeta, "predicciones_pesados.npz")
if os.path.exists(ruta_cache):
    PRED = dict(np.load(ruta_cache, allow_pickle=True))["PRED"].item()
for rama in ["células", "citoquinas"]:
    for nombre in CANDIDATOS:
        clave = f"{rama}|{nombre}"
        if clave in PRED:
            continue
        t = time.time()
        oof = np.zeros((REPETICIONES, len(y)))
        for r in range(REPETICIONES):
            for tr, va in SPLITS[r]:
                oof[r, va] = ajustar_predecir(nombre, rama, tr, va)
        oof_tr = np.zeros(len(idx_tr))           # fuera de muestra dentro del train (para la fusión del test)
        for a, b in SPLITS_TRAIN:
            oof_tr[b] = ajustar_predecir(nombre, rama, idx_tr[a], idx_tr[b])
        p_te = ajustar_predecir(nombre, rama, idx_tr, idx_te)
        PRED[clave] = {"oof": oof, "oof_tr": oof_tr, "test": p_te}
        np.savez(ruta_cache, PRED=np.array(PRED, dtype=object))
        mae = np.abs(oof - y).mean(axis=1).mean()
        log(f"{rama:<10s} | {nombre:<10s} | MAE CV {mae:.2f} | R² {np.mean([r2_score(y, oof[r]) for r in range(REPETICIONES)]):.3f} "
            f"| test {mean_absolute_error(y[idx_te], p_te):.2f} | {(time.time()-t)/60:.1f} min")

# ------------------------------------------------------------------------------
# Tablas: ramas y fusiones
# ------------------------------------------------------------------------------
def met(yv, p):
    return mean_absolute_error(yv, p), r2_score(yv, p), np.mean(np.abs(yv - p) <= 5) * 100

def combinar(claves):
    """Promedio de varios modelos de una misma rama."""
    return {k: np.mean([PRED[c][k] for c in claves], axis=0) for k in ["oof", "oof_tr", "test"]}

filas = []
ramas_pred = {}
for rama in ["células", "citoquinas"]:
    for nombre in CANDIDATOS:
        ramas_pred[f"{rama}|{nombre}"] = PRED[f"{rama}|{nombre}"]
    ramas_pred[f"{rama}|Actual + TabPFN v2"] = combinar([f"{rama}|Actual", f"{rama}|TabPFN v2"])
    ramas_pred[f"{rama}|Actual + TabPFN v2 + TabM + RealMLP"] = combinar([f"{rama}|{n}" for n in CANDIDATOS])
for clave, d in ramas_pred.items():
    rama, nombre = clave.split("|")
    cv = [met(y, d["oof"][r]) for r in range(REPETICIONES)]
    te = met(y[idx_te], d["test"])
    filas.append({"rama": rama, "modelo": nombre, "MAE_CV": np.mean([c[0] for c in cv]), "R2_CV": np.mean([c[1] for c in cv]),
                  "MAE_test": te[0], "R2_test": te[1], "acierto5_test": te[2]})
df_ramas = pd.DataFrame(filas).round(3)

def fusionar(pc, pk):
    """Fusión lineal: pesos aprendidos con los pacientes de entrenamiento de cada partición."""
    oof_f = np.zeros((REPETICIONES, len(y)))
    for r in range(REPETICIONES):
        for tr, va in SPLITS[r]:
            lr = LinearRegression().fit(np.column_stack([pc["oof"][r, tr], pk["oof"][r, tr]]), y[tr])
            oof_f[r, va] = lr.predict(np.column_stack([pc["oof"][r, va], pk["oof"][r, va]]))
    lr = LinearRegression().fit(np.column_stack([pc["oof_tr"], pk["oof_tr"]]), y[idx_tr])
    return oof_f, lr.predict(np.column_stack([pc["test"], pk["test"]]))

filas_f = []
for nc in [n for n in ramas_pred if n.startswith("células")]:
    for nk in [n for n in ramas_pred if n.startswith("citoquinas")]:
        oof_f, p_te = fusionar(ramas_pred[nc], ramas_pred[nk])
        cv = [met(y, oof_f[r]) for r in range(REPETICIONES)]
        te = met(y[idx_te], p_te)
        filas_f.append({"rama_celulas": nc.split("|")[1], "rama_citoquinas": nk.split("|")[1],
                        "MAE_CV": np.mean([c[0] for c in cv]), "R2_CV": np.mean([c[1] for c in cv]),
                        "acierto5_CV": np.mean([c[2] for c in cv]),
                        "MAE_test": te[0], "R2_test": te[1], "acierto5_test": te[2]})
df_f = pd.DataFrame(filas_f).sort_values("MAE_CV").round(3)
df_ramas.to_csv(os.path.join(carpeta, "ramas_modelos_pesados.csv"), index=False)
df_f.to_csv(os.path.join(carpeta, "fusiones_modelos_pesados.csv"), index=False)
pd.set_option("display.width", 250)
print("\n=== RAMAS ===\n" + df_ramas.to_string(index=False))
print("\n=== FUSIONES (ordenadas por MAE de CV) ===\n" + df_f.to_string(index=False))
ref = df_f[(df_f.rama_celulas == "Actual") & (df_f.rama_citoquinas == "Actual")].iloc[0]
print(f"\nReferencia (modelo final, mismas particiones 3x5): CV {ref['MAE_CV']:.2f} | test {ref['MAE_test']:.2f}")
log("Listo.")
