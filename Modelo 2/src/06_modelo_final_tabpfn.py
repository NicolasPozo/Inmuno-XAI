# ==============================================================================
# MODELO FINAL: TabPFN v2, UN SOLO MODELO (células + citoquinas juntas)
# Proyecto: Immuno-XAI - UTEM
#
# Datos  : Stanford 10K Immunomes (datos normalizados), 473 pacientes con células
#          y citoquinas. Train = 354 pacientes reales (sin ruido ni mixup),
#          test = 119 pacientes (los mismos de df_test_puro_k45.csv).
# Variables: los 45 biomarcadores del k45 (16 poblaciones celulares + 29 citoquinas) + sexo.
# Modelo : TabPFN v2 (transformer pre-entrenado para tablas), valores por defecto.
#
# Evaluaciones (mismas que el modelo anterior de dos ramas, para comparar):
#   1. CV agrupada por paciente 5x5 sobre los 473 pacientes
#   2. Test de 119 pacientes (modelo entrenado solo con los 354 de train)
#   3. Estudios no vistos (se deja un estudio completo fuera, 5 estudios)
#
# Requiere: Datasets/pareados.csv (paso 04).
# Uso:  python -X utf8 src/06_modelo_final_tabpfn.py
# Salida: modelos/ (modelo_tabpfn.joblib, variables_modelo.csv) y resultados/modelo_final/
# ==============================================================================

import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
from config import (
    ARCHIVO_ELISA_10K,
    ARCHIVO_MODELO_FINAL,
    ARCHIVO_PAREADOS,
    ARCHIVO_TEST_K45,
    ARCHIVO_VARIABLES_MODELO,
    CARPETA_RES_MODELO_DOS_RAMAS,
    CARPETA_RES_MODELO_FINAL,
)  # rutas del proyecto: src/config.py
carpeta = CARPETA_RES_MODELO_FINAL
os.makedirs(carpeta, exist_ok=True)
os.makedirs(ARCHIVO_MODELO_FINAL.parent, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)
REPETICIONES = 5

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

def tabpfn():
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)

def metricas(yv, p):
    return {"MAE": mean_absolute_error(yv, p), "RMSE": np.sqrt(mean_squared_error(yv, p)), "R2": r2_score(yv, p),
            "r": np.corrcoef(yv, p)[0, 1], "acierto_5a_%": np.mean(np.abs(yv - p) <= 5) * 100}

# ------------------------------------------------------------------------------
# Datos (mismos pacientes y estudios que el modelo anterior)
# ------------------------------------------------------------------------------
par = pd.read_csv(ARCHIVO_PAREADOS)
el = pd.read_csv(ARCHIVO_ELISA_10K)
k = el.assign(edad=el.age.round(2))[["subject_accession", "edad", "study_accession"]].drop_duplicates(["subject_accession", "edad"])
par = par.assign(edad=par.age.round(2)).merge(k, on=["subject_accession", "edad"], how="left")
y = par["age"].values.astype(float)
grupos = par["subject_accession"].values
estudios = par["study_accession"].fillna("otro").values
idx_tr = np.where(par["conjunto"] == "train")[0]
idx_te = np.where(par["conjunto"] == "test")[0]

test_k45 = pd.read_csv(ARCHIVO_TEST_K45)
VARIABLES = [c for c in test_k45.columns if c not in ["age", "gender", "source_type", "subject_accession", "race"]]

def matriz(d):
    """45 biomarcadores + sexo (1 = Male)."""
    return np.hstack([d[VARIABLES].astype(float).values,
                      (d["gender"] == "Male").astype(float).values.reshape(-1, 1)]).astype(np.float32)

X = matriz(par)
log(f"Pacientes {len(y)} (train {len(idx_tr)}, test {len(idx_te)}) | variables {X.shape[1]}")

filas = []

# ------------------------------------------------------------------------------
# 1. CV agrupada 5x5
# ------------------------------------------------------------------------------
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]
oof = np.zeros((REPETICIONES, len(y)))
for r in range(REPETICIONES):
    for tr, va in SPLITS[r]:
        oof[r, va] = tabpfn().fit(X[tr], y[tr]).predict(X[va])
    log(f"CV repetición {r+1}/{REPETICIONES}: MAE {mean_absolute_error(y, oof[r]):.2f}")
m_cv = [metricas(y, oof[r]) for r in range(REPETICIONES)]
fila = {"evaluacion": "CV 5x5 (473 pacientes)"}
for k_ in m_cv[0]:
    vals = [m[k_] for m in m_cv]
    fila[k_], fila[f"{k_}_sd"] = np.mean(vals), np.std(vals)
filas.append(fila)

# ------------------------------------------------------------------------------
# 2. Test de 119 pacientes -> MODELO FINAL (entrenado con los 354 de train)
# ------------------------------------------------------------------------------
modelo = tabpfn().fit(X[idx_tr], y[idx_tr])
p_te = modelo.predict(X[idx_te])
y_te = y[idx_te]
filas.append({"evaluacion": "Test (119 pacientes)", **metricas(y_te, p_te)})
log(f"Test: MAE {filas[-1]['MAE']:.2f} | R² {filas[-1]['R2']:.3f}")

# ------------------------------------------------------------------------------
# 3. Estudios no vistos
# ------------------------------------------------------------------------------
loso_est = ["SDY420", "SDY515", "SDY519", "SDY113", "SDY311"]
filas_l = []
for e in loso_est:
    tr, va = np.where(estudios != e)[0], np.where(estudios == e)[0]
    p = tabpfn().fit(X[tr], y[tr]).predict(X[va])
    yv = y[va]
    filas_l.append({"estudio_no_visto": e, "n": len(va), "edades": f"{yv.min():.0f}-{yv.max():.0f}", **metricas(yv, p)})
    log(f"Estudio no visto {e}: MAE {filas_l[-1]['MAE']:.2f}")
df_l = pd.DataFrame(filas_l)
filas.append({"evaluacion": "Estudios no vistos (promedio de 5)",
              **{k_: df_l[k_].mean() for k_ in ["MAE", "RMSE", "R2", "r", "acierto_5a_%"]}})

# ------------------------------------------------------------------------------
# Comparación con el modelo anterior (dos ramas) en el mismo test
# ------------------------------------------------------------------------------
ant = pd.read_csv(os.path.join(CARPETA_RES_MODELO_DOS_RAMAS, "predicciones_test.csv"))
ant = ant.set_index("subject_accession").loc[grupos[idx_te]]
e_ant, e_new = np.abs(ant["pred_fusion"].values - y_te), np.abs(p_te - y_te)
rng = np.random.default_rng(0)
dif = np.array([(e_ant[i] - e_new[i]).mean() for i in (rng.integers(0, len(y_te), len(y_te)) for _ in range(10000))])
comparacion = pd.DataFrame([{
    "MAE_test_modelo_anterior": e_ant.mean(), "MAE_test_tabpfn": e_new.mean(), "mejora_media_anos": dif.mean(),
    "IC95_bajo": np.percentile(dif, 2.5), "IC95_alto": np.percentile(dif, 97.5),
    "prob_mejora_%": np.mean(dif > 0) * 100}]).round(3)
log(f"Mejora vs modelo anterior: {dif.mean():.2f} años | IC95 [{np.percentile(dif, 2.5):.2f}, {np.percentile(dif, 97.5):.2f}]")

# ------------------------------------------------------------------------------
# Guardar
# ------------------------------------------------------------------------------
df = pd.DataFrame(filas).round(3)
df.to_csv(os.path.join(carpeta, "metricas_modelo_final.csv"), index=False)
df_l.round(3).to_csv(os.path.join(carpeta, "metricas_estudios_no_vistos.csv"), index=False)
comparacion.to_csv(os.path.join(carpeta, "comparacion_vs_modelo_anterior.csv"), index=False)
pd.DataFrame({"subject_accession": grupos[idx_te], "edad_real": y_te, "edad_predicha": p_te,
              "brecha_edad": p_te - y_te}).round(2).to_csv(os.path.join(carpeta, "predicciones_test.csv"), index=False)
np.savez(os.path.join(carpeta, "predicciones_cv.npz"), oof=oof, y=y, grupos=grupos)
try:
    joblib.dump({"modelo": modelo, "variables": VARIABLES + ["sexo_Male"],
                 "nota": "TabPFN v2 entrenado con los 354 pacientes de train. Entrada: 45 biomarcadores "
                         "(datos normalizados del 10K, orden de 'variables') + sexo (1 = Male)."},
                ARCHIVO_MODELO_FINAL)
    log("Modelo guardado en modelo_tabpfn.joblib")
except Exception as ex:
    log(f"No se pudo guardar el modelo con joblib: {type(ex).__name__}: {ex}")
pd.Series(VARIABLES + ["sexo_Male"], name="variable").to_csv(ARCHIVO_VARIABLES_MODELO, index=False)

pd.set_option("display.width", 250)
print("\n" + df.to_string(index=False))
print("\n" + df_l.round(3).to_string(index=False))
print("\n" + comparacion.to_string(index=False))
log("Listo.")
