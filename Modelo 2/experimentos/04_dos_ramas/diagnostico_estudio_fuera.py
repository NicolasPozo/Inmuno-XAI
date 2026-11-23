# ==============================================================================
# DIAGNÓSTICO: ¿FUNCIONA EL MODELO EN UN ESTUDIO QUE NO VIO?
# Proyecto: Immuno-XAI - UTEM
#
# Validación "dejando un estudio fuera": se entrena con todos los estudios menos
# uno y se evalúa en el estudio excluido (simula aplicar el modelo a otro
# laboratorio). Se compara con la CV normal (pacientes mezclados de todos los
# estudios) y con una predicción ingenua (edad media del train).
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
par = pd.read_csv(ruta("pareados.csv"))
el = pd.read_csv(ruta("datos_10k", "extraido", "Multiplex_ELISA", "Multiplex_ELISA", "Normalized_Files",
                              "multiplexELISA_normalized.csv"))
k = el.assign(edad=el.age.round(2))[["subject_accession", "edad", "study_accession"]].drop_duplicates(["subject_accession", "edad"])
par = par.assign(edad=par.age.round(2)).merge(k, on=["subject_accession", "edad"], how="left")
cuenta = par["study_accession"].value_counts()
par["estudio"] = par["study_accession"].where(par["study_accession"].map(cuenta) >= 20, "otros")

cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells", "Central_Memory_CD4_T_cells",
            "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells", "Effector_Memory_CD4_T_cells",
            "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes", "Memory_B_cells", "Monocytes", "Naive_B_cells",
            "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells", "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto", "edad", "study_accession", "estudio"]]
cols_cit = [c for c in todas_cit if par.loc[par.conjunto == "train", c].notna().mean() >= 0.8]
sexo = (par["gender"] == "Male").astype(float).values.reshape(-1, 1)
CONJ = {"Células": np.hstack([par[cols_cel].values, sexo]), "Citoquinas": np.hstack([par[cols_cit].values, sexo]),
        "Ambas": np.hstack([par[cols_cel].values, par[cols_cit].values, sexo])}
y = par["age"].values
modelo = lambda: lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=10,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=0, verbose=-1)
filas = []
for est in ["SDY420", "SDY515", "SDY519", "SDY113", "SDY311"]:
    va = (par["estudio"] == est).values
    tr = ~va
    fila = {"estudio_excluido": est, "n": int(va.sum()), "edad": f"{y[va].min():.0f}-{y[va].max():.0f}",
            "ingenuo_media_train": np.abs(y[va] - y[tr].mean()).mean()}
    for nombre, X in CONJ.items():
        p = modelo().fit(X[tr], y[tr]).predict(X[va])
        fila[f"MAE {nombre}"] = np.abs(y[va] - p).mean()
        fila[f"r {nombre}"] = np.corrcoef(y[va], p)[0, 1]
    filas.append(fila)
df = pd.DataFrame(filas).round(2)
pd.set_option("display.width", 250)
print("Dejando un estudio fuera (entrena con los demás, evalúa en el excluido):")
print(df.to_string(index=False))
print("\nr = correlación entre edad real y predicha dentro del estudio excluido (mide si ordena bien por edad,")
print("    sin importar el desplazamiento; útil en estudios con rango de edad estrecho).")
df.to_csv(ruta("resultados_dos_ramas", "diagnostico_estudio_fuera.csv"), index=False)
