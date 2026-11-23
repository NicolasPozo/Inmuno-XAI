# ==============================================================================
# DIAGNÓSTICO: ¿EL MODELO RECONOCE EL ESTUDIO DE ORIGEN?
# Proyecto: Immuno-XAI - UTEM
#
# Si las variables permiten adivinar de qué estudio viene cada paciente, y los
# estudios tienen edades distintas, el modelo puede usar la "firma" del estudio
# como atajo para estimar la edad. Eso funciona con pacientes de los mismos
# estudios (tu test) pero no con datos de otros laboratorios.
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_val_predict

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

cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells", "Central_Memory_CD4_T_cells",
            "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells", "Effector_Memory_CD4_T_cells",
            "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes", "Memory_B_cells", "Monocytes", "Naive_B_cells",
            "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells", "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto", "edad", "study_accession"]]
cols_cit = [c for c in todas_cit if par.loc[par.conjunto == "train", c].notna().mean() >= 0.8]

# Agrupar estudios pequeños
cuenta = par["study_accession"].value_counts()
par["estudio"] = par["study_accession"].where(par["study_accession"].map(cuenta) >= 20, "otros")
print("Pacientes y edad por estudio:")
print(par.groupby("estudio")["age"].agg(["size", "min", "median", "max"]).round(1).to_string())

g = par["subject_accession"].values
cv = GroupKFold(n_splits=5, shuffle=True, random_state=0)
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=0, verbose=-1)
azar = par["estudio"].value_counts(normalize=True).max()
for nombre, cols in [("citoquinas", cols_cit), ("células", cols_cel)]:
    pred = cross_val_predict(clf, par[cols].values, par["estudio"].values, cv=cv, groups=g)
    print(f"\n¿Se adivina el estudio con las {nombre}? acierto {np.mean(pred == par['estudio'].values)*100:.0f}% "
          f"(adivinando la clase más común: {azar*100:.0f}%)")

# ¿Cuánto error tendría un "modelo" que solo sabe el estudio (predice la edad media de su estudio)?
mae_estudio = []
for tr, va in cv.split(par, groups=g):
    medias = par.iloc[tr].groupby("estudio")["age"].mean()
    p = par.iloc[va]["estudio"].map(medias).fillna(par.iloc[tr]["age"].mean())
    mae_estudio.append(np.abs(p - par.iloc[va]["age"]).mean())
mae_global = np.abs(par["age"] - par["age"].mean()).mean()
print(f"\nError prediciendo siempre la edad media de todos: {mae_global:.2f} años")
print(f"Error sabiendo SOLO el estudio de origen (edad media del estudio): {np.mean(mae_estudio):.2f} años")
print("(como referencia, el mejor modelo con todas las variables logra ~8.3 años)")
