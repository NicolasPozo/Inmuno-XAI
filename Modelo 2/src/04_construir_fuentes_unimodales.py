# ==============================================================================
# FUENTES DE UNA SOLA MODALIDAD (PLAN B: DOS MODELOS + FUSIÓN)
# Proyecto: Immuno-XAI - UTEM
#
# Genera (en Datasets/) los archivos que usan el modelo final (pareados.csv) y
# los experimentos de dos ramas (experimentos/04_dos_ramas y 05_modelo_dos_ramas):
#   pareados.csv                       -> pacientes con células Y citoquinas
#                                         (tus 473: train 354 + test 119)
#   fuentes_citoquinas/10k_elisa.csv   -> visitas del 10K con citoquinas que NO
#                                         son las visitas pareadas
#   fuentes_celulas/10k_cytof.csv      -> personas del 10K con CyTOF que no están
#                                         entre los pareados
#   fuentes_celulas/10k_flujo.csv      -> citometría de flujo del 10K (mismo
#                                         pipeline y nombres que CyTOF; muchas
#                                         visitas solo miden T, CD4 y CD8)
#
# Todas las fuentes del 10K ya están en la misma escala (normalizadas por el 10K).
# Para agregar un dataset nuevo de una modalidad: deja un CSV en la carpeta
# correspondiente con columnas subject_accession, age, gender y las variables
# con los nombres del 10K (lo que falte queda como NaN). Debe estar en la
# escala del 10K (ver la función de alineación en
# experimentos/03_banco_pruebas/armonizar_soundlife.py).
# ==============================================================================

import os
import numpy as np
import pandas as pd

from config import (
    ARCHIVO_CYTOF_10K,
    ARCHIVO_ELISA_10K,
    ARCHIVO_FLUJO_10K,
    ARCHIVO_PAREADOS,
    ARCHIVO_TEST_COMPLETO,
    ARCHIVO_TRAIN_REALES_COMPLETO,
    CARPETA_FUENTES_CELULAS,
    CARPETA_FUENTES_CITOQUINAS,
)  # rutas del proyecto: src/config.py
dir_cit = CARPETA_FUENTES_CITOQUINAS
dir_cel = CARPETA_FUENTES_CELULAS
os.makedirs(dir_cit, exist_ok=True)
os.makedirs(dir_cel, exist_ok=True)

cytof = pd.read_csv(ARCHIVO_CYTOF_10K)
elisa = pd.read_csv(ARCHIVO_ELISA_10K)
flujo = pd.read_csv(ARCHIVO_FLUJO_10K)
flujo = flujo.rename(columns={"Central_memory_CD4_T_cells": "Central_Memory_CD4_T_cells",
                              "Central_memory_CD8_T_cells": "Central_Memory_CD8_T_cells",
                              "Effector_memory_CD4_T_cells": "Effector_Memory_CD4_T_cells",
                              "Effector_memory_CD8_T_cells": "Effector_Memory_CD8_T_cells",
                              "Gamma_delta_T_cells": "Gamma_Delta_T_cells"})

meta = ["study_accession", "age", "gender", "race", "subject_accession"]
cols_cel = [c for c in cytof.columns if c not in meta]
cols_cit = [c for c in elisa.columns if c not in meta]

# Pareados: datos crudos del 10K (sin imputar) para tus pacientes de train y test
tr = pd.read_csv(ARCHIVO_TRAIN_REALES_COMPLETO).assign(conjunto="train")
te = pd.read_csv(ARCHIVO_TEST_COMPLETO).assign(conjunto="test")
par = pd.concat([tr, te], ignore_index=True)
par[["subject_accession", "gender", "age", "conjunto"] + cols_cel + cols_cit].to_csv(
    ARCHIVO_PAREADOS, index=False)
print(f"Pareados: {len(par)} filas ({par['subject_accession'].nunique()} personas) | train {len(tr)} | test {len(te)}")

# Citoquinas: todas las visitas ELISA salvo las que ya son pareadas (mismo sujeto y edad)
clave_par = set(zip(par["subject_accession"], par["age"].round(2)))
e = elisa.copy()
e["_clave"] = list(zip(e["subject_accession"], e["age"].round(2)))
e_extra = e[~e["_clave"].isin(clave_par)].drop(columns="_clave")
e_extra = e_extra.assign(fuente="10K ELISA (" + e_extra["study_accession"] + ")")
e_extra[["subject_accession", "age", "gender", "fuente"] + cols_cit].to_csv(os.path.join(dir_cit, "10k_elisa.csv"), index=False)
nuevos_cit = set(e_extra["subject_accession"]) - set(par["subject_accession"])
print(f"Citoquinas extra: {len(e_extra)} visitas | {e_extra['subject_accession'].nunique()} personas "
      f"({len(nuevos_cit)} que no están entre los pareados)")

# Células CyTOF: personas que no están entre los pareados (los pareados ya usan su fila CyTOF)
c_extra = cytof[~cytof["subject_accession"].isin(set(par["subject_accession"]))]
c_extra = c_extra.assign(fuente="10K CyTOF (" + c_extra["study_accession"] + ")")
c_extra[["subject_accession", "age", "gender", "fuente"] + cols_cel].to_csv(os.path.join(dir_cel, "10k_cytof.csv"), index=False)
print(f"Células CyTOF extra: {len(c_extra)} personas")

# Células por citometría de flujo (todas las visitas; las del mismo sujeto se agrupan en la CV)
cols_flujo = [c for c in cols_cel if c in flujo.columns]
f_extra = flujo.assign(fuente="10K flujo (" + flujo["study_accession"] + ")")
f_extra[["subject_accession", "age", "gender", "fuente"] + cols_flujo].to_csv(os.path.join(dir_cel, "10k_flujo.csv"), index=False)
cob = f_extra[cols_flujo].notna().mean().round(2)
print(f"Células por flujo: {len(f_extra)} visitas | {f_extra['subject_accession'].nunique()} personas | "
      f"{len(set(f_extra['subject_accession']) - set(par['subject_accession']))} no pareadas")
print("   cobertura por variable:", cob.to_dict())
print("   faltan en flujo:", [c for c in cols_cel if c not in flujo.columns])
