# ==============================================================================
# AMPLIACIÓN DEL TRAIN CON PACIENTES NUEVOS DEL 10K IMMUNOMES
# Proyecto: Immuno-XAI - UTEM
#
# Fuente: github.com/buttelab/10kimmunomes_v2 (data_raw)
#   - cytof_pbmc_normalized.csv    (células, CyTOF)
#   - multiplexELISA_normalized.csv (citoquinas)
#
# Tu dataset une células y citoquinas del MISMO paciente en la MISMA visita
# (misma edad). Hay 74 pacientes con CyTOF (estudio SDY112) y citoquinas que
# quedaron fuera porque sus citoquinas se midieron en otra visita (~1 año
# después, estudios SDY311/312/315/478). Este script los incorpora uniendo cada
# paciente con su visita de citoquinas más cercana en edad.
#
# Solo se añaden a TRAIN. El test (119 pacientes) no se modifica.
# Tus archivos originales no se sobrescriben: se genera un archivo nuevo.
#
# En el pipeline del modelo final este paso importa por otra salida:
# Datasets/10k/ids_pacientes_actuales.csv (el ID de paciente de cada fila del
# train k45), que usan los pasos 02 y 03. El train ampliado
# (Datasets/df_train_aumentado_k45_ampliado.csv) fue un experimento descartado.
# ==============================================================================

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
from config import (
    ARCHIVO_CYTOF_10K,
    ARCHIVO_ELISA_10K,
    ARCHIVO_IDS_PACIENTES,
    ARCHIVO_PACIENTES_NUEVOS_10K,
    ARCHIVO_TEST_K45,
    ARCHIVO_TRAIN_AMPLIADO,
    ARCHIVO_TRAIN_AUMENTADO_K45,
)  # rutas del proyecto: src/config.py

archivo_train = ARCHIVO_TRAIN_AUMENTADO_K45
archivo_test = ARCHIVO_TEST_K45
archivo_cytof = ARCHIVO_CYTOF_10K
archivo_elisa = ARCHIVO_ELISA_10K

salida_train = ARCHIVO_TRAIN_AMPLIADO
salida_nuevos = ARCHIVO_PACIENTES_NUEVOS_10K
salida_ids = ARCHIVO_IDS_PACIENTES

# Parámetros de aumento deducidos de df_train_aumentado_k45.csv
RUIDO_FRACCION_STD = 0.03   # ruido gaussiano = 0.03 * desviación estándar de la columna
ALPHA_MIXUP = 0.3           # lambda ~ Beta(0.3, 0.3), mezcla de pacientes del mismo sexo
TOL_EDAD = 0.011            # tolerancia para identificar la visita por edad

df_train = pd.read_csv(archivo_train)
df_test = pd.read_csv(archivo_test)
cytof = pd.read_csv(archivo_cytof)
elisa = pd.read_csv(archivo_elisa)

features = [c for c in df_train.columns if c not in ("gender", "source_type", "age")]
cols_celulas = [c for c in features if c in cytof.columns]
cols_citoquinas = [c for c in features if c in elisa.columns]
assert len(cols_celulas) + len(cols_citoquinas) == len(features), "Hay biomarcadores que no están en las fuentes"

# ------------------------------------------------------------------------------
# 1. Identificar qué pacientes ya están en tu train y test
#    (tu edad y tus citoquinas coinciden exactamente con una fila del ELISA)
# ------------------------------------------------------------------------------
def identificar_pacientes(df):
    E = elisa[cols_citoquinas].to_numpy(float)
    ids = []
    for _, fila in df.iterrows():
        x = fila[cols_citoquinas].to_numpy(float)
        candidatos = np.where(np.isclose(elisa["age"], fila["age"], atol=TOL_EDAD))[0]
        encontrado = None
        for i in candidatos:
            ok = np.isfinite(E[i]) & np.isfinite(x)
            if ok.sum() >= 3 and np.allclose(E[i][ok], x[ok], rtol=1e-5, atol=1e-3):
                encontrado = elisa.at[i, "subject_accession"]
                break
        ids.append(encontrado)
    return pd.Series(ids, index=df.index)

train_orig = df_train[df_train["source_type"] == "original"].copy()
train_orig["subject_accession"] = identificar_pacientes(train_orig)
df_test = df_test.copy()
df_test["subject_accession"] = identificar_pacientes(df_test)

if train_orig["subject_accession"].isna().any() or df_test["subject_accession"].isna().any():
    raise RuntimeError("No se pudo identificar a todos los pacientes actuales en el ELISA del 10K.")

pd.concat([
    train_orig[["subject_accession", "age", "gender"]].assign(conjunto="train"),
    df_test[["subject_accession", "age", "gender"]].assign(conjunto="test"),
]).to_csv(salida_ids, index=False)

ids_test = set(df_test["subject_accession"])
ids_usados = set(train_orig["subject_accession"]) | ids_test
print(f"Pacientes actuales identificados: train={train_orig['subject_accession'].nunique()} | test={len(ids_test)}")

# ------------------------------------------------------------------------------
# 2. Pacientes nuevos: tienen CyTOF y citoquinas pero no están en tu dataset
# ------------------------------------------------------------------------------
ids_nuevos = sorted((set(cytof["subject_accession"]) & set(elisa["subject_accession"])) - ids_usados)
assert not (set(ids_nuevos) & ids_test), "Un paciente nuevo está en test"

filas = []
for sid in ids_nuevos:
    c = cytof[cytof["subject_accession"] == sid].iloc[0]
    e_all = elisa[elisa["subject_accession"] == sid].copy()
    e_all["dif_edad"] = (e_all["age"] - c["age"]).abs()
    e = e_all.sort_values("dif_edad").iloc[0]   # visita de citoquinas más cercana
    fila = {
        "subject_accession": sid,
        "gender": c["gender"],
        "source_type": "original_10k_nuevo",
        "estudio_cytof": c["study_accession"],
        "estudio_elisa": e["study_accession"],
        "edad_cytof": c["age"],
        "edad_elisa": e["age"],
        "dif_edad_visitas": round(float(e["dif_edad"]), 2),
        "age": round((c["age"] + e["age"]) / 2.0, 2),   # edad promedio entre ambas visitas
    }
    for col in cols_celulas:
        fila[col] = c[col]
    for col in cols_citoquinas:
        fila[col] = e[col]
    filas.append(fila)

df_nuevos = pd.DataFrame(filas)
print(f"Pacientes nuevos: {len(df_nuevos)} | diferencia de edad entre visitas: "
      f"mediana {df_nuevos['dif_edad_visitas'].median():.2f} años, máx {df_nuevos['dif_edad_visitas'].max():.2f}")

# Faltantes -> mediana de los pacientes originales de train (mismo criterio que tu dataset)
medianas = train_orig[features].median()
n_faltantes = int(df_nuevos[features].isna().sum().sum())
df_nuevos[features] = df_nuevos[features].fillna(medianas)
print(f"Valores faltantes imputados con la mediana de train: {n_faltantes}")
df_nuevos.to_csv(salida_nuevos, index=False)

# ------------------------------------------------------------------------------
# 3. Aumento de datos para los nuevos (misma receta que tu train)
# ------------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
cols_salida = list(df_train.columns)

nuevos_orig = df_nuevos[cols_salida].copy()
nuevos_orig["source_type"] = "original"

# 3a. Copia con ruido gaussiano por paciente (edad sin cambios, sin negativos)
std_cols = train_orig[features].std()
nuevos_ruido = nuevos_orig.copy()
ruido = rng.normal(0.0, 1.0, size=(len(nuevos_ruido), len(features))) * (RUIDO_FRACCION_STD * std_cols.values)
nuevos_ruido[features] = np.clip(nuevos_ruido[features].values + ruido, 0.0, None)
nuevos_ruido["source_type"] = "synthetic_noise"

# 3b. Mixup proporcional: tu train tiene 300 mezclas para 357 pacientes reales.
#     Se agregan mezclas en la misma proporción, entre pacientes reales del mismo
#     sexo, donde al menos uno de los dos es un paciente nuevo.
pool = pd.concat([train_orig[cols_salida], nuevos_orig], ignore_index=True)
n_reales_actuales = (df_train["source_type"] == "original").sum()
n_mixup_actuales = (df_train["source_type"] == "synthetic_mixup").sum()
proporcion_mixup = n_mixup_actuales / n_reales_actuales

mezclas = []
for sexo in ["Male", "Female"]:
    idx_pool = pool.index[pool["gender"] == sexo].to_numpy()
    idx_nuevos = pool.index[(pool["gender"] == sexo) & (pool.index >= len(train_orig))].to_numpy()
    n_mix = int(round(len(idx_nuevos) * proporcion_mixup))
    for _ in range(n_mix):
        i = rng.choice(idx_nuevos)
        j = rng.choice(idx_pool[idx_pool != i])
        lam = float(rng.beta(ALPHA_MIXUP, ALPHA_MIXUP))
        fila = pool.loc[i].copy()
        fila[features] = lam * pool.loc[i, features].astype(float) + (1 - lam) * pool.loc[j, features].astype(float)
        fila["age"] = round(lam * pool.at[i, "age"] + (1 - lam) * pool.at[j, "age"], 2)
        fila["source_type"] = "synthetic_mixup"
        mezclas.append(fila)
nuevos_mixup = pd.DataFrame(mezclas, columns=cols_salida)

# ------------------------------------------------------------------------------
# 4. Train ampliado = tu train completo + nuevos (reales + ruido + mixup)
# ------------------------------------------------------------------------------
df_ampliado = pd.concat([df_train, nuevos_orig, nuevos_ruido, nuevos_mixup], ignore_index=True)
df_ampliado.to_csv(salida_train, index=False)

print("\nResumen del train ampliado:")
print(df_ampliado["source_type"].value_counts().to_string())
print(f"Total: {len(df_train)} -> {len(df_ampliado)} filas "
      f"({n_reales_actuales} -> {n_reales_actuales + len(nuevos_orig)} pacientes reales)")
print(f"Guardado en: {salida_train}")
