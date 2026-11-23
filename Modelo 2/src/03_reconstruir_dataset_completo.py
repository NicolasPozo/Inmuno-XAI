# ==============================================================================
# RECONSTRUCCIÓN DEL DATASET CON TODAS LAS VARIABLES DEL 10K
# Proyecto: Immuno-XAI - UTEM
#
# Para los MISMOS pacientes y visitas de df_train_reales_k45.csv y
# df_test_puro_k45.csv, recupera todas las variables disponibles en el 10K:
#   - CyTOF normalizado: 24 poblaciones celulares
#   - Multiplex ELISA normalizado: todas las citoquinas
# Los valores faltantes se dejan como NaN (se imputan dentro de cada modelo).
#
# Genera: Datasets/df_train_reales_completo.csv y Datasets/df_test_completo.csv
# ==============================================================================

import os
import numpy as np
import pandas as pd

from config import (
    ARCHIVO_CYTOF_10K,
    ARCHIVO_ELISA_10K,
    ARCHIVO_IDS_PACIENTES,
    ARCHIVO_TEST_COMPLETO,
    ARCHIVO_TEST_K45,
    ARCHIVO_TRAIN_REALES_COMPLETO,
    ARCHIVO_TRAIN_REALES_K45,
)  # rutas del proyecto: src/config.py
cytof = pd.read_csv(ARCHIVO_CYTOF_10K)
elisa = pd.read_csv(ARCHIVO_ELISA_10K)

train = pd.read_csv(ARCHIVO_TRAIN_REALES_K45)
test = pd.read_csv(ARCHIVO_TEST_K45)
ids = pd.read_csv(ARCHIVO_IDS_PACIENTES)
test.insert(0, "subject_accession", ids.loc[ids["conjunto"] == "test", "subject_accession"].values)

meta = ["study_accession", "age", "gender", "race", "subject_accession"]
cols_cel_todas = [c for c in cytof.columns if c not in meta]
cols_cit_todas = [c for c in elisa.columns if c not in meta]
feats45 = [c for c in train.columns if c not in ("subject_accession", "gender", "source_type", "age")]
cit45 = [c for c in feats45 if c in cols_cit_todas]
cel45 = [c for c in feats45 if c in cols_cel_todas]

def fila_elisa(fila):
    """Fila del ELISA que corresponde a esta visita (mismo paciente, edad y citoquinas)."""
    cand = elisa[(elisa["subject_accession"] == fila["subject_accession"]) & ((elisa["age"] - fila["age"]).abs() <= 0.011)]
    if len(cand) == 1:
        return cand.index[0]
    mejor, mejor_err = None, np.inf
    for i, r in cand.iterrows():
        a, b = r[cit45].astype(float).values, fila[cit45].astype(float).values
        ok = np.isfinite(a)
        err = np.nanmean(np.abs(a[ok] - b[ok]) / (np.abs(b[ok]) + 1e-9)) if ok.any() else np.inf
        if err < mejor_err:
            mejor, mejor_err = i, err
    return mejor

def reconstruir(df):
    filas = []
    for _, f in df.iterrows():
        ie = fila_elisa(f)
        e = elisa.loc[ie]
        c = cytof[cytof["subject_accession"] == f["subject_accession"]]
        reg = {"subject_accession": f["subject_accession"], "gender": f["gender"], "age": f["age"],
               "estudio_elisa": e["study_accession"],
               "estudio_cytof": c["study_accession"].iloc[0] if len(c) else np.nan,
               "edad_cytof": c["age"].iloc[0] if len(c) else np.nan}
        for col in cols_cel_todas:
            reg[col] = c[col].iloc[0] if len(c) else np.nan
        for col in cols_cit_todas:
            reg[col] = e[col]
        filas.append(reg)
    return pd.DataFrame(filas)

tr_full = reconstruir(train)
te_full = reconstruir(test)

# Verificación: las 45 columnas deben coincidir con tus archivos (salvo valores imputados)
for nombre, orig, full in [("train", train, tr_full), ("test", test, te_full)]:
    print(f"\n[{nombre}] filas: {len(full)}")
    for grupo, cols in [("citoquinas", cit45), ("células", cel45)]:
        a, b = full[cols].astype(float).values, orig[cols].astype(float).values
        ok = np.isfinite(a)
        coincide = np.isclose(a[ok], b[ok], rtol=1e-5, atol=1e-6).mean()
        print(f"   {grupo}: {coincide*100:.1f}% de valores no faltantes coinciden | faltantes en fuente: {(~ok).sum()}")
    dif = (full["edad_cytof"] - full["age"]).abs()
    print(f"   visitas con edad CyTOF distinta a la edad ELISA (>0.05 años): {(dif > 0.05).sum()}")

cobertura = tr_full[cols_cel_todas + cols_cit_todas].notna().mean()
print(f"\nVariables totales: {len(cols_cel_todas)} células + {len(cols_cit_todas)} citoquinas")
print(f"Variables con >=80% de datos en train: {(cobertura >= 0.8).sum()} | >=50%: {(cobertura >= 0.5).sum()}")
print("Variables nuevas (no están en las 45) con >=80% de datos:")
nuevas = cobertura[(cobertura >= 0.8) & ~cobertura.index.isin(feats45)].sort_values(ascending=False)
print("   " + ", ".join(f"{k} ({v*100:.0f}%)" for k, v in nuevas.items()))

tr_full.to_csv(ARCHIVO_TRAIN_REALES_COMPLETO, index=False)
te_full.to_csv(ARCHIVO_TEST_COMPLETO, index=False)
print("\nGuardados: df_train_reales_completo.csv y df_test_completo.csv")
