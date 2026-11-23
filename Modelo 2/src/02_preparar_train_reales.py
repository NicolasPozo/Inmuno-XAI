# ==============================================================================
# TRAIN SOLO CON PACIENTES REALES (SIN RUIDO NI MIXUP)
# Proyecto: Immuno-XAI - UTEM
#
# - Toma de df_train_aumentado_k45.csv solo las filas "original".
# - Agrega el ID de paciente (subject_accession) para poder agrupar la
#   validación cruzada por paciente.
# - Quita a los pacientes que también están en el test (fuga de datos).
#
# Requiere Datasets/10k/ids_pacientes_actuales.csv (lo genera src/01_ampliar_dataset.py).
# No modifica tus archivos: genera Datasets/df_train_reales_k45.csv
# ==============================================================================

import os
import numpy as np
import pandas as pd

from config import (
    ARCHIVO_IDS_PACIENTES,
    ARCHIVO_TRAIN_AUMENTADO_K45,
    ARCHIVO_TRAIN_REALES_K45,
)  # rutas del proyecto: src/config.py
df_train = pd.read_csv(ARCHIVO_TRAIN_AUMENTADO_K45)
ids = pd.read_csv(ARCHIVO_IDS_PACIENTES)

reales = df_train[df_train["source_type"] == "original"].reset_index(drop=True)
ids_train = ids[ids["conjunto"] == "train"].reset_index(drop=True)
assert len(ids_train) == len(reales)
assert np.allclose(ids_train["age"], reales["age"]) and (ids_train["gender"] == reales["gender"]).all()
reales.insert(0, "subject_accession", ids_train["subject_accession"])

ids_test = set(ids.loc[ids["conjunto"] == "test", "subject_accession"])
en_test = reales["subject_accession"].isin(ids_test)
print(f"Pacientes de train que también están en test: {sorted(reales.loc[en_test, 'subject_accession'].unique())}")
reales = reales[~en_test].reset_index(drop=True)

repetidos = reales["subject_accession"].value_counts()
print(f"Filas: {len(reales)} | pacientes únicos: {reales['subject_accession'].nunique()} | "
      f"pacientes con 2 filas (visitas distintas, quedan en el mismo grupo de CV): {(repetidos > 1).sum()}")

salida = ARCHIVO_TRAIN_REALES_K45
reales.to_csv(salida, index=False)
print(f"Guardado en: {salida}")
