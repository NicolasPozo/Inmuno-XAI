# ==============================================================================
# PREDECIR LA EDAD EPIGENÉTICA CON EL MODELO FINAL (ElasticNet, 500 sondas CpG)
#
# Uso:
#   python -X utf8 predecir_epigenetico.py pacientes.csv   -> pacientes_predicciones.csv
#
# El CSV de entrada tiene una fila por paciente y una columna por sonda CpG
# (valores beta entre 0 y 1, array Illumina 450K o EPIC), más una columna
# "gender" ("Male" / "Female"; también acepta "M" / "F"). Las columnas que el
# modelo no usa se ignoran. Si trae "age", también se calcula la brecha
# (edad predicha - edad real).
# Las sondas que falten o vengan vacías se rellenan con la mediana del train
# (medianas_imputacion). Si a un paciente le falta más del 10 % de las 500
# sondas, la predicción pierde precisión y se avisa en la columna
# "sondas_imputadas".
#
# Requisitos: scikit-learn 1.9, pandas, numpy, joblib.
# ==============================================================================

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CARPETA = os.path.dirname(os.path.abspath(__file__))
paquete = joblib.load(os.path.join(CARPETA, "epigenetico_elasticnet", "modelo_elasticnet.joblib"))
modelo, variables = paquete["modelo"], paquete["columnas"]
sondas, medianas = variables[:-1], paquete["medianas_imputacion"]
MAX_FRACCION_IMPUTADA = 0.10


def predecir(datos: pd.DataFrame):
    """Devuelve (edad predicha, nº de sondas imputadas por paciente)."""
    if "gender" not in datos.columns:
        raise ValueError('Falta la columna "gender" ("Male" / "Female")')
    X = datos.reindex(columns=sondas).astype(float)
    n_imputadas = X.isna().sum(axis=1).values
    X = X.fillna(medianas).assign(Sexo_M=datos["gender"].isin(["Male", "M"]).astype(float).values)
    return modelo.predict(X[variables].values), n_imputadas


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python -X utf8 predecir_epigenetico.py pacientes.csv")
    entrada = sys.argv[1]
    datos = pd.read_csv(entrada)
    edad, n_imputadas = predecir(datos)
    datos["edad_predicha_epigenetica"] = edad.round(2)
    datos["sondas_imputadas"] = n_imputadas
    if "age" in datos.columns:
        datos["brecha_edad_epigenetica"] = (datos["edad_predicha_epigenetica"] - datos["age"]).round(2)
    salida = os.path.splitext(entrada)[0] + "_predicciones.csv"
    datos.to_csv(salida, index=False)
    muchas = int(np.sum(n_imputadas > MAX_FRACCION_IMPUTADA * len(sondas)))
    print(f"Listo: {len(datos)} pacientes -> {salida}")
    if muchas:
        print(f"Aviso: a {muchas} paciente(s) les falta más del 10 % de las 500 sondas; su predicción es menos precisa.")
