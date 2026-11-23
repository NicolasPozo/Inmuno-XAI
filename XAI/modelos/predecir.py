# ==============================================================================
# PREDECIR LA EDAD INMUNOLÓGICA CON EL MODELO FINAL (TabPFN v2)
#
# Uso:
#   python -X utf8 predecir.py pacientes.csv            -> pacientes_predicciones.csv
#
# El CSV de entrada necesita las 45 columnas de variables_modelo.csv (sin
# "sexo_Male") con valores en la escala de los archivos normalizados del
# Stanford 10K, y una columna "gender" ("Male" / "Female"). Si trae "age",
# también se calcula la brecha (edad predicha - edad real).
# Los valores faltantes pueden quedar vacíos: TabPFN los maneja.
#
# Requisitos: tabpfn 9.2 (pesos v2), torch, scikit-learn, pandas, numpy, joblib.
# ==============================================================================

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CARPETA = os.path.dirname(os.path.abspath(__file__))
paquete = joblib.load(os.path.join(CARPETA, "final_tabpfn", "modelo_tabpfn.joblib"))
modelo, variables = paquete["modelo"], paquete["variables"]


def predecir(datos: pd.DataFrame) -> np.ndarray:
    faltan = [v for v in variables[:-1] if v not in datos.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}")
    X = datos[variables[:-1]].astype(float).copy()
    X["sexo_Male"] = (datos["gender"] == "Male").astype(float)
    return modelo.predict(X.values.astype(np.float32))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python -X utf8 predecir.py pacientes.csv")
    entrada = sys.argv[1]
    datos = pd.read_csv(entrada)
    datos["edad_predicha"] = predecir(datos).round(2)
    if "age" in datos.columns:
        datos["brecha_edad"] = (datos["edad_predicha"] - datos["age"]).round(2)
    salida = os.path.splitext(entrada)[0] + "_predicciones.csv"
    datos.to_csv(salida, index=False)
    print(f"Listo: {len(datos)} pacientes -> {salida}")
