# ==============================================================================
# SHAP DEL MODELO INMUNOLÓGICO (TabPFN) PARA LOS PACIENTES DE TEST
#
#   python -X utf8 explicabilidad/shap_inmunologico.py        (alrededor de 1 hora y cuarto en CPU)
#
# TabPFN no es lineal, así que su SHAP no tiene fórmula: se estima con la
# librería shap (KernelExplainer), que solo necesita las predicciones del modelo.
#
#   - Referencia: un paciente con todas las variables en la mediana del train.
#     Su edad predicha es la base (unos 50,7 años).
#   - Valor SHAP de un biomarcador: cuánto mueve la edad predicha tener el valor
#     del paciente en vez del de referencia, promediado sobre las combinaciones
#     de los demás biomarcadores. Así se reparten también los efectos conjuntos.
#   - Base + suma de los 46 valores SHAP = edad predicha del paciente (exacto).
#   - Los biomarcadores vacíos se dejan vacíos, como los recibe el modelo.
#
# Con N_COMBINACIONES = 1000 el error de cada valor es de unos 0,1 años (se
# midió contra 6.000 combinaciones). Subirlo lo reduce y alarga el cálculo en
# proporción. El script guarda el avance: si se corta, al repetirlo sigue.
#
# Escribe:
#   resultados/explicabilidad/shap_inmunologico_test.csv   (una fila por paciente, una columna por variable)
#   dashboard/datos/shap_inmune.json                        (una fila por paciente y variable; lo usa la página)
#   dashboard/datos/shap_inmune_importancia.json            (media de |SHAP| por variable)
#   dashboard/datos/pacientes_test.json                     (le agrega el campo "base")
# ==============================================================================

import json
import os

import numpy as np
import pandas as pd

from comun import RESULTADOS, Inmune, api, guardar_json, leer_json, log

N_COMBINACIONES = 1000
SALIDA = os.path.join(RESULTADOS, "shap_inmunologico_test.csv")
AVANCE = os.path.join(RESULTADOS, f"shap_inmunologico_avance_{N_COMBINACIONES}.csv")

inm = Inmune()
n, k = len(inm.ids), len(inm.variables)

# ------------------------------------------------------------------------------
# 1. KernelSHAP paciente por paciente (con respaldo del avance)
# ------------------------------------------------------------------------------
hechos = {}
if os.path.exists(AVANCE):
    previo = pd.read_csv(AVANCE).set_index("id")
    hechos = {pid: previo.loc[pid] for pid in previo.index if pid in set(inm.ids)}
    log(f"Se retoma un cálculo anterior: {len(hechos)} pacientes ya hechos")

base = None
valores = np.zeros((n, k))
for i, pid in enumerate(inm.ids):
    if pid in hechos:
        base, valores[i] = float(hechos[pid]["base"]), hechos[pid][inm.nombres].values.astype(float)
        continue
    base, v = api.shap_kernel(inm.predecir, inm.X_test[i:i + 1], inm.medianas, N_COMBINACIONES, semilla=i)
    valores[i] = v[0]
    fila = pd.DataFrame([[pid, base, *valores[i]]], columns=["id", "base", *inm.nombres])
    fila.to_csv(AVANCE, mode="a", header=not os.path.exists(AVANCE), index=False)
    if (i + 1) % 5 == 0 or i + 1 == n:
        log(f"{i + 1} de {n} pacientes")

# ------------------------------------------------------------------------------
# 2. Comprobaciones
# ------------------------------------------------------------------------------
suma = np.abs(base + valores.sum(axis=1) - inm.pred).max()
log(f"Base (edad predicha del paciente de referencia): {base:.2f} años")
log(f"Base + suma de SHAP frente a la edad predicha: diferencia máxima {suma:.4f} años")
assert suma < 0.01, "Los valores SHAP no suman la edad predicha"

# ------------------------------------------------------------------------------
# 3. Archivos
# ------------------------------------------------------------------------------
tabla = pd.DataFrame(valores, columns=inm.nombres)
tabla.insert(0, "base", base)
tabla.insert(0, "edad_predicha", inm.pred)
tabla.insert(0, "edad_real", [r["edad_real"] for r in inm.pacientes])
tabla.insert(0, "id", inm.ids)
tabla.round(4).to_csv(SALIDA, index=False)
log(f"Guardado {SALIDA}")

z = (inm.X_test - inm.media) / inm.desv  # valor del paciente en desviaciones estándar del train
filas = []
for i, pid in enumerate(inm.ids):
    for j, (v, tipo) in enumerate(zip(inm.nombres, inm.tipos)):
        vacio = bool(np.isnan(inm.X_test[i, j]))
        filas.append({"clave": f"{pid}|{v}", "id": pid, "variable": v, "tipo": tipo,
                      "aporte_anos": round(float(valores[i, j]), 3),
                      "valor": None if vacio else round(float(inm.X_test[i, j]), 4),
                      "z": None if vacio or tipo == "Sexo" else round(float(z[i, j]), 3)})
guardar_json("shap_inmune", filas)

permutacion = {r["variable"]: r["puesto"] for r in leer_json("importancia")}
imp = pd.DataFrame({"variable": inm.nombres, "tipo": inm.tipos, "media_abs": np.abs(valores).mean(axis=0),
                    "media": valores.mean(axis=0)})
imp["puesto_permutacion"] = imp["variable"].map(permutacion)
imp = imp.sort_values("media_abs", ascending=False)
imp.insert(0, "puesto", range(1, len(imp) + 1))
guardar_json("shap_inmune_importancia", json.loads(imp.round(4).to_json(orient="records", force_ascii=False)))

pacientes = leer_json("pacientes_test")
for r in pacientes:
    r["base"] = round(base, 2)
guardar_json("pacientes_test", pacientes)

log(f"Listo: {inm.filas_evaluadas} filas evaluadas por TabPFN. Los 5 biomarcadores con más aporte medio: "
    + ", ".join(f"{r.variable} ({r.media_abs:.2f})" for r in imp.head(5).itertuples()))
