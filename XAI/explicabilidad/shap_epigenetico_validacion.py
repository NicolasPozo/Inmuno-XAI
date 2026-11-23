# ==============================================================================
# VALIDACIÓN DEL SHAP DEL MODELO EPIGENÉTICO CON LA LIBRERÍA shap
#
#   python -X utf8 explicabilidad/shap_epigenetico_validacion.py     (2-3 minutos)
#
# El dashboard calcula el aporte de cada sonda "a mano": coeficiente x valor
# estandarizado, que en un modelo lineal es su valor SHAP. Este script comprueba
# ese cálculo con la librería shap, de dos formas:
#
#   1. shap.LinearExplainer: el SHAP exacto de un modelo lineal. Se calcula para
#      los 347 pacientes de test y las 501 variables, con el train como referencia.
#   2. shap.KernelExplainer: el SHAP agnóstico al modelo, que solo ve las
#      predicciones (no los coeficientes). Se calcula para 10 pacientes de test.
#
# No entrena nada ni cambia el dashboard. Escribe:
#   resultados/explicabilidad/shap_epigenetico_validacion.csv
# ==============================================================================

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

XAI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_1 = os.path.join(os.path.dirname(XAI), "Modelo 1", "Datasets")
RES = os.path.join(XAI, "resultados", "modelo_epigenetico")
SALIDA = os.path.join(XAI, "resultados", "explicabilidad", "shap_epigenetico_validacion.csv")
N_KERNEL = 10  # pacientes de test para la comprobación con KernelExplainer

# ------------------------------------------------------------------------------
# 1. Modelo y datos (la misma lectura que dashboard/exportar_epigenetico.py)
# ------------------------------------------------------------------------------
paquete = joblib.load(os.path.join(XAI, "modelos", "epigenetico_elasticnet", "modelo_elasticnet.joblib"))
pipe, columnas = paquete["modelo"], list(paquete["columnas"])
sondas = columnas[:-1]
escalador, enet = pipe.named_steps["scaler"], pipe.named_steps["model"]
par = pd.read_csv(os.path.join(RES, "particion_pacientes_elasticnet.csv"))

print("Leyendo las 500 sondas desde las cachés de Modelo 1...")
todas = np.load(os.path.join(MODELO_1, "cache_matriz_completa", "sondas_cpg.npy"))
pos = pd.Series(np.arange(len(todas)), index=todas)[sondas].values
bloques = []
for carpeta, ds in (("cache_matriz_completa", "Hannum"), ("cache_GSE87571", "GSE87571")):
    M = np.load(os.path.join(MODELO_1, carpeta, "matriz_cpg_float32.npy"), mmap_mode="r")
    pac = pd.read_csv(os.path.join(MODELO_1, carpeta, "pacientes.csv"))
    sub = np.vstack([np.array(M[i:i + 32])[:, pos] for i in range(0, M.shape[0], 32)])
    B = pd.DataFrame(sub.astype(np.float64), columns=sondas, index=pac["ID_Paciente"].values)
    B = B.loc[par.loc[par.Dataset == ds, "ID_Paciente"]]
    ids_train = par.loc[(par.Dataset == ds) & (par.Conjunto == "train"), "ID_Paciente"]
    bloques.append(B.fillna(B.loc[ids_train].median()))
betas = pd.concat(bloques).loc[par["ID_Paciente"]]
X = betas.assign(Sexo_M=(par["Sexo"].values == "M").astype(float))[columnas].values
es_test = (par["Conjunto"] == "test").values
X_train, X_test = X[~es_test], X[es_test]
ids_test = par.loc[es_test, "ID_Paciente"].values
pred = pipe.predict(X_test)
print(f"Train: {len(X_train)} pacientes. Test: {len(X_test)} pacientes. Variables: {len(columnas)}.")

# ------------------------------------------------------------------------------
# 2. El cálculo del dashboard: aporte = coeficiente x valor estandarizado
# ------------------------------------------------------------------------------
Z_train, Z_test = escalador.transform(X_train), escalador.transform(X_test)
manual = Z_test * enet.coef_
base_manual = float(enet.intercept_)
filas = []


def anotar(prueba, valor, unidad, detalle):
    filas.append({"prueba": prueba, "valor": valor, "unidad": unidad, "detalle": detalle})
    print(f"   {prueba}: {valor:.3g} {unidad}  ({detalle})")


# ------------------------------------------------------------------------------
# 3. shap.LinearExplainer (SHAP exacto del modelo lineal, con el train como referencia)
# ------------------------------------------------------------------------------
print("\n1) shap.LinearExplainer, 347 pacientes de test x 501 variables")
# max_samples: por defecto shap usa solo 100 pacientes de referencia; aquí van los 1.038 del train
lineal = shap.LinearExplainer(enet, shap.maskers.Independent(Z_train, max_samples=len(Z_train)))
sv = np.asarray(lineal.shap_values(Z_test))
base_shap = float(np.ravel(lineal.expected_value)[0])
dif = np.abs(sv - manual)
anotar("LinearExplainer: diferencia máxima con el dashboard", dif.max(), "años",
       f"{sv.size} aportes: {len(X_test)} pacientes x {len(columnas)} variables")
anotar("LinearExplainer: diferencia media con el dashboard", dif.mean(), "años", "valor absoluto")
anotar("LinearExplainer: diferencia de la base", abs(base_shap - base_manual), "años",
       f"base de shap {base_shap:.4f}; intercepto del modelo {base_manual:.4f}")
anotar("LinearExplainer: base + suma de aportes − edad predicha (máx.)", np.abs(base_shap + sv.sum(axis=1) - pred).max(), "años",
       "propiedad de eficiencia de SHAP")
orden_shap = np.argsort(-np.abs(sv).mean(axis=0))[:20]
orden_manual = np.argsort(-np.abs(manual).mean(axis=0))[:20]
anotar("LinearExplainer: sondas del top 20 en el mismo puesto", float((orden_shap == orden_manual).sum()), "de 20",
       "ranking por media de |SHAP| en el test")

# Comparación con lo que muestra el dashboard (epi_aportes.json, redondeado a 4 decimales)
aportes = pd.DataFrame(json.load(open(os.path.join(XAI, "dashboard", "datos", "epi_aportes.json"), encoding="utf-8")))
aportes = aportes[aportes["tipo"] != "Resto"]
col = {("Sexo (hombre)" if c == "Sexo_M" else c): j for j, c in enumerate(columnas)}
fila = {pid: i for i, pid in enumerate(ids_test)}
de_shap = np.array([sv[fila[r.id], col[r.variable]] for r in aportes.itertuples()])
anotar("LinearExplainer: diferencia máxima con epi_aportes.json", np.abs(de_shap - aportes["aporte_anos"].values).max(), "años",
       f"{len(aportes)} aportes que dibuja la cascada; el archivo guarda 4 decimales")

# ------------------------------------------------------------------------------
# 4. shap.KernelExplainer (agnóstico: solo usa las predicciones del modelo completo)
# ------------------------------------------------------------------------------
print(f"\n2) shap.KernelExplainer, {N_KERNEL} pacientes de test (referencia: la media del train)")
elegidos = np.linspace(0, len(X_test) - 1, N_KERNEL).astype(int)
np.random.seed(0)
kernel = shap.KernelExplainer(pipe.predict, X_train.mean(axis=0, keepdims=True))
sk = np.asarray(kernel.shap_values(X_test[elegidos], nsamples=4 * len(columnas), l1_reg=False, silent=True))
dk = np.abs(sk - manual[elegidos])
anotar("KernelExplainer: diferencia máxima con el dashboard", dk.max(), "años",
       f"{sk.size} aportes: {N_KERNEL} pacientes x {len(columnas)} variables, {4 * len(columnas)} combinaciones por paciente")
anotar("KernelExplainer: diferencia de la base", abs(float(np.ravel(kernel.expected_value)[0]) - base_manual), "años",
       "predicción del paciente medio frente al intercepto")

pd.DataFrame(filas).to_csv(SALIDA, index=False)
print(f"\nGuardado en {SALIDA}")
