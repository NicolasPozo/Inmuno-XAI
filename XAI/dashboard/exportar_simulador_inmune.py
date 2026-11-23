# ==============================================================================
# SIMULADOR "QUÉ PASARÍA SI" DEL MODELO INMUNOLÓGICO (TabPFN)
#
#   python -X utf8 dashboard/exportar_simulador_inmune.py      (unos 10 minutos en CPU)
#
# TabPFN tarda unos segundos por predicción, así que el simulador no puede
# calcular en vivo. Este script precalcula, con el modelo real, cómo cambia la
# edad predicha de cada paciente de test al mover cada uno de sus 6
# biomarcadores con más efecto por los percentiles 1-99 del train (uno a la
# vez, el resto fijo). La página interpola entre esos puntos.
#
# El script también prueba si se podrían sumar los cambios de varios
# biomarcadores: mueve los 6 a la vez, al azar, y compara la suma con la
# predicción exacta. Con TabPFN la suma se aleja mucho (8,2 años en promedio
# en la corrida original), así que la página mueve un biomarcador a la vez.
#
# Escribe:
#   modelos/final_tabpfn/percentiles_train.csv   (la rejilla de cada biomarcador; la usa también el servidor web)
#   dashboard/datos/simulador_inmune.json         (los puntos de cada paciente de test)
#   resultados/explicabilidad/simulador_aditividad.csv (la comparación suma vs exacto)
# ==============================================================================

import json
import os
import sys
import time

import numpy as np
import pandas as pd

XAI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(XAI, "web"))
import modelos_api as api  # noqa: E402

TITULO = os.path.dirname(XAI)
DATOS = os.path.join(XAI, "dashboard", "datos")
T0 = time.time()
log = lambda m: print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)
leer = lambda n: json.load(open(os.path.join(DATOS, f"{n}.json"), encoding="utf-8"))

# ------------------------------------------------------------------------------
# 1. Datos: train para los percentiles y los 119 pacientes de test
# ------------------------------------------------------------------------------
import joblib  # noqa: E402
variables = list(joblib.load(os.path.join(api.MODELOS, "final_tabpfn", "modelo_tabpfn.joblib"))["variables"])
bio = variables[:-1]
par = pd.read_csv(os.path.join(TITULO, "Modelo 2", "Datasets", "pareados.csv"))
X = np.hstack([par[bio].astype(float).values, (par["gender"] == "Male").astype(float).values.reshape(-1, 1)]).astype(np.float32)
es_train = (par["conjunto"] == "train").values

pct = pd.DataFrame(np.nanpercentile(X[es_train][:, :-1], api.PERCENTILES_SIM, axis=0).T,
                   index=pd.Index(bio, name="variable"), columns=[f"p{p:02d}" for p in api.PERCENTILES_SIM])
pct.round(6).to_csv(api.ARCHIVO_PERCENTILES_INMUNE)
log(f"Percentiles del train -> {api.ARCHIVO_PERCENTILES_INMUNE}")

pt = leer("pacientes_test")
ids = [r["id"] for r in pt]
test = par[(par["conjunto"] == "test")].drop_duplicates("subject_accession").set_index("subject_accession")
M = np.hstack([test.loc[ids, bio].astype(float).values,
               (test.loc[ids, "gender"] == "Male").astype(float).values.reshape(-1, 1)]).astype(np.float32)

# Aportes por paciente: los mismos del dashboard (eligen los 6 biomarcadores de cada uno)
ap = pd.DataFrame(leer("aportes_paciente")).pivot(index="id", columns="variable", values="aporte_anos")
nombres = bio + ["Sexo"]
aportes = ap.loc[ids, nombres].values

# ------------------------------------------------------------------------------
# 2. Curvas del simulador (modelo real)
# ------------------------------------------------------------------------------
inm = api.Inmunologico()
inm.listo.wait()
if inm.error:
    sys.exit(inm.error)
log("Modelo cargado. Prediciendo la edad de los pacientes de test...")
pred = inm.modelo.predict(M)
dif = np.abs(pred - np.array([r["edad_predicha"] for r in pt])).max()
log(f"Diferencia máxima con el dashboard: {dif:.3f} años")
assert dif < 0.02, "El modelo no reproduce las predicciones del dashboard"

log(f"Calculando las curvas ({len(ids)} pacientes x {api.N_SIM} biomarcadores x {len(api.PERCENTILES_SIM)} valores)...")
filas = inm.simulacion(M, ids, aportes, pred)
with open(os.path.join(DATOS, "simulador_inmune.json"), "w", encoding="utf-8") as f:
    json.dump(filas, f, ensure_ascii=False, separators=(",", ":"))
log(f"simulador_inmune.json: {len(filas)} filas, {os.path.getsize(os.path.join(DATOS, 'simulador_inmune.json')) / 1024:.0f} KB")

# ------------------------------------------------------------------------------
# 3. ¿Qué tan buena es la suma? Mover los 6 a la vez y comparar con el modelo exacto
# ------------------------------------------------------------------------------
sim = pd.DataFrame(filas)
rng = np.random.default_rng(0)
Z, sumas = M.copy(), np.zeros(len(ids))
for i, pid in enumerate(ids):
    for v, g in sim[sim["id"] == pid].groupby("variable"):
        punto = g.iloc[rng.integers(len(g))]
        Z[i, bio.index(v)] = punto["valor"]
        sumas[i] += punto["delta"]
exacta = inm.modelo.predict(Z) - pred
comp = pd.DataFrame({"id": ids, "cambio_suma": sumas, "cambio_exacto": exacta, "diferencia": sumas - exacta})
comp.round(3).to_csv(os.path.join(XAI, "resultados", "explicabilidad", "simulador_aditividad.csv"), index=False)
log(f"Suma vs exacto al mover los 6 a la vez: diferencia media {np.abs(comp.diferencia).mean():.2f} años, "
    f"mediana {np.abs(comp.diferencia).median():.2f}, máxima {np.abs(comp.diferencia).max():.2f}; "
    f"cambio exacto medio {np.abs(exacta).mean():.2f} años")
