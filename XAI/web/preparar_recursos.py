# ==============================================================================
# PREPARA LO QUE NECESITA LA PÁGINA WEB (se corre una vez)
#
#   python -X utf8 web/preparar_recursos.py
#
# 1. Medianas del train del modelo inmunológico -> modelos/final_tabpfn/medianas_train.csv
#    (la explicación por paciente reemplaza cada variable por esa mediana).
# 2. Archivos de ejemplo para subir en la página -> web/publico/ejemplos/
#    (pacientes reales del test, con su edad).
# 3. Comprueba que el cálculo de la página da lo mismo que el dashboard para
#    esos pacientes.
#
# Lee "Modelo 2/Datasets/pareados.csv" y las cachés de "Modelo 1/Datasets".
# Esos archivos no hacen falta para correr ni desplegar la página.
# ==============================================================================

import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import modelos_api as api  # noqa: E402

XAI = api.XAI
TITULO = os.path.dirname(XAI)
EJEMPLOS = os.path.join(AQUI, "publico", "ejemplos")
DATOS = os.path.join(XAI, "dashboard", "datos")
os.makedirs(EJEMPLOS, exist_ok=True)
leer = lambda n: json.load(open(os.path.join(DATOS, f"{n}.json"), encoding="utf-8"))

# ------------------------------------------------------------------------------
# 1-2. Modelo inmunológico: medianas y ejemplo
# ------------------------------------------------------------------------------
import joblib  # noqa: E402
variables = list(joblib.load(os.path.join(api.MODELOS, "final_tabpfn", "modelo_tabpfn.joblib"))["variables"])
bio = variables[:-1]
par = pd.read_csv(os.path.join(TITULO, "Modelo 2", "Datasets", "pareados.csv"))
X = np.hstack([par[bio].astype(float).values, (par["gender"] == "Male").astype(float).values.reshape(-1, 1)]).astype(np.float32)
X_train = X[(par["conjunto"] == "train").values]
mediana = np.nanmedian(X_train, axis=0)  # igual que en 08_explicabilidad.py
# media y desviación: para mostrar cada valor en desviaciones estándar del train (color del enjambre SHAP)
pd.DataFrame({"variable": variables, "mediana": mediana, "media": np.nanmean(X_train, axis=0),
              "desviacion": np.nanstd(X_train, axis=0)}).to_csv(api.ARCHIVO_MEDIANAS_INMUNE, index=False)
print(f"Medianas, medias y desviaciones del train inmunológico -> {api.ARCHIVO_MEDIANAS_INMUNE}")

pt = pd.DataFrame(leer("pacientes_test")).sort_values("brecha_edad", ascending=False)
ids_inm = [pt["id"].iloc[0], pt["id"].iloc[len(pt) // 2]]  # la brecha más alta y una típica
ej = par[par["subject_accession"].isin(ids_inm) & (par["conjunto"] == "test")]
ej = ej.set_index("subject_accession").loc[ids_inm].reset_index()
ej_inm = ej.rename(columns={"subject_accession": "id"})[["id", "age", "gender"] + bio]
ej_inm.to_csv(os.path.join(EJEMPLOS, "ejemplo_inmunologico.csv"), index=False)
print(f"Ejemplo inmunológico: {ids_inm}")

# ------------------------------------------------------------------------------
# 2. Modelo epigenético: ejemplo leído de las cachés (solo las filas necesarias)
# ------------------------------------------------------------------------------
epi = api.Epigenetico()
ept = pd.DataFrame(leer("epi_pacientes_test")).sort_values("brecha_edad", ascending=False)
ids_epi = [ept["id"].iloc[0], ept["id"].iloc[len(ept) // 2]]
M1 = os.path.join(TITULO, "Modelo 1", "Datasets")
todas = np.load(os.path.join(M1, "cache_matriz_completa", "sondas_cpg.npy"), allow_pickle=True)
pos = pd.Series(np.arange(len(todas)), index=todas)[epi.sondas].values
filas = []
for carpeta in ("cache_matriz_completa", "cache_GSE87571"):
    M = np.load(os.path.join(M1, carpeta, "matriz_cpg_float32.npy"), mmap_mode="r")
    pac = pd.read_csv(os.path.join(M1, carpeta, "pacientes.csv"))
    for i, pid in enumerate(pac["ID_Paciente"]):
        if pid in ids_epi:
            filas.append(pd.Series(np.array(M[i])[pos].astype(float), index=epi.sondas, name=pid))
betas = pd.DataFrame(filas).loc[ids_epi]
info = ept.set_index("id").loc[ids_epi]
ej_epi = pd.concat([pd.DataFrame({"id": ids_epi, "age": info["edad_real"].values, "gender": info["sexo"].values}),
                    betas.round(5).reset_index(drop=True)], axis=1)
ej_epi.to_csv(os.path.join(EJEMPLOS, "ejemplo_epigenetico.csv"), index=False)
print(f"Ejemplo epigenético: {ids_epi}")

# ------------------------------------------------------------------------------
# 3. Comprobación contra el dashboard
# ------------------------------------------------------------------------------
def comparar(nombre, res, pacientes, aportes, campo_ap="aporte_anos", tol_pred=0.02, tol_ap=0.02, clave="aportes"):
    ref_p = {r["id"]: r for r in leer(pacientes)}
    ref_a = {(r["id"], r["variable"]): r[campo_ap] for r in leer(aportes)}
    peor_p = peor_a = 0
    for p in res["pacientes"]:
        orig = p["id"].split(" · ", 1)[1]
        peor_p = max(peor_p, abs(p["edad_predicha"] - ref_p[orig]["edad_predicha"]))
    for a in res[clave]:
        orig = a["id"].split(" · ", 1)[1]
        if (orig, a["variable"]) in ref_a and not a["variable"].startswith("Otras"):
            peor_a = max(peor_a, abs(a[campo_ap] - ref_a[(orig, a["variable"])]))
    ok = peor_p <= tol_pred and peor_a <= tol_ap
    print(f"{nombre}: diferencia máxima edad {peor_p:.3f} años, aportes {peor_a:.3f} años -> {'OK' if ok else 'REVISAR'}")
    print(f"   avisos: {res['avisos']}")
    return ok

ok_epi = comparar("Epigenético", epi.predecir(open(os.path.join(EJEMPLOS, "ejemplo_epigenetico.csv"), encoding="utf-8").read()),
                  "epi_pacientes_test", "epi_aportes")
print("Cargando el modelo inmunológico (TabPFN)...")
inm = api.Inmunologico()
ok_inm = comparar("Inmunológico", inm.predecir(open(os.path.join(EJEMPLOS, "ejemplo_inmunologico.csv"), encoding="utf-8").read()),
                  "pacientes_test", "aportes_paciente", clave="oclusion")  # los aportes que se muestran son SHAP (se estiman al azar)
sys.exit(0 if ok_epi and ok_inm else 1)
