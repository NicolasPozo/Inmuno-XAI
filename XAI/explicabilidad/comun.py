# ==============================================================================
# Lo que comparten shap_inmunologico.py y lime_inmunologico.py: el modelo
# inmunológico (TabPFN), los pacientes de train y de test, y las medianas.
#
# TabPFN repite el trabajo sobre los 354 pacientes de train en cada llamada
# (unos 10 segundos fijos). SHAP y LIME llaman al modelo una vez por paciente,
# así que aquí se usa una copia del modelo en modo "fit_with_cache", que guarda
# ese trabajo. La copia se entrena en memoria con los mismos datos y se
# comprueba que da las mismas predicciones que el modelo guardado; no se guarda.
# ==============================================================================

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

warnings.filterwarnings("ignore")

XAI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITULO = os.path.dirname(XAI)
DATOS = os.path.join(XAI, "dashboard", "datos")
RESULTADOS = os.path.join(XAI, "resultados", "explicabilidad")
sys.path.insert(0, os.path.join(XAI, "web"))
import modelos_api as api  # noqa: E402

T0 = time.time()


def log(mensaje):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {mensaje}", flush=True)


def leer_json(nombre):
    with open(os.path.join(DATOS, f"{nombre}.json"), encoding="utf-8") as f:
        return json.load(f)


def guardar_json(nombre, filas):
    ruta = os.path.join(DATOS, f"{nombre}.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, separators=(",", ":"))
    log(f"{nombre}.json: {len(filas)} filas, {os.path.getsize(ruta) / 1024:.0f} KB")


class Inmune:
    """Modelo inmunológico y sus datos, con los pacientes de test en el orden del dashboard."""

    def __init__(self):
        paquete = joblib.load(os.path.join(api.MODELOS, "final_tabpfn", "modelo_tabpfn.joblib"))
        guardado, self.variables = paquete["modelo"], list(paquete["variables"])  # 45 biomarcadores + sexo_Male
        self.bio = self.variables[:-1]
        self.nombres = self.bio + ["Sexo"]
        self.tipos = [api.Inmunologico.tipo(v) for v in self.nombres]

        par = pd.read_csv(os.path.join(TITULO, "Modelo 2", "Datasets", "pareados.csv"))
        matriz = lambda df: np.hstack([df[self.bio].astype(float).values,
                                       (df["gender"] == "Male").astype(float).values.reshape(-1, 1)]).astype(np.float32)
        train = par[par["conjunto"] == "train"]
        self.X_train, self.y_train = matriz(train), train["age"].values.astype(np.float32)

        self.pacientes = leer_json("pacientes_test")
        self.ids = [r["id"] for r in self.pacientes]
        test = par[par["conjunto"] == "test"].drop_duplicates("subject_accession").set_index("subject_accession")
        self.X_test = matriz(test.loc[self.ids])

        med = pd.read_csv(api.ARCHIVO_MEDIANAS_INMUNE).set_index("variable")["mediana"]
        self.medianas = med[self.variables].values.astype(np.float32)
        # Media y desviación del train, para expresar cada valor en desviaciones estándar
        self.media, self.desv = np.nanmean(self.X_train, axis=0), np.nanstd(self.X_train, axis=0)

        log("Preparando TabPFN con caché...")
        self.modelo = clone(guardado).set_params(fit_mode="fit_with_cache")
        self.modelo.fit(self.X_train, self.y_train)
        self.pred = self.predecir(self.X_test)
        dif = np.abs(self.pred - np.array([r["edad_predicha"] for r in self.pacientes])).max()
        log(f"Modelo listo. Diferencia máxima con las predicciones del dashboard: {dif:.3f} años")
        assert dif < 0.02, "El modelo con caché no reproduce las predicciones del dashboard"
        self.filas_evaluadas = 0

    def predecir(self, Z):
        Z = np.asarray(Z, dtype=np.float32)
        self.filas_evaluadas = getattr(self, "filas_evaluadas", 0) + len(Z)
        return self.modelo.predict(Z)
