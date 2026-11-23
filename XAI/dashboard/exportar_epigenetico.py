# ==============================================================================
# EXPORTA LOS DATOS DEL MODELO EPIGENÉTICO PARA EL DASHBOARD (JSON, una tabla por archivo)
#
# Lee el modelo guardado (ElasticNet, 500 sondas + sexo), los resultados de
# resultados/modelo_epigenetico/ y los valores beta de las 500 sondas desde las
# cachés de "Modelo 1/Datasets". No entrena nada. Tarda 1-2 minutos (lee ~2,5 GB).
#
# Uso:   python -X utf8 dashboard/exportar_epigenetico.py
# Salida: dashboard/datos/epi_*.json
#
# La explicación por paciente es exacta: en un modelo lineal, la edad predicha es
# la base (intercepto) más la suma de coeficiente x valor estandarizado de cada sonda.
# ==============================================================================

import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

XAI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_1 = os.path.join(os.path.dirname(XAI), "Modelo 1", "Datasets")
RES = os.path.join(XAI, "resultados", "modelo_epigenetico")
SALIDA = os.path.join(XAI, "dashboard", "datos")
N_APORTES = 15      # sondas con más efecto que se guardan por paciente (el resto va sumado en "Otras")
N_SONDAS_EDAD = 15  # sondas de mayor peso para el gráfico de metilación vs edad
N_SHAP = 20         # sondas con más aporte medio para el gráfico de enjambre (SHAP)
COHORTE = {"Hannum": "Hannum", "GSE87571": "GSE87571"}


def guardar(nombre, df):
    filas = json.loads(df.to_json(orient="records", force_ascii=False))
    with open(os.path.join(SALIDA, f"{nombre}.json"), "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{nombre}.json: {len(filas)} filas, {os.path.getsize(os.path.join(SALIDA, f'{nombre}.json')) / 1024:.0f} KB")


def bh(p):
    """Valores q de Benjamini-Hochberg."""
    p = np.asarray(p); n = len(p); orden = np.argsort(p)
    q = p[orden] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[orden] = np.minimum(q, 1)
    return out


# ------------------------------------------------------------------------------
# 0. Modelo y partición
# ------------------------------------------------------------------------------
paquete = joblib.load(os.path.join(XAI, "modelos", "epigenetico_elasticnet", "modelo_elasticnet.joblib"))
pipe, columnas = paquete["modelo"], paquete["columnas"]
sondas = columnas[:-1]
escalador, enet = pipe.named_steps["scaler"], pipe.named_steps["model"]
coef = enet.coef_
base = float(enet.intercept_)
par = pd.read_csv(os.path.join(RES, "particion_pacientes_elasticnet.csv"))

# ------------------------------------------------------------------------------
# 1. Valores beta de las 500 sondas para los 1.385 pacientes
# ------------------------------------------------------------------------------
print("Leyendo las 500 sondas desde las cachés de Modelo 1...")
todas = np.load(os.path.join(MODELO_1, "cache_matriz_completa", "sondas_cpg.npy"))
pos = pd.Series(np.arange(len(todas)), index=todas)[sondas].values
bloques = []
for carpeta, ds in (("cache_matriz_completa", "Hannum"), ("cache_GSE87571", "GSE87571")):
    M = np.load(os.path.join(MODELO_1, carpeta, "matriz_cpg_float32.npy"), mmap_mode="r")
    pac = pd.read_csv(os.path.join(MODELO_1, carpeta, "pacientes.csv"))
    # Lectura secuencial por bloques de filas (leer columnas sueltas del archivo es muy lento)
    sub = np.vstack([np.array(M[i:i + 32])[:, pos] for i in range(0, M.shape[0], 32)])
    print(f"   {ds}: {sub.shape[0]} pacientes leídos", flush=True)
    B = pd.DataFrame(sub.astype(np.float64), columns=sondas, index=pac["ID_Paciente"].values)
    B = B.loc[par.loc[par.Dataset == ds, "ID_Paciente"]]
    # Imputación igual que en Modelo 1: mediana del train de esa cohorte
    ids_train = par.loc[(par.Dataset == ds) & (par.Conjunto == "train"), "ID_Paciente"]
    B = B.fillna(B.loc[ids_train].median())
    bloques.append(B)
betas = pd.concat(bloques).loc[par["ID_Paciente"]]
X = betas.assign(Sexo_M=(par["Sexo"].values == "M").astype(float))[columnas].values

# Verificación: el modelo reproduce las predicciones guardadas del test
es_test = (par["Conjunto"] == "test").values
pred = pipe.predict(X)
dif = np.abs(pred[es_test] - par.loc[es_test, "Prediccion_ElasticNet"].values).max()
print(f"Diferencia máxima con las predicciones de test guardadas: {dif:.4f} años")
assert dif < 0.01, "Los datos leídos no reproducen las predicciones del modelo"

# ------------------------------------------------------------------------------
# 2. Métricas
# ------------------------------------------------------------------------------
met = pd.read_csv(os.path.join(RES, "metricas_modelo_final.csv"))
met.insert(0, "prueba", ["cv", "test", "hannum", "gse87571"])
met["evaluacion"] = ["Validación cruzada, 5 folds (1.038 de train)", "Test combinado (347 pacientes)",
                     "Test Hannum GSE40279 (164)", "Test GSE87571 (183)"]
guardar("epi_metricas", met.rename(columns={"acierto_5a_%": "acierto5"})
        [["prueba", "evaluacion", "N", "MAE", "RMSE", "R2", "r", "acierto5"]].round(3))

# 3. Comparación de los 5 algoritmos (combinado y por cohorte)
c5 = pd.read_csv(os.path.join(RES, "comparacion_5_modelos.csv"))
c5d = pd.read_csv(os.path.join(RES, "comparacion_5_modelos_por_dataset.csv"))
comb = pd.DataFrame({"cohorte": "Combinado", "modelo": c5["Modelo"], "n": 347,
                     "MAE": c5["MAE (Años)"], "R2": c5["R²"]})
porc = pd.DataFrame({"cohorte": c5d["Dataset"], "modelo": c5d["Modelo"], "n": c5d["N_test"],
                     "MAE": c5d["MAE (Años)"], "R2": c5d["R²"]})
mod = pd.concat([comb, porc], ignore_index=True)
mod.insert(0, "clave", mod["cohorte"] + "|" + mod["modelo"])
guardar("epi_modelos", mod.round(3))

# 4. Matriz de confusión (persona mayor, 60 años o más)
mc = pd.read_csv(os.path.join(RES, "metricas_matriz_confusion.csv"))
def separar(col):
    txt = mc[col].astype(str)
    return (txt.str.extract(r"^([\d.]+)%")[0].astype(float),
            txt.str.extract(r"\[([\d.]+),")[0].astype(float),
            txt.str.extract(r",\s*([\d.]+)\]")[0].astype(float))
conf = pd.DataFrame({"clave": mc["Test"] + "|" + mc["Modelo"], "cohorte": mc["Test"], "modelo": mc["Modelo"],
                     "N": mc["N"], "VP": mc["VP"], "FN": mc["FN"], "FP": mc["FP"], "VN": mc["VN"], "F1": mc["F1"]})
for col, nombre in [("Exactitud", "exactitud"), ("Sensibilidad", "sensibilidad"), ("Especificidad", "especificidad"),
                    ("VPP", "vpp"), ("VPN", "vpn")]:
    v, lo, hi = separar(col)
    conf[nombre], conf[f"{nombre}_lo"], conf[f"{nombre}_hi"] = v, lo, hi
guardar("epi_confusion", conf)

# 5-6. Curva de sondas y curva de aprendizaje
cs = pd.read_csv(os.path.join(RES, "curva_sondas.csv"))
cs.insert(0, "clave", cs["Modelo"] + "|" + cs["k"].astype(str))
guardar("epi_curva_sondas", cs.rename(columns={"Modelo": "modelo"}).round(4))
ca = pd.read_csv(os.path.join(RES, "curva_aprendizaje.csv"))
ca.insert(0, "clave", ca["Modelo"] + "|" + ca["Fraccion"].astype(str))
guardar("epi_curva_aprendizaje", ca.rename(columns={"Modelo": "modelo"}).round(4))

# ------------------------------------------------------------------------------
# 7. Coeficientes (explicación global)
# ------------------------------------------------------------------------------
idx_tr = ~es_test
edad = par["Edad"].values
rho_edad = [spearmanr(X[idx_tr, j], edad[idx_tr])[0] for j in range(len(columnas))]  # solo train
co = pd.DataFrame({"variable": [c if c != "Sexo_M" else "Sexo (hombre)" for c in columnas],
                   "tipo": ["Sonda"] * len(sondas) + ["Sexo"], "coeficiente": coef, "rho_con_edad": rho_edad,
                   "mediana_train": list(paquete["medianas_imputacion"].values) + [None],
                   # Para el simulador de la página: aporte = coeficiente x (valor - media) / escala
                   "media": escalador.mean_, "escala": escalador.scale_,
                   "p01": np.percentile(X[idx_tr], 1, axis=0), "p99": np.percentile(X[idx_tr], 99, axis=0)})
co["abs"] = co["coeficiente"].abs()
co = co.sort_values("abs", ascending=False).drop(columns="abs")
co.insert(0, "puesto", range(1, len(co) + 1))
guardar("epi_coeficientes", co.round({"coeficiente": 7, "rho_con_edad": 4, "mediana_train": 5,
                                      "media": 6, "escala": 6, "p01": 4, "p99": 4}))  # 7 decimales: hay coeficientes muy chicos pero distintos de 0

# 8. Metilación vs edad de las sondas de mayor peso (pacientes de test)
top = [v for v in co["variable"] if v.startswith("cg")][:N_SONDAS_EDAD]
pt = par[es_test].reset_index(drop=True)
se = pd.DataFrame([{"clave": f"{r.ID_Paciente}|{v}", "variable": v, "id": r.ID_Paciente, "cohorte": r.Dataset,
                    "edad_real": r.Edad, "beta": betas.loc[r.ID_Paciente, v]}
                   for v in top for r in pt.itertuples()])
guardar("epi_sondas_edad", se.round(4))

# ------------------------------------------------------------------------------
# 9-10. Pacientes de test y aporte de cada sonda (exacto)
# ------------------------------------------------------------------------------
Z = escalador.transform(X[es_test])
aportes = Z * coef                       # años que suma cada variable a la base
pred_test = base + aportes.sum(axis=1)
pac = pd.DataFrame({"id": pt["ID_Paciente"], "edad_real": pt["Edad"], "edad_predicha": pred_test,
                    "brecha_edad": pred_test - pt["Edad"], "sexo": pt["Sexo"].map({"M": "Male", "F": "Female"}),
                    "cohorte": pt["Dataset"], "base": base})
guardar("epi_pacientes_test", pac.round(2))
nombres = [c if c != "Sexo_M" else "Sexo (hombre)" for c in columnas]
filas = []
for i, pid in enumerate(pt["ID_Paciente"]):
    orden = np.argsort(-np.abs(aportes[i]))
    for j in orden[:N_APORTES]:
        filas.append({"clave": f"{pid}|{nombres[j]}", "id": pid, "variable": nombres[j],
                      "tipo": "Sexo" if columnas[j] == "Sexo_M" else "Sonda",
                      "beta": None if columnas[j] == "Sexo_M" else float(X[es_test][i, j]),
                      "aporte_anos": float(aportes[i, j])})
    filas.append({"clave": f"{pid}|otras", "id": pid, "variable": f"Otras {len(columnas) - N_APORTES} variables",
                  "tipo": "Resto", "beta": None, "aporte_anos": float(aportes[i, orden[N_APORTES:]].sum())})
guardar("epi_aportes", pd.DataFrame(filas).round(4))

# ------------------------------------------------------------------------------
# 10b. SHAP global. En un modelo lineal con estandarización, el aporte de cada
# variable (coeficiente x valor estandarizado) es exactamente su valor SHAP.
# ------------------------------------------------------------------------------
es_hannum = (pt["Dataset"] == "Hannum").values
imp = pd.DataFrame({"variable": nombres, "coeficiente": coef,
                    "media_abs": np.abs(aportes).mean(axis=0),
                    "media_abs_hannum": np.abs(aportes[es_hannum]).mean(axis=0),
                    "media_abs_gse": np.abs(aportes[~es_hannum]).mean(axis=0)})
imp = imp.sort_values("media_abs", ascending=False)
imp.insert(0, "puesto", range(1, len(imp) + 1))
guardar("epi_importancia", imp.round(5))

top_shap = imp["variable"].head(N_SHAP).tolist()
Xte = X[es_test]
filas = []
for v in top_shap:
    j = nombres.index(v)
    for i, pid in enumerate(pt["ID_Paciente"]):
        filas.append({"clave": f"{pid}|{v}", "id": pid, "variable": v, "aporte_anos": float(aportes[i, j]),
                      "beta": None if columnas[j] == "Sexo_M" else float(Xte[i, j]), "z": float(Z[i, j])})
guardar("epi_shap", pd.DataFrame(filas).round(4))

# ------------------------------------------------------------------------------
# 11-12. Brecha de edad (1.385 pacientes, todas las predicciones fuera de muestra)
# ------------------------------------------------------------------------------
pred_fuera = par["Prediccion_ElasticNet"].values
brecha = pred_fuera - edad
a, b = np.polyfit(edad, brecha, 1)
brecha_corr = brecha - (a * edad + b)
br = pd.DataFrame({"fila": range(len(par)), "id": par["ID_Paciente"], "conjunto": par["Conjunto"],
                   "cohorte": par["Dataset"], "sexo": par["Sexo"].map({"M": "Male", "F": "Female"}),
                   "edad_real": edad, "edad_predicha": pred_fuera, "brecha": brecha, "brecha_corregida": brecha_corr})
guardar("epi_brecha", br.round(2))

# Spearman parcial (descontando la edad) DENTRO de cada cohorte. Juntar las cohortes mezclaría
# el efecto de laboratorio, y la recta de corrección común deja algo de tendencia con la edad
# dentro de cada cohorte, que haría parecer asociada a cualquier sonda ligada a la edad.
from scipy.stats import rankdata, t as dist_t


def spearman_parcial(x, y, z):
    """Correlación de Spearman entre x e y descontando z, con su valor p."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1]); gl = len(x) - 3
    return r, float(2 * dist_t.sf(abs(r) * np.sqrt(gl / max(1e-12, 1 - r * r)), gl))


filas = []
masc = {ds: (par["Dataset"] == ds).values for ds in ("Hannum", "GSE87571")}
for j, v in enumerate(sondas):
    fila = {"variable": v, "coeficiente": coef[j], "rho_con_edad": spearmanr(X[:, j], edad)[0]}
    for ds, sufijo in (("Hannum", "hannum"), ("GSE87571", "gse")):
        m = masc[ds]
        fila[f"rho_{sufijo}"], fila[f"p_{sufijo}"] = spearman_parcial(X[m, j], brecha[m], edad[m])
    filas.append(fila)
bs = pd.DataFrame(filas)
bs["q_hannum"], bs["q_gse"] = bh(bs["p_hannum"].values), bh(bs["p_gse"].values)
# Asociada = q < 0,05 en las dos cohortes y con el mismo signo
bs["replicada"] = (bs["q_hannum"] < 0.05) & (bs["q_gse"] < 0.05) & (np.sign(bs["rho_hannum"]) == np.sign(bs["rho_gse"]))
bs["p_max"] = bs[["p_hannum", "p_gse"]].max(axis=1)
guardar("epi_brecha_sondas", bs.sort_values("p_max")[["variable", "coeficiente", "rho_hannum", "q_hannum", "rho_gse", "q_gse",
                                                      "replicada", "p_max", "rho_con_edad"]].round(4))
print(f"Corrección de la brecha: brecha = {a:.4f} x edad + {b:.3f}")
print(f"Brecha corregida media por cohorte: {br.groupby('cohorte')['brecha_corregida'].mean().round(2).to_dict()}")
print(f"Sondas asociadas a la brecha en ambas cohortes (q < 0,05, mismo signo): {bs['replicada'].sum()} de {len(bs)}")
