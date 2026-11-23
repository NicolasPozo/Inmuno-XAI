# ==============================================================================
# LIME DEL MODELO INMUNOLÓGICO (TabPFN) Y COMPARACIÓN CON SHAP Y OCLUSIÓN
#
#   python -X utf8 explicabilidad/lime_inmunologico.py        (alrededor de 1 hora y media en CPU)
#
# Hay que correr antes explicabilidad/shap_inmunologico.py: este script compara
# LIME con sus resultados.
#
# LIME explica a un paciente ajustando un modelo lineal simple a las
# predicciones de TabPFN en pacientes parecidos, que genera al azar. Se usa con
# su configuración habitual (lime.lime_tabular.LimeTabularExplainer):
#
#   - Cada biomarcador se divide en los cuartiles del train. El peso de LIME es
#     cuántos años suma o resta que el biomarcador esté en el cuartil del
#     paciente y no en otro.
#   - N_MUESTRAS pacientes generados por paciente explicado.
#   - LIME no admite valores vacíos: se rellenan con la mediana del train
#     (en SHAP se dejan vacíos, como los recibe el modelo).
#
# LIME queda como comparación para la tesis; la página muestra SHAP. Se
# comparan tres explicaciones por paciente (SHAP, LIME y oclusión: llevar cada
# biomarcador a la mediana) y, en el ranking global, también la importancia por
# permutación. Además, 10 pacientes se repiten con otra semilla para medir
# cuánto cambia cada método entre dos corridas.
#
# Escribe en resultados/explicabilidad/:
#   lime_inmunologico_test.csv            (una fila por paciente, una columna por variable)
#   comparacion_metodos_por_paciente.csv  (acuerdo entre métodos, paciente por paciente)
#   comparacion_metodos_global.csv        (ranking de biomarcadores según cada método)
#   comparacion_metodos_resumen.csv       (las cifras para la tesis)
# y el gráfico graficos/explicabilidad/comparacion_shap_lime.png
# ==============================================================================

import os

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from comun import RESULTADOS, XAI, Inmune, api, leer_json, log

N_MUESTRAS = 1000
N_REPETIDOS = 10  # pacientes que se repiten con otra semilla (estabilidad)
N_SHAP = 1000     # combinaciones de SHAP en la repetición (las mismas que shap_inmunologico.py)
TOP = 5
SALIDA = os.path.join(RESULTADOS, "lime_inmunologico_test.csv")
AVANCE = os.path.join(RESULTADOS, f"lime_inmunologico_avance_{N_MUESTRAS}.csv")
ARCHIVO_SHAP = os.path.join(RESULTADOS, "shap_inmunologico_test.csv")
GRAFICO = os.path.join(XAI, "graficos", "explicabilidad", "comparacion_shap_lime.png")

if not os.path.exists(ARCHIVO_SHAP):
    raise SystemExit("Falta resultados/explicabilidad/shap_inmunologico_test.csv. Corre antes explicabilidad/shap_inmunologico.py")

inm = Inmune()
n, k = len(inm.ids), len(inm.variables)
rellenar = lambda X: np.where(np.isnan(X), inm.medianas, X).astype(np.float64)
X_train, X_test = rellenar(inm.X_train), rellenar(inm.X_test)
log(f"Valores vacíos rellenados con la mediana en el test: {int(np.isnan(inm.X_test).sum())} "
    f"en {int(np.isnan(inm.X_test).any(axis=1).sum())} pacientes")


def lime_de(fila, semilla):
    """Pesos de LIME (años) de las 46 variables de un paciente, más el ajuste del modelo local."""
    explicador = LimeTabularExplainer(X_train, mode="regression", feature_names=inm.nombres, categorical_features=[k - 1],
                                      discretize_continuous=True, random_state=semilla)
    exp = explicador.explain_instance(fila, inm.predecir, num_features=k, num_samples=N_MUESTRAS)
    pesos = dict(exp.local_exp[1])
    return np.array([pesos[j] for j in range(k)]), float(exp.intercept[1]), float(exp.local_pred[0]), float(exp.score)


# ------------------------------------------------------------------------------
# 1. LIME paciente por paciente (con respaldo del avance)
# ------------------------------------------------------------------------------
hechos = {}
if os.path.exists(AVANCE):
    previo = pd.read_csv(AVANCE).set_index("id")
    hechos = {pid: previo.loc[pid] for pid in previo.index if pid in set(inm.ids)}
    log(f"Se retoma un cálculo anterior: {len(hechos)} pacientes ya hechos")

EXTRA = ["intercepto", "prediccion_local", "r2_local"]
lime, extra = np.zeros((n, k)), np.zeros((n, 3))
for i, pid in enumerate(inm.ids):
    if pid in hechos:
        lime[i], extra[i] = hechos[pid][inm.nombres].values.astype(float), hechos[pid][EXTRA].values.astype(float)
        continue
    lime[i], *extra[i] = lime_de(X_test[i], semilla=i)
    fila = pd.DataFrame([[pid, *extra[i], *lime[i]]], columns=["id", *EXTRA, *inm.nombres])
    fila.to_csv(AVANCE, mode="a", header=not os.path.exists(AVANCE), index=False)
    if (i + 1) % 5 == 0 or i + 1 == n:
        log(f"LIME: {i + 1} de {n} pacientes")

pred_rellena = inm.predecir(X_test)  # edad predicha con los vacíos rellenados (la que explica LIME)
tabla = pd.DataFrame(lime, columns=inm.nombres)
for j, c in enumerate(EXTRA):
    tabla.insert(j, c, extra[:, j])
tabla.insert(0, "edad_predicha_rellena", pred_rellena)
tabla.insert(0, "edad_predicha", inm.pred)
tabla.insert(0, "id", inm.ids)
tabla.round(4).to_csv(SALIDA, index=False)
log(f"Guardado {SALIDA}")

# ------------------------------------------------------------------------------
# 2. Estabilidad: los mismos pacientes con otra semilla, en LIME y en SHAP
# ------------------------------------------------------------------------------
shap = pd.read_csv(ARCHIVO_SHAP).set_index("id").loc[inm.ids, inm.nombres].values
repetidos = np.linspace(0, n - 1, N_REPETIDOS).astype(int)
log(f"Repitiendo {N_REPETIDOS} pacientes con otra semilla...")
lime2 = np.array([lime_de(X_test[i], semilla=10_000 + i)[0] for i in repetidos])
shap2 = np.vstack([api.shap_kernel(inm.predecir, inm.X_test[i:i + 1], inm.medianas, N_SHAP, semilla=10_000 + i)[1] for i in repetidos])


def comunes(a, b, top=TOP):
    """Cuántas variables comparten los 'top' de dos explicaciones (por valor absoluto)."""
    return len(set(np.argsort(-np.abs(a))[:top]) & set(np.argsort(-np.abs(b))[:top]))


# ------------------------------------------------------------------------------
# 3. Comparación paciente por paciente: SHAP, LIME y oclusión
# ------------------------------------------------------------------------------
ocl = pd.DataFrame(leer_json("aportes_paciente")).pivot(index="id", columns="variable", values="aporte_anos").loc[inm.ids, inm.nombres].values
metodos = {"shap": shap, "lime": lime, "oclusion": ocl}
pares = [("shap", "lime"), ("shap", "oclusion"), ("lime", "oclusion")]
filas = []
for i, pid in enumerate(inm.ids):
    f = {"id": pid, "edad_predicha": round(float(inm.pred[i]), 2), "biomarcadores_vacios": int(np.isnan(inm.X_test[i]).sum()),
         "lime_r2_local": round(float(extra[i, 2]), 3), "lime_error_local": round(float(abs(extra[i, 1] - pred_rellena[i])), 2)}
    principales = np.argsort(-np.abs(shap[i]))[:10]
    for a, b in pares:
        f[f"rho_{a}_{b}"] = round(float(spearmanr(metodos[a][i], metodos[b][i])[0]), 3)
        f[f"top{TOP}_{a}_{b}"] = comunes(metodos[a][i], metodos[b][i])
        f[f"signo_{a}_{b}"] = round(float((np.sign(metodos[a][i, principales]) == np.sign(metodos[b][i, principales])).mean()), 2)
    filas.append(f)
por_paciente = pd.DataFrame(filas)
por_paciente.to_csv(os.path.join(RESULTADOS, "comparacion_metodos_por_paciente.csv"), index=False)

# ------------------------------------------------------------------------------
# 4. Ranking global de biomarcadores según cada método
# ------------------------------------------------------------------------------
perm = pd.DataFrame(leer_json("importancia")).set_index("variable")["aumento_MAE"]
glob = pd.DataFrame({"variable": inm.nombres, "tipo": inm.tipos,
                     "shap_media_abs": np.abs(shap).mean(axis=0), "lime_media_abs": np.abs(lime).mean(axis=0),
                     "oclusion_media_abs": np.abs(ocl).mean(axis=0), "permutacion_aumento_MAE": perm[inm.nombres].values})
COLS = ["shap_media_abs", "lime_media_abs", "oclusion_media_abs", "permutacion_aumento_MAE"]
for c in COLS:
    glob[f"puesto_{c.split('_')[0]}"] = glob[c].rank(ascending=False).astype(int)
glob = glob.sort_values("shap_media_abs", ascending=False)
glob.round(4).to_csv(os.path.join(RESULTADOS, "comparacion_metodos_global.csv"), index=False)

# ------------------------------------------------------------------------------
# 5. Resumen
# ------------------------------------------------------------------------------
res = []


def anotar(medida, valor, detalle):
    res.append({"medida": medida, "valor": round(float(valor), 3), "detalle": detalle})
    log(f"   {medida}: {valor:.3f}  ({detalle})")


log("Resumen:")
NOMBRE = {"shap": "SHAP", "lime": "LIME", "oclusion": "oclusión"}
for a, b in pares:
    par = f"{NOMBRE[a]} y {NOMBRE[b]}"
    anotar(f"Correlación de Spearman por paciente, {par} (mediana)", por_paciente[f"rho_{a}_{b}"].median(),
           f"entre los 46 valores de cada paciente; mínimo {por_paciente[f'rho_{a}_{b}'].min():.2f}, máximo {por_paciente[f'rho_{a}_{b}'].max():.2f}")
    anotar(f"Biomarcadores en común en el top {TOP}, {par} (media)", por_paciente[f"top{TOP}_{a}_{b}"].mean(), f"de {TOP}, por paciente")
    anotar(f"Mismo signo en los 10 principales de SHAP, {par} (media)", por_paciente[f"signo_{a}_{b}"].mean(), "fracción, por paciente")
for a, b in [("shap", "lime"), ("shap", "oclusion"), ("shap", "permutacion"), ("lime", "permutacion")]:
    ca, cb = [next(c for c in COLS if c.startswith(m)) for m in (a, b)]
    nb = NOMBRE.get(b, "permutación")
    anotar(f"Correlación de Spearman del ranking global, {NOMBRE[a]} y {nb}", spearmanr(glob[ca], glob[cb])[0], "46 variables")
    anotar(f"Biomarcadores en común en el top 10 global, {NOMBRE[a]} y {nb}",
           len(set(glob.nlargest(10, ca)["variable"]) & set(glob.nlargest(10, cb)["variable"])), "de 10")
anotar("LIME: R² del modelo local (mediana)", por_paciente["lime_r2_local"].median(),
       f"qué tan bien la recta de LIME imita a TabPFN cerca del paciente; mínimo {por_paciente['lime_r2_local'].min():.2f}")
anotar("LIME: error del modelo local en el paciente (media)", por_paciente["lime_error_local"].mean(), "años, |predicción local − edad predicha|")
for nombre, uno, dos in [("LIME", lime[repetidos], lime2), ("SHAP", shap[repetidos], shap2)]:
    anotar(f"Estabilidad de {nombre}: diferencia media entre dos corridas", np.abs(uno - dos).mean(), f"años por variable, {N_REPETIDOS} pacientes")
    anotar(f"Estabilidad de {nombre}: diferencia máxima entre dos corridas", np.abs(uno - dos).max(), "años")
    anotar(f"Estabilidad de {nombre}: correlación de Spearman entre dos corridas (mediana)",
           np.median([spearmanr(a, b)[0] for a, b in zip(uno, dos)]), "por paciente")
    anotar(f"Estabilidad de {nombre}: biomarcadores en común en el top {TOP} (media)", np.mean([comunes(a, b) for a, b in zip(uno, dos)]), f"de {TOP}")
pd.DataFrame(res).to_csv(os.path.join(RESULTADOS, "comparacion_metodos_resumen.csv"), index=False)

# ------------------------------------------------------------------------------
# 6. Gráfico
# ------------------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs(os.path.dirname(GRAFICO), exist_ok=True)
AZUL, NARANJA, GRIS = "#2a6fd6", "#d9572b", "#8a8a8a"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
color = glob["tipo"].map({"Célula": AZUL, "Citoquina": NARANJA, "Sexo": GRIS})
a1.scatter(glob["shap_media_abs"], glob["lime_media_abs"], c=color, s=38, zorder=3)
for r in glob.head(6).itertuples():
    a1.annotate(r.variable, (r.shap_media_abs, r.lime_media_abs), xytext=(5, 4), textcoords="offset points", fontsize=8)
rho = spearmanr(glob["shap_media_abs"], glob["lime_media_abs"])[0]
a1.set_xlabel("SHAP: aporte medio en valor absoluto (años)")
a1.set_ylabel("LIME: peso medio en valor absoluto (años)")
a1.set_title(f"Importancia global de los 46 biomarcadores (Spearman {rho:.2f})".replace(".", ","), fontsize=11, loc="left")
for nombre, c in (("Célula", AZUL), ("Citoquina", NARANJA), ("Sexo", GRIS)):
    a1.scatter([], [], c=c, s=38, label=nombre)
a1.legend(frameon=False, loc="lower right")
datos = [por_paciente[f"rho_{a}_{b}"].values for a, b in pares]
cajas = a2.boxplot(datos, tick_labels=[f"{NOMBRE[a]}\ny {NOMBRE[b]}" for a, b in pares], patch_artist=True, widths=0.55,
                   medianprops={"color": "black"}, flierprops={"markersize": 3})
for caja, c in zip(cajas["boxes"], (AZUL, NARANJA, GRIS)):
    caja.set(facecolor=c, alpha=0.55)
a2.set_ylabel("Correlación de Spearman por paciente")
a2.set_title(f"Acuerdo entre métodos en cada paciente (n = {n})", fontsize=11, loc="left")
for ax in (a1, a2):
    ax.grid(alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(GRAFICO, dpi=160)
log(f"Gráfico guardado en {GRAFICO}")
log(f"Listo: {inm.filas_evaluadas} filas evaluadas por TabPFN")
