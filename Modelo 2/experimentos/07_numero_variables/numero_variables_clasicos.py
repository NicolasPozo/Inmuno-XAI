# ==============================================================================
# ¿CUÁNTAS VARIABLES NECESITA CADA MODELO? (etapa 12, modelos clásicos)
# Proyecto: Immuno-XAI - UTEM
#
# Repite la prueba de numero_variables.py (que se hizo solo con TabPFN) con los
# cinco modelos clásicos del paso 05: ElasticNet, SVR (RBF), Random Forest,
# LightGBM y XGBoost. Responde si los 45 biomarcadores favorecen a TabPFN o si
# algún modelo clásico rinde mejor con menos o más variables.
#
# Mismo diseño que con TabPFN:
#   - Candidatas: los 71 biomarcadores con >= 80 % de datos en el train. El sexo
#     siempre se incluye.
#   - Orden de las variables: |Spearman| con la edad, calculado solo con los
#     pacientes de entrenamiento de cada partición (sin fuga).
#   - Fase 1 (elegir, solo train): CV agrupada 5x3 sobre los 354 de train, con las
#     mismas particiones que la curva de TabPFN.
#       a) k mejores dentro de las 45 actuales (k = 15 ... 45)
#       b) k mejores de las 71 candidatas (k = 10 ... 71)
#   - Fase 2 (confirmar): para cada modelo cuyo mejor k no sea el actual, ese k se
#     evalúa como en el paso 05 (CV 5x5 sobre 473, test de 119, estudios no vistos)
#     y se compara con el mismo modelo con las 45 actuales.
# Hiperparámetros: los elegidos en el paso 05 (con las 45 variables), fijos para
# todos los k, como en la curva de sondas del modelo epigenético. Así la curva
# refleja solo el efecto del número de variables.
# El modelo de dos ramas no se incluye: usa su propio conjunto de variables.
#
# Requiere: Datasets/pareados.csv, resultados/comparacion_modelos/ (paso 05) y,
#           para el gráfico, resultados/curva_numero_variables.csv (numero_variables.py).
# Uso:  python -X utf8 experimentos/07_numero_variables/numero_variables_clasicos.py   (~3 min en CPU)
# Salida: experimentos/07_numero_variables/resultados/
# ==============================================================================

import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from config import CARPETA_RES_COMPARACION, EXP_NUMERO_VARIABLES  # noqa: E402
carpeta = EXP_NUMERO_VARIABLES / "resultados"
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)

# ------------------------------------------------------------------------------
# Datos, modelos y métricas del paso 05 (se ejecuta su primera parte)
# ------------------------------------------------------------------------------
ruta05 = SRC / "05_comparar_modelos.py"
codigo = ruta05.read_text(encoding="utf-8")
ns = {"__file__": str(ruta05), "__name__": "comparar"}
exec(compile(codigo[:codigo.index("for nombre, (est, grilla) in CLASICOS.items():")], str(ruta05), "exec"), ns)
par, y, grupos, estudios = ns["par"], ns["y"], ns["grupos"], ns["estudios"]
idx_tr, idx_te, VARS45, metricas = ns["idx_tr"], ns["idx_te"], ns["VARIABLES"], ns["metricas"]
LOSO = ns["LOSO_ESTUDIOS"]
sexo = (par["gender"] == "Male").astype(float).values.reshape(-1, 1)
meta = ["subject_accession", "gender", "age", "conjunto", "edad", "study_accession"]
extra = [c for c in par.columns if c not in meta + VARS45 and par.loc[idx_tr, c].notna().mean() >= 0.8]
POOL = VARS45 + extra
assert len(POOL) == 71, len(POOL)


def valor(v):
    """Convierte el texto de hiperparametros.csv al tipo de Python."""
    if pd.isna(v) or v == "":
        return None
    for tipo in (int, float):
        try:
            return tipo(v)
        except ValueError:
            pass
    return v


hp = pd.read_csv(CARPETA_RES_COMPARACION / "hiperparametros.csv", dtype=str, keep_default_na=False)
MODELOS = {}
for nombre, (est, _) in ns["CLASICOS"].items():
    params = {}
    for _, f in hp[hp.modelo == nombre].iterrows():
        clave = f.parametro if f.parametro in est.get_params() else f"m__{f.parametro}"
        params[clave] = valor(f.valor)
    MODELOS[nombre] = clone(est).set_params(**params)
    log(f"{nombre}: {params}")


def ranking(idx, pool):
    rho = par.iloc[idx][pool].corrwith(pd.Series(y[idx], index=par.index[idx]), method="spearman").abs()
    return rho.sort_values(ascending=False).index.tolist()


def matriz(cols, idx):
    return np.hstack([par.iloc[idx][cols].astype(float).values, sexo[idx]]).astype(np.float32)


def predecir(modelo, pool, k, tr, va):
    cols = ranking(tr, pool)[:k]
    return clone(modelo).fit(matriz(cols, tr), y[tr]).predict(matriz(cols, va))


# ------------------------------------------------------------------------------
# Fase 1: curvas con CV 5x3 solo sobre el train (mismas particiones que TabPFN)
# ------------------------------------------------------------------------------
REP1 = 3
SPLITS1 = [[(idx_tr[a], idx_tr[b]) for a, b in
            GroupKFold(n_splits=5, shuffle=True, random_state=r).split(idx_tr, groups=grupos[idx_tr])]
           for r in range(REP1)]
CONFIGS = ([("Las 45 actuales", VARS45, k) for k in [45, 40, 35, 30, 25, 20, 15]]
           + [("Candidatas (71)", POOL, k) for k in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, len(POOL)]])

cache = carpeta / "fase1_clasicos.npz"
F1 = dict(np.load(cache, allow_pickle=True))["F1"].item() if cache.exists() else {}
for nombre, modelo in MODELOS.items():
    t = time.time()
    pendientes = [(fam, pool, k) for fam, pool, k in CONFIGS if f"{nombre}|{fam}|{k}" not in F1]
    tareas = [(fam, k, r, tr, va, pool) for fam, pool, k in pendientes for r in range(REP1) for tr, va in SPLITS1[r]]
    salidas = Parallel(n_jobs=-1)(delayed(predecir)(modelo, pool, k, tr, va) for _, k, _, tr, va, pool in tareas)
    for (fam, k, r, _, va, _), p in zip(tareas, salidas):
        F1.setdefault(f"{nombre}|{fam}|{k}", np.full((REP1, len(y)), np.nan))[r, va] = p
    np.savez(cache, F1=np.array(F1, dtype=object))
    log(f"{nombre:<14s} fase 1 lista | {(time.time()-t)/60:.1f} min")

filas = []
for nombre in MODELOS:
    ref = F1[f"{nombre}|Las 45 actuales|45"]
    mae_ref = np.array([mean_absolute_error(y[idx_tr], ref[r, idx_tr]) for r in range(REP1)])
    for fam, _, k in CONFIGS:
        oof = F1[f"{nombre}|{fam}|{k}"]
        mae = np.array([mean_absolute_error(y[idx_tr], oof[r, idx_tr]) for r in range(REP1)])
        filas.append({"modelo": nombre, "candidatas": fam, "k": k, "variables": k + 1, "MAE_CV_train": mae.mean(),
                      "MAE_sd": mae.std(), "R2_CV_train": np.mean([r2_score(y[idx_tr], oof[r, idx_tr]) for r in range(REP1)]),
                      "dif_vs_45": (mae - mae_ref).mean(), "dif_vs_45_min": (mae - mae_ref).min(),
                      "dif_vs_45_max": (mae - mae_ref).max()})
curva = pd.DataFrame(filas).round(3)
curva.to_csv(carpeta / "curva_numero_variables_clasicos.csv", index=False)

# ------------------------------------------------------------------------------
# Fase 2: confirmar el mejor k de cada modelo (evaluación del paso 05)
# ------------------------------------------------------------------------------
REP2 = 5
SPLITS2 = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REP2)]
guardado_cv = np.load(CARPETA_RES_COMPARACION / "predicciones_cv.npz", allow_pickle=True)
guardado_te = pd.read_csv(CARPETA_RES_COMPARACION / "predicciones_test.csv")


def evaluar(modelo, pool, k):
    tareas = ([("cv", r, tr, va) for r in range(REP2) for tr, va in SPLITS2[r]] + [("test", None, idx_tr, idx_te)]
              + [("loso", e, np.where(estudios != e)[0], np.where(estudios == e)[0]) for e in LOSO])
    salidas = Parallel(n_jobs=-1)(delayed(predecir)(modelo, pool, k, tr, va) for _, _, tr, va in tareas)
    oof, te, lo = np.zeros((REP2, len(y))), None, {}
    for (tipo, r, _, va), p in zip(tareas, salidas):
        if tipo == "cv":
            oof[r, va] = p
        elif tipo == "test":
            te = p
        else:
            lo[r] = p
    return oof, te, lo


conf = []
for nombre, modelo in MODELOS.items():
    c = curva[curva.modelo == nombre]
    mejor = c.loc[c.MAE_CV_train.idxmin()]
    actual = c[(c.candidatas == "Las 45 actuales") & (c.k == 45)].iloc[0]
    fila = {"modelo": nombre, "mejor_config": f"{mejor.candidatas}, k = {int(mejor.k)}",
            "MAE_CV_train_45": actual.MAE_CV_train, "MAE_CV_train_mejor": mejor.MAE_CV_train}
    if mejor.candidatas == "Las 45 actuales" and mejor.k == 45:
        log(f"{nombre}: el mejor k es el actual (45); no hay nada que confirmar")
        conf.append(fila)
        continue
    pool = VARS45 if mejor.candidatas == "Las 45 actuales" else POOL
    log(f"{nombre}: confirmando {fila['mejor_config']}")
    res = {}
    for clave, (p_, k_) in {"45": (VARS45, 45), "mejor": (pool, int(mejor.k))}.items():
        oof, te, lo = evaluar(modelo, p_, k_)
        res[clave] = {"cv": np.mean([mean_absolute_error(y, oof[r]) for r in range(REP2)]),
                      "test": te, "loso": np.mean([metricas(y[estudios == e], lo[e])["MAE"] for e in LOSO])}
    # La versión con las 45 debe reproducir la del paso 05 (mismo modelo y particiones)
    cv05 = np.mean([mean_absolute_error(y, guardado_cv[nombre][r]) for r in range(REP2)])
    te05 = mean_absolute_error(y[idx_te], guardado_te[nombre].values)
    e45, eme = np.abs(res["45"]["test"] - y[idx_te]), np.abs(res["mejor"]["test"] - y[idx_te])
    rng = np.random.default_rng(0)
    dif = np.array([(eme[i] - e45[i]).mean() for i in (rng.integers(0, len(e45), len(e45)) for _ in range(10000))])
    fila.update({"MAE_CV473_45": res["45"]["cv"], "MAE_CV473_mejor": res["mejor"]["cv"],
                 "MAE_test_45": e45.mean(), "MAE_test_mejor": eme.mean(),
                 "dif_test": dif.mean(), "IC95_bajo": np.percentile(dif, 2.5), "IC95_alto": np.percentile(dif, 97.5),
                 "MAE_loso_45": res["45"]["loso"], "MAE_loso_mejor": res["mejor"]["loso"],
                 "MAE_CV473_paso05": cv05, "MAE_test_paso05": te05})
    log(f"  CV 473 {res['45']['cv']:.2f} -> {res['mejor']['cv']:.2f} (paso 05: {cv05:.2f}) | "
        f"test {e45.mean():.2f} -> {eme.mean():.2f} (paso 05: {te05:.2f}; dif {dif.mean():+.2f} "
        f"[{np.percentile(dif, 2.5):+.2f}, {np.percentile(dif, 97.5):+.2f}]) | "
        f"estudios no vistos {res['45']['loso']:.2f} -> {res['mejor']['loso']:.2f}")
    conf.append(fila)
confirmacion = pd.DataFrame(conf).round(3)
confirmacion.to_csv(carpeta / "confirmacion_clasicos.csv", index=False)

# ------------------------------------------------------------------------------
# Gráfico: una curva por modelo (TabPFN + 5 clásicos), mismo eje vertical
# ------------------------------------------------------------------------------
AZUL, NARANJA, GRIS, TINTA, TINTA2, FONDO = "#2a78d6", "#eb6834", "#b9b8b1", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
                     "ytick.color": TINTA2, "figure.facecolor": FONDO, "axes.facecolor": FONDO,
                     "axes.spines.top": False, "axes.spines.right": False})
tab = pd.read_csv(carpeta / "curva_numero_variables.csv").assign(modelo="TabPFN v2")
todas = pd.concat([tab, curva], ignore_index=True)
orden = ["TabPFN v2"] + list(MODELOS)
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True, sharey=True)
for ax, nombre in zip(axes.ravel(), orden):
    d = todas[todas.modelo == nombre]
    for fam, color, etiqueta in [("Candidatas (71)", AZUL, "k mejores de las 71 candidatas"),
                                 ("Las 45 actuales", NARANJA, "k mejores dentro de las 45 actuales")]:
        s = d[d.candidatas == fam].sort_values("k")
        ax.fill_between(s.k, s.MAE_CV_train - s.MAE_sd, s.MAE_CV_train + s.MAE_sd, color=color, alpha=0.12, lw=0)
        ax.plot(s.k, s.MAE_CV_train, color=color, lw=1.8, marker="o", ms=4.5, mec=FONDO, mew=1.2, label=etiqueta)
    a = d[(d.candidatas == "Las 45 actuales") & (d.k == 45)].iloc[0]
    m = d.loc[d.MAE_CV_train.idxmin()]
    ax.scatter([45], [a.MAE_CV_train], s=90, facecolor="none", edgecolor=TINTA, lw=1.4, zorder=5)
    ax.set_title(f"{nombre}\n45 actuales: {a.MAE_CV_train + 1e-9:.2f} | mejor: {m.MAE_CV_train + 1e-9:.2f} (k = {int(m.k)})",
                 color=TINTA, loc="left", fontsize=10, fontweight="bold" if nombre == "TabPFN v2" else "normal")
    ax.grid(color="#e8e7e3", lw=0.6)
    ax.set_axisbelow(True)
for ax in axes[1]:
    ax.set_xlabel("Número de biomarcadores (k, más el sexo)")
for ax in axes[:, 0]:
    ax.set_ylabel("MAE CV 5×3 en el train (años)")
axes[0, 0].legend(frameon=False, loc="upper right", fontsize=8.5)
fig.suptitle("Error según el número de biomarcadores en cada modelo (solo train; círculo = 45 actuales; banda = ±1 DE)",
             color=TINTA, x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(carpeta / "curva_numero_variables_todos.png", dpi=200)
plt.close(fig)

pd.set_option("display.width", 250)
resumen = curva.loc[curva.groupby("modelo").MAE_CV_train.idxmin(), ["modelo", "candidatas", "k", "MAE_CV_train"]]
print("\nMejor configuración por modelo (fase 1):\n" + resumen.to_string(index=False))
print("\n" + confirmacion.to_string(index=False))
log("Listo.")
