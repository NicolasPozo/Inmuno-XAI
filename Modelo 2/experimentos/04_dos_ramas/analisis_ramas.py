# ==============================================================================
# ANÁLISIS DEL PLAN B: MODELOS POR RAMA Y "¿QUÉ PASA CON MÁS DATOS?"
# Proyecto: Immuno-XAI - UTEM
#
# Parte A - Cada modelo candidato en cada rama (solo pacientes pareados):
#           ElasticNet, SVR, Random Forest, LightGBM, XGBoost (los 5 originales),
#           KNN y el ensamble de 4 (LightGBM raíz(edad) + XGBoost + SVR + KNN).
#           Afinados con tu train; evaluados con CV agrupada 5x5 sobre los 473
#           pareados y en el test de 119.
#
# Parte B - Simulación de "más datos por separado" (LightGBM por rama + fusión):
#           una rama se entrena con 25/50/75/100% de las personas de entrenamiento
#           mientras la otra usa el 100%, y también ambas a la vez. La fusión
#           siempre se aprende con todos los pareados de entrenamiento. Se ajusta
#           una curva MAE = a + b * n^(-c) y se extrapola a 2x y 4x datos.
#
# Requiere: pareados.csv (construir_fuentes_unimodales.py)
# ==============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RANDOM_STATE, REPETICIONES, N_ITER = 42, 5, 30
FRACCIONES = [0.25, 0.50, 0.75, 1.00]
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_dos_ramas")
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

# ------------------------------------------------------------------------------
# Datos (mismas variables que modelo_dos_ramas.py)
# ------------------------------------------------------------------------------
par = pd.read_csv(ruta("pareados.csv"))
y = par["age"].values.astype(float)
grupos = par["subject_accession"].values
es_train = (par["conjunto"] == "train").values
cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells",
            "Central_Memory_CD4_T_cells", "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells",
            "Effector_Memory_CD4_T_cells", "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes",
            "Memory_B_cells", "Monocytes", "Naive_B_cells", "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells",
            "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto"]]
cob = par.loc[es_train, todas_cit].notna().mean()
cols_cit = [c for c in todas_cit if cob[c] >= 0.8]

def razones(df):
    e = 0.01
    g = lambda c: df[c].astype(float)
    return pd.DataFrame({
        "r_CD4_CD8": g("CD4_T_cells") / (g("CD8_T_cells") + e),
        "r_naiveCD8_CD8": g("Naive_CD8_T_cells") / (g("CD8_T_cells") + e),
        "r_naiveCD4_CD4": g("Naive_CD4_T_cells") / (g("CD4_T_cells") + e),
        "r_naive_mem_CD8": g("Naive_CD8_T_cells") / (g("Central_Memory_CD8_T_cells") + g("Effector_Memory_CD8_T_cells") + e),
        "r_naive_mem_CD4": g("Naive_CD4_T_cells") / (g("Central_Memory_CD4_T_cells") + g("Effector_Memory_CD4_T_cells") + e),
        "r_EM_CD8": g("Effector_Memory_CD8_T_cells") / (g("CD8_T_cells") + e),
        "r_Eff_CD8": g("Effector_CD8_T_cells") / (g("CD8_T_cells") + e),
        "r_Bnaive_Bmem": g("Naive_B_cells") / (g("Memory_B_cells") + e),
        "r_Treg_CD4": g("Tregs") / (g("CD4_T_cells") + e),
        "r_mono_lymph": g("Monocytes") / (g("Lymphocytes") + e),
        "r_naiveCD8_T": g("Naive_CD8_T_cells") / (g("T_cells") + e),
    }, index=df.index)

sexo = (par["gender"] == "Male").astype(float).values.reshape(-1, 1)
XC = np.hstack([par[cols_cel].astype(float).values, razones(par).values, sexo])
XK = np.hstack([par[cols_cit].astype(float).values, sexo])
RAMAS = {"Células": XC, "Citoquinas": XK}

# ------------------------------------------------------------------------------
# Modelos
# ------------------------------------------------------------------------------
def log1p_positivo(Z):
    return np.log1p(np.clip(Z, 0, None))

def lineal(m):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(log1p_positivo)),
                     ("esc", StandardScaler()), ("m", m)])

def lgbm(**kw):
    return lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **kw)

class Promedio(BaseEstimator, RegressorMixin):
    def __init__(self, modelos=None):
        self.modelos = modelos
    def fit(self, X, y):
        self.ajustados_ = [clone(m).fit(X, y) for m in self.modelos]
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ajustados_], axis=0)

GRILLA_LGB = {"n_estimators": [200, 400, 600, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
              "max_depth": [3, 4, 5, 6, -1], "num_leaves": [7, 15, 31], "min_child_samples": [5, 10, 20, 30],
              "subsample": [0.7, 0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
              "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.0, 1.0, 5.0]}
GRILLA_XGB = {"n_estimators": [200, 400, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
              "max_depth": [2, 3, 4, 5], "min_child_weight": [1, 3, 5, 10], "subsample": [0.7, 0.8, 1.0],
              "colsample_bytree": [0.3, 0.5, 0.7, 0.9], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0.5, 1, 5]}
CANDIDATOS = {
    "ElasticNet": (lineal(ElasticNet(max_iter=20000)),
                   {"m__alpha": [0.01, 0.03, 0.1, 0.3, 1.0], "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}),
    "SVR": (lineal(SVR()), {"m__C": [5, 10, 20, 50, 100, 200], "m__gamma": ["scale", 0.003, 0.01, 0.03],
                            "m__epsilon": [0.1, 0.5, 1, 2, 5]}),
    "Random Forest": (Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))]),
                      {"m__n_estimators": [300, 600], "m__max_features": [0.2, 0.33, 0.5, 0.8],
                       "m__min_samples_leaf": [1, 2, 4, 8], "m__max_depth": [None, 8, 12]}),
    "LightGBM": (lgbm(), GRILLA_LGB),
    "XGBoost": (xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1), GRILLA_XGB),
    "KNN": (lineal(KNeighborsRegressor()), {"m__n_neighbors": [5, 10, 15, 20, 30], "m__weights": ["uniform", "distance"], "m__p": [1, 2]}),
    "LightGBM raíz(edad)": (TransformedTargetRegressor(lgbm(), func=np.sqrt, inverse_func=np.square),
                            {f"regressor__{k}": v for k, v in GRILLA_LGB.items()}),
}

def afinar(est, grilla, X):
    s = RandomizedSearchCV(est, grilla, n_iter=N_ITER, scoring="neg_mean_absolute_error",
                           cv=GroupKFold(n_splits=5, shuffle=True, random_state=100),
                           random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    s.fit(X[es_train], y[es_train], groups=grupos[es_train])
    return clone(est).set_params(**s.best_params_)

SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]
idx_tr, idx_te = np.where(es_train)[0], np.where(~es_train)[0]

def cv_oof(modelo, X):
    tareas = [(r, tr, va) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
    preds = Parallel(n_jobs=-1)(delayed(lambda m, a, b: clone(m).fit(X[a], y[a]).predict(X[b]))(modelo, tr, va)
                                for _, tr, va in tareas)
    oof = np.zeros((REPETICIONES, len(y)))
    for (r, _, va), p in zip(tareas, preds):
        oof[r, va] = p
    return oof

# ------------------------------------------------------------------------------
# PARTE A - modelos por rama
# ------------------------------------------------------------------------------
filas_a, AFINADOS = [], {}
for rama, X in RAMAS.items():
    AFINADOS[rama] = {}
    for nombre, (est, grilla) in CANDIDATOS.items():
        AFINADOS[rama][nombre] = afinar(est, grilla, X)
    AFINADOS[rama]["Ensamble 4 (LGBM raíz + XGB + SVR + KNN)"] = Promedio(
        [AFINADOS[rama][k] for k in ["LightGBM raíz(edad)", "XGBoost", "SVR", "KNN"]])
    for nombre, modelo in AFINADOS[rama].items():
        oof = cv_oof(modelo, X)
        p_te = clone(modelo).fit(X[idx_tr], y[idx_tr]).predict(X[idx_te])
        filas_a.append({"rama": rama, "modelo": nombre,
                        "MAE_CV": np.abs(oof - y).mean(axis=1).mean(),
                        "R2_CV": np.mean([r2_score(y, oof[r]) for r in range(REPETICIONES)]),
                        "MAE_test": np.abs(y[idx_te] - p_te).mean(), "R2_test": r2_score(y[idx_te], p_te)})
        log(f"A | {rama:<10s} | {nombre:<42s} | MAE CV {filas_a[-1]['MAE_CV']:.2f} | R² {filas_a[-1]['R2_CV']:.3f} "
            f"| test {filas_a[-1]['MAE_test']:.2f} (R² {filas_a[-1]['R2_test']:.3f})")
df_a = pd.DataFrame(filas_a).round(3)
df_a.to_csv(os.path.join(carpeta, "modelos_por_rama.csv"), index=False)

# ------------------------------------------------------------------------------
# PARTE B - simulación de más datos por rama (LightGBM por rama + fusión lineal)
# ------------------------------------------------------------------------------
M_CEL = AFINADOS["Células"]["LightGBM"]
M_CIT = AFINADOS["Citoquinas"]["LightGBM"]

def subconjunto(tr, frac, semilla):
    if frac >= 1.0:
        return tr
    sujetos = np.unique(grupos[tr])
    elegidos = np.random.default_rng(semilla).choice(sujetos, size=int(round(frac * len(sujetos))), replace=False)
    return tr[np.isin(grupos[tr], elegidos)]

def fusion_con_fracciones(tr, va, f_cel, f_cit, semilla):
    sub_c, sub_k = set(subconjunto(tr, f_cel, semilla)), set(subconjunto(tr, f_cit, semilla + 1))
    oof = np.zeros((len(tr), 2))
    for itr, iva in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(tr, groups=grupos[tr]):
        a, b = tr[itr], tr[iva]
        ac = np.array([i for i in a if i in sub_c]); ak = np.array([i for i in a if i in sub_k])
        oof[iva, 0] = clone(M_CEL).fit(XC[ac], y[ac]).predict(XC[b])
        oof[iva, 1] = clone(M_CIT).fit(XK[ak], y[ak]).predict(XK[b])
    fus = LinearRegression().fit(oof, y[tr])
    tc, tk = np.array(sorted(sub_c)), np.array(sorted(sub_k))
    pc = clone(M_CEL).fit(XC[tc], y[tc]).predict(XC[va])
    pk = clone(M_CIT).fit(XK[tk], y[tk]).predict(XK[va])
    return va, fus.predict(np.column_stack([pc, pk])), pc, pk, len(np.unique(grupos[tc])), len(np.unique(grupos[tk]))

escenarios = ([("Solo crecen las células", f, 1.0) for f in FRACCIONES]
              + [("Solo crecen las citoquinas", 1.0, f) for f in FRACCIONES[:-1]]
              + [("Crecen ambas", f, f) for f in FRACCIONES[:-1]])
filas_b = []
for esc, fc, fk in escenarios:
    tareas = [(r, i, tr, va) for r in range(REPETICIONES) for i, (tr, va) in enumerate(SPLITS[r])]
    res = Parallel(n_jobs=-1)(delayed(fusion_con_fracciones)(tr, va, fc, fk, 1000 * r + i) for r, i, tr, va in tareas)
    oof = np.zeros((3, REPETICIONES, len(y))); nc, nk = [], []
    for (r, _, _, _), (va, pf, pc, pk, a, b) in zip(tareas, res):
        oof[0, r, va], oof[1, r, va], oof[2, r, va] = pf, pc, pk
        nc.append(a); nk.append(b)
    m = lambda o: np.abs(o - y).mean(axis=1).mean()
    filas_b.append({"escenario": esc, "frac_celulas": fc, "frac_citoquinas": fk,
                    "personas_celulas": np.mean(nc), "personas_citoquinas": np.mean(nk),
                    "MAE_fusion": m(oof[0]), "MAE_rama_celulas": m(oof[1]), "MAE_rama_citoquinas": m(oof[2]),
                    "R2_fusion": np.mean([r2_score(y, oof[0, r]) for r in range(REPETICIONES)])})
    log(f"B | {esc:<28s} | células {fc:.0%} citoquinas {fk:.0%} | fusión {filas_b[-1]['MAE_fusion']:.2f} | "
        f"rama células {filas_b[-1]['MAE_rama_celulas']:.2f} | rama citoquinas {filas_b[-1]['MAE_rama_citoquinas']:.2f}")
df_b = pd.DataFrame(filas_b)
completo = df_b[(df_b.frac_celulas == 1) & (df_b.frac_citoquinas == 1)].iloc[0]
for esc in ["Solo crecen las citoquinas", "Crecen ambas"]:
    df_b = pd.concat([df_b, completo.to_frame().T.assign(escenario=esc)], ignore_index=True)
df_b = df_b.sort_values(["escenario", "frac_celulas", "frac_citoquinas"]).reset_index(drop=True)
for c in df_b.columns[1:]:
    df_b[c] = df_b[c].astype(float)

# Extrapolación
def potencia(n, a, b, c):
    return a + b * np.power(n, -c)

extrap = []
for esc, col_n, col_mae in [("Solo crecen las células", "personas_celulas", "MAE_fusion"),
                            ("Solo crecen las citoquinas", "personas_citoquinas", "MAE_fusion"),
                            ("Crecen ambas", "personas_celulas", "MAE_fusion"),
                            ("Solo crecen las células", "personas_celulas", "MAE_rama_celulas"),
                            ("Solo crecen las citoquinas", "personas_citoquinas", "MAE_rama_citoquinas")]:
    d = df_b[df_b.escenario == esc]
    n, mae = d[col_n].values, d[col_mae].values
    try:
        (a, b, c), _ = curve_fit(potencia, n, mae, p0=[mae.min() * 0.8, 50, 0.5], bounds=([0, 0, 0.05], [30, 1e5, 2]), maxfev=50000)
        n0 = n.max()
        extrap.append({"escenario": esc, "medida": col_mae, "personas_actuales": n0, "MAE_actual": potencia(n0, a, b, c),
                       "MAE_x2": potencia(2 * n0, a, b, c), "MAE_x4": potencia(4 * n0, a, b, c), "piso_estimado": a})
    except RuntimeError:
        pass
df_e = pd.DataFrame(extrap).round(2)
df_b.round(3).to_csv(os.path.join(carpeta, "simulacion_mas_datos.csv"), index=False)
df_e.to_csv(os.path.join(carpeta, "extrapolacion_mas_datos.csv"), index=False)

# Gráfico
TINTA, TINTA2, MUTED, GRID, SUP = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
COL = {"Solo crecen las células": "#2a78d6", "Solo crecen las citoquinas": "#eb6834", "Crecen ambas": "#1baf7a"}
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=120)
fig.patch.set_facecolor(SUP); ax.set_facecolor(SUP)
for esc, color in COL.items():
    d = df_b[df_b.escenario == esc].sort_values("frac_celulas" if esc != "Solo crecen las citoquinas" else "frac_citoquinas")
    x = (d["frac_celulas"] if esc != "Solo crecen las citoquinas" else d["frac_citoquinas"]) * 100
    ax.plot(x, d["MAE_fusion"], color=color, lw=2, marker="o", markersize=8, markeredgecolor=SUP, markeredgewidth=2, label=esc)
    ax.annotate(f"{d['MAE_fusion'].iloc[0]:.2f}", xy=(x.iloc[0], d["MAE_fusion"].iloc[0]), xytext=(-10, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8.5, color=TINTA2)
ax.annotate(f"Todo al 100%: {completo['MAE_fusion']:.2f}", xy=(100, completo["MAE_fusion"]), xytext=(8, 0),
            textcoords="offset points", va="center", fontsize=9, color=TINTA)
fig.suptitle("Plan B: error de la fusión según cuántos datos tiene cada rama", x=0.01, ha="left", fontsize=12, fontweight="bold", color=TINTA)
fig.text(0.01, 0.91, "LightGBM por rama + fusión lineal · CV agrupada 5×5 sobre 473 pacientes · la fusión siempre usa todos los pareados",
         ha="left", fontsize=8.5, color=TINTA2)
ax.set_xlabel("% de las personas de entrenamiento usadas por la(s) rama(s) que crece(n)", color=TINTA2)
ax.set_ylabel("MAE de la fusión (años)", color=TINTA2)
ax.set_xticks([25, 50, 75, 100]); ax.set_xlim(15, 118)
ax.grid(axis="y", color=GRID, lw=0.8); ax.grid(axis="x", visible=False)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color("#c3c2b7")
ax.tick_params(colors=MUTED, labelsize=9)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], color=c, lw=2, ls="-", marker="o", markersize=7, label=e) for e, c in COL.items()],
          frameon=False, fontsize=9, labelcolor=TINTA, loc="upper right")
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(carpeta, "simulacion_mas_datos.png"), dpi=300, facecolor=SUP)

pd.set_option("display.width", 250)
print("\n" + "=" * 110 + "\nPARTE A - MODELOS POR RAMA (solo pareados)\n" + "=" * 110)
print(df_a.to_string(index=False))
print("\n" + "=" * 110 + "\nPARTE B - SIMULACIÓN DE MÁS DATOS\n" + "=" * 110)
print(df_b.round(3).to_string(index=False))
print("\nExtrapolación (curva de potencia; orientativa):")
print(df_e.to_string(index=False))
log("Listo.")
