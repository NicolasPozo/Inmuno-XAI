# ==============================================================================
# PLAN B: DOS MODELOS SEPARADOS (CÉLULAS / CITOQUINAS) + FUSIÓN
# Proyecto: Immuno-XAI - UTEM
#
#   Rama células     : entrena con TODOS los datos de células disponibles
#   Rama citoquinas  : entrena con TODOS los datos de citoquinas disponibles
#   Fusión           : edad = a + w1 * pred_células + w2 * pred_citoquinas,
#                      aprendida con los pacientes que tienen ambas modalidades
#                      (usando predicciones fuera de muestra, sin fuga). Al tener
#                      intercepto y pesos libres, también corrige el sesgo de
#                      regresión a la media.
#
# Datos extra: todos los CSV de fuentes_celulas/ y fuentes_citoquinas/
# (generados por construir_fuentes_unimodales.py; puedes agregar más).
#
# Evaluación: CV agrupada por paciente 5x5 sobre los 473 pacientes pareados.
# Las personas de validación se excluyen también de las fuentes extra.
# Se compara contra la receta R8 (un solo modelo con todo) con los mismos datos.
# ==============================================================================

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RANDOM_STATE, REPETICIONES, N_ITER = 42, 5, 30
MIN_CELULAS_FLUJO = 6     # visitas de flujo con al menos 6 tipos celulares medidos
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_dos_ramas")
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

# ------------------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------------------
par = pd.read_csv(ruta("pareados.csv"))
y = par["age"].values.astype(float)
grupos = par["subject_accession"].values
es_train = (par["conjunto"] == "train").values

def leer_fuentes(carpeta_fuentes):
    dfs = []
    for p in sorted(glob.glob(ruta(carpeta_fuentes, "*.csv"))):
        d = pd.read_csv(p)
        d["archivo"] = os.path.basename(p)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

ext_cit = leer_fuentes("fuentes_citoquinas")
ext_cel = leer_fuentes("fuentes_celulas")

cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells",
            "Central_Memory_CD4_T_cells", "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells",
            "Effector_Memory_CD4_T_cells", "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes",
            "Memory_B_cells", "Monocytes", "Naive_B_cells", "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells",
            "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto"]]
cob = par.loc[es_train, todas_cit].notna().mean()
cols_cit = [c for c in todas_cit if cob[c] >= 0.8]

for d, cols in [(ext_cit, cols_cit), (ext_cel, cols_cel)]:
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
n_cel_medidas = ext_cel[cols_cel].notna().sum(axis=1)
es_flujo = ext_cel["archivo"].str.contains("flujo")
ext_cel = ext_cel[~es_flujo | (n_cel_medidas >= MIN_CELULAS_FLUJO)].reset_index(drop=True)

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

sexo = lambda df: (df["gender"] == "Male").astype(float).values.reshape(-1, 1)
def X_cel(df):
    return np.hstack([df[cols_cel].astype(float).values, razones(df).values, sexo(df)])
def X_cit(df):
    return np.hstack([df[cols_cit].astype(float).values, sexo(df)])
def X_todo(df):
    return np.hstack([df[cols_cel].astype(float).values, razones(df).values, df[cols_cit].astype(float).values, sexo(df)])

XC, XK, XT = X_cel(par), X_cit(par), X_todo(par)
EC, EK = X_cel(ext_cel), X_cit(ext_cit)
yEC, yEK = ext_cel["age"].values.astype(float), ext_cit["age"].values.astype(float)
gEC, gEK = ext_cel["subject_accession"].values, ext_cit["subject_accession"].values
log(f"Pareados: {len(par)} | rama células: {XC.shape[1]} variables, extra {len(EC)} filas "
    f"({ext_cel['archivo'].value_counts().to_dict()}) | rama citoquinas: {XK.shape[1]} variables, extra {len(EK)} filas")

# ------------------------------------------------------------------------------
# Modelos
# ------------------------------------------------------------------------------
def log1p_positivo(Z):
    return np.log1p(np.clip(Z, 0, None))

def lgbm(**kw):
    return lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **kw)

def lineal(m):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(log1p_positivo)),
                     ("esc", StandardScaler()), ("m", m)])

class Promedio(BaseEstimator, RegressorMixin):
    def __init__(self, modelos=None):
        self.modelos = modelos
    def fit(self, X, y):
        self.ajustados_ = [clone(m).fit(X, y) for m in self.modelos]
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ajustados_], axis=0)

class CorreccionSesgo(BaseEstimator, RegressorMixin):
    def __init__(self, base=None):
        self.base = base
    def fit(self, X, y, groups=None):
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(X, y, groups):
            oof[va] = clone(self.base).fit(X[tr], y[tr]).predict(X[va])
        self.b_, self.a_ = np.polyfit(oof, y, 1)
        self.modelo_ = clone(self.base).fit(X, y)
        return self
    def predict(self, X):
        return self.a_ + self.b_ * self.modelo_.predict(X)

GRILLA_LGB = {"n_estimators": [200, 400, 600, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
              "max_depth": [3, 4, 5, 6, -1], "num_leaves": [7, 15, 31], "min_child_samples": [5, 10, 20, 30],
              "subsample": [0.7, 0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
              "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.0, 1.0, 5.0]}
GRILLA_XGB = {"n_estimators": [200, 400, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
              "max_depth": [2, 3, 4, 5], "min_child_weight": [1, 3, 5, 10], "subsample": [0.7, 0.8, 1.0],
              "colsample_bytree": [0.3, 0.5, 0.7, 0.9], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0.5, 1, 5]}
GRILLA_SVR = {"m__C": [5, 10, 20, 50, 100, 200], "m__gamma": ["scale", 0.003, 0.01, 0.03], "m__epsilon": [0.1, 0.5, 1, 2, 5]}
GRILLA_KNN = {"m__n_neighbors": [5, 10, 15, 20, 30], "m__weights": ["uniform", "distance"], "m__p": [1, 2]}
pref = lambda g, p: {f"{p}{k}": v for k, v in g.items()}

def afinar(est, grilla, X):
    """Afinado con el train pareado (354 filas, sin fuentes extra) -> conservador para los extras."""
    s = RandomizedSearchCV(est, grilla, n_iter=N_ITER, scoring="neg_mean_absolute_error",
                           cv=GroupKFold(n_splits=5, shuffle=True, random_state=100),
                           random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    s.fit(X[es_train], y[es_train], groups=grupos[es_train])
    return clone(est).set_params(**s.best_params_)

def ensamble_afinado(X):
    return Promedio([
        afinar(TransformedTargetRegressor(lgbm(), func=np.sqrt, inverse_func=np.square), pref(GRILLA_LGB, "regressor__"), X),
        afinar(xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1), GRILLA_XGB, X),
        afinar(lineal(SVR()), GRILLA_SVR, X),
        afinar(lineal(KNeighborsRegressor()), GRILLA_KNN, X),
    ])

log("Afinando la rama de células...")
MODELO_CEL = ensamble_afinado(XC)
log("Afinando la rama de citoquinas...")
MODELO_CIT = ensamble_afinado(XK)
log("Afinando la receta R8 (un solo modelo con todo, referencia)...")
sel = lambda: SelectFromModel(lgbm(n_estimators=300, learning_rate=0.03, num_leaves=15, colsample_bytree=0.7),
                              max_features=35, threshold=-np.inf)
R8 = CorreccionSesgo(Promedio([
    afinar(Pipeline([("sel", sel()), ("m", TransformedTargetRegressor(lgbm(), func=np.sqrt, inverse_func=np.square))]),
           pref(GRILLA_LGB, "m__regressor__"), XT),
    afinar(Pipeline([("sel", sel()), ("m", xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1))]), pref(GRILLA_XGB, "m__"), XT),
    afinar(Pipeline([("sel", sel()), ("m", lineal(SVR()))]), pref(GRILLA_SVR, "m__"), XT),
    afinar(Pipeline([("sel", sel()), ("m", lineal(KNeighborsRegressor()))]), pref(GRILLA_KNN, "m__"), XT),
]))
log("Afinado listo.")

# ------------------------------------------------------------------------------
# Dos ramas + fusión para una partición (sin fuga)
# ------------------------------------------------------------------------------
def entrenar_rama(modelo, Xp, yp, Xe, ye, ge, excluir):
    """Entrena con pareados + extras cuyos sujetos no están en 'excluir'."""
    if Xe is not None and len(Xe):
        ok = ~np.isin(ge, list(excluir))
        Xp, yp = np.vstack([Xp, Xe[ok]]), np.concatenate([yp, ye[ok]])
    return clone(modelo).fit(Xp, yp)

def dos_ramas(tr, va, usar_cel, usar_cit):
    Ec = (EC, yEC, gEC) if usar_cel else (None, None, None)
    Ek = (EK, yEK, gEK) if usar_cit else (None, None, None)
    excluir_va = set(grupos[va])
    # Predicciones fuera de muestra en el train pareado -> para aprender la fusión
    oof = np.zeros((len(tr), 2))
    for itr, iva in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(tr, groups=grupos[tr]):
        a, b = tr[itr], tr[iva]
        excl = excluir_va | set(grupos[b])
        oof[iva, 0] = entrenar_rama(MODELO_CEL, XC[a], y[a], *Ec, excl).predict(XC[b])
        oof[iva, 1] = entrenar_rama(MODELO_CIT, XK[a], y[a], *Ek, excl).predict(XK[b])
    fusion = LinearRegression().fit(oof, y[tr])
    pc = entrenar_rama(MODELO_CEL, XC[tr], y[tr], *Ec, excluir_va).predict(XC[va])
    pk = entrenar_rama(MODELO_CIT, XK[tr], y[tr], *Ek, excluir_va).predict(XK[va])
    pf = fusion.predict(np.column_stack([pc, pk]))
    return pf, pc, pk, np.r_[fusion.intercept_, fusion.coef_]

def r8(tr, va):
    m = clone(R8).fit(XT[tr], y[tr], groups=grupos[tr])
    p = m.predict(XT[va])
    return p, p, p, np.array([np.nan, np.nan, np.nan])

VARIANTES = [
    ("R8: un solo modelo con todo (referencia)", lambda tr, va: r8(tr, va)),
    ("Dos ramas, solo pareados", lambda tr, va: dos_ramas(tr, va, False, False)),
    ("Dos ramas + citoquinas extra", lambda tr, va: dos_ramas(tr, va, False, True)),
    ("Dos ramas + citoquinas extra + células extra (CyTOF y flujo)", lambda tr, va: dos_ramas(tr, va, True, True)),
]

# ------------------------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------------------------
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]
idx_tr, idx_te = np.where(es_train)[0], np.where(~es_train)[0]
filas, OOF = [], {}
for nombre, fn in VARIANTES:
    tareas = [(r, tr, va) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
    res = Parallel(n_jobs=-1)(delayed(fn)(tr, va) for _, tr, va in tareas)
    oof = np.zeros((3, REPETICIONES, len(y)))
    pesos = []
    for (r, _, va), (pf, pc, pk, w) in zip(tareas, res):
        oof[0, r, va], oof[1, r, va], oof[2, r, va] = pf, pc, pk
        pesos.append(w)
    OOF[nombre] = oof[0]
    pf_te, pc_te, pk_te, w_te = fn(idx_tr, idx_te)
    mae = lambda o: np.abs(o - y).mean(axis=1)
    r2 = lambda o: np.array([r2_score(y, o[r]) for r in range(REPETICIONES)])
    y_te = y[idx_te]
    w = np.nanmean(pesos, axis=0)
    fila = {"variante": nombre, "MAE_CV_473": mae(oof[0]).mean(), "MAE_CV_473_sd": mae(oof[0]).std(),
            "R2_CV": r2(oof[0]).mean(),
            "MAE_rama_celulas": mae(oof[1]).mean(), "R2_rama_celulas": r2(oof[1]).mean(),
            "MAE_rama_citoquinas": mae(oof[2]).mean(), "R2_rama_citoquinas": r2(oof[2]).mean(),
            "acierto5_CV": (np.abs(oof[0] - y) <= 5).mean() * 100,
            "MAE_test_119": np.abs(y_te - pf_te).mean(), "R2_test_119": r2_score(y_te, pf_te),
            "MAE_test_celulas": np.abs(y_te - pc_te).mean(), "R2_test_celulas": r2_score(y_te, pc_te),
            "MAE_test_citoquinas": np.abs(y_te - pk_te).mean(), "R2_test_citoquinas": r2_score(y_te, pk_te),
            "fusion_a": w[0], "fusion_w_celulas": w[1], "fusion_w_citoquinas": w[2]}
    np.savez(os.path.join(carpeta, f"predicciones_{len(filas)}.npz"), variante=nombre, y=y, oof_fusion=oof[0],
             oof_celulas=oof[1], oof_citoquinas=oof[2], y_test=y_te, test_fusion=pf_te,
             test_celulas=pc_te, test_citoquinas=pk_te)
    filas.append(fila)
    extra = "" if nombre.startswith("R8") else (f" | rama células {fila['MAE_rama_celulas']:.2f} | rama citoquinas "
                                                 f"{fila['MAE_rama_citoquinas']:.2f} | pesos {w[1]:.2f}/{w[2]:.2f}")
    log(f"{nombre:<62s} | MAE CV {fila['MAE_CV_473']:.2f} ± {fila['MAE_CV_473_sd']:.2f} | test {fila['MAE_test_119']:.2f}{extra}")

# Comparaciones pareadas
ref = list(OOF)[0]
rng = np.random.default_rng(0)
idx_b = rng.integers(0, len(y), size=(5000, len(y)))
for f in filas:
    d = np.abs(OOF[f["variante"]] - y).mean(axis=0) - np.abs(OOF[ref] - y).mean(axis=0)
    db = d[idx_b].mean(axis=1)
    f["dif_vs_R8"], f["IC95_bajo"], f["IC95_alto"] = d.mean(), *np.percentile(db, [2.5, 97.5])
    f["P_mejora_vs_R8"] = (db < 0).mean() * 100

df = pd.DataFrame(filas).round(3)
df.to_csv(os.path.join(carpeta, "evaluacion_dos_ramas.csv"), index=False)
pd.set_option("display.width", 260)
print("\n" + "=" * 130)
print("PLAN B - DOS RAMAS + FUSIÓN (CV agrupada 5x5 sobre 473 pacientes; dif < 0 = mejor que R8)")
print("=" * 130)
print(df[["variante", "MAE_CV_473", "MAE_CV_473_sd", "dif_vs_R8", "IC95_bajo", "IC95_alto", "P_mejora_vs_R8",
          "acierto5_CV", "fusion_w_celulas", "fusion_w_citoquinas"]].to_string(index=False))
print("\nMAE (años) y R² de cada rama y de la fusión")
print(df[["variante", "MAE_rama_celulas", "R2_rama_celulas", "MAE_rama_citoquinas", "R2_rama_citoquinas",
          "MAE_CV_473", "R2_CV", "MAE_test_celulas", "R2_test_celulas", "MAE_test_citoquinas", "R2_test_citoquinas",
          "MAE_test_119", "R2_test_119"]].to_string(index=False))
log("Listo.")
