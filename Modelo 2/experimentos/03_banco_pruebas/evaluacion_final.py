# ==============================================================================
# EVALUACIÓN FINAL RIGUROSA DE LAS MEJORES IDEAS DEL BANCO DE PRUEBAS
# Proyecto: Immuno-XAI - UTEM
#
# Problema: el banco de pruebas eligió ideas con la misma validación cruzada
# que usó para medirlas (optimista), y el test de 119 pacientes es muy ruidoso
# (±0.7 años) para distinguir mejoras pequeñas.
#
# Solución:
#   1. Se fijan de antemano 9 "recetas" (variables + modelo + extras).
#   2. Sus hiperparámetros se afinan SOLO con el train (354 filas).
#   3. Cada receta se evalúa con validación cruzada agrupada por paciente,
#      repetida 5 veces x 5 folds, sobre los 473 pacientes (train + test).
#      La selección de variables y la corrección de sesgo ocurren DENTRO de
#      cada partición, así que no hay fuga.
#   4. Se compara cada receta contra la base con las mismas particiones
#      (diferencia pareada + intervalo de confianza bootstrap).
#
# Requiere: df_train_reales_k45.csv, df_test_puro_k45.csv,
#           df_train_reales_completo.csv, df_test_completo.csv
# ==============================================================================

import os
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
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
REPETICIONES = 5
N_ITER = 30
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_banco_pruebas")
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()

def log(msg):
    print(f"[{(time.time()-T0)/60:5.1f} min] {msg}", flush=True)

# ------------------------------------------------------------------------------
# Datos (train + test juntos para la evaluación; el afinado usa solo train)
# ------------------------------------------------------------------------------
tr45 = pd.read_csv(ruta("df_train_reales_k45.csv"))
te45 = pd.read_csv(ruta("df_test_puro_k45.csv"))
trF = pd.read_csv(ruta("df_train_reales_completo.csv"))
teF = pd.read_csv(ruta("df_test_completo.csv"))
feats45 = [c for c in tr45.columns if c not in ("subject_accession", "gender", "source_type", "age")]
trF[feats45] = tr45[feats45].values
teF[feats45] = te45[feats45].values
meta = ["subject_accession", "gender", "age", "estudio_elisa", "estudio_cytof", "edad_cytof"]
cob = trF.drop(columns=meta).notna().mean()
feats71 = [c for c in cob.index if cob[c] >= 0.8]

todo = pd.concat([trF.assign(origen="train"), teF.assign(origen="test")], ignore_index=True)
es_train = (todo["origen"] == "train").values
y = todo["age"].values.astype(float)
grupos = todo["subject_accession"].values

def razones(df):
    e = 0.01
    r = pd.DataFrame(index=df.index)
    r["r_CD4_CD8"] = df["CD4_T_cells"] / (df["CD8_T_cells"] + e)
    r["r_naiveCD8_CD8"] = df["Naive_CD8_T_cells"] / (df["CD8_T_cells"] + e)
    r["r_naiveCD4_CD4"] = df["Naive_CD4_T_cells"] / (df["CD4_T_cells"] + e)
    r["r_naive_mem_CD8"] = df["Naive_CD8_T_cells"] / (df["Central_Memory_CD8_T_cells"] + df["Effector_Memory_CD8_T_cells"] + e)
    r["r_naive_mem_CD4"] = df["Naive_CD4_T_cells"] / (df["Central_Memory_CD4_T_cells"] + df["Effector_Memory_CD4_T_cells"] + e)
    r["r_EM_CD8"] = df["Effector_Memory_CD8_T_cells"] / (df["CD8_T_cells"] + e)
    r["r_Eff_CD8"] = df["Effector_CD8_T_cells"] / (df["CD8_T_cells"] + e)
    r["r_Bnaive_Bmem"] = df["Naive_B_cells"] / (df["Memory_B_cells"] + e)
    r["r_Treg_CD4"] = df["Tregs"] / (df["CD4_T_cells"] + e)
    r["r_mono_lymph"] = df["Monocytes"] / (df["Lymphocytes"] + e)
    r["r_naiveCD8_T"] = df["Naive_CD8_T_cells"] / (df["T_cells"] + e)
    return r

def matriz(cols, con_razones=False):
    X = todo[cols].astype(float).copy()
    if con_razones:
        X = pd.concat([X, razones(todo)], axis=1)
    X["gender_male"] = (todo["gender"] == "Male").astype(float)
    return X.values

CONJ = {"45": matriz(feats45), "45+razones": matriz(feats45, True),
        "71": matriz(feats71), "71+razones": matriz(feats71, True)}

# ------------------------------------------------------------------------------
# Piezas
# ------------------------------------------------------------------------------
def log1p_positivo(Z):
    return np.log1p(np.clip(Z, 0, None))

def lgbm(**kw):
    return lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **kw)

def selector(k=35):
    return SelectFromModel(lgbm(n_estimators=300, learning_rate=0.03, num_leaves=15, colsample_bytree=0.7),
                           max_features=k, threshold=-np.inf)

def lineal(modelo):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("log", FunctionTransformer(log1p_positivo)),
                     ("esc", StandardScaler()), ("m", modelo)])

class Promedio(BaseEstimator, RegressorMixin):
    """Promedia las predicciones de varios modelos."""
    def __init__(self, modelos=None):
        self.modelos = modelos
    def fit(self, X, y):
        self.ajustados_ = [clone(m).fit(X, y) for m in self.modelos]
        return self
    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ajustados_], axis=0)

class CorreccionSesgo(BaseEstimator, RegressorMixin):
    """edad = a + b * predicción; a y b con predicciones fuera de muestra dentro del train recibido."""
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

def afinar(est, grilla, X):
    """Afinado SOLO con el train original (354 filas), CV agrupada por paciente."""
    s = RandomizedSearchCV(est, grilla, n_iter=N_ITER, scoring="neg_mean_absolute_error",
                           cv=GroupKFold(n_splits=5, shuffle=True, random_state=100),
                           random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    s.fit(X[es_train], y[es_train], groups=grupos[es_train])
    return clone(est).set_params(**s.best_params_)

def con_prefijo(grilla, p):
    return {f"{p}{k}": v for k, v in grilla.items()}

# ------------------------------------------------------------------------------
# Afinado (solo train) y definición de recetas
# ------------------------------------------------------------------------------
log("Afinando hiperparámetros con el train (354 filas)...")
lgb45 = afinar(lgbm(), GRILLA_LGB, CONJ["45"]);             log("  LGBM 45 listo")
lgb71 = afinar(lgbm(), GRILLA_LGB, CONJ["71"]);             log("  LGBM 71 listo")
lgb45r = afinar(lgbm(), GRILLA_LGB, CONJ["45+razones"]);    log("  LGBM 45+razones listo")
lgb71r = afinar(lgbm(), GRILLA_LGB, CONJ["71+razones"]);    log("  LGBM 71+razones listo")

# Modelos sobre las 35 variables elegidas dentro de cada fold (de 71 + razones)
X71r = CONJ["71+razones"]
lgb_sqrt_sel = afinar(Pipeline([("sel", selector()), ("m", TransformedTargetRegressor(lgbm(), func=np.sqrt, inverse_func=np.square))]),
                      con_prefijo(GRILLA_LGB, "m__regressor__"), X71r); log("  LGBM raíz(edad) top-35 listo")
xgb_sel = afinar(Pipeline([("sel", selector()), ("m", xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1))]),
                 con_prefijo(GRILLA_XGB, "m__"), X71r); log("  XGBoost top-35 listo")
svr_sel = afinar(Pipeline([("sel", selector()), ("m", lineal(SVR()))]), con_prefijo(GRILLA_SVR, "m__"), X71r); log("  SVR top-35 listo")
knn_sel = afinar(Pipeline([("sel", selector()), ("m", lineal(KNeighborsRegressor()))]), con_prefijo(GRILLA_KNN, "m__"), X71r); log("  KNN top-35 listo")

# Modelos sobre las 45 originales para un ensamble sin cambiar variables
xgb45 = afinar(xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1), GRILLA_XGB, CONJ["45"]); log("  XGBoost 45 listo")
svr45 = afinar(lineal(SVR()), GRILLA_SVR, CONJ["45"]); log("  SVR 45 listo")
knn45 = afinar(lineal(KNeighborsRegressor()), GRILLA_KNN, CONJ["45"]); log("  KNN 45 listo")

ens_sel = Promedio([lgb_sqrt_sel, xgb_sel, svr_sel, knn_sel])
ens45 = Promedio([lgb45, xgb45, svr45, knn45])

RECETAS = [
    ("R0 Base: LightGBM | 45 originales", lgb45, "45"),
    ("R1 LightGBM | 71 variables", lgb71, "71"),
    ("R2 LightGBM | 45 + razones inmunes", lgb45r, "45+razones"),
    ("R3 LightGBM | 71 + razones", lgb71r, "71+razones"),
    ("R4 LightGBM + corrección de sesgo | 45", CorreccionSesgo(lgb45), "45"),
    ("R5 Ensamble 4 modelos | 45", ens45, "45"),
    ("R6 Ensamble 4 modelos + corrección de sesgo | 45", CorreccionSesgo(ens45), "45"),
    ("R7 Ensamble 4 modelos | top-35 de 71+razones", ens_sel, "71+razones"),
    ("R8 Ensamble 4 modelos + corrección de sesgo | top-35 de 71+razones", CorreccionSesgo(ens_sel), "71+razones"),
]

# ------------------------------------------------------------------------------
# Evaluación: CV agrupada 5x5 sobre los 473 pacientes
# ------------------------------------------------------------------------------
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]

def ajustar_predecir(modelo, X, tr, va):
    m = clone(modelo)
    if isinstance(m, CorreccionSesgo):
        m.fit(X[tr], y[tr], groups=grupos[tr])
    else:
        m.fit(X[tr], y[tr])
    return m.predict(X[va])

filas, OOF = [], {}
idx_tr, idx_te = np.where(es_train)[0], np.where(~es_train)[0]
for nombre, modelo, conj in RECETAS:
    X = CONJ[conj]
    tareas = [(r, tr, va) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
    preds = Parallel(n_jobs=-1)(delayed(ajustar_predecir)(modelo, X, tr, va) for _, tr, va in tareas)
    oof = np.zeros((REPETICIONES, len(y)))
    for (r, _, va), p in zip(tareas, preds):
        oof[r, va] = p
    OOF[nombre] = oof
    maes = np.abs(oof - y).mean(axis=1)
    p_test = ajustar_predecir(modelo, X, idx_tr, idx_te)   # entrenado en train, evaluado en tu test original
    filas.append({"receta": nombre, "MAE_CV_473": maes.mean(), "MAE_CV_473_sd": maes.std(),
                  "acierto5_CV_473": (np.abs(oof - y) <= 5).mean() * 100,
                  "MAE_test_119": np.abs(y[idx_te] - p_test).mean()})
    log(f"{nombre:<70s} | MAE CV(473) {maes.mean():.2f} ± {maes.std():.2f} | test(119) {filas[-1]['MAE_test_119']:.2f}")

# Comparación pareada contra la base (error medio por paciente sobre las 5 repeticiones)
err_base = np.abs(OOF[RECETAS[0][0]] - y).mean(axis=0)
rng = np.random.default_rng(0)
idx_boot = rng.integers(0, len(y), size=(5000, len(y)))
for f in filas:
    d = np.abs(OOF[f["receta"]] - y).mean(axis=0) - err_base
    db = d[idx_boot].mean(axis=1)
    f["dif_vs_base"] = d.mean()
    f["IC95_bajo"], f["IC95_alto"] = np.percentile(db, [2.5, 97.5])
    f["P_mejora"] = (db < 0).mean() * 100
    f["significativo"] = "Sí" if f["IC95_alto"] < 0 else ("Peor" if f["IC95_bajo"] > 0 else "No")

df = pd.DataFrame(filas).round(3)
df.to_csv(os.path.join(carpeta, "evaluacion_final.csv"), index=False)
pd.set_option("display.width", 250)
print("\n" + "=" * 120)
print("EVALUACIÓN FINAL: CV agrupada 5x5 sobre 473 pacientes (dif < 0 = mejor que la base)")
print("=" * 120)
print(df[["receta", "MAE_CV_473", "MAE_CV_473_sd", "dif_vs_base", "IC95_bajo", "IC95_alto", "P_mejora",
          "significativo", "acierto5_CV_473", "MAE_test_119"]].to_string(index=False))
log("Listo.")
