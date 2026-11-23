# ==============================================================================
# ¿AYUDA SOUND LIFE (ALLEN INSTITUTE) A PREDECIR LA EDAD EN TUS PACIENTES?
# Proyecto: Immuno-XAI - UTEM
#
# Misma evaluación rigurosa que evaluacion_final.py (CV agrupada 5x5 sobre los
# 473 pacientes del 10K), comparando cada receta:
#   - sin datos extra
#   - + Sound Life, 1 visita basal por persona (96 filas)
#   - + Sound Life, todas sus visitas basales (348 filas, agrupadas por persona)
# Sound Life SOLO se agrega al entrenamiento de cada partición; nunca se evalúa
# sobre él. Los hiperparámetros se afinan con tu train (sin Sound Life) y son los
# mismos en ambas versiones, así la única diferencia son los datos extra.
#
# Requiere: datos_soundlife/soundlife_armonizado.csv (armonizar_soundlife.py)
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
RANDOM_STATE, REPETICIONES, N_ITER = 42, 5, 30
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_banco_pruebas")
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

# ------------------------------------------------------------------------------
# Datos
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

sl = pd.read_csv(ruta("datos_soundlife", "soundlife_armonizado.csv"))
for c in feats71 + feats45:
    if c not in sl.columns:
        sl[c] = np.nan
orden = {"Flu Year 1 Day 0": 0, "Immune Variation Day 0": 1, "Flu Year 2 Day 0": 2}
sl["_orden"] = sl["visita"].map(orden).fillna(9)
EXTRA = {
    "1 visita/persona": sl.sort_values(["subject_accession", "_orden"]).drop_duplicates("subject_accession").reset_index(drop=True),
    "todas las visitas basales": sl.reset_index(drop=True),
}

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

def matriz(df, cols, con_razones=False):
    X = df[cols].astype(float).copy()
    if con_razones:
        X = pd.concat([X, razones(df)], axis=1)
    X["gender_male"] = (df["gender"] == "Male").astype(float)
    return X.values

CONJ = {"45": lambda d: matriz(d, feats45), "71+razones": lambda d: matriz(d, feats71, True)}

# ------------------------------------------------------------------------------
# Piezas (idénticas a evaluacion_final.py)
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
    s = RandomizedSearchCV(est, grilla, n_iter=N_ITER, scoring="neg_mean_absolute_error",
                           cv=GroupKFold(n_splits=5, shuffle=True, random_state=100),
                           random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    s.fit(X[es_train], y[es_train], groups=grupos[es_train])
    return clone(est).set_params(**s.best_params_)

log("Afinando hiperparámetros con tu train (sin Sound Life)...")
X45, X71r = CONJ["45"](todo), CONJ["71+razones"](todo)
lgb45 = afinar(lgbm(), GRILLA_LGB, X45)
xgb45 = afinar(xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1), GRILLA_XGB, X45)
svr45 = afinar(lineal(SVR()), GRILLA_SVR, X45)
knn45 = afinar(lineal(KNeighborsRegressor()), GRILLA_KNN, X45)
lgb_sqrt_sel = afinar(Pipeline([("sel", selector()), ("m", TransformedTargetRegressor(lgbm(), func=np.sqrt, inverse_func=np.square))]),
                      pref(GRILLA_LGB, "m__regressor__"), X71r)
xgb_sel = afinar(Pipeline([("sel", selector()), ("m", xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1))]), pref(GRILLA_XGB, "m__"), X71r)
svr_sel = afinar(Pipeline([("sel", selector()), ("m", lineal(SVR()))]), pref(GRILLA_SVR, "m__"), X71r)
knn_sel = afinar(Pipeline([("sel", selector()), ("m", lineal(KNeighborsRegressor()))]), pref(GRILLA_KNN, "m__"), X71r)
log("Afinado listo.")

RECETAS = [
    ("R0 LightGBM | 45", lgb45, "45", ["1 visita/persona", "todas las visitas basales"]),
    ("R6 Ensamble + corrección de sesgo | 45", CorreccionSesgo(Promedio([lgb45, xgb45, svr45, knn45])), "45", ["1 visita/persona"]),
    ("R8 Ensamble + corrección de sesgo | top-35 de 71+razones",
     CorreccionSesgo(Promedio([lgb_sqrt_sel, xgb_sel, svr_sel, knn_sel])), "71+razones", ["1 visita/persona", "todas las visitas basales"]),
]

# ------------------------------------------------------------------------------
# Evaluación
# ------------------------------------------------------------------------------
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]

def ajustar_predecir(modelo, X, tr, va, Xe, ye, ge):
    Xtr, ytr, gtr = X[tr], y[tr], grupos[tr]
    if Xe is not None:
        Xtr, ytr, gtr = np.vstack([Xtr, Xe]), np.concatenate([ytr, ye]), np.concatenate([gtr, ge])
    m = clone(modelo)
    m.fit(Xtr, ytr, groups=gtr) if isinstance(m, CorreccionSesgo) else m.fit(Xtr, ytr)
    return m.predict(X[va])

filas, OOF = [], {}
idx_tr, idx_te = np.where(es_train)[0], np.where(~es_train)[0]
for nombre, modelo, conj, extras in RECETAS:
    X = CONJ[conj](todo)
    for extra in [None] + extras:
        if extra is None:
            Xe = ye = ge = None
        else:
            d = EXTRA[extra]
            Xe, ye, ge = CONJ[conj](d), d["age"].values.astype(float), d["subject_accession"].values
        tareas = [(r, tr, va) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
        preds = Parallel(n_jobs=-1)(delayed(ajustar_predecir)(modelo, X, tr, va, Xe, ye, ge) for _, tr, va in tareas)
        oof = np.zeros((REPETICIONES, len(y)))
        for (r, _, va), p in zip(tareas, preds):
            oof[r, va] = p
        etiqueta = f"{nombre} | {'sin datos extra' if extra is None else '+ Sound Life (' + extra + ')'}"
        OOF[etiqueta] = oof
        maes = np.abs(oof - y).mean(axis=1)
        p_te = ajustar_predecir(modelo, X, idx_tr, idx_te, Xe, ye, ge)
        fila = {"receta": nombre, "datos": "sin extra" if extra is None else f"+ Sound Life ({extra})",
                "MAE_CV_473": maes.mean(), "MAE_CV_473_sd": maes.std(), "MAE_test_119": np.abs(y[idx_te] - p_te).mean()}
        if extra is not None:
            base_err = np.abs(OOF[f"{nombre} | sin datos extra"] - y).mean(axis=0)
            d_i = np.abs(oof - y).mean(axis=0) - base_err
            db = d_i[np.random.default_rng(0).integers(0, len(y), size=(5000, len(y)))].mean(axis=1)
            fila.update({"dif_vs_sin_extra": d_i.mean(), "IC95_bajo": np.percentile(db, 2.5),
                         "IC95_alto": np.percentile(db, 97.5), "P_mejora": (db < 0).mean() * 100})
        filas.append(fila)
        log(f"{etiqueta:<95s} | MAE CV {maes.mean():.2f} | test {fila['MAE_test_119']:.2f}"
            + (f" | dif {fila['dif_vs_sin_extra']:+.2f} [{fila['IC95_bajo']:+.2f}, {fila['IC95_alto']:+.2f}]" if extra else ""))

df = pd.DataFrame(filas).round(3)
df.to_csv(os.path.join(carpeta, "evaluacion_soundlife.csv"), index=False)
pd.set_option("display.width", 250)
print("\n" + "=" * 120)
print("¿AYUDA SOUND LIFE? (dif < 0 = mejora; CV agrupada 5x5 sobre los 473 pacientes del 10K)")
print("=" * 120)
print(df.to_string(index=False))
log("Listo.")
