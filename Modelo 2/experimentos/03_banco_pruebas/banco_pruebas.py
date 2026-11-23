# ==============================================================================
# BANCO DE PRUEBAS: IDEAS PARA BAJAR EL ERROR
# Proyecto: Immuno-XAI - UTEM
#
# Regla de oro: las ideas se ELIGEN por validación cruzada (CV) agrupada por
# paciente, repetida 3 veces con las mismas particiones para todas. El test
# (119 pacientes) se reporta, pero no se usa para decidir.
#
# Fases:
#   1. Conjuntos de variables (45 originales, 70 completas, + razones inmunes)
#   2. Selección de variables dentro de la CV (sin fuga)
#   3. Función de pérdida / transformación de la edad
#   4. Otros modelos (cada uno afinado con CV agrupada)
#   5. Ensambles (promedios de modelos)
#   6. Corrección del sesgo de regresión a la media
#
# Requiere: df_train_reales_k45.csv, df_test_puro_k45.csv,
#           df_train_reales_completo.csv, df_test_completo.csv
# ==============================================================================

import os
import itertools
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
REPETICIONES = 3
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
# Datos
# ------------------------------------------------------------------------------
tr45 = pd.read_csv(ruta("df_train_reales_k45.csv"))
te45 = pd.read_csv(ruta("df_test_puro_k45.csv"))
trF = pd.read_csv(ruta("df_train_reales_completo.csv"))
teF = pd.read_csv(ruta("df_test_completo.csv"))
assert (trF["subject_accession"].values == tr45["subject_accession"].values).all()

feats45 = [c for c in tr45.columns if c not in ("subject_accession", "gender", "source_type", "age")]
# Para las 45 originales se usan exactamente tus valores (incluida tu imputación)
trF[feats45] = tr45[feats45].values
teF[feats45] = te45[feats45].values

meta = ["subject_accession", "gender", "age", "estudio_elisa", "estudio_cytof", "edad_cytof"]
cobertura = trF.drop(columns=meta).notna().mean()
feats70 = [c for c in cobertura.index if cobertura[c] >= 0.8]

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
    return X

y = trF["age"].values.astype(float)
y_te = teF["age"].values.astype(float)
grupos = trF["subject_accession"].values

# Particiones fijas (iguales para todos los candidatos)
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(trF, y, grupos)) for r in range(REPETICIONES)]
CV_AFINAR = GroupKFold(n_splits=5, shuffle=True, random_state=100)

# ------------------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------------------
RESULTADOS, OOF, PRED_TE = [], {}, {}

class CorreccionSesgo(BaseEstimator, RegressorMixin):
    """Corrige la regresión a la media: edad = a + b * predicción, con a y b
    estimados con predicciones fuera de muestra DENTRO del train (CV agrupada)."""
    def __init__(self, base=None):
        self.base = base
    def fit(self, X, y, groups=None):
        X, y = np.asarray(X), np.asarray(y)
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(X, y, groups):
            oof[va] = clone(self.base).fit(X[tr], y[tr]).predict(X[va])
        self.b_, self.a_ = np.polyfit(oof, y, 1)
        self.modelo_ = clone(self.base).fit(X, y)
        return self
    def predict(self, X):
        return self.a_ + self.b_ * self.modelo_.predict(np.asarray(X))

def _ajustar(modelo, X, yv, g, Xp):
    m = clone(modelo)
    if isinstance(m, CorreccionSesgo):
        m.fit(X, yv, groups=g)
    else:
        m.fit(X, yv)
    return m.predict(Xp)

def evaluar(fase, nombre, modelo, Xtr, Xte, guardar=True):
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    tareas = [(r, tr, va) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
    preds = Parallel(n_jobs=-1)(delayed(_ajustar)(modelo, Xtr[tr], y[tr], grupos[tr], Xtr[va]) for _, tr, va in tareas)
    oof = np.zeros((REPETICIONES, len(y)))
    for (r, _, va), p in zip(tareas, preds):
        oof[r, va] = p
    p_te = _ajustar(modelo, Xtr, y, grupos, Xte)
    maes = np.abs(oof - y).mean(axis=1)
    fila = {"fase": fase, "candidato": nombre, "n_variables": Xtr.shape[1],
            "MAE_CV": round(maes.mean(), 3), "MAE_CV_sd": round(maes.std(), 3),
            "MAE_test": round(np.abs(y_te - p_te).mean(), 3),
            "acierto5_test": round((np.abs(y_te - p_te) <= 5).mean() * 100, 1)}
    if guardar:
        RESULTADOS.append(fila); OOF[nombre] = oof; PRED_TE[nombre] = p_te
        pd.DataFrame(RESULTADOS).to_csv(os.path.join(carpeta, "resultados_banco.csv"), index=False)
    log(f"{fase:<11s} | {nombre:<42s} | MAE CV {fila['MAE_CV']:.2f} ± {fila['MAE_CV_sd']:.2f} | test {fila['MAE_test']:.2f}")
    return fila

def afinar(modelo, grilla, X, n_iter=40):
    s = RandomizedSearchCV(modelo, grilla, n_iter=n_iter, scoring="neg_mean_absolute_error", cv=CV_AFINAR,
                           random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    s.fit(np.asarray(X, float), y, groups=grupos)
    return s.best_params_

def lgbm(**kw):
    return lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1, **kw)

GRILLA_LGB = {"n_estimators": [200, 400, 600, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
              "max_depth": [3, 4, 5, 6, -1], "num_leaves": [7, 15, 31], "min_child_samples": [5, 10, 20, 30],
              "subsample": [0.7, 0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
              "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [0.0, 1.0, 5.0]}

def log1p_positivo(Z):
    return np.log1p(np.clip(Z, 0, None))

def prepro_lineal():
    return [("imp", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(log1p_positivo)),
            ("esc", StandardScaler())]

# ------------------------------------------------------------------------------
# FASE 1 - Conjuntos de variables (LightGBM afinado en las 45)
# ------------------------------------------------------------------------------
log(f"Variables: 45 originales | {len(feats70)} con >=80% de datos | train {len(y)} filas, test {len(y_te)}")
X45, X45te = matriz(trF, feats45), matriz(teF, feats45)
p_lgb45 = afinar(lgbm(), GRILLA_LGB, X45, n_iter=60)
log(f"LightGBM afinado (45): {p_lgb45}")
evaluar("1-variables", "LGBM | 45 originales (base)", lgbm(**p_lgb45), X45, X45te)

conjuntos = {
    "45 + razones": (matriz(trF, feats45, True), matriz(teF, feats45, True)),
    f"{len(feats70)} completas": (matriz(trF, feats70), matriz(teF, feats70)),
    f"{len(feats70)} completas + razones": (matriz(trF, feats70, True), matriz(teF, feats70, True)),
}
for nombre, (Xa, Xb) in conjuntos.items():
    evaluar("1-variables", f"LGBM | {nombre}", lgbm(**p_lgb45), Xa, Xb)
conjuntos["45 originales"] = (X45, X45te)

# Re-afinar LightGBM en el mejor conjunto
mejor_conj = min([r for r in RESULTADOS if r["fase"] == "1-variables"], key=lambda r: r["MAE_CV"])["candidato"].split("| ")[1]
mejor_conj = "45 originales" if "45 originales" in mejor_conj else mejor_conj
Xb_tr, Xb_te = conjuntos[mejor_conj]
p_lgbB = afinar(lgbm(), GRILLA_LGB, Xb_tr, n_iter=60)
evaluar("1-variables", f"LGBM re-afinado | {mejor_conj}", lgbm(**p_lgbB), Xb_tr, Xb_te)

# ------------------------------------------------------------------------------
# FASE 2 - Selección de variables dentro de cada fold (sin fuga)
# ------------------------------------------------------------------------------
Xg_tr, Xg_te = conjuntos[f"{len(feats70)} completas + razones"]
for k in [15, 25, 35, 50]:
    sel = SelectFromModel(lgbm(n_estimators=300, learning_rate=0.03, num_leaves=15, colsample_bytree=0.7),
                          max_features=k, threshold=-np.inf)
    evaluar("2-selección", f"LGBM | top-{k} elegidas dentro de cada fold",
            Pipeline([("sel", sel), ("m", lgbm(**p_lgbB))]), Xg_tr, Xg_te)

# Conjunto de trabajo para las fases siguientes = el mejor hasta ahora (por CV)
cands = [r for r in RESULTADOS if r["fase"] in ("1-variables", "2-selección")]
mejor = min(cands, key=lambda r: r["MAE_CV"])
log(f">>> Mejor configuración de variables por CV: {mejor['candidato']} (MAE CV {mejor['MAE_CV']:.2f})")
if "top-" in mejor["candidato"]:
    k = int(mejor["candidato"].split("top-")[1].split()[0])
    sel = SelectFromModel(lgbm(n_estimators=300, learning_rate=0.03, num_leaves=15, colsample_bytree=0.7),
                          max_features=k, threshold=-np.inf).fit(np.asarray(Xg_tr, float), y)
    cols_sel = list(np.asarray(Xg_tr.columns)[sel.get_support()])
    XW_tr, XW_te = Xg_tr[cols_sel], Xg_te[cols_sel]
    nombre_W = f"top-{k} de {len(feats70)}+razones"
else:
    XW_tr, XW_te = Xb_tr, Xb_te
    nombre_W = mejor_conj
pd.Series(list(XW_tr.columns)).to_csv(os.path.join(carpeta, "variables_conjunto_trabajo.csv"), index=False, header=["variable"])
p_lgbW = afinar(lgbm(), GRILLA_LGB, XW_tr, n_iter=60)
evaluar("2-selección", f"LGBM re-afinado | {nombre_W}", lgbm(**p_lgbW), XW_tr, XW_te)

# ------------------------------------------------------------------------------
# FASE 3 - Función de pérdida y transformación de la edad
# ------------------------------------------------------------------------------
evaluar("3-objetivo", f"LGBM pérdida L1 | {nombre_W}", lgbm(**{**p_lgbW, "objective": "l1"}), XW_tr, XW_te)
evaluar("3-objetivo", f"LGBM pérdida Huber | {nombre_W}", lgbm(**{**p_lgbW, "objective": "huber", "alpha": 10.0}), XW_tr, XW_te)
evaluar("3-objetivo", f"LGBM predice log(edad) | {nombre_W}",
        TransformedTargetRegressor(lgbm(**p_lgbW), func=np.log, inverse_func=np.exp), XW_tr, XW_te)
evaluar("3-objetivo", f"LGBM predice raíz(edad) | {nombre_W}",
        TransformedTargetRegressor(lgbm(**p_lgbW), func=np.sqrt, inverse_func=np.square), XW_tr, XW_te)

# ------------------------------------------------------------------------------
# FASE 4 - Otros modelos (afinados con CV agrupada en el conjunto de trabajo)
# ------------------------------------------------------------------------------
X4, X4te = XW_tr, XW_te
modelos = {
    "XGBoost": (xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1),
                {"n_estimators": [200, 400, 800, 1200], "learning_rate": [0.005, 0.01, 0.02, 0.04],
                 "max_depth": [2, 3, 4, 5], "min_child_weight": [1, 3, 5, 10], "subsample": [0.7, 0.8, 1.0],
                 "colsample_bytree": [0.3, 0.5, 0.7, 0.9], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0.5, 1, 5]}, 50),
    "ExtraTrees": (Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("m", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1))]),
                   {"m__n_estimators": [300, 600], "m__max_features": [0.2, 0.33, 0.5, 0.8],
                    "m__min_samples_leaf": [1, 2, 4, 8], "m__max_depth": [None, 8, 12]}, 30),
    "Random Forest": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))]),
                      {"m__n_estimators": [300, 600], "m__max_features": [0.2, 0.33, 0.5, 0.8],
                       "m__min_samples_leaf": [1, 2, 4, 8], "m__max_depth": [None, 8, 12]}, 30),
    "HistGB (pérdida L1)": (HistGradientBoostingRegressor(loss="absolute_error", random_state=RANDOM_STATE),
                            {"learning_rate": [0.02, 0.05, 0.1], "max_iter": [200, 400, 800], "max_leaf_nodes": [7, 15, 31],
                             "min_samples_leaf": [5, 10, 20], "l2_regularization": [0, 0.1, 1.0]}, 30),
    "SVR (log + escalado)": (Pipeline(prepro_lineal() + [("m", SVR())]),
                             {"m__C": [5, 10, 20, 50, 100, 200], "m__gamma": ["scale", 0.003, 0.01, 0.03],
                              "m__epsilon": [0.1, 0.5, 1, 2, 5]}, 40),
    "ElasticNet (log + escalado)": (Pipeline(prepro_lineal() + [("m", ElasticNet(max_iter=20000))]),
                                    {"m__alpha": [0.01, 0.03, 0.1, 0.3, 1.0], "m__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}, 30),
    "KNN (log + escalado)": (Pipeline(prepro_lineal() + [("m", KNeighborsRegressor())]),
                             {"m__n_neighbors": [5, 10, 15, 20, 30], "m__weights": ["uniform", "distance"], "m__p": [1, 2]}, 20),
    "Red neuronal MLP (log + escalado)": (Pipeline(prepro_lineal() + [("m", MLPRegressor(max_iter=3000, early_stopping=True,
                                                                                   random_state=RANDOM_STATE))]),
                                          {"m__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                                           "m__alpha": [0.001, 0.01, 0.1, 1.0, 3.0], "m__learning_rate_init": [0.001, 0.003, 0.01]}, 20),
}
PARAMS = {"LightGBM": p_lgbW}
for nombre, (est, grilla, n_iter) in modelos.items():
    p = afinar(est, grilla, X4, n_iter=n_iter)
    PARAMS[nombre] = p
    evaluar("4-modelos", f"{nombre} | {nombre_W}", clone(est).set_params(**p), X4, X4te)

# SVR y ElasticNet también con las 45 originales (para ver si el cambio de variables les ayuda)
for nombre in ["SVR (log + escalado)", "ElasticNet (log + escalado)"]:
    est, grilla, n_iter = modelos[nombre]
    p = afinar(est, grilla, X45, n_iter=n_iter)
    evaluar("4-modelos", f"{nombre} | 45 originales", clone(est).set_params(**p), X45, X45te)

# ------------------------------------------------------------------------------
# FASE 5 - Ensambles (promedio de predicciones; se eligen por CV)
# ------------------------------------------------------------------------------
# Solo modelos entrenados sobre el mismo conjunto de trabajo, uno por familia (el mejor de cada una)
cand = [r for r in RESULTADOS if r["fase"] in ("2-selección", "3-objetivo", "4-modelos") and r["candidato"].endswith(f"| {nombre_W}")]
por_familia = {}
for r in sorted(cand, key=lambda r: r["MAE_CV"]):
    familia = "LightGBM" if r["candidato"].startswith("LGBM") else r["candidato"].split(" |")[0]
    por_familia.setdefault(familia, r["candidato"])
cand_ens = sorted(por_familia.values(), key=lambda c: next(r["MAE_CV"] for r in RESULTADOS if r["candidato"] == c))[:7]
log(f"Candidatos para ensamble (mejores 7 por CV): {cand_ens}")
ens_res = []
for k in [2, 3, 4, 5]:
    for combo in itertools.combinations(cand_ens, k):
        oof = np.mean([OOF[c] for c in combo], axis=0)
        ens_res.append((np.abs(oof - y).mean(axis=1).mean(), combo))
ens_res.sort(key=lambda t: t[0])
for mae_cv, combo in ens_res[:5]:
    oof = np.mean([OOF[c] for c in combo], axis=0)
    p_te = np.mean([PRED_TE[c] for c in combo], axis=0)
    maes = np.abs(oof - y).mean(axis=1)
    nombre = "Promedio: " + " + ".join(c.split(" |")[0] for c in combo)
    fila = {"fase": "5-ensamble", "candidato": nombre, "n_variables": XW_tr.shape[1], "MAE_CV": round(maes.mean(), 3),
            "MAE_CV_sd": round(maes.std(), 3), "MAE_test": round(np.abs(y_te - p_te).mean(), 3),
            "acierto5_test": round((np.abs(y_te - p_te) <= 5).mean() * 100, 1), "miembros": " || ".join(combo)}
    RESULTADOS.append(fila); OOF[nombre] = oof; PRED_TE[nombre] = p_te
    log(f"5-ensamble  | {nombre[:60]:<60s} | MAE CV {fila['MAE_CV']:.2f} | test {fila['MAE_test']:.2f}")
pd.DataFrame(RESULTADOS).to_csv(os.path.join(carpeta, "resultados_banco.csv"), index=False)

# ------------------------------------------------------------------------------
# FASE 6 - Corrección del sesgo (regresión a la media)
# ------------------------------------------------------------------------------
mejor_single = min([r for r in RESULTADOS if r["fase"] in ("2-selección", "3-objetivo", "4-modelos")
                    and r["candidato"].endswith(f"| {nombre_W}")], key=lambda r: r["MAE_CV"])
log(f"Mejor modelo individual por CV: {mejor_single['candidato']}")
evaluar("6-sesgo", "LGBM + corrección de sesgo", CorreccionSesgo(lgbm(**p_lgbW)), XW_tr, XW_te)
# Corrección de sesgo para el mejor ensamble (a y b estimados con sus predicciones fuera de muestra)
mejor_ens = min([r for r in RESULTADOS if r["fase"] == "5-ensamble"], key=lambda r: r["MAE_CV"])
oof_e = OOF[mejor_ens["candidato"]]
# a,b estimados en la repetición 0 y evaluados en las otras (evita usar las mismas predicciones)
b, a = np.polyfit(oof_e[0], y, 1)
maes_corr = np.abs(a + b * oof_e[1:] - y).mean(axis=1)
p_te_corr = a + b * PRED_TE[mejor_ens["candidato"]]
fila = {"fase": "6-sesgo", "candidato": mejor_ens["candidato"] + " + corrección de sesgo", "n_variables": XW_tr.shape[1],
        "MAE_CV": round(maes_corr.mean(), 3), "MAE_CV_sd": round(maes_corr.std(), 3),
        "MAE_test": round(np.abs(y_te - p_te_corr).mean(), 3),
        "acierto5_test": round((np.abs(y_te - p_te_corr) <= 5).mean() * 100, 1), "miembros": mejor_ens.get("miembros", "")}
RESULTADOS.append(fila); PRED_TE[fila["candidato"]] = p_te_corr
log(f"6-sesgo     | {fila['candidato'][:60]:<60s} | MAE CV {fila['MAE_CV']:.2f} | test {fila['MAE_test']:.2f}")

# ------------------------------------------------------------------------------
# Resumen
# ------------------------------------------------------------------------------
df = pd.DataFrame(RESULTADOS)
df.to_csv(os.path.join(carpeta, "resultados_banco.csv"), index=False)
pd.DataFrame({"edad_real": y_te, **PRED_TE}).to_csv(os.path.join(carpeta, "predicciones_test_banco.csv"), index=False)
pd.Series({k: str(v) for k, v in PARAMS.items()}).to_csv(os.path.join(carpeta, "hiperparametros.csv"), header=["parametros"])

base = df.iloc[0]
ganador = df.loc[df["MAE_CV"].idxmin()]
err_b = np.abs(y_te - PRED_TE[base["candidato"]])
err_g = np.abs(y_te - PRED_TE[ganador["candidato"]])
d = (err_g - err_b)[np.random.default_rng(0).integers(0, len(y_te), size=(5000, len(y_te)))].mean(axis=1)
print("\n" + "=" * 100)
print("TOP 12 POR MAE DE VALIDACIÓN CRUZADA")
print("=" * 100)
print(df.sort_values("MAE_CV").head(12)[["fase", "candidato", "n_variables", "MAE_CV", "MAE_CV_sd", "MAE_test", "acierto5_test"]].to_string(index=False))
print(f"\nBase:    {base['candidato']} -> CV {base['MAE_CV']:.2f} | test {base['MAE_test']:.2f}")
print(f"Ganador: {ganador['candidato']} -> CV {ganador['MAE_CV']:.2f} | test {ganador['MAE_test']:.2f}")
print(f"Diferencia en test (ganador - base): {err_g.mean()-err_b.mean():.2f} años | IC95% "
      f"[{np.percentile(d, 2.5):.2f}, {np.percentile(d, 97.5):.2f}] | P(mejora) {np.mean(d < 0)*100:.0f}%")
log("Listo.")
