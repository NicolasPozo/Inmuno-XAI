# ==============================================================================
# MODELO FINAL: DOS RAMAS (CÉLULAS / CITOQUINAS) + FUSIÓN - SOLO 10K IMMUNOMES
# Proyecto: Immuno-XAI - UTEM
#
# Datos (todo del 10K Immunomes):
#   - 473 pacientes con células Y citoquinas (train 354 / test 119)
#   - Rama células   : + personas del 10K con CyTOF sin citoquinas (10k_cytof.csv)
#   - Rama citoquinas: solo los pareados (modelo final). Con el argumento
#     "con_elisa" agrega las visitas del 10K con citoquinas sin CyTOF (10k_elisa.csv)
#   (la citometría de flujo del 10K no se usa: casi siempre mide solo 2-3 tipos celulares)
#
# Modelos:
#   - Rama células    : ensamble de 4 (LightGBM raíz(edad) + XGBoost + SVR + KNN)
#   - Rama citoquinas : LightGBM
#   - Fusión          : edad = a + w1*pred_células + w2*pred_citoquinas, aprendida con
#                       predicciones fuera de muestra de los pacientes pareados
#
# Evaluación (sin fuga: las personas evaluadas nunca entran a ninguna rama ni a
# la fusión, tampoco a través de los datos extra):
#   1. CV agrupada por paciente 5x5 sobre los 473 pareados
#   2. Test de 119 pacientes (entrenado solo con el train)
#   3. Estudios no vistos (se deja un estudio completo fuera)
#
# Salidas en resultados_modelo_final/: métricas, predicciones, gráfico y el
# modelo entrenado (modelo_dos_ramas.joblib).
# ==============================================================================

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
# Uso: python modelo_final_dos_ramas.py [con_elisa]
#   (por defecto) MODELO FINAL: la rama de citoquinas usa solo los pareados
#   con_elisa -> variante alternativa: la rama de citoquinas agrega el ELISA extra
#                del 10K (más robusta en estudios nuevos, algo menos precisa en el 10K)
#   immport   -> agrega a cada rama las fuentes de ImmPort (fuentes_immport/, generadas
#                por incorporar_immport.py)
import sys
SIN_ELISA = "con_elisa" not in sys.argv[1:]
CON_IMMPORT = "immport" in sys.argv[1:]
carpeta = ruta("resultados_modelo_final" + ("" if SIN_ELISA else "_con_elisa")
                       + ("_immport" if CON_IMMPORT else ""))
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:5.1f} min] {m}", flush=True)

# ------------------------------------------------------------------------------
# Datos
# ------------------------------------------------------------------------------
par = pd.read_csv(ruta("pareados.csv"))
el = pd.read_csv(ruta("datos_10k", "extraido", "Multiplex_ELISA", "Multiplex_ELISA", "Normalized_Files",
                              "multiplexELISA_normalized.csv"))
k = el.assign(edad=el.age.round(2))[["subject_accession", "edad", "study_accession"]].drop_duplicates(["subject_accession", "edad"])
par = par.assign(edad=par.age.round(2)).merge(k, on=["subject_accession", "edad"], how="left")
y = par["age"].values.astype(float)
grupos = par["subject_accession"].values
estudios = par["study_accession"].fillna("otro").values
es_train = (par["conjunto"] == "train").values
idx_tr, idx_te = np.where(es_train)[0], np.where(~es_train)[0]

ext_cit = pd.read_csv(ruta("fuentes_citoquinas", "10k_elisa.csv"))
if SIN_ELISA:
    ext_cit = ext_cit.iloc[0:0].copy()
ext_cel = pd.read_csv(ruta("fuentes_celulas", "10k_cytof.csv"))
for d in (ext_cit, ext_cel):
    d["estudio"] = d["fuente"].str.extract(r"\((SDY\d+)\)")[0]
if CON_IMMPORT:
    imm_cit = pd.read_csv(ruta("fuentes_immport", "citoquinas_immport.csv"))
    imm_cel = pd.read_csv(ruta("fuentes_immport", "celulas_immport.csv"))
    ext_cit = pd.concat([ext_cit, imm_cit], ignore_index=True)
    ext_cel = pd.concat([ext_cel, imm_cel], ignore_index=True)

cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells",
            "Central_Memory_CD4_T_cells", "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells",
            "Effector_Memory_CD4_T_cells", "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes",
            "Memory_B_cells", "Monocytes", "Naive_B_cells", "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells",
            "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto", "edad", "study_accession"]]
cols_cit = [c for c in todas_cit if par.loc[es_train, c].notna().mean() >= 0.8]

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

sexo = lambda d: (d["gender"] == "Male").astype(float).values.reshape(-1, 1)
X_cel = lambda d: np.hstack([d[cols_cel].astype(float).values, razones(d).values, sexo(d)])
X_cit = lambda d: np.hstack([d[cols_cit].astype(float).values, sexo(d)])
XC, XK = X_cel(par), X_cit(par)
EC, EK = X_cel(ext_cel), X_cit(ext_cit)
yEC, yEK = ext_cel["age"].values.astype(float), ext_cit["age"].values.astype(float)
log(f"Pareados {len(par)} | extra células {len(EC)} | extra citoquinas {len(EK)} | "
    f"variables: células {XC.shape[1]}, citoquinas {XK.shape[1]}")

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
    """Promedia las predicciones de varios modelos."""
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
GRILLA_SVR = {"m__C": [5, 10, 20, 50, 100, 200], "m__gamma": ["scale", 0.003, 0.01, 0.03], "m__epsilon": [0.1, 0.5, 1, 2, 5]}
GRILLA_KNN = {"m__n_neighbors": [5, 10, 15, 20, 30], "m__weights": ["uniform", "distance"], "m__p": [1, 2]}

def afinar(est, grilla, X):
    """Hiperparámetros elegidos solo con el train pareado (354 filas)."""
    s = RandomizedSearchCV(est, grilla, n_iter=N_ITER, scoring="neg_mean_absolute_error",
                           cv=GroupKFold(n_splits=5, shuffle=True, random_state=100),
                           random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    s.fit(X[es_train], y[es_train], groups=grupos[es_train])
    return clone(est).set_params(**s.best_params_)

log("Afinando la rama de células (ensamble de 4)...")
MODELO_CEL = Promedio([
    afinar(TransformedTargetRegressor(lgbm(), func=np.sqrt, inverse_func=np.square),
           {f"regressor__{k}": v for k, v in GRILLA_LGB.items()}, XC),
    afinar(xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1), GRILLA_XGB, XC),
    afinar(lineal(SVR()), GRILLA_SVR, XC),
    afinar(lineal(KNeighborsRegressor()), GRILLA_KNN, XC),
])
log("Afinando la rama de citoquinas (LightGBM)...")
MODELO_CIT = afinar(lgbm(), GRILLA_LGB, XK)

# ------------------------------------------------------------------------------
# Dos ramas + fusión para un conjunto de entrenamiento/evaluación (sin fuga)
# ------------------------------------------------------------------------------
def entrenar(modelo, Xp, yp, Xe, ye, ge, ee, excluir_sujetos, excluir_estudio):
    ok = ~np.isin(ge, list(excluir_sujetos))
    if excluir_estudio is not None:
        ok &= (ee != excluir_estudio)
    return clone(modelo).fit(np.vstack([Xp, Xe[ok]]), np.concatenate([yp, ye[ok]]))

def dos_ramas(tr, va, excluir_estudio=None, devolver_modelos=False):
    excl = set(grupos[va])
    gEC, eEC = ext_cel["subject_accession"].values, ext_cel["estudio"].values
    gEK, eEK = ext_cit["subject_accession"].values, ext_cit["estudio"].values
    oof = np.zeros((len(tr), 2))
    for itr, iva in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(tr, groups=grupos[tr]):
        a, b = tr[itr], tr[iva]
        ex = excl | set(grupos[b])
        oof[iva, 0] = entrenar(MODELO_CEL, XC[a], y[a], EC, yEC, gEC, eEC, ex, excluir_estudio).predict(XC[b])
        oof[iva, 1] = entrenar(MODELO_CIT, XK[a], y[a], EK, yEK, gEK, eEK, ex, excluir_estudio).predict(XK[b])
    fusion = LinearRegression().fit(oof, y[tr])
    mc = entrenar(MODELO_CEL, XC[tr], y[tr], EC, yEC, gEC, eEC, excl, excluir_estudio)
    mk = entrenar(MODELO_CIT, XK[tr], y[tr], EK, yEK, gEK, eEK, excl, excluir_estudio)
    pc, pk = mc.predict(XC[va]), mk.predict(XK[va])
    pf = fusion.predict(np.column_stack([pc, pk]))
    if devolver_modelos:
        return pf, pc, pk, (mc, mk, fusion)
    return pf, pc, pk, np.r_[fusion.intercept_, fusion.coef_]

def metricas(yv, p):
    return {"MAE": mean_absolute_error(yv, p), "RMSE": np.sqrt(mean_squared_error(yv, p)), "R2": r2_score(yv, p),
            "r": np.corrcoef(yv, p)[0, 1], "acierto_5a_%": np.mean(np.abs(yv - p) <= 5) * 100}

# ------------------------------------------------------------------------------
# 1. CV agrupada 5x5 sobre los 473 pareados
# ------------------------------------------------------------------------------
log("Validación cruzada 5x5...")
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]
tareas = [(r, tr, va) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
res = Parallel(n_jobs=-1)(delayed(dos_ramas)(tr, va) for _, tr, va in tareas)
oof = np.zeros((3, REPETICIONES, len(y))); pesos = []
for (r, _, va), (pf, pc, pk, w) in zip(tareas, res):
    oof[0, r, va], oof[1, r, va], oof[2, r, va] = pf, pc, pk
    pesos.append(w)
filas = []
for i, modelo in enumerate(["Rama células", "Rama citoquinas", "Fusión"]):
    j = {0: 1, 1: 2, 2: 0}[i]
    m = [metricas(y, oof[j, r]) for r in range(REPETICIONES)]
    fila = {"evaluacion": "CV 5x5 (473 pacientes)", "modelo": modelo}
    for k_ in m[0]:
        vals = [mm[k_] for mm in m]
        fila[k_], fila[f"{k_}_sd"] = np.mean(vals), np.std(vals)
    filas.append(fila)
w_cv = np.mean(pesos, axis=0)

# ------------------------------------------------------------------------------
# 2. Test de 119 pacientes (modelo entrenado solo con el train) -> MODELO FINAL
# ------------------------------------------------------------------------------
log("Entrenando el modelo final con el train y evaluando en el test...")
pf_te, pc_te, pk_te, (mc, mk, fusion) = dos_ramas(idx_tr, idx_te, devolver_modelos=True)
y_te = y[idx_te]
for modelo, p in [("Rama células", pc_te), ("Rama citoquinas", pk_te), ("Fusión", pf_te)]:
    filas.append({"evaluacion": "Test (119 pacientes)", "modelo": modelo, **metricas(y_te, p)})

# ------------------------------------------------------------------------------
# 3. Estudios no vistos (se deja un estudio completo fuera)
# ------------------------------------------------------------------------------
log("Evaluando en estudios no vistos...")
loso_est = ["SDY420", "SDY515", "SDY519", "SDY113", "SDY311"]
res_l = Parallel(n_jobs=-1)(delayed(dos_ramas)(np.where(estudios != e)[0], np.where(estudios == e)[0], e) for e in loso_est)
filas_l = []
for e, (pf, pc, pk, _) in zip(loso_est, res_l):
    yv = y[estudios == e]
    for modelo, p in [("Rama células", pc), ("Rama citoquinas", pk), ("Fusión", pf)]:
        filas_l.append({"estudio_no_visto": e, "n": len(yv), "edades": f"{yv.min():.0f}-{yv.max():.0f}", "modelo": modelo,
                        **metricas(yv, p)})
df_l = pd.DataFrame(filas_l)
for modelo in ["Rama células", "Rama citoquinas", "Fusión"]:
    d = df_l[df_l.modelo == modelo]
    filas.append({"evaluacion": "Estudios no vistos (promedio de 5)", "modelo": modelo,
                  **{k_: d[k_].mean() for k_ in ["MAE", "RMSE", "R2", "r", "acierto_5a_%"]}})

# ------------------------------------------------------------------------------
# Guardar
# ------------------------------------------------------------------------------
df = pd.DataFrame(filas).round(3)
df.to_csv(os.path.join(carpeta, "metricas_modelo_final.csv"), index=False)
df_l.round(3).to_csv(os.path.join(carpeta, "metricas_estudios_no_vistos.csv"), index=False)
pd.DataFrame({"subject_accession": grupos[idx_te], "edad_real": y_te, "pred_celulas": pc_te,
              "pred_citoquinas": pk_te, "pred_fusion": pf_te, "brecha_edad_fusion": pf_te - y_te}).round(2).to_csv(
    os.path.join(carpeta, "predicciones_test.csv"), index=False)
joblib.dump({"modelo_celulas": mc, "modelo_citoquinas": mk, "fusion": fusion,
             "columnas_celulas": cols_cel, "columnas_citoquinas": cols_cit,
             "nota": "Variables de células = columnas_celulas + 11 razones inmunes + sexo (1 = Male); "
                     "citoquinas = columnas_citoquinas + sexo. Ver X_cel / X_cit en modelo_final_dos_ramas.py."},
            os.path.join(carpeta, "modelo_dos_ramas.joblib"))

# Gráfico: edad real vs predicha en el test
TINTA, TINTA2, MUTED, GRID, SUP = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), dpi=120, sharex=True, sharey=True)
fig.patch.set_facecolor(SUP)
lim = [min(y_te.min(), pf_te.min(), pc_te.min(), pk_te.min()) - 3, max(y_te.max(), pf_te.max(), pc_te.max(), pk_te.max()) + 3]
for ax, (titulo, p, color) in zip(axes, [("Rama células", pc_te, "#2a78d6"), ("Rama citoquinas", pk_te, "#eb6834"),
                                          ("Fusión", pf_te, "#1baf7a")]):
    m = metricas(y_te, p)
    ax.set_facecolor(SUP)
    ax.plot(lim, lim, color="#c3c2b7", lw=1, ls="--", zorder=1)
    ax.scatter(y_te, p, s=34, color=color, alpha=0.85, edgecolor=SUP, linewidth=1.2, zorder=2)
    ax.set_title(f"{titulo}\nMAE {m['MAE']:.2f} años · R² {m['R2']:.2f} · ±5 años {m['acierto_5a_%']:.0f}%",
                 fontsize=10.5, color=TINTA, loc="left")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.grid(color=GRID, lw=0.7)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("Edad real (años)", color=TINTA2)
axes[0].set_ylabel("Edad predicha (años)", color=TINTA2)
fig.suptitle("Modelo final de dos ramas (solo 10K Immunomes): test de 119 pacientes", x=0.01, ha="left",
             fontsize=12.5, fontweight="bold", color=TINTA)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(carpeta, "test_real_vs_predicha.png"), dpi=300, facecolor=SUP)

pd.set_option("display.width", 220)
print("\n" + "=" * 110)
print("MÉTRICAS DEL MODELO FINAL (dos ramas + fusión, solo 10K)")
print("=" * 110)
print(df[["evaluacion", "modelo", "MAE", "RMSE", "R2", "r", "acierto_5a_%"]].to_string(index=False))
print(f"\nFusión (CV): edad = {w_cv[0]:.2f} + {w_cv[1]:.2f}·células + {w_cv[2]:.2f}·citoquinas")
print(f"Fusión (modelo final): edad = {fusion.intercept_:.2f} + {fusion.coef_[0]:.2f}·células + {fusion.coef_[1]:.2f}·citoquinas")
print("\nPor estudio no visto:")
print(df_l[["estudio_no_visto", "n", "edades", "modelo", "MAE", "R2", "r"]].round(2).to_string(index=False))
log("Listo.")
