# ==============================================================================
# RAMA DE CITOQUINAS ROBUSTA AL ESTUDIO DE ORIGEN
# Proyecto: Immuno-XAI - UTEM
#
# Problema: las citoquinas del 10K permiten adivinar el estudio (97%) y su señal
# de edad no se transfiere a estudios no vistos.
#
# Transformaciones probadas (todas sobre log2(pg/ml + 1)):
#   crudo   : sin cambios
#   CLR     : log-ratio centrado dentro de cada muestra (resta la media de un
#             núcleo de citoquinas medidas en casi todas las muestras); elimina
#             desplazamientos globales de cada muestra/placa
#   rangos  : posición de cada citoquina dentro de su propia muestra (0-1)
#   + selección por consistencia: solo citoquinas cuya relación con la edad
#             tiene el mismo signo DENTRO de cada estudio de entrenamiento
#             (calculada solo con los estudios de entrenamiento, sin fuga)
#
# Evaluación:
#   - Firma del estudio: % de acierto adivinando el estudio (menos = mejor)
#   - Dejando un estudio fuera (LOSO): MAE y r en el estudio excluido
#   - CV agrupada 5x5 sobre los 473 pareados (lo que veníamos usando)
# Después: las fuentes externas de citoquinas con la mejor transformación, y la
# fusión células + citoquinas en estudios no vistos.
# ==============================================================================

import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

warnings.filterwarnings("ignore")
REPETICIONES = 5
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_dos_ramas")
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
cuenta = par["study_accession"].value_counts()
par["estudio"] = par["study_accession"].where(par["study_accession"].map(cuenta) >= 20, "otros")
y = par["age"].values.astype(float)
grupos = par["subject_accession"].values
estudios = par["estudio"].values
ESTUDIOS_LOSO = ["SDY420", "SDY515", "SDY519", "SDY113", "SDY311"]

cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells", "Central_Memory_CD4_T_cells",
            "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells", "Effector_Memory_CD4_T_cells",
            "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes", "Memory_B_cells", "Monocytes", "Naive_B_cells",
            "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells", "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto", "edad", "study_accession", "estudio"]]
cols_cit = [c for c in todas_cit if par.loc[par.conjunto == "train", c].notna().mean() >= 0.8]

def leer_fuentes():
    fuentes = {}
    for p in sorted(glob.glob(ruta("fuentes_citoquinas", "*.csv"))):
        d = pd.read_csv(p)
        for c in cols_cit:
            if c not in d.columns:
                d[c] = np.nan
        d["estudio"] = d["fuente"].str.extract(r"\((SDY\d+)\)")[0].fillna(os.path.basename(p)) if "fuente" in d else os.path.basename(p)
        fuentes[os.path.basename(p).replace(".csv", "")] = d.reset_index(drop=True)
    return fuentes
FUENTES = leer_fuentes()

# Núcleo para el CLR: citoquinas medidas en >=95% de los pareados y en >=90% de cada fuente externa
nucleo = [c for c in cols_cit if par[c].notna().mean() >= 0.95
          and all(d[c].notna().mean() >= 0.9 for d in FUENTES.values())]

# ------------------------------------------------------------------------------
# Transformaciones (fila a fila: no usan información de otras muestras -> sin fuga)
# ------------------------------------------------------------------------------
def T_crudo(d):
    return np.log2(d[cols_cit].astype(float).clip(lower=0) + 1).values

def T_clr(d):
    L = T_crudo(d)
    idx = [cols_cit.index(c) for c in nucleo]
    centro = np.nanmean(L[:, idx], axis=1, keepdims=True)
    return L - centro

def T_rangos(d):
    L = pd.DataFrame(T_crudo(d))
    return L.rank(axis=1, pct=True).values

TRANSF = {"crudo": T_crudo, "CLR": T_clr, "rangos": T_rangos}
sexo = lambda d: (d["gender"] == "Male").astype(float).values.reshape(-1, 1)

def seleccion_consistente(Z, edad, est, min_r=0.15, min_n=20, min_rango=15):
    """Columnas cuya correlación con la edad tiene el mismo signo en todos los estudios
    de entrenamiento informativos y |r| medio >= min_r."""
    rs = []
    for e in np.unique(est):
        m = est == e
        if m.sum() < min_n or np.ptp(edad[m]) < min_rango:
            continue
        rs.append([spearmanr(Z[m, j], edad[m], nan_policy="omit")[0] for j in range(Z.shape[1])])
    rs = np.array(rs)
    if len(rs) < 2:
        return np.arange(Z.shape[1])
    signo = np.sign(np.nanmean(rs, axis=0))
    consistente = np.all(np.sign(rs) == signo, axis=0)
    fuerza = np.nanmean(np.abs(rs), axis=0)
    elegidas = np.where(consistente & (fuerza >= min_r))[0]
    if len(elegidas) < 5:
        elegidas = np.argsort(-(consistente * fuerza))[:10]
    return elegidas

def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.5, random_state=0, n_jobs=1, verbose=-1)

# ------------------------------------------------------------------------------
# Entrenar/predecir la rama de citoquinas con una transformación (+ selección opcional)
# ------------------------------------------------------------------------------
def rama_cit(transf, seleccionar, tr, va, extra=None, excluir_estudio=None):
    Zp = TRANSF[transf](par)
    Ztr, ytr, etr = Zp[tr], y[tr], estudios[tr]
    cols = seleccion_consistente(Ztr, ytr, etr) if seleccionar else np.arange(Zp.shape[1])
    Xtr = np.hstack([Ztr[:, cols], sexo(par)[tr]])
    if extra is not None:
        d = extra
        ok = ~d["subject_accession"].isin(set(grupos[va])).values
        if excluir_estudio is not None:
            ok &= (d["estudio"] != excluir_estudio).values
        Ze = TRANSF[transf](d[ok])
        Xtr = np.vstack([Xtr, np.hstack([Ze[:, cols], sexo(d[ok])])])
        ytr = np.concatenate([ytr, d.loc[ok, "age"].values.astype(float)])
    Xva = np.hstack([Zp[va][:, cols], sexo(par)[va]])
    return lgbm().fit(Xtr, ytr).predict(Xva), len(cols)

def loso(fn):
    out = {}
    for e in ESTUDIOS_LOSO:
        va = np.where(estudios == e)[0]; tr = np.where(estudios != e)[0]
        p = fn(tr, va, e)
        out[e] = (np.abs(y[va] - p).mean(), np.corrcoef(y[va], p)[0, 1])
    return out

SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]
def cv_pooled(fn):
    tareas = [(r, tr, va) for r in range(REPETICIONES) for tr, va in SPLITS[r]]
    preds = Parallel(n_jobs=-1)(delayed(fn)(tr, va, None) for _, tr, va in tareas)
    oof = np.zeros((REPETICIONES, len(y)))
    for (r, _, va), p in zip(tareas, preds):
        oof[r, va] = p
    return np.abs(oof - y).mean(axis=1).mean(), np.mean([r2_score(y, oof[r]) for r in range(REPETICIONES)]), oof

# ------------------------------------------------------------------------------
# 1. Comparación de transformaciones
# ------------------------------------------------------------------------------
log(f"Citoquinas: {len(cols_cit)} | núcleo CLR: {len(nucleo)} | estudios LOSO: {ESTUDIOS_LOSO}")
clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=0, verbose=-1)
filas = []
for transf in TRANSF:
    Z = TRANSF[transf](par)
    acierto = np.mean(cross_val_predict(clf, Z, estudios, cv=GroupKFold(5, shuffle=True, random_state=0), groups=grupos) == estudios)
    for sel in [False, True]:
        nombre = f"{transf}{' + selección consistente' if sel else ''}"
        L = loso(lambda tr, va, e: rama_cit(transf, sel, tr, va)[0])
        mae_cv, r2_cv, _ = cv_pooled(lambda tr, va, e: rama_cit(transf, sel, tr, va)[0])
        n_sel = rama_cit(transf, sel, np.arange(len(y)), np.arange(5))[1]
        fila = {"transformacion": nombre, "n_citoquinas": n_sel, "acierto_estudio_%": round(acierto * 100, 1),
                "LOSO_r_medio": np.mean([v[1] for v in L.values()]), "LOSO_MAE_medio": np.mean([v[0] for v in L.values()]),
                **{f"r_{e}": L[e][1] for e in ESTUDIOS_LOSO}, "MAE_CV_473": mae_cv, "R2_CV_473": r2_cv}
        filas.append(fila)
        log(f"{nombre:<32s} | estudio adivinado {fila['acierto_estudio_%']:5.1f}% | LOSO r medio {fila['LOSO_r_medio']:+.2f} "
            f"MAE {fila['LOSO_MAE_medio']:.2f} | CV MAE {mae_cv:.2f} R² {r2_cv:.3f} | {n_sel} citoquinas")
df_t = pd.DataFrame(filas).round(3)
df_t.to_csv(os.path.join(carpeta, "citoquinas_transformaciones.csv"), index=False)

# Mejor transformación = mayor r medio en estudios no vistos
mejor = df_t.loc[df_t["LOSO_r_medio"].idxmax()]
T_MEJOR, SEL_MEJOR = mejor["transformacion"].split(" + ")[0], "selección" in mejor["transformacion"]
log(f">>> Mejor en estudios no vistos: {mejor['transformacion']} (r medio {mejor['LOSO_r_medio']:+.2f})")
if SEL_MEJOR:
    elegidas = seleccion_consistente(TRANSF[T_MEJOR](par), y, estudios)
    pd.Series([cols_cit[j] for j in elegidas], name="citoquina").to_csv(os.path.join(carpeta, "citoquinas_consistentes.csv"), index=False)
    log(f"    citoquinas consistentes entre estudios: {[cols_cit[j] for j in elegidas]}")

# ------------------------------------------------------------------------------
# 2. Fuentes externas con la transformación cruda vs la mejor
# ------------------------------------------------------------------------------
filas_f = []
for transf, sel, etiqueta in [("crudo", False, "crudo"), (T_MEJOR, SEL_MEJOR, mejor["transformacion"])]:
    base_L = loso(lambda tr, va, e: rama_cit(transf, sel, tr, va)[0])
    base_cv = cv_pooled(lambda tr, va, e: rama_cit(transf, sel, tr, va)[0])
    for nombre_f in [None] + list(FUENTES):
        if nombre_f is None:
            L, (mae_cv, r2_cv, oof) = base_L, base_cv
        else:
            d = FUENTES[nombre_f]
            L = loso(lambda tr, va, e: rama_cit(transf, sel, tr, va, d, e)[0])
            mae_cv, r2_cv, oof = cv_pooled(lambda tr, va, e: rama_cit(transf, sel, tr, va, d, None)[0])
        fila = {"transformacion": etiqueta, "fuente": "sin fuentes" if nombre_f is None else f"+ {nombre_f}",
                "LOSO_r_medio": np.mean([v[1] for v in L.values()]), "LOSO_MAE_medio": np.mean([v[0] for v in L.values()]),
                "MAE_CV_473": mae_cv, "R2_CV_473": r2_cv,
                "dif_CV_vs_sin_fuentes": mae_cv - base_cv[0],
                "dif_LOSO_MAE_vs_sin_fuentes": np.mean([v[0] for v in L.values()]) - np.mean([v[0] for v in base_L.values()])}
        filas_f.append(fila)
        log(f"{etiqueta:<32s} | {fila['fuente']:<22s} | LOSO r {fila['LOSO_r_medio']:+.2f} MAE {fila['LOSO_MAE_medio']:.2f} "
            f"({fila['dif_LOSO_MAE_vs_sin_fuentes']:+.2f}) | CV MAE {mae_cv:.2f} ({fila['dif_CV_vs_sin_fuentes']:+.2f})")
df_f = pd.DataFrame(filas_f).round(3)
df_f.to_csv(os.path.join(carpeta, "citoquinas_robustas_fuentes.csv"), index=False)

# ------------------------------------------------------------------------------
# 3. Fusión células + citoquinas en estudios no vistos
# ------------------------------------------------------------------------------
def razones(df):
    e = 0.01
    g = lambda c: df[c].astype(float)
    return pd.DataFrame({"r_CD4_CD8": g("CD4_T_cells") / (g("CD8_T_cells") + e),
                         "r_naiveCD8_CD8": g("Naive_CD8_T_cells") / (g("CD8_T_cells") + e),
                         "r_naiveCD4_CD4": g("Naive_CD4_T_cells") / (g("CD4_T_cells") + e),
                         "r_naiveCD8_T": g("Naive_CD8_T_cells") / (g("T_cells") + e)}, index=df.index)
XC = np.hstack([par[cols_cel].astype(float).values, razones(par).values, sexo(par)])

def fusion(transf, sel, tr, va, solo_celulas=False):
    oof = np.zeros((len(tr), 2))
    for itr, iva in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(tr, groups=grupos[tr]):
        a, b = tr[itr], tr[iva]
        oof[iva, 0] = lgbm().fit(XC[a], y[a]).predict(XC[b])
        oof[iva, 1] = rama_cit(transf, sel, a, b)[0]
    pc = lgbm().fit(XC[tr], y[tr]).predict(XC[va])
    if solo_celulas:
        return pc
    pk = rama_cit(transf, sel, tr, va)[0]
    return LinearRegression().fit(oof, y[tr]).predict(np.column_stack([pc, pk]))

filas_u = []
for etiqueta, fn in [("Solo células", lambda tr, va, e: fusion("crudo", False, tr, va, True)),
                     ("Fusión con citoquinas crudas", lambda tr, va, e: fusion("crudo", False, tr, va)),
                     (f"Fusión con citoquinas {mejor['transformacion']}", lambda tr, va, e: fusion(T_MEJOR, SEL_MEJOR, tr, va))]:
    L = loso(fn)
    mae_cv, r2_cv, _ = cv_pooled(fn)
    fila = {"modelo": etiqueta, "LOSO_r_medio": np.mean([v[1] for v in L.values()]),
            "LOSO_MAE_medio": np.mean([v[0] for v in L.values()]), **{f"MAE_{e}": L[e][0] for e in ESTUDIOS_LOSO},
            "MAE_CV_473": mae_cv, "R2_CV_473": r2_cv}
    filas_u.append(fila)
    log(f"{etiqueta:<55s} | LOSO r {fila['LOSO_r_medio']:+.2f} MAE {fila['LOSO_MAE_medio']:.2f} | CV MAE {mae_cv:.2f} R² {r2_cv:.3f}")
df_u = pd.DataFrame(filas_u).round(3)
df_u.to_csv(os.path.join(carpeta, "fusion_estudios_no_vistos.csv"), index=False)

# ------------------------------------------------------------------------------
# 4. Fusión con fuentes externas en AMBAS ramas (estudios no vistos y CV normal)
# ------------------------------------------------------------------------------
EXTRA_CIT = pd.concat(list(FUENTES.values()), ignore_index=True)
ext_cel = pd.read_csv(ruta("fuentes_celulas", "10k_cytof.csv"))
ext_cel["estudio"] = ext_cel["fuente"].str.extract(r"\((SDY\d+)\)")[0]
Xcel = lambda d: np.hstack([d[cols_cel].astype(float).values, razones(d).values, sexo(d)])
XCE = Xcel(ext_cel)

def rama_cel(a, b, usar_extra, excluir_sujetos, excluir_estudio):
    Xt, yt = XC[a], y[a]
    if usar_extra:
        ok = ~ext_cel["subject_accession"].isin(excluir_sujetos | set(grupos[b])).values
        if excluir_estudio is not None:
            ok &= (ext_cel["estudio"] != excluir_estudio).values
        Xt, yt = np.vstack([Xt, XCE[ok]]), np.concatenate([yt, ext_cel.loc[ok, "age"].values.astype(float)])
    return lgbm().fit(Xt, yt).predict(XC[b])

def rama_cit_ext(transf, sel, a, b, usar_extra, excluir_sujetos, excluir_estudio):
    if not usar_extra:
        return rama_cit(transf, sel, a, b)[0]
    d = EXTRA_CIT[~EXTRA_CIT["subject_accession"].isin(excluir_sujetos)]
    return rama_cit(transf, sel, a, b, d, excluir_estudio)[0]

def fusion_ext(transf, sel, tr, va, e, usar_extra):
    excl = set(grupos[va])
    oof = np.zeros((len(tr), 2))
    for itr, iva in GroupKFold(n_splits=5, shuffle=True, random_state=7).split(tr, groups=grupos[tr]):
        a, b = tr[itr], tr[iva]
        oof[iva, 0] = rama_cel(a, b, usar_extra, excl, e)
        oof[iva, 1] = rama_cit_ext(transf, sel, a, b, usar_extra, excl, e)
    pc = rama_cel(tr, va, usar_extra, excl, e)
    pk = rama_cit_ext(transf, sel, tr, va, usar_extra, excl, e)
    return LinearRegression().fit(oof, y[tr]).predict(np.column_stack([pc, pk]))

filas_x = []
for etiqueta, transf, sel, extra in [("Fusión citoquinas crudas, sin fuentes", "crudo", False, False),
                                     ("Fusión citoquinas crudas + todas las fuentes", "crudo", False, True),
                                     (f"Fusión citoquinas {mejor['transformacion']}, sin fuentes", T_MEJOR, SEL_MEJOR, False),
                                     (f"Fusión citoquinas {mejor['transformacion']} + todas las fuentes", T_MEJOR, SEL_MEJOR, True)]:
    L = loso(lambda tr, va, e: fusion_ext(transf, sel, tr, va, e, extra))
    mae_cv, r2_cv, _ = cv_pooled(lambda tr, va, e: fusion_ext(transf, sel, tr, va, None, extra))
    fila = {"modelo": etiqueta, "LOSO_r_medio": np.mean([v[1] for v in L.values()]),
            "LOSO_MAE_medio": np.mean([v[0] for v in L.values()]), **{f"MAE_{e}": L[e][0] for e in ESTUDIOS_LOSO},
            "MAE_CV_473": mae_cv, "R2_CV_473": r2_cv}
    filas_x.append(fila)
    log(f"{etiqueta:<62s} | LOSO r {fila['LOSO_r_medio']:+.2f} MAE {fila['LOSO_MAE_medio']:.2f} | CV MAE {mae_cv:.2f} R² {r2_cv:.3f}")
df_x = pd.DataFrame(filas_x).round(3)
df_x.to_csv(os.path.join(carpeta, "fusion_con_fuentes_estudios_no_vistos.csv"), index=False)

pd.set_option("display.width", 260)
print("\n=== 4. Fusión con fuentes externas ===\n" + df_x.to_string(index=False))
print("\n=== 1. Transformaciones de citoquinas ===\n" + df_t.to_string(index=False))
print("\n=== 2. Fuentes externas ===\n" + df_f.to_string(index=False))
print("\n=== 3. Fusión en estudios no vistos ===\n" + df_u.to_string(index=False))
log("Listo.")
