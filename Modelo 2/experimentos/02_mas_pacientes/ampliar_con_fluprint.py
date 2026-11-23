# ==============================================================================
# AMPLIACIÓN DEL TRAIN CON DONANTES NUEVOS DE FLUPRINT
# Proyecto: Immuno-XAI - UTEM
#
# Fuente: FluPRINT (Tomic et al. 2019, Scientific Data), Stanford HIMC.
#   Zenodo 3213899 - raw_data_for_import.zip (licencia CC-BY 4.0)
#   CyTOF (gating HIMC, % del nivel padre) + Luminex 50/51/62-plex (Z.log2)
#
# Pasos:
#   1. Construye las 45 variables de tu modelo para cada muestra basal
#      (día 0, antes de vacunar) que tenga CyTOF y Luminex:
#        - células: % del total viable (producto de la cadena de gating)
#        - citoquinas: renombradas a los nombres del 10K (MIG -> CXCL9, ...)
#   2. Excluye a cualquier donante que pueda estar en tu dataset (10K):
#        - mismo sexo y misma edad (±0.06 años) que algún sujeto del 10K, o
#        - perfil de citoquinas casi idéntico (correlación >= 0.75)
#      Si una muestra de una persona queda excluida, se excluye la persona.
#   3. Lleva las variables a la escala del 10K usando solo TRAIN (el test no
#      se usa para nada):
#        - citoquinas: recta calibrada con las muestras que FluPRINT y el 10K
#          comparten (misma muestra medida en ambos), sin sujetos de test
#        - células: alineación de distribuciones tipo ComBat (media y
#          dispersión) ajustando por edad y sexo, con tu train como referencia
#   4. Agrega los donantes nuevos a train con la misma receta de aumento
#      (copia con ruido + mixup proporcional).
#
# No modifica tus archivos: genera df_train_aumentado_k45_fluprint.csv
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd

RANDOM_STATE = 42
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
DIR_FP = ruta("datos_fluprint", "extraido")
DIR_10K = ruta("datos_10k", "extraido")

archivo_train = ruta("df_train_aumentado_k45.csv")
archivo_ids = ruta("datos_10k", "ids_pacientes_actuales.csv")  # generado por ampliar_dataset.py
archivo_cytof = os.path.join(DIR_10K, "CyTOF_PBMC", "CyTOF_PBMC", "cytof_pbmc_normalized.csv")
archivo_elisa = os.path.join(DIR_10K, "Multiplex_ELISA", "Multiplex_ELISA", "Normalized_Files", "multiplexELISA_normalized.csv")

salida_train = ruta("df_train_aumentado_k45_fluprint.csv")
salida_nuevos = ruta("datos_fluprint", "donantes_nuevos_fluprint.csv")
salida_calib = ruta("datos_fluprint", "calibracion_fluprint.csv")

EDAD_MINIMA = 11.0          # rango de edades de tu train (mín. 11.5)
TOL_EDAD = 0.06             # misma visita = misma edad con 2 decimales
CORR_SOLAPE = 0.75          # perfil de citoquinas casi idéntico = misma muestra
RUIDO_FRACCION_STD = 0.03
ALPHA_MIXUP = 0.3

df_train = pd.read_csv(archivo_train)
ids = pd.read_csv(archivo_ids)
cytof = pd.read_csv(archivo_cytof)
elisa = pd.read_csv(archivo_elisa)

features = [c for c in df_train.columns if c not in ("gender", "source_type", "age")]
cols_celulas = [c for c in features if c in cytof.columns]
cols_citoquinas = [c for c in features if c in elisa.columns]
ids_test = set(ids.loc[ids["conjunto"] == "test", "subject_accession"])
train_orig = df_train[df_train["source_type"] == "original"].reset_index(drop=True)

# ------------------------------------------------------------------------------
# 1. Tabla de FluPRINT con las 45 variables (muestras basales)
# ------------------------------------------------------------------------------
frames = []
for p in glob.glob(os.path.join(DIR_FP, "*", "*.csv")):
    d = pd.read_csv(p, low_memory=False)
    if "Assay" in d.columns:
        frames.append(d.rename(columns={"units": "Units"}))
fp_raw = pd.concat(frames, ignore_index=True)
fp_raw["Age"] = pd.to_numeric(fp_raw["Age"], errors="coerce")
fp_raw["N"] = pd.to_numeric(fp_raw["N"], errors="coerce")
basal = fp_raw[fp_raw["Day"].isin(["Day00", "V1"])]

cy = basal[basal["Assay"] == "CyTOF phenotyping"]
par = cy[cy["Units"] == "% of Parent"].pivot_table(index="Sample", columns="Analyte Generic Name", values="N", aggfunc="mean")
via = cy[cy["Units"] == "% of Viable"].pivot_table(index="Sample", columns="Analyte Generic Name", values="N", aggfunc="mean")
f = lambda c: par[c] / 100.0
lymph, mono = f("lymphocytes"), f("monocytes")
T = lymph * f("T cells")
nonT = lymph * f("non-T lymphocytes")
CD4, CD8 = T * f("CD4+ T cells"), T * f("CD8+ T cells")
celulas = pd.DataFrame({
    "Naive_CD8_T_cells": CD8 * f("naive CD8+ T cells"),
    "Central_Memory_CD8_T_cells": CD8 * f("central memory CD8+ T cells"),
    "Monocytes": mono,
    "NKT_cells": lymph * f("NKT cells"),
    "Effector_CD4_T_cells": CD4 * f("effector CD4+ T cells"),
    "B_cells": nonT * f("B cells"),
    "CD16_neg_monocytes": mono * f("CD16- monocytes"),
    "Tregs": CD4 * f("Tregs"),
    "Central_Memory_CD4_T_cells": CD4 * f("central memory CD4+ T cells"),
    "CD16_pos_monocytes": mono * f("CD16+ monocytes"),
    "Naive_CD4_T_cells": CD4 * f("naive CD4+ T cells"),
    "Effector_CD8_T_cells": CD8 * f("effector CD8+ T cells"),
    "Plasmablasts": nonT * f("CD20- CD3- lymphocytes") * f("plasmablasts"),
    "Gamma_Delta_T_cells": lymph * f("gamma-delta T cells"),
    "CD4_T_cells": CD4,
    "T_cells": T,
}) * 100.0
# Donde FluPRINT ya trae "% of Viable", se usa directamente
directos = {"Naive_CD8_T_cells": "naive CD8+ T cells", "Central_Memory_CD8_T_cells": "central memory CD8+ T cells",
            "Monocytes": "monocytes", "NKT_cells": "NKT cells", "Effector_CD4_T_cells": "effector CD4+ T cells",
            "B_cells": "B cells", "CD16_neg_monocytes": "CD16- monocytes", "Tregs": "Tregs",
            "Central_Memory_CD4_T_cells": "central memory CD4+ T cells", "CD16_pos_monocytes": "CD16+ monocytes",
            "Naive_CD4_T_cells": "naive CD4+ T cells", "Effector_CD8_T_cells": "effector CD8+ T cells",
            "Plasmablasts": "plasmablasts", "Gamma_Delta_T_cells": "gamma-delta T cells",
            "CD4_T_cells": "CD4+ T cells", "T_cells": "T cells"}
for col, nombre in directos.items():
    if nombre in via.columns:
        idx = via.index[via[nombre].notna()].intersection(celulas.index)
        celulas.loc[idx, col] = via.loc[idx, nombre]

mapa_citoquinas = {"IL6": "IL6", "VCAM1": "VCAM1", "HGF": "HGF", "IL8": "CXCL8", "IFNA": "IFN1.", "IFNG": "IFNG",
                   "IL1RA": "IL1R1", "MIG": "CXCL9", "IL13": "IL13", "NGF": "NGF", "ICAM1": "ICAM1", "FGFB": "FGF2",
                   "FASL": "FASLG", "MIP1A": "CCL3", "IL15": "IL15", "LEPTIN": "LEP", "MCP3": "CCL7", "MIP1B": "CCL4",
                   "TGFB": "TGFB1", "IL2": "IL2", "IFNB": "IFNB1", "RESISTIN": "RETN", "CD40L": "CD40LG",
                   "ENA78": "CXCL5", "IL4": "IL4", "TNFA": "TNF", "IL5": "IL5", "MCP1": "CCL2", "MCSF": "CSF1"}
lu = basal[basal["Assay"].str.contains("Human Luminex", regex=False) & basal["Analyte Generic Name"].isin(mapa_citoquinas)]
citoquinas = lu.pivot_table(index="Sample", columns="Analyte Generic Name", values="N", aggfunc="mean").rename(columns=mapa_citoquinas)
assert set(cols_citoquinas) <= set(citoquinas.columns) and set(cols_celulas) <= set(celulas.columns)

meta = basal.drop_duplicates("Sample").set_index("Sample")[["project", "Year", "Person", "Ethnicity", "Gender", "Age"]]
fp = meta.join(celulas[cols_celulas], how="inner").join(citoquinas[cols_citoquinas], how="inner").reset_index()
fp = fp.dropna(subset=["Age", "Gender"])
print(f"FluPRINT: {len(fp)} muestras basales con CyTOF y Luminex ({fp['Person'].nunique()} personas)")

# ------------------------------------------------------------------------------
# 2. Detección de donantes que ya están en el 10K
# ------------------------------------------------------------------------------
k10 = pd.concat([elisa[["subject_accession", "age", "gender"]], cytof[["subject_accession", "age", "gender"]]]).drop_duplicates()
fp["match_demografico"] = [
    bool(((k10["gender"] == g) & ((k10["age"] - a).abs() <= TOL_EDAD)).any()) for g, a in zip(fp["Gender"], fp["Age"])
]

zf = fp[cols_citoquinas].astype(float)
zf = ((zf - zf.mean()) / zf.std()).to_numpy()
le = np.log2(elisa[cols_citoquinas].astype(float).clip(lower=0) + 1)
ze = ((le - le.mean()) / le.std()).to_numpy()
mejor_corr, mejor_fila = [], []
for i in range(len(fp)):
    ok = np.isfinite(zf[i])[None, :] & np.isfinite(ze)
    a = np.where(ok, zf[i][None, :], np.nan)
    b = np.where(ok, ze, np.nan)
    a = a - np.nanmean(a, 1, keepdims=True)
    b = b - np.nanmean(b, 1, keepdims=True)
    c = np.nansum(a * b, 1) / np.sqrt(np.nansum(a ** 2, 1) * np.nansum(b ** 2, 1))
    c[ok.sum(1) < 15] = np.nan
    j = int(np.nanargmax(c))
    mejor_corr.append(c[j])
    mejor_fila.append(j)
fp["corr_citoquinas_10k"] = mejor_corr
fp["fila_elisa_10k"] = mejor_fila
fp["match_perfil"] = fp["corr_citoquinas_10k"] >= CORR_SOLAPE

fp["en_10k"] = fp["match_demografico"] | fp["match_perfil"]
personas_10k = set(fp.loc[fp["en_10k"], "Person"])
fp["persona_en_10k"] = fp["Person"].isin(personas_10k)
print(f"  posibles coincidencias con el 10K: {fp['en_10k'].sum()} muestras -> {len(personas_10k)} personas excluidas")

# ------------------------------------------------------------------------------
# 3a. Calibración de citoquinas con muestras compartidas (sin sujetos de test)
#     log2(pg/ml + 1) = a + b * Z.log2, con b = sd_y / sd_x (preserva la dispersión)
# ------------------------------------------------------------------------------
pares = fp[fp["match_perfil"]].copy()
pares["subject_10k"] = elisa["subject_accession"].values[pares["fila_elisa_10k"]]
pares["edad_10k"] = elisa["age"].values[pares["fila_elisa_10k"]]
pares = pares[((pares["Age"] - pares["edad_10k"]).abs() <= TOL_EDAD) & ~pares["subject_10k"].isin(ids_test)]
y10 = np.log2(elisa.iloc[pares["fila_elisa_10k"].values][cols_citoquinas].astype(float).clip(lower=0).reset_index(drop=True) + 1)
xfp = pares[cols_citoquinas].astype(float).reset_index(drop=True)
print(f"  pares para calibrar citoquinas (misma muestra, sin test): {len(pares)}")

filas_calib = []
cit_cal = {}
for c in cols_citoquinas:
    ok = xfp[c].notna() & y10[c].notna()
    x, y = xfp.loc[ok, c], y10.loc[ok, c]
    r = float(np.corrcoef(x, y)[0, 1])
    b = float(y.std() / x.std()) * np.sign(r)
    a = float(y.mean() - b * x.mean())
    cit_cal[c] = (a, b)
    filas_calib.append({"variable": c, "tipo": "citoquina", "metodo": "pares", "n": int(ok.sum()), "r_pares": round(r, 3),
                        "a": round(a, 4), "b": round(b, 4)})

# ------------------------------------------------------------------------------
# 3b. Calibración de células: alineación tipo ComBat (referencia = tu train)
#     En log2(% + 0.1): se iguala la dispersión residual y se corrige el
#     desplazamiento controlando por edad, edad^2 y sexo.
# ------------------------------------------------------------------------------
# Referencia FluPRINT = todas sus muestras, salvo las que podrían ser sujetos de test
sujeto_perfil = pd.Series(elisa["subject_accession"].values[fp["fila_elisa_10k"]], index=fp.index)
posible_test_perfil = fp["match_perfil"] & sujeto_perfil.isin(ids_test)
k10_test = k10[k10["subject_accession"].isin(ids_test)]
posible_test_demo = pd.Series([
    bool(((k10_test["gender"] == g) & ((k10_test["age"] - a).abs() <= TOL_EDAD)).any()) for g, a in zip(fp["Gender"], fp["Age"])
], index=fp.index)
fp_ref = fp[~(posible_test_perfil | posible_test_demo)]
print(f"  muestras FluPRINT usadas para calibrar células: {len(fp_ref)}")

def diseno(edad, sexo_male):
    e = (np.asarray(edad, float) - 50.0) / 20.0
    return np.column_stack([np.ones_like(e), e, e ** 2, np.asarray(sexo_male, float)])

L = lambda v: np.log2(np.clip(np.asarray(v, float), 0, None) + 0.1)
Xr = diseno(train_orig["age"], train_orig["gender"] == "Male")
Xf = diseno(fp_ref["Age"], fp_ref["Gender"] == "Male")
cel_cal = {}
for c in cols_celulas:
    yr, yf = L(train_orig[c]), L(fp_ref[c])
    okr, okf = np.isfinite(yr), np.isfinite(yf)
    br = np.linalg.lstsq(Xr[okr], yr[okr], rcond=None)[0]
    bf = np.linalg.lstsq(Xf[okf], yf[okf], rcond=None)[0]
    sr = np.std(yr[okr] - Xr[okr] @ br)
    sf = np.std(yf[okf] - Xf[okf] @ bf)
    escala = sr / sf
    # Tras escalar, desplazamiento = diferencia de interceptos con la curva de edad del train
    yf_esc = (yf - Xf @ bf) * escala + Xf @ (bf * escala)
    resid = yf_esc[okf] - Xf[okf] @ br
    shift = float(np.mean(resid))
    cel_cal[c] = (bf, escala, shift)
    filas_calib.append({"variable": c, "tipo": "celula", "metodo": "ComBat (edad, sexo)", "n": int(okf.sum()),
                        "r_pares": np.nan, "a": round(shift, 4), "b": round(float(escala), 4)})
pd.DataFrame(filas_calib).to_csv(salida_calib, index=False)

def calibrar(df):
    out = df.copy()
    for c in cols_citoquinas:
        a, b = cit_cal[c]
        out[c] = np.clip(2.0 ** (a + b * df[c].astype(float)) - 1.0, 0.0, None)
    Xd = diseno(df["Age"], df["Gender"] == "Male")
    for c in cols_celulas:
        bf, escala, shift = cel_cal[c]
        y = L(df[c])
        y_adj = (y - Xd @ bf) * escala + Xd @ (bf * escala) - shift
        out[c] = np.clip(2.0 ** y_adj - 0.1, 0.0, None)
    return out

# ------------------------------------------------------------------------------
# 4. Donantes nuevos: no están en el 10K, edad dentro del rango, 1 muestra por persona
# ------------------------------------------------------------------------------
nuevos = fp[~fp["persona_en_10k"] & (fp["Age"] >= EDAD_MINIMA)].sort_values(["Person", "Year"]).drop_duplicates("Person")
nuevos = calibrar(nuevos)
medianas = train_orig[features].median()
n_falt = int(nuevos[features].isna().sum().sum())
nuevos[features] = nuevos[features].fillna(medianas)
nuevos = nuevos.rename(columns={"Gender": "gender", "Age": "age"})
nuevos["age"] = nuevos["age"].round(2)
nuevos.to_csv(salida_nuevos, index=False)
print(f"Donantes nuevos agregados: {len(nuevos)} | edad mediana {nuevos['age'].median():.1f} "
      f"(rango {nuevos['age'].min():.1f}-{nuevos['age'].max():.1f}) | faltantes imputados: {n_falt}")

# Control: tras calibrar, la mediana de cada variable vs tu train (mismo rango de edad)
rango = (train_orig["age"] >= nuevos["age"].min()) & (train_orig["age"] <= nuevos["age"].max())
ratio = (nuevos[features].median() / train_orig.loc[rango, features].median()).replace([np.inf, -np.inf], np.nan)
print(f"  mediana nuevos / mediana train (misma edad): p10={ratio.quantile(.1):.2f} "
      f"mediana={ratio.median():.2f} p90={ratio.quantile(.9):.2f}")

# ------------------------------------------------------------------------------
# 5. Aumento (misma receta) y train ampliado
# ------------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
cols_salida = list(df_train.columns)
nuevos_orig = nuevos.assign(source_type="original")[cols_salida].copy()

std_cols = train_orig[features].std()
nuevos_ruido = nuevos_orig.copy()
ruido = rng.normal(0.0, 1.0, size=(len(nuevos_ruido), len(features))) * (RUIDO_FRACCION_STD * std_cols.values)
nuevos_ruido[features] = np.clip(nuevos_ruido[features].values + ruido, 0.0, None)
nuevos_ruido["source_type"] = "synthetic_noise"

pool = pd.concat([train_orig[cols_salida], nuevos_orig], ignore_index=True)
proporcion_mixup = (df_train["source_type"] == "synthetic_mixup").sum() / len(train_orig)
mezclas = []
for sexo in ["Male", "Female"]:
    idx_pool = pool.index[pool["gender"] == sexo].to_numpy()
    idx_nuevos = pool.index[(pool["gender"] == sexo) & (pool.index >= len(train_orig))].to_numpy()
    for _ in range(int(round(len(idx_nuevos) * proporcion_mixup))):
        i = rng.choice(idx_nuevos)
        j = rng.choice(idx_pool[idx_pool != i])
        lam = float(rng.beta(ALPHA_MIXUP, ALPHA_MIXUP))
        fila = pool.loc[i].copy()
        fila[features] = lam * pool.loc[i, features].astype(float) + (1 - lam) * pool.loc[j, features].astype(float)
        fila["age"] = round(lam * pool.at[i, "age"] + (1 - lam) * pool.at[j, "age"], 2)
        fila["source_type"] = "synthetic_mixup"
        mezclas.append(fila)
nuevos_mixup = pd.DataFrame(mezclas, columns=cols_salida)

df_ampliado = pd.concat([df_train, nuevos_orig, nuevos_ruido, nuevos_mixup], ignore_index=True)
df_ampliado.to_csv(salida_train, index=False)
print("\nTrain ampliado con FluPRINT:")
print(df_ampliado["source_type"].value_counts().to_string())
print(f"Total: {len(df_train)} -> {len(df_ampliado)} filas | Guardado en: {salida_train}")
