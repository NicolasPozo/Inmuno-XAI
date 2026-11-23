# ==============================================================================
# FUENTES EXTERNAS POR RAMA (PLAN B)
# Proyecto: Immuno-XAI - UTEM
#
# Rama citoquinas:
#   fuentes_citoquinas/fluprint_luminex.csv  FluPRINT (Stanford HIMC), Luminex basal
#       (día 0). Calibrado a la escala del 10K con muestras medidas en ambos
#       datasets (recta por analito, sin sujetos de test).
#   fuentes_citoquinas/soundlife_olink.csv   Sound Life (Allen), Olink, 1 visita
#       basal por persona, alineado tipo ComBat (de armonizar_soundlife.py).
# Rama células:
#   fuentes_celulas/fluprint_cytof.csv       FluPRINT CyTOF basal (% del total
#       viable), alineado tipo ComBat a tu train.
#   fuentes_celulas/soundlife_scrna.csv      Sound Life single-cell, 1 visita basal.
#
# En todos los casos se EXCLUYE a cualquier persona que pueda estar en el 10K
# (misma edad exacta y sexo, o perfil de citoquinas casi idéntico). Los sujetos
# de test nunca se usan para calibrar.
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
DIR_FP = ruta("datos_fluprint", "extraido")
DIR_10K = ruta("datos_10k", "extraido")
dir_cit = ruta("fuentes_citoquinas")
dir_cel = ruta("fuentes_celulas")
EDAD_MINIMA, TOL_EDAD, CORR_SOLAPE = 11.0, 0.06, 0.75

par = pd.read_csv(ruta("pareados.csv"))
ref = par[par["conjunto"] == "train"].reset_index(drop=True)          # referencia de escala = tu train
ids_test = set(par.loc[par["conjunto"] == "test", "subject_accession"])
cytof = pd.read_csv(os.path.join(DIR_10K, "CyTOF_PBMC", "CyTOF_PBMC", "cytof_pbmc_normalized.csv"))
elisa = pd.read_csv(os.path.join(DIR_10K, "Multiplex_ELISA", "Multiplex_ELISA", "Normalized_Files", "multiplexELISA_normalized.csv"))
flujo = pd.read_csv(os.path.join(DIR_10K, "Flow_Cytometry_PBMC", "Flow_Cytometry_PBMC", "Normalized_Files", "flow_pbmc_normalized.csv"))

cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells",
            "Central_Memory_CD4_T_cells", "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells",
            "Effector_Memory_CD4_T_cells", "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes",
            "Memory_B_cells", "Monocytes", "Naive_B_cells", "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells",
            "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto"]]
cob = ref[todas_cit].notna().mean()
cols_cit = [c for c in todas_cit if cob[c] >= 0.8]

# Todas las personas del 10K (para detectar solapes)
k10 = pd.concat([elisa[["subject_accession", "age", "gender"]], cytof[["subject_accession", "age", "gender"]],
                 flujo[["subject_accession", "age", "gender"]]]).drop_duplicates()

def en_10k_demografico(gen, edad):
    return np.array([((k10["gender"] == g) & ((k10["age"] - a).abs() <= TOL_EDAD)).any() for g, a in zip(gen, edad)])

def diseno(edad, male):
    e = (np.asarray(edad, float) - 50.0) / 20.0
    return np.column_stack([np.ones_like(e), e, e ** 2, np.asarray(male, float)])

def alinear_combat(nuevo, cols, offset, nuevo_ya_log=False):
    """Lleva 'nuevo' a la escala de tu train: log2, misma dispersión residual y
    desplazamiento respecto a la curva edad/sexo del train."""
    out = nuevo.copy()
    Xr, Xn = diseno(ref["age"], ref["gender"] == "Male"), diseno(nuevo["age"], nuevo["gender"] == "Male")
    for c in cols:
        yr = np.log2(ref[c].astype(float).clip(lower=0) + offset).values
        yn = nuevo[c].astype(float).values if nuevo_ya_log else np.log2(nuevo[c].astype(float).clip(lower=0) + offset).values
        okr, okn = np.isfinite(yr), np.isfinite(yn)
        if okn.sum() < 20 or okr.sum() < 20:
            out[c] = np.nan
            continue
        br = np.linalg.lstsq(Xr[okr], yr[okr], rcond=None)[0]
        bn = np.linalg.lstsq(Xn[okn], yn[okn], rcond=None)[0]
        esc = np.std(yr[okr] - Xr[okr] @ br) / np.std(yn[okn] - Xn[okn] @ bn)
        desp = np.mean(esc * yn[okn] - Xn[okn] @ br)
        out[c] = np.clip(2.0 ** (esc * yn - desp) - offset, 0, None)
    return out

# ------------------------------------------------------------------------------
# FluPRINT: lectura
# ------------------------------------------------------------------------------
frames = []
for p in glob.glob(os.path.join(DIR_FP, "*", "*.csv")):
    d = pd.read_csv(p, low_memory=False)
    if "Assay" in d.columns:
        frames.append(d.rename(columns={"units": "Units"}))
fp = pd.concat(frames, ignore_index=True)
fp["Age"] = pd.to_numeric(fp["Age"], errors="coerce")
fp["N"] = pd.to_numeric(fp["N"], errors="coerce")
basal = fp[fp["Day"].isin(["Day00", "V1"])]
meta_fp = basal.drop_duplicates("Sample").set_index("Sample")[["Person", "Gender", "Age", "Year"]]

# --- Citoquinas FluPRINT ---
mapa = {"IL6": "IL6", "VCAM1": "VCAM1", "HGF": "HGF", "IL8": "CXCL8", "IFNA": "IFN1.", "IFNG": "IFNG", "IL1RA": "IL1R1",
        "MIG": "CXCL9", "IL13": "IL13", "NGF": "NGF", "ICAM1": "ICAM1", "FGFB": "FGF2", "FASL": "FASLG", "MIP1A": "CCL3",
        "IL15": "IL15", "LEPTIN": "LEP", "MCP3": "CCL7", "MIP1B": "CCL4", "TGFB": "TGFB1", "IL2": "IL2", "IFNB": "IFNB1",
        "RESISTIN": "RETN", "CD40L": "CD40LG", "ENA78": "CXCL5", "IL4": "IL4", "TNFA": "TNF", "IL5": "IL5", "MCP1": "CCL2",
        "MCSF": "CSF1", "EOTAXIN": "CCL11", "RANTES": "CCL5", "IP10": "CXCL10", "PAI1": "SERPINE1", "GROA": "CXCL1",
        "SCF": "KITLG", "IL7": "IL7", "LIF": "LIF", "TRAIL": "TNFSF10", "GMCSF": "CSF2", "VEGF": "VEGFA", "IL1A": "IL1A",
        "GCSF": "CSF3", "IL12P40": "IL12B", "IL1B": "IL1B", "IL17F": "IL17F", "IL17A": "IL17A", "IL17": "IL17A", "TGFA": "TGFA",
        "IL10": "IL10", "IL12P70": "IL12A", "IL18": "IL18", "IL9": "IL9", "LIF": "LIF", "TNFB": "LTA", "PDGFBB": "PDGFB",
        "SDF1A": "CXCL12", "IL21": "IL21", "IL22": "IL22", "IL23": "IL23A", "IL27": "IL27", "IL31": "IL31", "PIGF1": "PIGF",
        "VEGFD": "FIGF", "BDNF": "BDNF", "EGF": "EGF", "IFNA2": "IFNA2"}
lu = basal[basal["Assay"].str.contains("Human Luminex", regex=False) & basal["Analyte Generic Name"].isin(mapa)].copy()
lu["var"] = lu["Analyte Generic Name"].map(mapa)
cit_fp = lu[lu["var"].isin(cols_cit)].pivot_table(index="Sample", columns="var", values="N", aggfunc="mean")
cit_fp = meta_fp.join(cit_fp, how="inner").reset_index().dropna(subset=["Age", "Gender"])
for c in cols_cit:
    if c not in cit_fp.columns:
        cit_fp[c] = np.nan

# Perfil de citoquinas vs ELISA del 10K (para detectar la misma muestra)
zf = cit_fp[cols_cit].astype(float)
zf = ((zf - zf.mean()) / zf.std()).to_numpy()
le = np.log2(elisa[cols_cit].astype(float).clip(lower=0) + 1)
ze = ((le - le.mean()) / le.std()).to_numpy()
mejor_c, mejor_j = [], []
for i in range(len(cit_fp)):
    ok = np.isfinite(zf[i])[None, :] & np.isfinite(ze)
    a = np.where(ok, zf[i][None, :], np.nan); b = np.where(ok, ze, np.nan)
    a = a - np.nanmean(a, 1, keepdims=True); b = b - np.nanmean(b, 1, keepdims=True)
    c = np.nansum(a * b, 1) / np.sqrt(np.nansum(a ** 2, 1) * np.nansum(b ** 2, 1))
    c[ok.sum(1) < 12] = np.nan
    j = int(np.nanargmax(c)) if np.isfinite(c).any() else 0
    mejor_c.append(c[j] if np.isfinite(c).any() else np.nan); mejor_j.append(j)
cit_fp["corr_10k"], cit_fp["fila_10k"] = mejor_c, mejor_j
cit_fp["solapa"] = en_10k_demografico(cit_fp["Gender"], cit_fp["Age"]) | (cit_fp["corr_10k"] >= CORR_SOLAPE)

# Calibración por analito con las muestras compartidas (misma edad, sin sujetos de test)
pares = cit_fp[(cit_fp["corr_10k"] >= CORR_SOLAPE)].copy()
pares["sujeto_10k"] = elisa["subject_accession"].values[pares["fila_10k"]]
pares["edad_10k"] = elisa["age"].values[pares["fila_10k"]]
pares = pares[((pares["Age"] - pares["edad_10k"]).abs() <= TOL_EDAD) & ~pares["sujeto_10k"].isin(ids_test)]
y10 = np.log2(elisa.iloc[pares["fila_10k"].values][cols_cit].astype(float).clip(lower=0).reset_index(drop=True) + 1)
xfp = pares[cols_cit].astype(float).reset_index(drop=True)
calib = []
for c in cols_cit:
    ok = xfp[c].notna() & y10[c].notna()
    if ok.sum() < 15:
        calib.append((c, np.nan, np.nan, int(ok.sum()), np.nan))
        continue
    r = float(np.corrcoef(xfp.loc[ok, c], y10.loc[ok, c])[0, 1])
    b = float(y10.loc[ok, c].std() / xfp.loc[ok, c].std()) * np.sign(r)
    a = float(y10.loc[ok, c].mean() - b * xfp.loc[ok, c].mean())
    calib.append((c, a, b, int(ok.sum()), r))
calib = pd.DataFrame(calib, columns=["variable", "a", "b", "n_pares", "r_pares"])
calib.to_csv(ruta("datos_fluprint", "calibracion_citoquinas_fluprint.csv"), index=False)
utiles = calib[(calib["r_pares"] >= 0.5)]["variable"].tolist()   # analitos con calibración confiable

personas_solapan = set(cit_fp.loc[cit_fp["solapa"], "Person"])
nuevos = cit_fp[~cit_fp["Person"].isin(personas_solapan) & (cit_fp["Age"] >= EDAD_MINIMA)]
nuevos = nuevos.sort_values(["Person", "Year"]).drop_duplicates("Person").copy()
for c in cols_cit:
    fila = calib.set_index("variable").loc[c]
    nuevos[c] = np.clip(2.0 ** (fila["a"] + fila["b"] * nuevos[c].astype(float)) - 1, 0, None) if c in utiles else np.nan
out = nuevos.rename(columns={"Person": "subject_accession", "Age": "age", "Gender": "gender"})
out["subject_accession"] = "FP_" + out["subject_accession"].astype(str)
out["fuente"] = "FluPRINT Luminex"
out[["subject_accession", "age", "gender", "fuente"] + cols_cit].to_csv(os.path.join(dir_cit, "fluprint_luminex.csv"), index=False)
print(f"FluPRINT Luminex: {len(cit_fp)} muestras basales | {cit_fp['Person'].nunique()} personas | "
      f"{len(personas_solapan)} personas posiblemente en el 10K (excluidas) | pares de calibración: {len(pares)}")
print(f"   -> {len(out)} personas nuevas (edad {out['age'].min():.0f}-{out['age'].max():.0f}, mediana {out['age'].median():.0f}) | "
      f"{len(utiles)}/{len(cols_cit)} citoquinas con calibración r>=0.5")

# --- Células FluPRINT (CyTOF) ---
cy = basal[basal["Assay"] == "CyTOF phenotyping"]
par_cy = cy[cy["Units"] == "% of Parent"].pivot_table(index="Sample", columns="Analyte Generic Name", values="N", aggfunc="mean")
via = cy[cy["Units"] == "% of Viable"].pivot_table(index="Sample", columns="Analyte Generic Name", values="N", aggfunc="mean")
f = lambda c: par_cy[c] / 100.0 if c in par_cy.columns else np.nan
lymph, mono = f("lymphocytes"), f("monocytes")
T, nonT = lymph * f("T cells"), lymph * f("non-T lymphocytes")
CD4, CD8 = T * f("CD4+ T cells"), T * f("CD8+ T cells")
B = nonT * f("B cells")
cel_fp = pd.DataFrame({
    "B_cells": B, "CD16_neg_monocytes": mono * f("CD16- monocytes"), "CD16_pos_monocytes": mono * f("CD16+ monocytes"),
    "CD4_T_cells": CD4, "CD8_T_cells": CD8,
    "Central_Memory_CD4_T_cells": CD4 * f("central memory CD4+ T cells"),
    "Central_Memory_CD8_T_cells": CD8 * f("central memory CD8+ T cells"),
    "Effector_CD4_T_cells": CD4 * f("effector CD4+ T cells"), "Effector_CD8_T_cells": CD8 * f("effector CD8+ T cells"),
    "Effector_Memory_CD4_T_cells": CD4 * f("effector memory CD4+ T cells"),
    "Effector_Memory_CD8_T_cells": CD8 * f("effector memory CD8+ T cells"),
    "Gamma_Delta_T_cells": lymph * f("gamma-delta T cells"), "Lymphocytes": lymph,
    "Memory_B_cells": B * f("memory B cells"), "Monocytes": mono, "Naive_B_cells": B * f("naive B cells"),
    "Naive_CD4_T_cells": CD4 * f("naive CD4+ T cells"), "Naive_CD8_T_cells": CD8 * f("naive CD8+ T cells"),
    "NK_cells": nonT * f("NK cells"), "NKT_cells": lymph * f("NKT cells"),
    "Plasmablasts": nonT * f("CD20- CD3- lymphocytes") * f("plasmablasts"), "T_cells": T,
    "Transitional_B_cells": B * f("transitional B cells"), "Tregs": CD4 * f("Tregs"),
}) * 100.0
directos = {"Naive_CD8_T_cells": "naive CD8+ T cells", "Monocytes": "monocytes", "T_cells": "T cells", "B_cells": "B cells",
            "Plasmablasts": "plasmablasts", "Tregs": "Tregs", "Gamma_Delta_T_cells": "gamma-delta T cells", "NKT_cells": "NKT cells",
            "CD4_T_cells": "CD4+ T cells", "CD8_T_cells": "CD8+ T cells", "NK_cells": "NK cells", "Lymphocytes": "lymphocytes"}
for col, nom in directos.items():
    if nom in via.columns:
        idx = via.index[via[nom].notna()].intersection(cel_fp.index)
        cel_fp.loc[idx, col] = via.loc[idx, nom]
cel_fp = meta_fp.join(cel_fp, how="inner").reset_index().dropna(subset=["Age", "Gender"])
personas_solapan_cel = set(cel_fp.loc[en_10k_demografico(cel_fp["Gender"], cel_fp["Age"]), "Person"]) | personas_solapan
nuevos_c = cel_fp[~cel_fp["Person"].isin(personas_solapan_cel) & (cel_fp["Age"] >= EDAD_MINIMA)]
nuevos_c = nuevos_c.sort_values(["Person", "Year"]).drop_duplicates("Person").rename(
    columns={"Person": "subject_accession", "Age": "age", "Gender": "gender"})
# Referencia FluPRINT para alinear: todas sus muestras CyTOF salvo posibles sujetos de test
k10t = k10[k10["subject_accession"].isin(ids_test)]
es_test_fp = np.array([((k10t["gender"] == g) & ((k10t["age"] - a).abs() <= TOL_EDAD)).any()
                       for g, a in zip(cel_fp["Gender"], cel_fp["Age"])])
ref_fp = cel_fp[~es_test_fp].rename(columns={"Age": "age", "Gender": "gender"})
alin = alinear_combat(pd.concat([ref_fp.assign(_nuevo=False), nuevos_c.assign(_nuevo=True)], ignore_index=True), cols_cel, 0.1)
nuevos_c = alin[alin["_nuevo"]].copy()
nuevos_c["subject_accession"] = "FP_" + nuevos_c["subject_accession"].astype(str)
nuevos_c["fuente"] = "FluPRINT CyTOF"
nuevos_c[["subject_accession", "age", "gender", "fuente"] + cols_cel].to_csv(os.path.join(dir_cel, "fluprint_cytof.csv"), index=False)
print(f"FluPRINT CyTOF: {len(cel_fp)} muestras basales -> {len(nuevos_c)} personas nuevas "
      f"(edad {nuevos_c['age'].min():.0f}-{nuevos_c['age'].max():.0f}, mediana {nuevos_c['age'].median():.0f})")

# ------------------------------------------------------------------------------
# Sound Life (ya alineado por armonizar_soundlife.py) -> 1 visita basal por persona
# ------------------------------------------------------------------------------
sl = pd.read_csv(ruta("datos_soundlife", "soundlife_armonizado.csv"))
orden = {"Flu Year 1 Day 0": 0, "Immune Variation Day 0": 1, "Flu Year 2 Day 0": 2}
sl = sl.assign(_o=sl["visita"].map(orden).fillna(9)).sort_values(["subject_accession", "_o"]).drop_duplicates("subject_accession")
for c in cols_cel + cols_cit:
    if c not in sl.columns:
        sl[c] = np.nan
sl.assign(fuente="Sound Life single-cell")[["subject_accession", "age", "gender", "fuente"] + cols_cel].to_csv(
    os.path.join(dir_cel, "soundlife_scrna.csv"), index=False)
sl.assign(fuente="Sound Life Olink")[["subject_accession", "age", "gender", "fuente"] + cols_cit].to_csv(
    os.path.join(dir_cit, "soundlife_olink.csv"), index=False)
print(f"Sound Life: {len(sl)} personas (células y citoquinas)")
