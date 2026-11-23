# ==============================================================================
# FUENTES NUEVAS DESDE IMMPORT (estudios de Stanford no incluidos en el 10K)
# Proyecto: Immuno-XAI - UTEM
#
# Lee los archivos *_Tab.zip descomprimidos en datos_immport/extraido/ y genera:
#   fuentes_immport/citoquinas_immport.csv  (Luminex en suero, día 0)
#   fuentes_immport/celulas_immport.csv     (CyTOF, día 0; solo estudios con
#                                            poblaciones calculadas por el laboratorio)
# Reglas:
#   - Se descarta cualquier persona cuyo ID (SUB...) esté en el 10K o en tu test.
#   - Solo visita basal (día 0, antes de vacunar) y personas de 11 años o más.
#   - Si una persona aparece en varios estudios, se usa una sola vez.
#   - Alineación a la escala de tu train estudio por estudio (log2; tipo ComBat
#     con edad y sexo), igual que con las otras fuentes externas.
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
DIR_IMM = ruta("datos_immport", "extraido")
DIR_10K = ruta("datos_10k", "extraido")
salida = ruta("fuentes_immport")
os.makedirs(salida, exist_ok=True)
EDAD_MINIMA = 11.0

par = pd.read_csv(ruta("pareados.csv"))
ref = par[par["conjunto"] == "train"].reset_index(drop=True)
k10 = set(pd.read_csv(os.path.join(DIR_10K, "Multiplex_ELISA", "Multiplex_ELISA", "Normalized_Files", "multiplexELISA_normalized.csv")).subject_accession) \
    | set(pd.read_csv(os.path.join(DIR_10K, "CyTOF_PBMC", "CyTOF_PBMC", "cytof_pbmc_normalized.csv")).subject_accession) \
    | set(pd.read_csv(os.path.join(DIR_10K, "Flow_Cytometry_PBMC", "Flow_Cytometry_PBMC", "Normalized_Files", "flow_pbmc_normalized.csv")).subject_accession) \
    | set(par.subject_accession)

cols_cel = ["B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells",
            "Central_Memory_CD4_T_cells", "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells",
            "Effector_Memory_CD4_T_cells", "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes",
            "Memory_B_cells", "Monocytes", "Naive_B_cells", "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells",
            "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"]
todas_cit = [c for c in par.columns if c not in cols_cel + ["subject_accession", "gender", "age", "conjunto"]]
cols_cit = [c for c in todas_cit if ref[c].notna().mean() >= 0.8]

MAPA = {"IL6": "IL6", "VCAM1": "VCAM1", "VCAM-1": "VCAM1", "HGF": "HGF", "IL8": "CXCL8", "IL-8": "CXCL8", "IFNA": "IFN1.",
        "IFNA2": "IFN1.", "IFNG": "IFNG", "IL1RA": "IL1R1", "MIG": "CXCL9", "IL13": "IL13", "NGF": "NGF", "ICAM1": "ICAM1",
        "ICAM-1": "ICAM1", "FGFB": "FGF2", "FASL": "FASLG", "MIP1A": "CCL3", "IL15": "IL15", "LEPTIN": "LEP", "MCP3": "CCL7",
        "MIP1B": "CCL4", "TGFB": "TGFB1", "TGFB1": "TGFB1", "IL2": "IL2", "IFNB": "IFNB1", "RESISTIN": "RETN", "CD40L": "CD40LG",
        "ENA78": "CXCL5", "IL4": "IL4", "TNFA": "TNF", "IL5": "IL5", "MCP1": "CCL2", "MCSF": "CSF1", "EOTAXIN": "CCL11",
        "RANTES": "CCL5", "IP10": "CXCL10", "PAI1": "SERPINE1", "GROA": "CXCL1", "SCF": "KITLG", "IL7": "IL7", "LIF": "LIF",
        "TRAIL": "TNFSF10", "GMCSF": "CSF2", "VEGF": "VEGFA", "IL1A": "IL1A", "GCSF": "CSF3", "IL12P40": "IL12B",
        "IL1B": "IL1B", "IL17F": "IL17F", "IL17A": "IL17A", "IL17": "IL17A", "TGFA": "TGFA"}
norm = lambda s: str(s).upper().replace(" ", "").replace("_", "").replace("Α", "A").replace("Β", "B")

def diseno(edad, male):
    e = (np.asarray(edad, float) - 50.0) / 20.0
    return np.column_stack([np.ones_like(e), e, e ** 2, np.asarray(male, float)])

def alinear(nuevo, cols, offset):
    """Escala de tu train: log2, misma dispersión residual y desplazamiento respecto a la curva edad/sexo."""
    out = nuevo.copy()
    Xr, Xn = diseno(ref["age"], ref["gender"] == "Male"), diseno(nuevo["age"], nuevo["gender"] == "Male")
    for c in cols:
        yr = np.log2(ref[c].astype(float).clip(lower=0) + offset).values
        yn = np.log2(nuevo[c].astype(float).clip(lower=0) + offset).values
        okr, okn = np.isfinite(yr), np.isfinite(yn)
        if okn.sum() < 15:
            out[c] = np.nan
            continue
        br = np.linalg.lstsq(Xr[okr], yr[okr], rcond=None)[0]
        bn = np.linalg.lstsq(Xn[okn], yn[okn], rcond=None)[0]
        esc = np.std(yr[okr] - Xr[okr] @ br) / max(np.std(yn[okn] - Xn[okn] @ bn), 1e-6)
        desp = np.mean(esc * yn[okn] - Xn[okn] @ br)
        out[c] = np.clip(2.0 ** (esc * yn - desp) - offset, 0, None)
    return out

def leer(carpeta, nombre):
    p = glob.glob(os.path.join(carpeta, "**", nombre), recursive=True)
    return pd.read_csv(p[0], sep="\t", low_memory=False) if p else None

# ------------------------------------------------------------------------------
# Lectura por estudio
# ------------------------------------------------------------------------------
filas_cit, filas_cel = [], []
for carpeta in sorted(glob.glob(os.path.join(DIR_IMM, "*"))):
    sdy = os.path.basename(carpeta).split("-")[0]
    subj = leer(carpeta, "subject.txt")
    a2s = leer(carpeta, "arm_2_subject.txt")
    edad = a2s.groupby("SUBJECT_ACCESSION")["MIN_SUBJECT_AGE_IN_YEARS"].min()
    sexo = subj.set_index("SUBJECT_ACCESSION")["GENDER"]

    # --- Citoquinas (Luminex, suero, día 0) ---
    mb = leer(carpeta, "mbaa_result.txt")
    bs = leer(carpeta, "biosample.txt")[["BIOSAMPLE_ACCESSION", "TYPE"]]
    if mb is not None:
        mb = mb.merge(bs, on="BIOSAMPLE_ACCESSION", how="left")
        mb = mb[(pd.to_numeric(mb["STUDY_TIME_COLLECTED"], errors="coerce") == 0) & mb["TYPE"].isin(["Serum", "Plasma"])]
        mb["var"] = mb["ANALYTE_REPORTED"].map(lambda a: MAPA.get(norm(a)))
        mb = mb[mb["var"].isin(cols_cit)]
        mb["valor"] = pd.to_numeric(mb["CONCENTRATION_VALUE_PREFERRED"], errors="coerce")
        t = mb.pivot_table(index="SUBJECT_ACCESSION", columns="var", values="valor", aggfunc="median")
        t["estudio"] = sdy
        filas_cit.append(t.reset_index())
        print(f"{sdy}: citoquinas día 0 en suero para {len(t)} personas | {t.shape[1]-2} de {len(cols_cit)} citoquinas")

    # --- Células (CyTOF con poblaciones calculadas, día 0) ---
    fcs = leer(carpeta, "fcs_analyzed_result.txt")
    if fcs is not None:
        fcs = fcs[pd.to_numeric(fcs["STUDY_TIME_COLLECTED"], errors="coerce") == 0]
        fcs["valor"] = pd.to_numeric(fcs["POPULATION_STATISTIC_REPORTED"], errors="coerce") / 100.0
        p = fcs.pivot_table(index="SUBJECT_ACCESSION", columns="POPULATION_NAME_REPORTED", values="valor", aggfunc="mean")
        def f(c):
            # Acepta el nombre en plural (FluPRINT) o en singular (ImmPort): "T cells" / "T cell"
            for cand in (c, c[:-1] if c.endswith("s") else c + "s", c.replace("cells", "cell")):
                if cand in p.columns:
                    return p[cand]
            return np.nan
        lymph, mono = f("lymphocytes"), f("monocytes")
        T, nonT = lymph * f("T cells"), lymph * f("non-T lymphocytes")
        CD4, CD8 = T * f("CD4+ T cells"), T * f("CD8+ T cells")
        B = nonT * f("B cells")
        cel = pd.DataFrame({
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
        cel["estudio"] = sdy
        filas_cel.append(cel.reset_index().rename(columns={"index": "SUBJECT_ACCESSION"}))
        print(f"{sdy}: células día 0 para {len(cel)} personas | poblaciones con datos: {int(cel[cols_cel].notna().any().sum())}/24")

    for lista in (filas_cit, filas_cel):
        if lista and lista[-1]["estudio"].iloc[0] == sdy:
            d = lista[-1]
            d["age"] = d["SUBJECT_ACCESSION"].map(edad).astype(float)
            d["gender"] = d["SUBJECT_ACCESSION"].map(sexo)

def finalizar(filas, cols, offset, nombre):
    if not filas:
        print(f"{nombre}: sin datos")
        return
    d = pd.concat(filas, ignore_index=True).rename(columns={"SUBJECT_ACCESSION": "subject_accession"})
    n0 = d["subject_accession"].nunique()
    d = d[~d["subject_accession"].isin(k10)]
    d = d[(d["age"] >= EDAD_MINIMA) & d["gender"].isin(["Male", "Female"])]
    d = d.drop_duplicates("subject_accession")
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    alineados = [alinear(g, cols, offset) for _, g in d.groupby("estudio")]
    d = pd.concat(alineados, ignore_index=True)
    d["fuente"] = "ImmPort " + d["estudio"]
    d[["subject_accession", "age", "gender", "fuente", "estudio"] + cols].to_csv(os.path.join(salida, nombre), index=False)
    print(f"{nombre}: {n0} personas -> {len(d)} nuevas (no en el 10K, >= {EDAD_MINIMA:.0f} años) | "
          f"edad {d['age'].min():.0f}-{d['age'].max():.0f}, mediana {d['age'].median():.0f} | por estudio {d['estudio'].value_counts().to_dict()}")

finalizar(filas_cit, cols_cit, 1.0, "citoquinas_immport.csv")
finalizar(filas_cel, cols_cel, 0.1, "celulas_immport.csv")
