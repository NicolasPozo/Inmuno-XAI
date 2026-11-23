# ==============================================================================
# ARMONIZACIÓN DE SOUND LIFE (ALLEN INSTITUTE) A LAS VARIABLES DEL MODELO
# Proyecto: Immuno-XAI - UTEM
#
# Fuente (acceso abierto): Allen Institute for Immunology, "Dynamics of Immune
# Health and Age" (Gong et al., Nature 2025). Cohorte Sound Life (Seattle):
# 96 adultos sanos de 25-36 y 55-67 años. Tecnologías DISTINTAS a tu dataset:
#   - Células: proporciones por single-cell RNA-seq (AIFI L2/L3), no CyTOF
#   - Citoquinas: Olink Explore 1536 (NPX, log2), no Luminex
#
# Pasos:
#   1. Visitas basales (día 0 y visitas sueltas; se descartan día 7 y 90 post-vacuna)
#   2. Células -> % de PBMC (sin plaquetas ni eritrocitos), mapeadas a tus nombres
#   3. Citoquinas -> NPX de Olink con tus nombres (IL1RN de Olink = "IL1R1" del 10K)
#   4. Alineación a la escala de tu train (tipo ComBat, ajustando por edad y sexo):
#      log2 de cada variable; se iguala la dispersión residual y se corrige el
#      desplazamiento respecto a la curva de edad de tu train.
#
# Genera: datos_soundlife/soundlife_armonizado.csv
#
# NOTA: los CSV crudos (sound_life_*.csv, ~300 MB) se borraron para ahorrar
# espacio. Su resultado, soundlife_armonizado.csv, sigue en Datasets/soundlife/.
# Para volver a ejecutar este script hay que descargarlos de nuevo desde el
# Allen Institute (Human Immune Health Atlas / Sound Life).
# ==============================================================================

import os
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
DIR_SL = ruta("datos_soundlife")

ref = pd.read_csv(ruta("df_train_reales_completo.csv"))     # tu train (referencia de escala)
ref45 = pd.read_csv(ruta("df_train_reales_k45.csv"))
feats45 = [c for c in ref45.columns if c not in ("subject_accession", "gender", "source_type", "age")]
ref[feats45] = ref45[feats45].values

# ------------------------------------------------------------------------------
# 1-2. Células (fracciones de single-cell)
# ------------------------------------------------------------------------------
def tabla_frac(nivel):
    f = pd.read_csv(os.path.join(DIR_SL, f"sound_life_AIFI_{nivel}_frequencies.csv"))
    meta = f.drop_duplicates("sample.sampleKitGuid").set_index("sample.sampleKitGuid")[
        ["subject.subjectGuid", "subject.biologicalSex", "sample.subjectAgeAtDraw", "sample.visitName"]]
    p = f.pivot_table(index="sample.sampleKitGuid", columns=f"AIFI_{nivel}", values=f"AIFI_{nivel}_count", aggfunc="sum").fillna(0)
    excluir = [c for c in p.columns if c in ("Platelet", "Erythrocyte")]
    p = p.drop(columns=excluir)
    return p.div(p.sum(axis=1), axis=0) * 100.0, meta

L2, meta = tabla_frac("L2")
L3, _ = tabla_frac("L3")
L3 = L3.reindex(L2.index).fillna(0)
s2 = lambda *cols: sum(L2[c] for c in cols if c in L2.columns)
s3 = lambda *cols: sum(L3[c] for c in cols if c in L3.columns)

tregs_cd4 = s3("Naive CD4 Treg", "Memory CD4 Treg", "GZMK+ memory CD4 Treg", "KLRB1+ memory CD4 Treg")
cd4 = s2("Naive CD4 T cell", "Memory CD4 T cell") + tregs_cd4 + s3("CD4 MAIT")
cd8 = s2("Naive CD8 T cell", "Memory CD8 T cell", "CD8aa") + s3("Memory CD8 Treg", "KLRB1+ memory CD8 Treg", "CD8 MAIT")
T = s2("Naive CD4 T cell", "Memory CD4 T cell", "Treg", "Naive CD8 T cell", "Memory CD8 T cell", "MAIT", "gdT",
       "DN T cell", "CD8aa", "Proliferating T cell")
B = s2("Naive B cell", "Memory B cell", "Effector B cell", "Transitional B cell")
NK = s2("CD56bright NK cell", "CD56dim NK cell", "Proliferating NK cell")
celulas = pd.DataFrame({
    "Naive_CD8_T_cells": s2("Naive CD8 T cell"),
    "Central_Memory_CD8_T_cells": s3("CM CD8 T cell"),
    "Monocytes": s2("CD14 monocyte", "CD16 monocyte", "Intermediate monocyte"),
    "NKT_cells": np.nan,   # single-cell no define células NKT (CD3+CD56+) -> faltante
    "Effector_CD4_T_cells": s3("KLRF1- GZMB+ CD27- memory CD4 T cell"),
    "B_cells": B,
    "CD16_neg_monocytes": s2("CD14 monocyte"),
    "Tregs": tregs_cd4,
    "Central_Memory_CD4_T_cells": s3("CM CD4 T cell"),
    "CD16_pos_monocytes": s2("CD16 monocyte", "Intermediate monocyte"),
    "Naive_CD4_T_cells": s2("Naive CD4 T cell"),
    "Effector_CD8_T_cells": s3("KLRF1+ GZMB+ CD27- EM CD8 T cell", "KLRF1- GZMB+ CD27- EM CD8 T cell"),
    "Plasmablasts": s2("Plasma cell"),
    "Gamma_Delta_T_cells": s2("gdT"),
    "CD4_T_cells": cd4,
    "T_cells": T,
    # variables extra del conjunto de 71
    "CD8_T_cells": cd8,
    "Effector_Memory_CD4_T_cells": s3("GZMB- CD27+ EM CD4 T cell", "GZMB- CD27- EM CD4 T cell"),
    "Effector_Memory_CD8_T_cells": s3("GZMK+ CD27+ EM CD8 T cell", "GZMK- CD27+ EM CD8 T cell"),
    "Lymphocytes": T + B + NK + s2("ILC", "Plasma cell"),
    "Memory_B_cells": s2("Memory B cell"),
    "Naive_B_cells": s2("Naive B cell"),
    "Transitional_B_cells": s2("Transitional B cell"),
    "NK_cells": NK,
}, index=L2.index)

# ------------------------------------------------------------------------------
# 3. Citoquinas (Olink NPX normalizado)
# ------------------------------------------------------------------------------
mapa_olink = {c: c for c in ["IL6", "VCAM1", "HGF", "CXCL8", "IFNG", "CXCL9", "IL13", "NGF", "ICAM1", "FGF2", "FASLG",
                              "CCL3", "IL15", "LEP", "CCL7", "CCL4", "TGFB1", "IL2", "RETN", "CD40LG", "CXCL5", "IL4",
                              "TNF", "IL5", "CCL2", "CSF1", "CCL11", "CCL5", "CXCL10", "SERPINE1", "CXCL1", "KITLG",
                              "IL7", "LIF", "TNFSF10", "VEGFA", "IL1A", "CSF3", "IL12B", "IL1B", "IL17F", "IL17A", "TGFA"]}
mapa_olink["IL1RN"] = "IL1R1"   # el "IL1R1" del 10K corresponde al antagonista IL-1Ra (IL1RN)
olink = pd.read_csv(os.path.join(DIR_SL, "sound_life_all_olink.csv"),
                    usecols=["sample.sampleKitGuid", "olink.assay", "olink.NPX_norm"])
olink = olink[olink["olink.assay"].isin(mapa_olink)]
cit = olink.pivot_table(index="sample.sampleKitGuid", columns="olink.assay", values="olink.NPX_norm", aggfunc="mean")
cit = cit.rename(columns=mapa_olink)

# ------------------------------------------------------------------------------
# Unión y visitas basales
# ------------------------------------------------------------------------------
sl = meta.join(celulas, how="inner").join(cit, how="inner")
sl = sl[sl["sample.visitName"].str.contains("Day 0|Stand-Alone", regex=True)].copy()
sl = sl.rename(columns={"subject.subjectGuid": "subject_accession", "subject.biologicalSex": "gender",
                        "sample.subjectAgeAtDraw": "age", "sample.visitName": "visita"})
sl["subject_accession"] = "SL_" + sl["subject_accession"].astype(str)
print(f"Sound Life: {len(sl)} visitas basales de {sl['subject_accession'].nunique()} personas | "
      f"edad {sl['age'].min()}-{sl['age'].max()} | sexo {sl['gender'].value_counts().to_dict()}")

# ------------------------------------------------------------------------------
# 4. Alineación a la escala de tu train (log2; referencia = tu train)
# ------------------------------------------------------------------------------
def diseno(edad, male):
    e = (np.asarray(edad, float) - 50.0) / 20.0
    return np.column_stack([np.ones_like(e), e, e ** 2, np.asarray(male, float)])

cols_cel = [c for c in celulas.columns if c in ref.columns]
cols_cit = [c for c in cit.columns if c in ref.columns]
Xr = diseno(ref["age"], ref["gender"] == "Male")
Xn = diseno(sl["age"], sl["gender"] == "Male")
resumen = []
for c in cols_cel + cols_cit:
    es_cel = c in cols_cel
    yr = np.log2(ref[c].astype(float).clip(lower=0) + (0.1 if es_cel else 1.0)).values
    yn = (np.log2(sl[c].astype(float).clip(lower=0) + 0.1) if es_cel else sl[c].astype(float)).values  # Olink ya es log2
    okr, okn = np.isfinite(yr), np.isfinite(yn)
    if okn.sum() < 20:
        sl[c] = np.nan
        continue
    br = np.linalg.lstsq(Xr[okr], yr[okr], rcond=None)[0]
    bn = np.linalg.lstsq(Xn[okn], yn[okn], rcond=None)[0]
    escala = np.std(yr[okr] - Xr[okr] @ br) / np.std(yn[okn] - Xn[okn] @ bn)
    desplaz = np.mean(escala * yn[okn] - Xn[okn] @ br)
    y_adj = escala * yn - desplaz
    sl[c] = np.clip(2.0 ** y_adj - (0.1 if es_cel else 1.0), 0, None)
    # correlación con la edad antes (Sound Life) y en tu train, para ver si la señal coincide
    r_sl = np.corrcoef(yn[okn], sl["age"].values[okn])[0, 1]
    r_ref = np.corrcoef(yr[okr], ref["age"].values[okr])[0, 1]
    resumen.append({"variable": c, "tipo": "célula" if es_cel else "citoquina", "escala": round(escala, 3),
                    "r_edad_soundlife": round(r_sl, 2), "r_edad_tu_train": round(r_ref, 2)})

res = pd.DataFrame(resumen)
res.to_csv(os.path.join(DIR_SL, "alineacion_soundlife.csv"), index=False)
mismo_signo = (np.sign(res["r_edad_soundlife"]) == np.sign(res["r_edad_tu_train"]))
fuertes = res[(res["r_edad_tu_train"].abs() >= 0.2)]
print(f"Variables armonizadas: {len(cols_cel)} células + {len(cols_cit)} citoquinas "
      f"(faltan: NKT_cells, IFN1., IFNB1, CSF2)")
print(f"Relación con la edad del mismo signo en ambas cohortes: {mismo_signo.mean()*100:.0f}% de las variables "
      f"({(np.sign(fuertes['r_edad_soundlife']) == np.sign(fuertes['r_edad_tu_train'])).mean()*100:.0f}% entre las que "
      f"se asocian con la edad en tu train, |r|>=0.2)")
print(fuertes.sort_values("r_edad_tu_train").to_string(index=False))

columnas = ["subject_accession", "gender", "age", "visita"] + [c for c in ref.columns if c in sl.columns and c not in
                                                              ("subject_accession", "gender", "age")]
for c in feats45:
    if c not in sl.columns:
        sl[c] = np.nan
        columnas.append(c)
sl[columnas].to_csv(os.path.join(DIR_SL, "soundlife_armonizado.csv"), index=False)
print(f"\nGuardado: datos_soundlife/soundlife_armonizado.csv ({len(sl)} filas)")
