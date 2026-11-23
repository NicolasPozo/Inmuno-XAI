# ==============================================================================
# EXPLICABILIDAD DEL MODELO FINAL (TabPFN un modelo, 45 biomarcadores + sexo)
# Proyecto: Immuno-XAI - UTEM
#
# 1. Importancia por permutación (test de 119): cuánto sube el MAE si se
#    desordena una variable entre pacientes (5 repeticiones).
# 2. Importancia por bloque: células, citoquinas y sexo.
# 3. Perfiles de dependencia parcial: edad predicha media al mover un
#    biomarcador por su rango (percentiles 5-95 del train), con el resto fijo.
# 4. Explicación individual: para cada paciente de test, cuánto cambia su edad
#    predicha si un biomarcador se reemplaza por la mediana del train.
# 5. Brecha de edad inmunológica (edad predicha - edad real) en los 473
#    pacientes (predicciones fuera de muestra de la CV 5x5), corregida por edad,
#    y su relación con cada biomarcador (Spearman + corrección de Benjamini-Hochberg).
#
# Requiere haber ejecutado src/06_modelo_final_tabpfn.py
# Salida: resultados/explicabilidad/
# ==============================================================================

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
from config import (
    ARCHIVO_ELISA_10K,
    ARCHIVO_MODELO_FINAL,
    ARCHIVO_PAREADOS,
    CARPETA_RES_EXPLICABILIDAD,
    CARPETA_RES_MODELO_FINAL,
)  # rutas del proyecto: src/config.py
carpeta = CARPETA_RES_EXPLICABILIDAD
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)
REP = 5

# ------------------------------------------------------------------------------
# Datos y modelo
# ------------------------------------------------------------------------------
paquete = joblib.load(ARCHIVO_MODELO_FINAL)
modelo = paquete["modelo"]
VARIABLES = paquete["variables"][:-1]
NOMBRES = VARIABLES + ["Sexo"]
CELULAS = {"B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells",
           "Central_Memory_CD4_T_cells", "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells",
           "Effector_Memory_CD4_T_cells", "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes",
           "Memory_B_cells", "Monocytes", "Naive_B_cells", "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells",
           "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"}
tipo = lambda v: "Sexo" if v == "Sexo" else ("Célula" if v in CELULAS else "Citoquina")

par = pd.read_csv(ARCHIVO_PAREADOS)
el = pd.read_csv(ARCHIVO_ELISA_10K)
k = el.assign(edad=el.age.round(2))[["subject_accession", "edad", "study_accession"]].drop_duplicates(["subject_accession", "edad"])
par = par.assign(edad=par.age.round(2)).merge(k, on=["subject_accession", "edad"], how="left")
par["study_accession"] = par["study_accession"].fillna("otro")
X = np.hstack([par[VARIABLES].astype(float).values, (par["gender"] == "Male").astype(float).values.reshape(-1, 1)]).astype(np.float32)
y = par["age"].values.astype(float)
idx_tr = np.where(par["conjunto"] == "train")[0]
idx_te = np.where(par["conjunto"] == "test")[0]
Xte, yte = X[idx_te], y[idx_te]
mediana_tr = np.nanmedian(X[idx_tr], axis=0)
nt = len(idx_te)

def predecir(bloques, tam=2380):
    """Predice muchas matrices de una vez (TabPFN tiene un costo fijo por llamada)."""
    Z = np.vstack(bloques)
    out = np.concatenate([modelo.predict(Z[i:i + tam]) for i in range(0, len(Z), tam)])
    return np.split(out, np.cumsum([len(b) for b in bloques])[:-1])

p0 = modelo.predict(Xte)
mae0 = np.mean(np.abs(p0 - yte))
log(f"MAE test base {mae0:.3f}")

# ------------------------------------------------------------------------------
# 1 y 2. Importancia por permutación (variables y bloques)
# ------------------------------------------------------------------------------
rng = np.random.default_rng(0)
grupos_bloque = {"Células (16)": [j for j, v in enumerate(NOMBRES) if tipo(v) == "Célula"],
                 "Citoquinas (29)": [j for j, v in enumerate(NOMBRES) if tipo(v) == "Citoquina"],
                 "Sexo": [len(NOMBRES) - 1]}
tareas, bloques = [], []
for j in range(X.shape[1]):
    for r in range(REP):
        Z = Xte.copy(); Z[:, j] = Z[rng.permutation(nt), j]
        tareas.append(("var", j, r)); bloques.append(Z)
for g, cols in grupos_bloque.items():
    for r in range(REP):
        Z = Xte.copy(); perm = rng.permutation(nt); Z[:, cols] = Z[perm][:, cols]
        tareas.append(("bloque", g, r)); bloques.append(Z)
preds = predecir(bloques)
log("Permutaciones listas")
inc = {}
for (t, key, r), p in zip(tareas, preds):
    inc.setdefault((t, key), []).append(np.mean(np.abs(p - yte)) - mae0)
imp = pd.DataFrame([{"variable": NOMBRES[j], "tipo": tipo(NOMBRES[j]),
                     "aumento_MAE": np.mean(inc[("var", j)]), "aumento_MAE_sd": np.std(inc[("var", j)])}
                    for j in range(X.shape[1])]).sort_values("aumento_MAE", ascending=False)
imp_bloque = pd.DataFrame([{"bloque": g, "aumento_MAE": np.mean(inc[("bloque", g)]), "aumento_MAE_sd": np.std(inc[("bloque", g)])}
                           for g in grupos_bloque])
imp.round(4).to_csv(os.path.join(carpeta, "importancia_permutacion.csv"), index=False)
imp_bloque.round(4).to_csv(os.path.join(carpeta, "importancia_bloques.csv"), index=False)

# ------------------------------------------------------------------------------
# 3. Perfiles de dependencia parcial (top 10 biomarcadores, sin el sexo)
# ------------------------------------------------------------------------------
top = [v for v in imp["variable"] if v != "Sexo"][:10]
tareas, bloques = [], []
for v in top:
    j = NOMBRES.index(v)
    grid = np.unique(np.nanpercentile(X[idx_tr, j], np.linspace(5, 95, 12)))
    for g in grid:
        Z = Xte.copy(); Z[:, j] = g
        tareas.append((v, g)); bloques.append(Z)
preds = predecir(bloques)
pdp = pd.DataFrame([{"variable": v, "valor": g, "edad_predicha_media": p.mean()} for (v, g), p in zip(tareas, preds)])
pdp.round(4).to_csv(os.path.join(carpeta, "dependencia_parcial.csv"), index=False)
log("Perfiles listos")

# ------------------------------------------------------------------------------
# 4. Explicación individual (reemplazo por la mediana del train)
# ------------------------------------------------------------------------------
bloques = []
for j in range(X.shape[1]):
    Z = Xte.copy(); Z[:, j] = mediana_tr[j]; bloques.append(Z)
preds = predecir(bloques)
contrib = np.column_stack([p0 - p for p in preds])  # + = ese valor empuja la edad predicha hacia arriba
pd.DataFrame(contrib, columns=NOMBRES).assign(subject_accession=par["subject_accession"].values[idx_te]).round(3).to_csv(
    os.path.join(carpeta, "explicacion_individual_test.csv"), index=False)
log("Explicaciones individuales listas")

# ------------------------------------------------------------------------------
# 5. Brecha de edad inmunológica (473 pacientes, predicciones fuera de muestra)
# ------------------------------------------------------------------------------
cv = np.load(os.path.join(CARPETA_RES_MODELO_FINAL, "predicciones_cv.npz"), allow_pickle=True)
assert (cv["grupos"] == par["subject_accession"].values).all()
pred_cv = cv["oof"].mean(axis=0)
brecha = pred_cv - y
a, b = np.polyfit(y, brecha, 1)
brecha_corr = brecha - (a * y + b)
brechas = pd.DataFrame({"subject_accession": par["subject_accession"], "conjunto": par["conjunto"],
                        "estudio": par["study_accession"], "sexo": par["gender"], "edad_real": y,
                        "edad_predicha": pred_cv, "brecha": brecha, "brecha_corregida": brecha_corr})
brechas.round(3).to_csv(os.path.join(carpeta, "brecha_edad_473.csv"), index=False)

filas = []
for j, v in enumerate(VARIABLES):
    ok = ~np.isnan(X[:, j])
    rho, pv = spearmanr(X[ok, j], brecha_corr[ok])
    rho_e, _ = spearmanr(X[ok, j], y[ok])
    filas.append({"variable": v, "tipo": tipo(v), "rho_con_brecha": rho, "p": pv, "rho_con_edad": rho_e, "n": ok.sum()})
cor = pd.DataFrame(filas).sort_values("p")
m = len(cor)
q = cor["p"].values * m / np.arange(1, m + 1)
cor["q_BH"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
cor.round(4).to_csv(os.path.join(carpeta, "brecha_vs_biomarcadores.csv"), index=False)
p_sexo = mannwhitneyu(brecha_corr[par.gender == "Male"], brecha_corr[par.gender != "Male"]).pvalue
resumen_brecha = {
    "pendiente_brecha_vs_edad": a, "intercepto": b,
    "brecha_corr_sd": float(np.std(brecha_corr)),
    "brecha_corr_hombres": float(brecha_corr[par.gender == "Male"].mean()),
    "brecha_corr_mujeres": float(brecha_corr[par.gender != "Male"].mean()),
    "p_sexo_mannwhitney": float(p_sexo),
}
log(f"Brecha: pendiente {a:.3f} | hombres {resumen_brecha['brecha_corr_hombres']:.2f} | "
    f"mujeres {resumen_brecha['brecha_corr_mujeres']:.2f} | p {p_sexo:.3f}")

# ------------------------------------------------------------------------------
# Datos para el dashboard
# ------------------------------------------------------------------------------
met = pd.read_csv(os.path.join(CARPETA_RES_MODELO_FINAL, "metricas_modelo_final.csv"))
loso = pd.read_csv(os.path.join(CARPETA_RES_MODELO_FINAL, "metricas_estudios_no_vistos.csv"))
te = brechas.iloc[idx_te].reset_index(drop=True)
datos = {
    "metricas": met.round(4).to_dict("records"),
    "estudios_no_vistos": loso.round(4).to_dict("records"),
    "importancia": imp.round(4).to_dict("records"),
    "importancia_bloques": imp_bloque.round(4).to_dict("records"),
    "dependencia_parcial": {v: pdp[pdp.variable == v][["valor", "edad_predicha_media"]].round(4).values.tolist() for v in top},
    "test": [{"id": r.subject_accession, "edad": round(r.edad_real, 2), "pred_test": round(float(p0[i]), 2),
              "brecha_test": round(float(p0[i] - r.edad_real), 2), "sexo": r.sexo, "estudio": r.estudio,
              "contrib": [round(float(c), 3) for c in contrib[i]]} for i, r in te.iterrows()],
    "nombres": NOMBRES, "tipos": [tipo(v) for v in NOMBRES],
    "brecha_473": brechas[["edad_real", "edad_predicha", "brecha", "brecha_corregida", "sexo", "estudio", "conjunto"]].round(2).to_dict("records"),
    "brecha_vs_biomarcadores": cor.round(4).to_dict("records"),
    "resumen_brecha": {k_: round(v_, 4) for k_, v_ in resumen_brecha.items()},
    "mae_test_base": round(float(mae0), 4),
}
json.dump(datos, open(os.path.join(carpeta, "datos_dashboard.json"), "w", encoding="utf-8"), ensure_ascii=False)

pd.set_option("display.width", 200)
print("\n=== IMPORTANCIA (top 15) ===\n" + imp.head(15).round(3).to_string(index=False))
print("\n=== BLOQUES ===\n" + imp_bloque.round(3).to_string(index=False))
print("\n=== BRECHA vs BIOMARCADORES (top 10) ===\n" + cor.head(10).round(4).to_string(index=False))
log("Listo.")
