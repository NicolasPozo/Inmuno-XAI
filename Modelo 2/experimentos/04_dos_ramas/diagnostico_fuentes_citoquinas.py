import os, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
par = pd.read_csv(ruta("pareados.csv"))
ext = pd.read_csv(ruta("fuentes_citoquinas", "10k_elisa.csv"))
el = pd.read_csv(ruta("datos_10k", "extraido", "Multiplex_ELISA", "Multiplex_ELISA", "Normalized_Files", "multiplexELISA_normalized.csv"))
cols_cel = ["B_cells","CD16_neg_monocytes","CD16_pos_monocytes","CD4_T_cells","CD8_T_cells","Central_Memory_CD4_T_cells","Central_Memory_CD8_T_cells","Effector_CD4_T_cells","Effector_CD8_T_cells","Effector_Memory_CD4_T_cells","Effector_Memory_CD8_T_cells","Gamma_Delta_T_cells","Lymphocytes","Memory_B_cells","Monocytes","Naive_B_cells","Naive_CD4_T_cells","Naive_CD8_T_cells","NK_cells","NKT_cells","Plasmablasts","T_cells","Transitional_B_cells","Tregs"]
todas = [c for c in par.columns if c not in cols_cel + ["subject_accession","gender","age","conjunto"]]
cob = par.loc[par.conjunto=="train", todas].notna().mean(); cit = [c for c in todas if cob[c] >= 0.8]
ext["estudio"] = ext["fuente"].str.extract(r"\((SDY\d+)\)")[0]
# estudio de los pareados (por sujeto+edad en el ELISA)
k = el.assign(edad=el.age.round(2))[["subject_accession","edad","study_accession"]].drop_duplicates(["subject_accession","edad"])
par = par.assign(edad=par.age.round(2)).merge(k, on=["subject_accession","edad"], how="left")
print("Estudios de los pareados:", par.study_accession.value_counts().to_dict())
print("\nCitoquinas extra por estudio:")
res = ext.groupby("estudio").agg(filas=("age","size"), personas=("subject_accession","nunique"),
      edad_min=("age","min"), edad_med=("age","median"), edad_max=("age","max"))
res["cobertura_cit"] = ext.groupby("estudio")[cit].apply(lambda d: d.notna().mean().mean()).round(2)
res["nuevas_personas"] = ext.groupby("estudio").subject_accession.apply(lambda s: len(set(s) - set(par.subject_accession)))
print(res.to_string())

X = lambda d: np.hstack([d[cit].astype(float).values, (d.gender=="Male").astype(float).values[:,None]])
Xp, yp, gp = X(par), par.age.values, par.subject_accession.values
Xe, ye, ge = X(ext), ext.age.values, ext.subject_accession.values
P = dict(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.5, random_state=42, n_jobs=1, verbose=-1)
SPL = [list(GroupKFold(5, shuffle=True, random_state=r).split(Xp, yp, gp)) for r in range(3)]
def fold(tr, va, mascara_extra):
    ok = mascara_extra & ~np.isin(ge, list(set(gp[va])))
    Xt = np.vstack([Xp[tr], Xe[ok]]) if ok.any() else Xp[tr]
    yt = np.concatenate([yp[tr], ye[ok]]) if ok.any() else yp[tr]
    return va, lgb.LGBMRegressor(**P).fit(Xt, yt).predict(Xp[va])
def evaluar(mascara):
    out = Parallel(n_jobs=-1)(delayed(fold)(tr, va, mascara) for s in SPL for tr, va in s)
    oof = np.zeros((3, len(yp)))
    for i, (va, p) in enumerate(out):
        oof[i // 5, va] = p
    return np.abs(oof - yp).mean(axis=1).mean(), np.abs(oof - yp).mean(axis=0)
base, e0 = evaluar(np.zeros(len(ye), bool))
todo, e1 = evaluar(np.ones(len(ye), bool))
print(f"\nRama citoquinas (LightGBM, CV 3x5 en pareados): solo pareados {base:.2f} | + todos los extra {todo:.2f}")
print("Agregando UN estudio extra a la vez:")
for est in res.index:
    m, e = evaluar((ext.estudio == est).values)
    print(f"   + {est:7s} ({res.loc[est,'filas']:3d} filas): {m:.2f}  ({m-base:+.2f})")
# Solo adultos (>=18) y estudios con buena cobertura
mask_ad = (ext.age >= 18).values
m, _ = evaluar(mask_ad); print(f"\n+ extras solo adultos (>=18 años): {m:.2f} ({m-base:+.2f})")
buenos = res[res.cobertura_cit >= 0.8].index
m, _ = evaluar(ext.estudio.isin(buenos).values); print(f"+ extras de estudios con cobertura >=80% {list(buenos)}: {m:.2f} ({m-base:+.2f})")
# Solo personas nuevas (no pareadas)
m, _ = evaluar(~ext.subject_accession.isin(set(gp)).values); print(f"+ extras solo de personas nuevas: {m:.2f} ({m-base:+.2f})")
