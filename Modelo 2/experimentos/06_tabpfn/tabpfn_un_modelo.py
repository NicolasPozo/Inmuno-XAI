# ==============================================================================
# TabPFN v2 COMO UN SOLO MODELO (células + citoquinas juntas, como al inicio)
# Proyecto: Immuno-XAI - UTEM
#
# A) Igual que al inicio: train = df_train_aumentado_k45.csv (con ruido y mixup),
#    test = df_test_puro_k45.csv. 45 biomarcadores + sexo.
# B) Mismo k45 pero solo pacientes reales (df_train_reales_k45.csv).
# C) Un solo modelo sobre los 473 pareados, con las mismas particiones CV 3x5 que
#    la prueba de dos ramas (comparación directa con TabPFN de dos ramas):
#      C1 = 45 biomarcadores del k45 + sexo
#      C2 = todas las variables (24 células + 11 razones + 46 citoquinas + sexo)
# ==============================================================================

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta = ruta("resultados_tabpfn_un_modelo")
os.makedirs(carpeta, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time()-T0)/60:6.1f} min] {m}", flush=True)

os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"  # el train aumentado tiene 1014 filas (límite CPU = 1000)
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

def tabpfn():
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)

def ajustar_predecir(Xt, yt, Xv):
    return tabpfn().fit(Xt.astype(np.float32), yt).predict(Xv.astype(np.float32))

def met(yv, p):
    return {"MAE": mean_absolute_error(yv, p), "RMSE": np.sqrt(mean_squared_error(yv, p)),
            "R2": r2_score(yv, p), "acierto5": np.mean(np.abs(yv - p) <= 5) * 100}

filas = []

# ------------------------------------------------------------------------------
# A y B: archivos k45 originales
# ------------------------------------------------------------------------------
test_k45 = pd.read_csv(ruta("df_test_puro_k45.csv"))
excluir = ["age", "gender", "source_type", "subject_accession", "race"]
cols_k45 = [c for c in test_k45.columns if c not in excluir]

def matriz_k45(d):
    return np.hstack([d[cols_k45].astype(float).values, (d["gender"] == "Male").astype(float).values.reshape(-1, 1)])

for nombre, archivo in [("A) k45 aumentado (como al inicio)", "df_train_aumentado_k45.csv"),
                        ("B) k45 solo pacientes reales", "df_train_reales_k45.csv")]:
    tr = pd.read_csv(ruta(archivo))
    p = ajustar_predecir(matriz_k45(tr), tr["age"].values, matriz_k45(test_k45))
    m = met(test_k45["age"].values, p)
    filas.append({"experimento": nombre, "filas_train": len(tr), "variables": len(cols_k45) + 1,
                  "MAE_CV": np.nan, "R2_CV": np.nan, **{f"{k}_test": v for k, v in m.items()}})
    pd.DataFrame({"edad_real": test_k45["age"], "edad_predicha": p}).to_csv(
        os.path.join(carpeta, f"predicciones_test_{archivo.replace('.csv', '')}.csv"), index=False)
    log(f"{nombre}: MAE test {m['MAE']:.2f} | R² {m['R2']:.3f}")

# ------------------------------------------------------------------------------
# C: un solo modelo sobre los 473 pareados (mismas particiones que dos ramas)
# ------------------------------------------------------------------------------
sys.argv = [sys.argv[0]]
codigo = open(ruta("modelo_final_dos_ramas.py"), encoding="utf-8").read()
corte = codigo.index("# ------------------------------------------------------------------------------\n# Modelos")
ns = {"__file__": ruta("modelo_final_dos_ramas.py"), "__name__": "mf"}
exec(compile(codigo[:corte], "modelo_final_dos_ramas.py", "exec"), ns)
par, y, grupos = ns["par"], ns["y"], ns["grupos"]
idx_tr, idx_te = ns["idx_tr"], ns["idx_te"]
XC, XK = ns["XC"], ns["XK"]
X_todas = np.hstack([XC, XK[:, :-1]])  # células + razones + sexo + citoquinas (sexo una sola vez)
X_45 = np.hstack([par[cols_k45].astype(float).values, (par["gender"] == "Male").astype(float).values.reshape(-1, 1)])

REPETICIONES = 3
SPLITS = [list(GroupKFold(n_splits=5, shuffle=True, random_state=r).split(y, y, grupos)) for r in range(REPETICIONES)]

PRED = {}
for nombre, X in [("C1) 473 pareados | 45 biomarcadores", X_45), ("C2) 473 pareados | todas las variables", X_todas)]:
    oof = np.zeros((REPETICIONES, len(y)))
    for r in range(REPETICIONES):
        for tr, va in SPLITS[r]:
            oof[r, va] = ajustar_predecir(X[tr], y[tr], X[va])
    p_te = ajustar_predecir(X[idx_tr], y[idx_tr], X[idx_te])
    PRED[nombre] = p_te
    cv = [met(y, oof[r]) for r in range(REPETICIONES)]
    m = met(y[idx_te], p_te)
    filas.append({"experimento": nombre, "filas_train": len(idx_tr), "variables": X.shape[1],
                  "MAE_CV": np.mean([c["MAE"] for c in cv]), "R2_CV": np.mean([c["R2"] for c in cv]),
                  **{f"{k}_test": v for k, v in m.items()}})
    log(f"{nombre}: MAE CV {filas[-1]['MAE_CV']:.2f} | MAE test {m['MAE']:.2f} | R² test {m['R2']:.3f}")

# ------------------------------------------------------------------------------
# Comparación con TabPFN de dos ramas (mismas particiones y mismo test)
# ------------------------------------------------------------------------------
P2 = dict(np.load(ruta("resultados_modelos_pesados", "predicciones_pesados.npz"),
                  allow_pickle=True))["PRED"].item()
pc, pk = P2["células|TabPFN v2"], P2["citoquinas|TabPFN v2"]
lr = LinearRegression().fit(np.column_stack([pc["oof_tr"], pk["oof_tr"]]), y[idx_tr])
p_dos = lr.predict(np.column_stack([pc["test"], pk["test"]]))
e_dos = np.abs(p_dos - y[idx_te])
e_uno = np.abs(PRED["C2) 473 pareados | todas las variables"] - y[idx_te])
rng = np.random.default_rng(0)
dif = np.array([(e_uno[i] - e_dos[i]).mean() for i in (rng.integers(0, len(e_uno), len(e_uno)) for _ in range(5000))])

df = pd.DataFrame(filas).round(3)
df.to_csv(os.path.join(carpeta, "metricas_tabpfn_un_modelo.csv"), index=False)
pd.set_option("display.width", 250)
print("\n" + df.to_string(index=False))
print(f"\nDos ramas TabPFN (referencia): MAE test {e_dos.mean():.2f}")
print(f"Un modelo (todas) - dos ramas: {dif.mean():+.2f} años | IC95 [{np.percentile(dif, 2.5):+.2f}, {np.percentile(dif, 97.5):+.2f}]"
      f" | P(dos ramas mejor) {np.mean(dif > 0)*100:.1f}%")
log("Listo.")
