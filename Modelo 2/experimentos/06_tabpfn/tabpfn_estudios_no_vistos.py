# ==============================================================================
# TabPFN EN ESTUDIOS NO VISTOS (se deja un estudio completo fuera)
# Proyecto: Immuno-XAI - UTEM
#
# Misma evaluación que modelo_final_dos_ramas.py (sección 3), con 5 estudios:
# SDY420, SDY515, SDY519, SDY113, SDY311. Se entrena con todos los pareados de
# los demás estudios y se evalúa en el estudio dejado fuera.
#   - TabPFN un modelo, 45 biomarcadores + sexo (solo pacientes reales)
#   - TabPFN un modelo, todas las variables
#   - TabPFN dos ramas + fusión (células con CyTOF extra del 10K)
# Se compara con el modelo final (resultados_modelo_final/metricas_estudios_no_vistos.csv).
# ==============================================================================

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch

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

# Datos, variables y función dos_ramas del modelo final
sys.argv = [sys.argv[0]]
codigo = open(ruta("modelo_final_dos_ramas.py"), encoding="utf-8").read()
corte = codigo.index("# ------------------------------------------------------------------------------\n# 1. CV agrupada")
ns = {"__file__": ruta("modelo_final_dos_ramas.py"), "__name__": "mf"}
exec(compile(codigo[:corte], "modelo_final_dos_ramas.py", "exec"), ns)
par, y, estudios = ns["par"], ns["y"], ns["estudios"]
XC, XK, metricas = ns["XC"], ns["XK"], ns["metricas"]

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

def tabpfn():
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=0)

# Las dos ramas usan TabPFN (dos_ramas clona MODELO_CEL / MODELO_CIT)
ns["MODELO_CEL"], ns["MODELO_CIT"] = tabpfn(), tabpfn()

test_k45 = pd.read_csv(ruta("df_test_puro_k45.csv"))
cols_k45 = [c for c in test_k45.columns if c not in ["age", "gender", "source_type", "subject_accession", "race"]]
X_45 = np.hstack([par[cols_k45].astype(float).values, (par["gender"] == "Male").astype(float).values.reshape(-1, 1)])
X_todas = np.hstack([XC, XK[:, :-1]])

loso_est = ["SDY420", "SDY515", "SDY519", "SDY113", "SDY311"]
filas = []
for e in loso_est:
    tr, va = np.where(estudios != e)[0], np.where(estudios == e)[0]
    yv = y[va]
    info = {"estudio_no_visto": e, "n": len(va), "edades": f"{yv.min():.0f}-{yv.max():.0f}"}
    for nombre, X in [("TabPFN un modelo (45)", X_45), ("TabPFN un modelo (todas)", X_todas)]:
        p = tabpfn().fit(X[tr].astype(np.float32), y[tr]).predict(X[va].astype(np.float32))
        filas.append({**info, "modelo": nombre, **metricas(yv, p)})
    pf, pc, pk, _ = ns["dos_ramas"](tr, va, e)
    for nombre, p in [("TabPFN rama células", pc), ("TabPFN rama citoquinas", pk), ("TabPFN dos ramas (fusión)", pf)]:
        filas.append({**info, "modelo": nombre, **metricas(yv, p)})
    log(f"{e} listo: " + " | ".join(f"{f['modelo']} {f['MAE']:.1f}" for f in filas[-5:]))

df = pd.DataFrame(filas)
ref = pd.read_csv(ruta("resultados_modelo_final", "metricas_estudios_no_vistos.csv"))
ref["modelo"] = "Modelo final: " + ref["modelo"]
todo = pd.concat([ref, df], ignore_index=True)
todo.round(3).to_csv(os.path.join(carpeta, "estudios_no_vistos_tabpfn.csv"), index=False)

orden = list(ref["modelo"].unique()) + list(df["modelo"].unique())
resumen = todo.groupby("modelo")[["MAE", "RMSE", "R2", "r", "acierto_5a_%"]].mean().loc[orden].round(3)
tabla_mae = todo.pivot(index="modelo", columns="estudio_no_visto", values="MAE").loc[orden, loso_est].round(1)
resumen.to_csv(os.path.join(carpeta, "estudios_no_vistos_resumen.csv"))
pd.set_option("display.width", 250)
print("\n=== PROMEDIO DE 5 ESTUDIOS NO VISTOS ===\n" + resumen.to_string())
print("\n=== MAE POR ESTUDIO ===\n" + tabla_mae.to_string())
log("Listo.")
