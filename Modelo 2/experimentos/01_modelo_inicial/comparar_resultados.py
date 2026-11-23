# ==============================================================================
# COMPARACIÓN: TRAIN ACTUAL vs TRAIN AMPLIADO (MISMO TEST DE 119 PACIENTES)
# Uso: python comparar_resultados.py [experimento]
#   experimento = sufijo del train ampliado (por defecto "ampliado"), p. ej.:
#     python Entrenamiento.py df_train_aumentado_k45_fluprint.csv
#     python comparar_resultados.py fluprint
# Requiere haber ejecutado antes: python Entrenamiento.py  (train actual)
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
experimento = sys.argv[1] if len(sys.argv) > 1 else "ampliado"
carpeta_exp = ruta(f"graficos_entrenamiento_multimodal_{experimento}")
ruta_base = ruta("graficos_entrenamiento_multimodal", "predicciones_test.csv")
ruta_ampl = os.path.join(carpeta_exp, "predicciones_test.csv")
ruta_salida = os.path.join(carpeta_exp, f"comparacion_actual_vs_{experimento}.csv")

N_BOOTSTRAP = 5000
rng = np.random.default_rng(42)

base = pd.read_csv(ruta_base)
ampl = pd.read_csv(ruta_ampl)
assert np.allclose(base["edad_real"], ampl["edad_real"]), "Los test no coinciden"

y = base["edad_real"].to_numpy()
n = len(y)
modelos = [c for c in base.columns if c != "edad_real"]

filas = []
for m in modelos:
    err_b = np.abs(y - base[m].to_numpy())
    err_a = np.abs(y - ampl[m].to_numpy())
    dif = err_a - err_b  # negativo = el ampliado se equivoca menos

    # Bootstrap pareado: remuestrea pacientes y recalcula la diferencia de MAE
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    difs_boot = dif[idx].mean(axis=1)
    ic_bajo, ic_alto = np.percentile(difs_boot, [2.5, 97.5])
    p_mejora = float((difs_boot < 0).mean())

    filas.append({
        "Modelo": m,
        "MAE actual": round(err_b.mean(), 2),
        f"MAE {experimento}": round(err_a.mean(), 2),
        "Diferencia MAE": round(dif.mean(), 2),
        "IC95% bajo": round(ic_bajo, 2),
        "IC95% alto": round(ic_alto, 2),
        "Prob. de que mejore": f"{p_mejora*100:.0f}%",
        "R² actual": f"{r2_score(y, base[m])*100:.1f}%",
        f"R² {experimento}": f"{r2_score(y, ampl[m])*100:.1f}%",
        "±5a actual": f"{(err_b <= 5).mean()*100:.1f}%",
        f"±5a {experimento}": f"{(err_a <= 5).mean()*100:.1f}%",
        "Significativo": "Sí" if (ic_alto < 0 or ic_bajo > 0) else "No",
    })

df = pd.DataFrame(filas)
pd.set_option("display.width", 250)
print(f"Diferencia MAE < 0 => el modelo '{experimento}' se equivoca menos que el actual.")
print("Significativo = el intervalo de confianza 95% no incluye el 0.\n")
print(df.to_string(index=False))
df.to_csv(ruta_salida, index=False)
print(f"\nGuardado en: {ruta_salida}")
