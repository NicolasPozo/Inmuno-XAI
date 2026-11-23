# ==============================================================================
# CURVA DE APRENDIZAJE - LightGBM
# Proyecto: Immuno-XAI - UTEM
#
# ¿Más pacientes reducirían el error? Se entrena LightGBM con el 25%, 50%,
# 75% y 100% de los pacientes reales de train, regenerando su aumento
# (copia con ruido + mixup) en cada caso, y se evalúa SIEMPRE en el mismo
# test de 119 pacientes.
#
# - Hiperparámetros: los mejores de LightGBM en tu train completo (misma
#   búsqueda que Entrenamiento.py), fijos para todos los tamaños.
# - Cada tamaño se repite 5 veces con distintos subconjuntos de pacientes
#   (en el 100% solo cambia la semilla del aumento).
# - El muestreo es por paciente (los repetidos van juntos) y por sexo.
# - También se mide la curva SIN aumento (solo pacientes reales).
#
# Requiere datos_10k/ids_pacientes_actuales.csv (lo genera ampliar_dataset.py).
# ==============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
FRACCIONES = [0.25, 0.50, 0.75, 1.00]
REPETICIONES = 5
RUIDO_FRACCION_STD = 0.03
ALPHA_MIXUP = 0.3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)
carpeta_salida = ruta("graficos_curva_aprendizaje")
os.makedirs(carpeta_salida, exist_ok=True)

df_train = pd.read_csv(ruta("df_train_aumentado_k45.csv"))
df_test = pd.read_csv(ruta("df_test_puro_k45.csv"))
ids = pd.read_csv(ruta("datos_10k", "ids_pacientes_actuales.csv"))

col_target = "age"
cols_excluir = [col_target, "gender", "source_type", "subject_accession", "race"]
biomarcadores = [c for c in df_train.columns if c not in cols_excluir and np.issubdtype(df_train[c].dtype, np.number)]

def matriz(df):
    return np.hstack([df[biomarcadores].values, (df["gender"] == "Male").astype(np.float32).values.reshape(-1, 1)])

X_test = matriz(df_test)
y_test = df_test[col_target].values.astype(np.float32)

orig = df_train[df_train["source_type"] == "original"].reset_index(drop=True)
ids_train = ids[ids["conjunto"] == "train"].reset_index(drop=True)
assert len(ids_train) == len(orig) and np.allclose(ids_train["age"], orig["age"]) and (ids_train["gender"] == orig["gender"]).all()
orig["subject_accession"] = ids_train["subject_accession"]
n_mix_full = df_train[df_train["source_type"] == "synthetic_mixup"]["gender"].value_counts().to_dict()
n_orig_full = orig["gender"].value_counts().to_dict()

# ------------------------------------------------------------------------------
# 1. Hiperparámetros de LightGBM (misma búsqueda que Entrenamiento.py)
# ------------------------------------------------------------------------------
print(">>> Buscando hiperparámetros de LightGBM en el train completo (igual que Entrenamiento.py)...")
param_lgb = {
    "n_estimators": [200, 400, 600, 800],
    "learning_rate": [0.005, 0.01, 0.02, 0.04],
    "max_depth": [3, 4, 5, 6],
    "num_leaves": [15, 31, 45],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.1, 1.0, 2.0, 5.0],
}
search = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1), param_lgb, n_iter=70,
    scoring="neg_mean_absolute_error", cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    random_state=RANDOM_STATE, n_jobs=-1, refit=True,
)
search.fit(matriz(df_train), df_train[col_target].values.astype(np.float32))
mejores = search.best_params_
mae_referencia = mean_absolute_error(y_test, search.best_estimator_.predict(X_test))
print(f"  -> Mejores parámetros: {mejores}")
print(f"  -> MAE en test con tu archivo original completo: {mae_referencia:.2f} años")

# ------------------------------------------------------------------------------
# 2. Aumento (misma receta deducida de df_train_aumentado_k45.csv)
# ------------------------------------------------------------------------------
def aumentar(reales, rng):
    reales = reales[df_train.columns].copy()
    std_cols = reales[biomarcadores].std()
    ruido = reales.copy()
    ruido[biomarcadores] = np.clip(
        reales[biomarcadores].values
        + rng.normal(0, 1, size=(len(reales), len(biomarcadores))) * (RUIDO_FRACCION_STD * std_cols.values), 0, None)
    ruido["source_type"] = "synthetic_noise"
    mezclas = []
    for sexo in ["Male", "Female"]:
        idx = reales.index[reales["gender"] == sexo].to_numpy()
        n_mix = int(round(len(idx) * n_mix_full.get(sexo, 0) / n_orig_full[sexo]))
        for _ in range(n_mix):
            i, j = rng.choice(idx, size=2, replace=False)
            lam = float(rng.beta(ALPHA_MIXUP, ALPHA_MIXUP))
            fila = reales.loc[i].copy()
            fila[biomarcadores] = lam * reales.loc[i, biomarcadores].astype(float) + (1 - lam) * reales.loc[j, biomarcadores].astype(float)
            fila[col_target] = lam * reales.at[i, col_target] + (1 - lam) * reales.at[j, col_target]
            fila["source_type"] = "synthetic_mixup"
            mezclas.append(fila)
    return pd.concat([reales, ruido, pd.DataFrame(mezclas, columns=reales.columns)], ignore_index=True)

def submuestra(frac, rng):
    """Elige pacientes (no filas) por sexo; los pacientes repetidos van juntos."""
    sujetos = orig.drop_duplicates("subject_accession")
    elegidos = []
    for sexo, grupo in sujetos.groupby("gender"):
        n = int(round(frac * len(grupo)))
        elegidos += list(rng.choice(grupo["subject_accession"].values, size=n, replace=False))
    return orig[orig["subject_accession"].isin(elegidos)].reset_index(drop=True)

def evaluar(df_fit):
    modelo = lgb.LGBMRegressor(**mejores, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    modelo.fit(matriz(df_fit), df_fit[col_target].values.astype(np.float32))
    p = modelo.predict(X_test)
    return mean_absolute_error(y_test, p), r2_score(y_test, p), float(np.mean(np.abs(y_test - p) <= 5) * 100)

# ------------------------------------------------------------------------------
# 3. Curva
# ------------------------------------------------------------------------------
print("\n>>> Entrenando la curva de aprendizaje...")
filas = []
for frac in FRACCIONES:
    for rep in range(REPETICIONES):
        rng = np.random.default_rng(1000 * rep + int(frac * 100))
        reales = submuestra(frac, rng)
        for variante, df_fit in [("Con aumento", aumentar(reales, rng)), ("Sin aumento", reales)]:
            mae, r2, acc5 = evaluar(df_fit)
            filas.append({"fraccion": frac, "repeticion": rep + 1, "variante": variante,
                          "pacientes_reales": reales["subject_accession"].nunique(), "filas_train": len(df_fit),
                          "MAE": round(mae, 3), "R2": round(r2, 4), "acierto_5a": round(acc5, 1)})
        print(f"   {int(frac*100):3d}% rep {rep+1}: {filas[-2]['pacientes_reales']} pacientes | "
              f"MAE con aumento {filas[-2]['MAE']:.2f} | sin aumento {filas[-1]['MAE']:.2f}")

df_runs = pd.DataFrame(filas)
df_runs.to_csv(os.path.join(carpeta_salida, "curva_aprendizaje_repeticiones.csv"), index=False)
resumen = (df_runs.groupby(["variante", "fraccion"])
           .agg(pacientes_reales=("pacientes_reales", "mean"), MAE_media=("MAE", "mean"), MAE_sd=("MAE", "std"),
                R2_media=("R2", "mean"), acierto_5a_media=("acierto_5a", "mean"))
           .round(3).reset_index())
resumen.to_csv(os.path.join(carpeta_salida, "curva_aprendizaje_resumen.csv"), index=False)

# ------------------------------------------------------------------------------
# 4. Extrapolación: MAE(n) = a + b * n^(-c)   (ajustada con todas las repeticiones)
# ------------------------------------------------------------------------------
def ley_potencia(n, a, b, c):
    return a + b * np.power(n, -c)

ca = df_runs[df_runs["variante"] == "Con aumento"]
n_obs, mae_obs = ca["pacientes_reales"].values.astype(float), ca["MAE"].values
extrapolacion = None
try:
    (a, b, c), _ = curve_fit(ley_potencia, n_obs, mae_obs, p0=[7.0, 30.0, 0.5],
                             bounds=([0.0, 0.0, 0.05], [20.0, 1e4, 2.0]), maxfev=20000)
    n_actual = orig["subject_accession"].nunique()
    extrapolacion = {"a": a, "b": b, "c": c, "n_actual": n_actual,
                     "mae_actual": ley_potencia(n_actual, a, b, c),
                     "mae_x2": ley_potencia(2 * n_actual, a, b, c),
                     "mae_x4": ley_potencia(4 * n_actual, a, b, c)}
except RuntimeError:
    pass

# Bootstrap del MAE del modelo de referencia (para saber qué diferencia es "ruido")
rng_b = np.random.default_rng(0)
p_ref = search.best_estimator_.predict(X_test)
err_ref = np.abs(y_test - p_ref)
boot = err_ref[rng_b.integers(0, len(err_ref), size=(5000, len(err_ref)))].mean(1)
ic_ref = np.percentile(boot, [2.5, 97.5])

print("\n" + "=" * 80)
print("RESUMEN (media de 5 repeticiones; MAE en el mismo test de 119 pacientes)")
print("=" * 80)
print(resumen.to_string(index=False))
print(f"\nReferencia (tu archivo completo): MAE {mae_referencia:.2f} años (IC95% bootstrap {ic_ref[0]:.2f}-{ic_ref[1]:.2f})")
if extrapolacion:
    e = extrapolacion
    print(f"Ajuste MAE = {e['a']:.2f} + {e['b']:.1f}·n^(-{e['c']:.2f})  -> piso estimado {e['a']:.2f} años")
    print(f"  con {e['n_actual']} pacientes: {e['mae_actual']:.2f} | x2 ({2*e['n_actual']}): {e['mae_x2']:.2f} | "
          f"x4 ({4*e['n_actual']}): {e['mae_x4']:.2f}")

# ------------------------------------------------------------------------------
# 5. Gráfico
# ------------------------------------------------------------------------------
TINTA, TINTA2, MUTED, GRID, SUPERFICIE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
COLORES = {"Con aumento": "#2a78d6", "Sin aumento": "#eb6834"}

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(9, 5.4), dpi=120)
fig.patch.set_facecolor(SUPERFICIE)
ax.set_facecolor(SUPERFICIE)

x_max = 2 * orig["subject_accession"].nunique()
if extrapolacion:
    xs = np.linspace(n_obs.min(), x_max, 200)
    ax.plot(xs, ley_potencia(xs, extrapolacion["a"], extrapolacion["b"], extrapolacion["c"]),
            color=COLORES["Con aumento"], lw=1.5, ls=(0, (4, 3)), alpha=0.8, zorder=2)
    ax.annotate(f"Extrapolación aprox. con 2× pacientes: {extrapolacion['mae_x2']:.2f} años",
                xy=(x_max, extrapolacion["mae_x2"]), xytext=(0, -10), textcoords="offset points",
                ha="right", va="top", color=TINTA2, fontsize=9)
    ax.set_ylim(min(extrapolacion["mae_x2"], df_runs["MAE"].min()) - 0.5,
                (resumen["MAE_media"] + resumen["MAE_sd"].fillna(0)).max() + 0.3)

for variante, color in COLORES.items():
    r = resumen[resumen["variante"] == variante]
    ax.fill_between(r["pacientes_reales"], r["MAE_media"] - r["MAE_sd"], r["MAE_media"] + r["MAE_sd"],
                    color=color, alpha=0.15, lw=0, zorder=1)
    ax.plot(r["pacientes_reales"], r["MAE_media"], color=color, lw=2, marker="o", markersize=8,
            markeredgecolor=SUPERFICIE, markeredgewidth=2, label=variante, zorder=3)
    ultimo = r.iloc[-1]
    ax.annotate(f"{variante}: {ultimo['MAE_media']:.2f}", xy=(ultimo["pacientes_reales"], ultimo["MAE_media"]),
                xytext=(10, 0), textcoords="offset points", va="center", color=TINTA, fontsize=9)

ax.axhline(mae_referencia, color=MUTED, lw=1, ls=":", zorder=1)
ax.annotate(f"Modelo actual (tu archivo): {mae_referencia:.2f}", xy=(n_obs.min(), mae_referencia),
            xytext=(0, -12), textcoords="offset points", color=TINTA2, fontsize=9, va="top")

ax.set_title("Curva de aprendizaje de LightGBM: error en test según pacientes reales en train",
             fontsize=11.5, fontweight="bold", color=TINTA, loc="left")
ax.set_xlabel("Pacientes reales en train", color=TINTA2, fontsize=10)
ax.set_ylabel("MAE en test (años)", color=TINTA2, fontsize=10)
ax.set_xlim(n_obs.min() * 0.85, x_max * 1.03)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.grid(axis="x", visible=False)
for lado in ["top", "right"]:
    ax.spines[lado].set_visible(False)
for lado in ["left", "bottom"]:
    ax.spines[lado].set_color("#c3c2b7")
ax.tick_params(colors=MUTED, labelsize=9)
from matplotlib.lines import Line2D
manejadores = [Line2D([0], [0], color=c, lw=2, ls="-", marker="o", markersize=7, label=v) for v, c in COLORES.items()]
if extrapolacion:
    manejadores.append(Line2D([0], [0], color=COLORES["Con aumento"], lw=1.5, ls=(0, (4, 3)), label="Extrapolación (con aumento)"))
ax.legend(handles=manejadores, frameon=False, loc="upper right", fontsize=9, labelcolor=TINTA)
fig.text(0.01, 0.01, "Banda = ±1 desviación estándar entre 5 repeticiones. Test fijo de 119 pacientes; "
         f"IC95% del modelo actual: {ic_ref[0]:.2f}-{ic_ref[1]:.2f} años.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
ruta_fig = os.path.join(carpeta_salida, "curva_aprendizaje_lightgbm.png")
fig.savefig(ruta_fig, dpi=300, facecolor=SUPERFICIE)
plt.close(fig)
print(f"\nGráfico: {ruta_fig}")
