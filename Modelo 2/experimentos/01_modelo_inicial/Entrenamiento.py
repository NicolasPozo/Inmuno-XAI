# ==============================================================================
# SCRIPT: 3.py - MODELADO (IMMUNO-XAI - SIN ENSAMBLE)
# Proyecto: Immuno-XAI - UTEM (Células y Proteínas / CyTOF & ELISA)
# Autores: Constanza Olivos Fernández & Nicolás Pozo Villagrán
#
# Usa directamente los datasets ya filtrados con los 45 biomarcadores
# seleccionados (df_train_aumentado_k45.csv y df_test_puro_k45.csv).
# Este script solo lee esos archivos: no los modifica ni los sobrescribe.
# ==============================================================================

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-Learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Boosting Avanzado
import lightgbm as lgb
import xgboost as xgb

# ==============================================================================
# 0. CONFIGURACIÓN GLOBAL Y ENTORNO DE REPRODUCIBILIDAD
# ==============================================================================
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 120

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config import ruta  # noqa: E402  (rutas de datos y resultados: src/config.py)

# Uso: python Entrenamiento.py [archivo_train.csv]
# Sin argumento usa df_train_aumentado_k45.csv. El test es siempre df_test_puro_k45.csv.
nombre_train = sys.argv[1] if len(sys.argv) > 1 else "df_train_aumentado_k45.csv"
archivo_train = ruta(nombre_train)
archivo_test = ruta("df_test_puro_k45.csv")

if nombre_train == "df_train_aumentado_k45.csv":
    carpeta_salida = ruta("graficos_entrenamiento_multimodal")
else:
    sufijo = os.path.splitext(os.path.basename(nombre_train))[0]
    for prefijo in ("df_train_aumentado_k45_", "df_train_"):
        if sufijo.startswith(prefijo):
            sufijo = sufijo[len(prefijo):]
            break
    carpeta_salida = ruta(f"graficos_entrenamiento_multimodal_{sufijo}")
os.makedirs(carpeta_salida, exist_ok=True)

def guardar_figura_forzada(ruta_archivo, dpi=300):
    """Elimina el archivo preexistente antes de guardar para forzar la actualización."""
    if os.path.exists(ruta_archivo):
        try:
            os.remove(ruta_archivo)
        except OSError:
            pass
    plt.savefig(ruta_archivo, dpi=dpi, bbox_inches="tight")
    plt.close()

# ==============================================================================
# 1. CARGA DE LOS DATASETS FILTRADOS (k=45)
# ==============================================================================
print(">>> [FASE 1] Cargando datasets filtrados Train y Test (k=45)...")
if not os.path.exists(archivo_train) or not os.path.exists(archivo_test):
    raise FileNotFoundError(
        f"Error crítico: No se encontraron '{nombre_train}' o 'df_test_puro_k45.csv' "
        "en la carpeta del script."
    )

df_train = pd.read_csv(archivo_train)
df_test = pd.read_csv(archivo_test)

col_target = "age"
cols_excluir = [col_target, "gender", "source_type", "subject_accession", "race"]

# Biomarcadores = todas las columnas numéricas que no son target ni metadatos
biomarcadores = [c for c in df_train.columns if c not in cols_excluir and np.issubdtype(df_train[c].dtype, np.number)]

faltantes_test = [c for c in biomarcadores if c not in df_test.columns]
if faltantes_test:
    raise ValueError(f"Error crítico: El test no contiene estos biomarcadores del train: {faltantes_test}")

y_train = df_train[col_target].values.astype(np.float32)
y_test = df_test[col_target].values.astype(np.float32)

genero_tr = (df_train["gender"] == "Male").astype(np.float32).values.reshape(-1, 1)
genero_te = (df_test["gender"] == "Male").astype(np.float32).values.reshape(-1, 1)

# Mismas columnas y mismo orden en train y test, más el género al final
X_train_final = np.hstack([df_train[biomarcadores].values, genero_tr])
X_test_final = np.hstack([df_test[biomarcadores].values, genero_te])

n_biomarcadores = len(biomarcadores)

n_reales_train = int((df_train["source_type"] == "original").sum()) if "source_type" in df_train.columns else len(df_train)
print(f"  -> Train: {len(df_train)} filas ({n_reales_train} pacientes reales) | Test Puro: {len(df_test)} pacientes reales")
print(f"  -> Biomarcadores utilizados: {n_biomarcadores} (+ género = {X_train_final.shape[1]} variables)")

# ==============================================================================
# 2. DEFINICIÓN DEL BENCHMARK OPTIMIZADO (BÚSQUEDA PROFUNDA DE HIPERPARÁMETROS)
# ==============================================================================
# Si el train trae el ID de paciente, la validación cruzada se agrupa por paciente
# (todas las filas de un mismo paciente caen en el mismo fold). Si no, KFold normal.
if "subject_accession" in df_train.columns:
    kf_tuning = GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grupos_cv = df_train["subject_accession"].values
    print("  -> Validación cruzada: GroupKFold (5 folds agrupados por paciente)")
else:
    kf_tuning = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grupos_cv = None
    print("  -> Validación cruzada: KFold (5 folds por fila)")

pipe_enet = Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=10000, tol=0.001))])
param_enet = {"model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0], "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}

pipe_svr = Pipeline([("scaler", StandardScaler()), ("model", SVR())])
param_svr = {"model__C": [0.5, 1.0, 5.0, 10.0, 20.0, 50.0], "model__gamma": ["scale", "auto", 0.001, 0.01, 0.05, 0.1], "model__epsilon": [0.01, 0.05, 0.1, 0.2]}

rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
param_rf = {"n_estimators": [100, 200, 300], "max_depth": [4, 6, 8, 12, None], "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2", 0.8]}

lgb_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
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

xgb_model = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="mae")
param_xgb = {
    "n_estimators": [200, 400, 600, 800],
    "learning_rate": [0.005, 0.01, 0.02, 0.04],
    "max_depth": [3, 4, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

config_modelos = {
    "ElasticNet": (pipe_enet, param_enet, 30),
    "SVR (RBF)": (pipe_svr, param_svr, 30),
    "Random Forest": (rf, param_rf, 30),
    "LightGBM": (lgb_model, param_lgb, 70),
    "XGBoost": (xgb_model, param_xgb, 70),
}

# ==============================================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN CIEGA EN TEST (SIN ENSAMBLE)
# ==============================================================================
predicciones_test = {}
modelos_entrenados = {}
metricas_finales = []
tiempos_entrenamiento = {}

print("\n>>> [FASE 2] Optimizando en Train (5-Fold CV) y evaluando a ciegas en Test...")

for nombre, (estimador, grid, n_iter) in config_modelos.items():
    t_inicio = time.time()

    search = RandomizedSearchCV(
        estimator=estimador,
        param_distributions=grid,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=kf_tuning,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train_final, y_train, groups=grupos_cv)
    best_mod = search.best_estimator_
    mae_cv = -search.best_score_
    modelos_entrenados[nombre] = best_mod

    t_total_segundos = time.time() - t_inicio
    tiempos_entrenamiento[nombre] = round(t_total_segundos, 2)

    preds = best_mod.predict(X_test_final)
    predicciones_test[nombre] = preds

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    acierto_5_anos = (np.sum(np.abs(y_test - preds) <= 5.0) / len(y_test)) * 100.0

    metricas_finales.append({
        "Modelo": nombre,
        "Tiempo (s)": round(t_total_segundos, 2),
        "MAE CV Train (Años)": round(mae_cv, 2),
        "MAE Test (Años)": round(mae, 2),
        "RMSE Test": round(rmse, 2),
        "R² (Varianza Explicada)": f"{round(r2 * 100, 1)}%",
        "Exactitud (±5 Años)": f"{round(acierto_5_anos, 1)}%",
    })

    print(f"   -> {nombre:15s} | Tiempo: {t_total_segundos:5.2f}s | MAE CV: {mae_cv:4.2f}a | R²: {r2*100:5.1f}% | MAE: {mae:4.2f}a | Acierto ±5a: {acierto_5_anos:5.1f}%")

# ==============================================================================
# 4. EXPORTACIÓN DE RESULTADOS Y GRÁFICO MULTIPANEL CON TIEMPOS EN TEST PURO
# ==============================================================================
df_resumen = pd.DataFrame(metricas_finales)
print("\n" + "=" * 80)
print(f"TABLA COMPARATIVA FINAL (EVALUADA SOBRE EL TEST PURO - {len(y_test)} PACIENTES REALES)")
print("=" * 80)
print(df_resumen.to_string(index=False))

ruta_resumen = os.path.join(carpeta_salida, "comparativa_modelos_multimodal.csv")
df_resumen.to_csv(ruta_resumen, index=False)

# Predicciones de cada modelo por paciente de test (para comparar experimentos)
df_preds = pd.DataFrame({"edad_real": y_test, **{n: p for n, p in predicciones_test.items()}})
df_preds.to_csv(os.path.join(carpeta_salida, "predicciones_test.csv"), index=False)

n_plots = len(predicciones_test)
fig, axes = plt.subplots(1, n_plots, figsize=(4.8 * n_plots, 4.8))
colores = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974"]

for idx, (nombre, preds) in enumerate(predicciones_test.items()):
    ax = axes[idx]
    r2_val = r2_score(y_test, preds)
    mae_val = mean_absolute_error(y_test, preds)
    tiempo_str = f"{tiempos_entrenamiento[nombre]}s"

    ax.scatter(y_test, preds, alpha=0.7, edgecolors="k", s=35, color=colores[idx % len(colores)])
    linea = [y_test.min(), y_test.max()]
    ax.plot(linea, linea, "r--", lw=2, label="Ideal")

    ax.set_title(f"{nombre} (⏱ {tiempo_str})\nR² = {r2_val*100:.1f}% | MAE = {mae_val:.2f}a", fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Edad Real (años)")
    ax.set_ylabel("Edad Predicha (años)")
    ax.legend(loc="upper left")

plt.suptitle(f"Evaluación en Pacientes Reales de Test (k={n_biomarcadores} Biomarcadores Seleccionados)", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()

ruta_fig_test = os.path.join(carpeta_salida, "2_evaluacion_test_multimodal.png")
guardar_figura_forzada(ruta_fig_test)

print("\n" + "=" * 80)
print(">>> BENCHMARK COMPLETADO EXITOSAMENTE (SIN ENSAMBLE)")
print(f"  * Artefactos guardados en: '{carpeta_salida}/'")
print("=" * 80)
