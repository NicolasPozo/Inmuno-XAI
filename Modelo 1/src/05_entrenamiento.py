"""
PASO 5 - Entrenamiento y evaluación de 5 modelos de reloj epigenético.

Modelos : ElasticNet, SVR (RBF), Random Forest, LightGBM, XGBoost
Entrada : Datasets fusionados de datos.DATASETS (Hannum + GSE87571, solo pacientes reales)
Salida  : resultados/entrenamiento/
          - auditoria_alucinaciones_modelos.csv  (test combinado)
          - auditoria_por_dataset.csv            (test de cada dataset por separado)
          - metricas_matriz_confusion.csv        (persona mayor >= EDAD_CORTE: sensibilidad, especificidad...)
          - 11_evaluacion_test_real_puro.png
          - 12_distribucion_residuos_alucinacion.png
          - 17_matriz_confusion.png

Protocolo sin fuga de datos (el test nunca participa en ninguna decisión):
  1-4. Partición e imputación por dataset y folds sin fuga: ver src/datos.py.
       Solo pacientes reales (sin datos sintéticos).
  5. Selección de las N_SONDAS_TOP sondas solo con train.
  6. La parada temprana de LightGBM/XGBoost se calcula en un fold de validación
     de train y luego se reentrena con todo train con ese número de árboles.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Modelos y métricas
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    max_error
)
import xgboost as xgb
import lightgbm as lgb

from config import (
    CARPETA_RESULTADOS_ENTRENAMIENTO,
    configurar_estilo_graficos,
    crear_carpeta,
)
from datos import (
    N_SONDAS_TOP,
    RANDOM_STATE,
    armar_matrices,
    preparar_particion,
    rankear_sondas,
    splits_sin_fuga,
)

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================
EARLY_STOPPING_ROUNDS = 30  # Ventana de paciencia
EDAD_CORTE = 60             # Persona mayor en Chile (Ley 19.828); clase positiva de la matriz de confusión
# N_SONDAS_TOP (sondas usadas por el modelo) se define en src/datos.py

configurar_estilo_graficos()
carpeta_salida = crear_carpeta(CARPETA_RESULTADOS_ENTRENAMIENTO)

# ==============================================================================
# 1. CARGA DE PACIENTES REALES Y PARTICIÓN TRAIN / TEST
# ==============================================================================
print("1. Cargando pacientes reales y separando train / test...")
p = preparar_particion()
train_ds, test_ds = p.por_dataset(p.idx_train), p.por_dataset(p.idx_test)
for nombre in p.nombres_lote:
    print(f"   -> {nombre:<9} Train: {len(train_ds[nombre])} | Test: {len(test_ds[nombre])}")
print(f"   -> Total     Train: {len(p.idx_train)} | Test: {len(p.idx_test)}")

# ==============================================================================
# 2. SELECCIÓN DE SONDAS (SOLO CON TRAIN)
# ==============================================================================
print(f"\n2. Seleccionando las {N_SONDAS_TOP} sondas más correlacionadas con la edad (solo train)...")
top_features = rankear_sondas(p, p.idx_train)[:N_SONDAS_TOP]
nombres_cols = p.sondas[top_features].tolist() + ['Sexo_M']
print(f"   -> Sondas candidatas: {len(p.sondas)} | Top 5: {nombres_cols[:5]}")

# ==============================================================================
# 3. MATRICES DE TRAIN / TEST
# ==============================================================================
print("\n3. Armando matrices de train y test...")
X_train, y_train, X_test, y_test = armar_matrices(p, top_features)
lote_test = p.lote[p.idx_test]  # dataset de origen de cada paciente de test
cv_splits = splits_sin_fuga(p)

print(f"   -> Muestras en Train: {X_train.shape[0]} pacientes")
print(f"   -> Muestras en Test:  {X_test.shape[0]} pacientes no vistos")

# ==============================================================================
# 4. ESPACIO DE BÚSQUEDA PARA LOS 5 MODELOS
# ==============================================================================
pipe_enet = Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(random_state=RANDOM_STATE, max_iter=2000))])
param_enet = {'model__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0], 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

pipe_svr = Pipeline([('scaler', StandardScaler()), ('model', SVR())])
param_svr = {'model__C': [1.0, 5.0, 10.0, 20.0, 50.0], 'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1], 'model__epsilon': [0.01, 0.05, 0.1, 0.2]}

rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=2)
param_rf = {'n_estimators': [100, 150, 200], 'max_depth': [6, 10, 15, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

lgb_model = lgb.LGBMRegressor(n_estimators=600, random_state=RANDOM_STATE, n_jobs=2, verbose=-1)
param_lgb = {
    'learning_rate': [0.01, 0.03, 0.05, 0.08],
    'max_depth': [3, 4, 5, 6],
    'num_leaves': [15, 31, 45],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0.0, 0.1, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0]
}

xgb_model = xgb.XGBRegressor(n_estimators=600, random_state=RANDOM_STATE, n_jobs=2)
param_xgb = {
    'learning_rate': [0.01, 0.03, 0.05, 0.08],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0.0, 0.1, 1.0],
    'reg_lambda': [1.0, 3.0, 5.0]
}

config_modelos = {
    "ElasticNet": (pipe_enet, param_enet, 15),
    "SVR (RBF)": (pipe_svr, param_svr, 15),
    "Random Forest": (rf, param_rf, 12),
    "LightGBM": (lgb_model, param_lgb, 25),
    "XGBoost": (xgb_model, param_xgb, 25)
}

# ==============================================================================
# 5. TUNING Y ENTRENAMIENTO CON MEDICIÓN DE TIEMPO
# ==============================================================================
predicciones_test = {}
tiempos_entrenamiento = {}

# Fold usado para decidir el número de árboles en LightGBM / XGBoost
pos_es_train, pos_es_val = cv_splits[0]


def n_arboles_optimo(nombre, modelo):
    """Ajusta con parada temprana sobre un fold de validación de train y devuelve los árboles óptimos."""
    modelo_es = clone(modelo)
    X_es, y_es = X_train[pos_es_train], y_train[pos_es_train]
    eval_set = [(X_train[pos_es_val], y_train[pos_es_val])]
    if nombre == "LightGBM":
        callbacks = [lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
        modelo_es.fit(X_es, y_es, eval_set=eval_set, eval_metric="l1", callbacks=callbacks)
        return max(1, modelo_es.best_iteration_)
    modelo_es.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_metric="mae")
    modelo_es.fit(X_es, y_es, eval_set=eval_set, verbose=False)
    return modelo_es.best_iteration + 1  # best_iteration de XGBoost empieza en 0


print("\n4. Optimizando hiperparámetros sobre train y validando en test...")

for nombre, (estimador, grid, n_iter) in config_modelos.items():
    print(f"\n   -> Optimizando {nombre} ({n_iter} combinaciones)...")
    
    # Inicia cronómetro de alta precisión
    t_inicio = time.perf_counter()
    
    search = RandomizedSearchCV(
        estimator=estimador,
        param_distributions=grid,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=cv_splits,
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_mod = search.best_estimator_

    # Parada temprana en validación de train y reentrenamiento con todo train
    if nombre in ("LightGBM", "XGBoost"):
        n_arboles = n_arboles_optimo(nombre, best_mod)
        best_mod.set_params(n_estimators=n_arboles)
        best_mod.fit(X_train, y_train)
        print(f"      * Árboles por parada temprana (validación de train): {n_arboles}")

    # Detiene cronómetro
    t_fin = time.perf_counter()
    tiempo_total = t_fin - t_inicio
    tiempos_entrenamiento[nombre] = tiempo_total

    preds = best_mod.predict(X_test)
    predicciones_test[nombre] = preds
    
    print(f"      * Tiempo total de entrenamiento/tuning: {tiempo_total:.2f} s")

# ==============================================================================
# 6. AUDITORÍA DIAGNÓSTICA DE ALUCINACIONES Y RENDIMIENTO (CON TIEMPO)
# ==============================================================================
def evaluar_alucinacion_y_rendimiento(y_real, y_pred, nombre_modelo, duracion_seg):
    mae = mean_absolute_error(y_real, y_pred)
    med_ae = median_absolute_error(y_real, y_pred)
    max_err = max_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    wape = (np.sum(np.abs(y_real - y_pred)) / np.sum(y_real)) * 100
    spearman_rho, _ = stats.spearmanr(y_real, y_pred)

    min_real, max_real = y_real.min(), y_real.max()
    fuera_de_rango = np.sum((y_pred < min_real - 5) | (y_pred > max_real + 5))
    pct_alucinacion = (fuera_de_rango / len(y_real)) * 100

    return {
        'Modelo': nombre_modelo,
        'MAE (Años)': round(mae, 2),
        'MedAE (Típico)': round(med_ae, 2),
        'MaxAE (Peor Caso)': round(max_err, 2),
        'RMSE': round(rmse, 2),
        'WAPE (%)': f"{wape:.1f}%",
        'R²': round(r2, 3),
        'Spearman ρ': round(spearman_rho, 3),
        'Alucinación Rango (%)': f"{pct_alucinacion:.1f}%",
        'Tiempo (s)': round(duracion_seg, 2)
    }

metricas_diagnostico = [
    evaluar_alucinacion_y_rendimiento(y_test, preds, nombre, tiempos_entrenamiento[nombre])
    for nombre, preds in predicciones_test.items()
]

df_diagnostico = pd.DataFrame(metricas_diagnostico)
print("\n" + "="*95)
print("AUDITORÍA DE PRECISIÓN, CONTROL DE ALUCINACIONES Y EFICIENCIA TEMPORAL (TEST REAL)")
print("="*95)
print(df_diagnostico.to_string(index=False))
df_diagnostico.to_csv(os.path.join(carpeta_salida, "auditoria_alucinaciones_modelos.csv"), index=False)

# Rendimiento en el test de cada dataset por separado
filas_ds = []
for i, nombre_ds in enumerate(p.nombres_lote):
    mascara = lote_test == i
    for nombre, preds in predicciones_test.items():
        filas_ds.append({
            'Dataset': nombre_ds,
            'Modelo': nombre,
            'N_test': int(mascara.sum()),
            'MAE (Años)': round(mean_absolute_error(y_test[mascara], preds[mascara]), 2),
            'R²': round(r2_score(y_test[mascara], preds[mascara]), 3),
            'Sesgo (Años)': round(float(np.mean(preds[mascara] - y_test[mascara])), 2),
        })
df_por_dataset = pd.DataFrame(filas_ds)
df_por_dataset.to_csv(os.path.join(carpeta_salida, "auditoria_por_dataset.csv"), index=False)
if len(p.nombres_lote) > 1:
    print("\nMAE (años) en el test de cada dataset:")
    print(df_por_dataset.pivot(index='Modelo', columns='Dataset', values='MAE (Años)')
          .loc[list(predicciones_test), list(p.nombres_lote)].to_string())

# ==============================================================================
# 6B. MATRIZ DE CONFUSIÓN: ¿PERSONA MAYOR (>= EDAD_CORTE)?
# ==============================================================================
# El modelo predice una edad continua; para las métricas de prueba diagnóstica
# se clasifica a cada paciente como positivo si tiene EDAD_CORTE años o más,
# tanto con la edad real como con la predicha.
def ic_wilson(aciertos, total, z=1.96):
    """IC 95 % de Wilson para una proporción (recomendado por STARD para pruebas diagnósticas)."""
    if total == 0:
        return np.nan, np.nan
    prop = aciertos / total
    centro = (prop + z**2 / (2 * total)) / (1 + z**2 / total)
    margen = z * np.sqrt(prop * (1 - prop) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return centro - margen, centro + margen


def proporcion_con_ic(aciertos, total):
    if total == 0:
        return "—"
    inf, sup = ic_wilson(aciertos, total)
    return f"{100 * aciertos / total:.1f}% [{100 * inf:.1f}, {100 * sup:.1f}]"


def metricas_confusion(y_real, y_pred, nombre_test, nombre_modelo):
    real_pos, pred_pos = y_real >= EDAD_CORTE, y_pred >= EDAD_CORTE
    vp = int(np.sum(real_pos & pred_pos))
    vn = int(np.sum(~real_pos & ~pred_pos))
    fp = int(np.sum(~real_pos & pred_pos))
    fn = int(np.sum(real_pos & ~pred_pos))
    f1 = 2 * vp / (2 * vp + fp + fn) if vp else 0.0
    return {
        'Test': nombre_test,
        'Modelo': nombre_modelo,
        'N': len(y_real),
        f'Prevalencia >= {EDAD_CORTE} (%)': round(100 * real_pos.mean(), 1),
        'VP': vp, 'FN': fn, 'FP': fp, 'VN': vn,
        'Exactitud': proporcion_con_ic(vp + vn, len(y_real)),
        'Sensibilidad': proporcion_con_ic(vp, vp + fn),
        'Especificidad': proporcion_con_ic(vn, vn + fp),
        'VPP': proporcion_con_ic(vp, vp + fp),
        'VPN': proporcion_con_ic(vn, vn + fn),
        'F1': round(f1, 3),
    }


filas_conf = []
for nombre, preds in predicciones_test.items():
    filas_conf.append(metricas_confusion(y_test, preds, "Combinado", nombre))
for i, nombre_ds in enumerate(p.nombres_lote):
    mascara = lote_test == i
    for nombre, preds in predicciones_test.items():
        filas_conf.append(metricas_confusion(y_test[mascara], preds[mascara], nombre_ds, nombre))
df_confusion = pd.DataFrame(filas_conf)
df_confusion.to_csv(os.path.join(carpeta_salida, "metricas_matriz_confusion.csv"), index=False)

print("\n" + "="*95)
print(f"MÉTRICAS DE PRUEBA DIAGNÓSTICA: ¿PERSONA MAYOR (>= {EDAD_CORTE} AÑOS)? [IC 95% de Wilson]")
print("="*95)
for nombre_test in ["Combinado", *p.nombres_lote]:
    print(f"\nTest {nombre_test}:")
    print(df_confusion[df_confusion['Test'] == nombre_test].drop(columns=['Test']).to_string(index=False))

# ==============================================================================
# 7. VISUALIZACIÓN: DISPERSIÓN + DISTRIBUCIÓN RESIDUAL
# ==============================================================================
# Figura 1: Comparativa multipanel (5 modelos)
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
colores = ['#4c72b0', '#55a868', '#c44e52', '#8172b3', '#ccb974']

for idx, (nombre, preds) in enumerate(predicciones_test.items()):
    ax = axes[idx]
    r2_val = r2_score(y_test, preds)
    mae_val = mean_absolute_error(y_test, preds)
    t_val = tiempos_entrenamiento[nombre]

    for i, nombre_ds in enumerate(p.nombres_lote):
        mascara = lote_test == i
        ax.scatter(y_test[mascara], preds[mascara], alpha=0.65, edgecolors='k', s=40, color=colores[idx],
                   marker=['o', '^', 's', 'D'][i % 4], label=nombre_ds)
    linea = [y_test.min(), y_test.max()]
    ax.plot(linea, linea, 'r--', lw=2, label='Ideal')
    ax.set_title(f"{nombre}\nR² = {r2_val:.3f} | MAE = {mae_val:.2f}a\nTiempo: {t_val:.1f}s", fontweight='bold')
    ax.set_xlabel("Edad Real (Pacientes Reales)")
    ax.set_ylabel("Edad Predicha")
    ax.legend(loc='upper left')

plt.suptitle(f"Evaluación de Modelos Epigenéticos (Test Independiente: {' + '.join(p.nombres_lote)})",
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "11_evaluacion_test_real_puro.png"), bbox_inches='tight', dpi=300)

# Figura 2: Curvas de densidad residual (Detección visual de sesgo y alucinaciones)
plt.figure(figsize=(10, 5))
for nombre, preds in predicciones_test.items():
    residuos = preds - y_test
    sns.kdeplot(residuos, label=nombre, fill=True, alpha=0.15)

plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Sin Error (Ideal)')
plt.title("Distribución de Errores Residuales (y_pred - y_real) en Pacientes Reales", fontsize=12, fontweight='bold')
plt.xlabel("Diferencia en Años (Error Residual)")
plt.ylabel("Densidad")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "12_distribucion_residuos_alucinacion.png"), dpi=300)

# Figura 3: Matrices de confusión (test combinado) para la clasificación >= EDAD_CORTE
etiquetas = [f"< {EDAD_CORTE}", f">= {EDAD_CORTE}"]
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
for idx, (nombre, preds) in enumerate(predicciones_test.items()):
    fila = df_confusion[(df_confusion['Test'] == "Combinado") & (df_confusion['Modelo'] == nombre)].iloc[0]
    matriz = np.array([[fila['VN'], fila['FP']], [fila['FN'], fila['VP']]])
    porcentajes = 100 * matriz / matriz.sum(axis=1, keepdims=True)  # % sobre la edad real (fila)
    anotaciones = [[f"{matriz[i, j]}\n({porcentajes[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
    sns.heatmap(matriz, annot=anotaciones, fmt="", cmap="Blues", cbar=False, ax=axes[idx],
                xticklabels=etiquetas, yticklabels=etiquetas, annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                linewidths=1, linecolor='white')
    axes[idx].set_title(f"{nombre}\nSens. = {fila['Sensibilidad'].split('%')[0]}% | "
                        f"Espec. = {fila['Especificidad'].split('%')[0]}%", fontweight='bold')
    axes[idx].set_xlabel("Edad predicha")
    axes[idx].set_ylabel("Edad real")
plt.suptitle(f"Matriz de confusión: clasificación de persona mayor (>= {EDAD_CORTE} años) en el test independiente",
             fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig(os.path.join(carpeta_salida, "17_matriz_confusion.png"), bbox_inches='tight', dpi=300)

print(f"\nResultados y figuras guardados exitosamente en: {carpeta_salida}")
plt.show()