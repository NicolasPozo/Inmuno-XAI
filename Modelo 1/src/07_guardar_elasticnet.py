"""
PASO 7 - Entrena solo ElasticNet (el modelo rector) y lo guarda para reutilizarlo.

Reproduce exactamente el ElasticNet de 05_entrenamiento.py (misma partición, mismas
500 sondas, misma búsqueda de hiperparámetros y semilla), pero en ~1 minuto.

Salida  : modelos/
          - elasticnet_reloj_epigenetico.joblib  (modelo + todo lo necesario para predecir)
          - particion_pacientes_elasticnet.csv   (ID de cada paciente, train/test y predicción)

Contenido del .joblib (diccionario):
  modelo               Pipeline StandardScaler + ElasticNet entrenado con todo el train
  columnas             Orden exacto de las 501 variables de entrada (500 sondas + 'Sexo_M')
  medianas_imputacion  Mediana de cada sonda en el train fusionado (para rellenar NaN)
  hiperparametros, metricas_test, datasets, n_train, n_test, random_state, version_sklearn

Para fusionar con otro modelo (stacking), usar 'particion_pacientes_elasticnet.csv':
  - Los pacientes de train traen una predicción FUERA DE PLIEGUE (out-of-fold): cada una
    la hace un modelo que no vio a ese paciente; el ranking de sondas también se recalcula
    sin él. Son las que deben usarse para entrenar el modelo fusionado, sin fuga.
  - Los pacientes de test traen la predicción del modelo final y no deben usarse para
    entrenar nada.

Uso del modelo guardado con pacientes nuevos:
    import joblib, pandas as pd
    paquete = joblib.load("modelos/elasticnet_reloj_epigenetico.joblib")
    # betas: DataFrame (filas = pacientes, columnas = sondas cg...); sexo: 'M' / 'F'
    X = (betas.reindex(columns=paquete['columnas'][:-1])
              .fillna(paquete['medianas_imputacion'])
              .assign(Sexo_M=(np.asarray(sexo) == 'M').astype(float)))
    edad_predicha = paquete['modelo'].predict(X[paquete['columnas']].values)
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CARPETA_MODELOS, crear_carpeta
from datos import (
    N_FOLDS,
    N_SONDAS_TOP,
    RANDOM_STATE,
    armar_matrices,
    matriz_reales,
    preparar_particion,
    rankear_sondas,
    splits_sin_fuga,
)

carpeta_salida = crear_carpeta(CARPETA_MODELOS)
ARCHIVO_MODELO = os.path.join(carpeta_salida, "elasticnet_reloj_epigenetico.joblib")
ARCHIVO_PARTICION = os.path.join(carpeta_salida, "particion_pacientes_elasticnet.csv")

# ==============================================================================
# 1. PARTICIÓN, SONDAS Y MATRICES (idéntico a 05_entrenamiento.py)
# ==============================================================================
print("1. Cargando pacientes reales y separando train / test...")
p = preparar_particion()
print(f"   -> Train: {len(p.idx_train)} | Test: {len(p.idx_test)}")

print(f"\n2. Seleccionando las {N_SONDAS_TOP} sondas más correlacionadas con la edad (solo train)...")
top_features = rankear_sondas(p, p.idx_train)[:N_SONDAS_TOP]
columnas = p.sondas[top_features].tolist() + ['Sexo_M']
X_train, y_train, X_test, y_test = armar_matrices(p, top_features)
cv_splits = splits_sin_fuga(p)

# ==============================================================================
# 2. BÚSQUEDA DE HIPERPARÁMETROS Y MODELO FINAL (mismo espacio y semilla que 05)
# ==============================================================================
print("\n3. Optimizando ElasticNet (15 combinaciones x 5 folds)...")
pipe_enet = Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(random_state=RANDOM_STATE, max_iter=2000))])
param_enet = {'model__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0], 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

t_inicio = time.perf_counter()
search = RandomizedSearchCV(
    estimator=pipe_enet,
    param_distributions=param_enet,
    n_iter=15,
    scoring='neg_mean_absolute_error',
    cv=cv_splits,
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbose=0
)
search.fit(X_train, y_train)
modelo = search.best_estimator_
tiempo = time.perf_counter() - t_inicio
print(f"   -> Mejores hiperparámetros: {search.best_params_} ({tiempo:.1f} s)")

# ==============================================================================
# 3. EVALUACIÓN EN TEST (debe coincidir con 05_entrenamiento.py)
# ==============================================================================
pred_test = modelo.predict(X_test)
lote_test = p.lote[p.idx_test]
metricas_test = {'Combinado': {'N': len(y_test),
                               'MAE': round(mean_absolute_error(y_test, pred_test), 3),
                               'R2': round(r2_score(y_test, pred_test), 4)}}
for i, nombre_ds in enumerate(p.nombres_lote):
    mascara = lote_test == i
    metricas_test[nombre_ds] = {'N': int(mascara.sum()),
                                'MAE': round(mean_absolute_error(y_test[mascara], pred_test[mascara]), 3),
                                'R2': round(r2_score(y_test[mascara], pred_test[mascara]), 4)}
print("\n4. Desempeño en test:")
for nombre_test, m in metricas_test.items():
    print(f"   -> {nombre_test:<9} N = {m['N']:>3} | MAE = {m['MAE']:.2f} años | R² = {m['R2']:.3f}")

# ==============================================================================
# 4. PREDICCIONES FUERA DE PLIEGUE PARA TRAIN (para fusionar modelos sin fuga)
# ==============================================================================
print(f"\n5. Predicciones fuera de pliegue para train ({N_FOLDS} folds, ranking de sondas por fold)...")
pred_oof = np.empty(len(p.idx_train))
for k, (pos_tr, pos_val) in enumerate(cv_splits):
    sondas_fold = rankear_sondas(p, p.idx_train[pos_tr])[:N_SONDAS_TOP]
    X_fold = matriz_reales(p, sondas_fold)[p.idx_train]
    modelo_fold = clone(modelo).fit(X_fold[pos_tr], y_train[pos_tr])
    pred_oof[pos_val] = modelo_fold.predict(X_fold[pos_val])
    print(f"   -> Fold {k + 1}: MAE = {mean_absolute_error(y_train[pos_val], pred_oof[pos_val]):.2f} años")
print(f"   -> MAE fuera de pliegue (train): {mean_absolute_error(y_train, pred_oof):.2f} años")

# ==============================================================================
# 5. GUARDADO
# ==============================================================================
medianas = np.median(p.matriz[np.ix_(p.idx_train, top_features)], axis=0)
paquete = {
    'modelo': modelo,
    'columnas': columnas,
    'medianas_imputacion': pd.Series(medianas, index=columnas[:-1]),
    'hiperparametros': search.best_params_,
    'metricas_test': metricas_test,
    'datasets': p.nombres_lote,
    'n_train': len(p.idx_train),
    'n_test': len(p.idx_test),
    'random_state': RANDOM_STATE,
    'version_sklearn': sklearn.__version__,
}
joblib.dump(paquete, ARCHIVO_MODELO)

idx_todos = np.concatenate([p.idx_train, p.idx_test])
df_particion = pd.DataFrame({
    'ID_Paciente': p.ids[idx_todos],
    'Dataset': np.array(p.nombres_lote)[p.lote[idx_todos]],
    'Edad': p.edades[idx_todos],
    'Sexo': p.sexos[idx_todos],
    'Conjunto': ['train'] * len(p.idx_train) + ['test'] * len(p.idx_test),
    'Fold': p.fold_paciente[idx_todos],
    'Prediccion_ElasticNet': np.round(np.concatenate([pred_oof, pred_test]), 3),
    'Tipo_Prediccion': ['fuera_de_pliegue'] * len(p.idx_train) + ['modelo_final'] * len(p.idx_test),
})
df_particion.to_csv(ARCHIVO_PARTICION, index=False)

# Verificación: el modelo cargado desde disco predice lo mismo
recargado = joblib.load(ARCHIVO_MODELO)
assert np.allclose(recargado['modelo'].predict(X_test), pred_test), "El modelo guardado no reproduce las predicciones"

print(f"\nModelo guardado en   : {ARCHIVO_MODELO} ({os.path.getsize(ARCHIVO_MODELO) / 1024:.0f} KB)")
print(f"Partición guardada en: {ARCHIVO_PARTICION}")
