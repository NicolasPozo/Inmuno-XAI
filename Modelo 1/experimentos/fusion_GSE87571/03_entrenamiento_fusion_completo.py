"""
EXPERIMENTO - Paso 3: pipeline completo con los datasets fusionados.

  1. Partición train/test dentro de cada dataset (Hannum: la misma del pipeline; GSE87571: 75/25).
  2. Fusión de los trains de Hannum + GSE87571.
  3. Selección de las N_SONDAS_EXPERIMENTO (130) sondas por |Spearman ρ| sobre el train fusionado.
  4. Aumentación Mixup estratificada por sexo DENTRO de cada dataset (no se mezclan
     pacientes de laboratorios distintos).
  5. Tuning de los 5 modelos con validación cruzada sin fuga (los sintéticos con un
     padre en el fold de validación se excluyen) y parada temprana para LightGBM/XGBoost,
     igual que src/05_entrenamiento.py.

Escenarios comparados (evaluados en test Hannum, test GSE87571 y test combinado):
  - Definitivo (solo Hannum) : el modelo previo a la fusión (con Mixup), validado en GSE87571.
  - Fusión sin Mixup         : es el diseño que quedó como pipeline definitivo.
  - Fusión con Mixup

El Mixup ya no forma parte del pipeline (no aportaba); se mantiene aquí solo como registro.
No modifica nada del proyecto principal.

Salida  : resultados/entrenamiento_fusion_completo.csv
          resultados/16_entrenamiento_fusion_completo.png
"""
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from comun import (
    CARPETA_RESULTADOS,
    N_SONDAS_EXPERIMENTO as N_SONDAS_TOP,
    armar_matrices_con_mixup,
    cargar_gse87571,
    cargar_hannum,
    rankear_sondas_fuentes,
    splits_sin_fuga_con_mixup,
)
from config import configurar_estilo_graficos, crear_carpeta
from datos import N_FOLDS, RANDOM_STATE, rankear_sondas
from mixup import interpolar, planificar_mixup_estratificado

EARLY_STOPPING_ROUNDS = 30
crear_carpeta(CARPETA_RESULTADOS)
configurar_estilo_graficos()


# ==============================================================================
# MODELOS Y ESPACIOS DE BÚSQUEDA (idénticos a src/05_entrenamiento.py)
# ==============================================================================
def config_modelos():
    return {
        "ElasticNet": (
            Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(random_state=RANDOM_STATE, max_iter=2000))]),
            {'model__alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0], 'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}, 15),
        "SVR (RBF)": (
            Pipeline([('scaler', StandardScaler()), ('model', SVR())]),
            {'model__C': [1.0, 5.0, 10.0, 20.0, 50.0], 'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
             'model__epsilon': [0.01, 0.05, 0.1, 0.2]}, 15),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=2),
            {'n_estimators': [100, 150, 200], 'max_depth': [6, 10, 15, None], 'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4]}, 12),
        "LightGBM": (
            lgb.LGBMRegressor(n_estimators=600, random_state=RANDOM_STATE, n_jobs=2, verbose=-1),
            {'learning_rate': [0.01, 0.03, 0.05, 0.08], 'max_depth': [3, 4, 5, 6], 'num_leaves': [15, 31, 45],
             'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.7, 0.8, 0.9], 'reg_alpha': [0.0, 0.1, 1.0],
             'reg_lambda': [0.1, 1.0, 5.0]}, 25),
        "XGBoost": (
            xgb.XGBRegressor(n_estimators=600, random_state=RANDOM_STATE, n_jobs=2),
            {'learning_rate': [0.01, 0.03, 0.05, 0.08], 'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9, 1.0],
             'colsample_bytree': [0.7, 0.8, 0.9], 'reg_alpha': [0.0, 0.1, 1.0], 'reg_lambda': [1.0, 3.0, 5.0]}, 25),
    }


def n_arboles_optimo(nombre, modelo, X_train, y_train, split):
    """Parada temprana en un fold de validación de train; devuelve el número de árboles."""
    pos_tr, pos_val = split
    modelo_es = clone(modelo)
    eval_set = [(X_train[pos_val], y_train[pos_val])]
    if nombre == "LightGBM":
        modelo_es.fit(X_train[pos_tr], y_train[pos_tr], eval_set=eval_set, eval_metric="l1",
                      callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)])
        return max(1, modelo_es.best_iteration_)
    modelo_es.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_metric="mae")
    modelo_es.fit(X_train[pos_tr], y_train[pos_tr], eval_set=eval_set, verbose=False)
    return modelo_es.best_iteration + 1


def metricas(y, pred):
    return {
        'MAE': mean_absolute_error(y, pred),
        'MedAE': median_absolute_error(y, pred),
        'MaxAE': max_error(y, pred),
        'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
        'R2': r2_score(y, pred),
        'Sesgo': float(np.mean(pred - y)),
    }


ESCENARIOS = ["Definitivo (solo Hannum)", "Fusión sin Mixup", "Fusión con Mixup"]
MODELOS = list(config_modelos())
TESTS = ["Test Hannum", "Test GSE87571", "Test combinado"]
ARCHIVO_CSV = CARPETA_RESULTADOS / "entrenamiento_fusion_completo.csv"


def graficar(df):
    """Barras de MAE por modelo y escenario en cada test (leyenda común debajo)."""
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']
    fig, axes = plt.subplots(1, len(TESTS), figsize=(22, 5.5), sharey=True)
    ancho = 0.8 / len(ESCENARIOS)
    x = np.arange(len(MODELOS))
    for ax, test in zip(axes, TESTS):
        d = df[df['Test'] == test]
        for i, (esc, color) in enumerate(zip(ESCENARIOS, colores)):
            valores = d[d['Escenario'] == esc].set_index('Modelo').loc[MODELOS, 'MAE']
            barras = ax.bar(x + (i - 1) * ancho, valores, ancho, label=esc, color=color, edgecolor='black', alpha=0.85)
            ax.bar_label(barras, fmt='%.2f', fontsize=7.5, padding=2)
        ax.set_xticks(x, MODELOS, fontsize=9)
        ax.set_title(test, fontweight='bold')
    axes[0].set_ylabel("Error Absoluto Medio - MAE (Años)")
    axes[0].set_ylim(0, df['MAE'].max() * 1.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(ESCENARIOS), fontsize=11, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Pipeline Completo con Datasets Fusionados (tuning + Mixup, k={N_SONDAS_TOP} sondas)",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(CARPETA_RESULTADOS / "16_entrenamiento_fusion_completo.png", bbox_inches='tight', dpi=300)


# Regenerar solo la figura desde el CSV, sin reentrenar:  python 03_entrenamiento_fusion_completo.py --solo-grafico
if "--solo-grafico" in sys.argv:
    graficar(pd.read_csv(ARCHIVO_CSV))
    print(f"Figura regenerada en: {CARPETA_RESULTADOS}")
    sys.exit(0)


def entrenar_y_evaluar(escenario, X_train, y_train, cv_splits, tests):
    filas = []
    for nombre, (estimador, grid, n_iter) in config_modelos().items():
        t_inicio = time.perf_counter()
        search = RandomizedSearchCV(estimador, grid, n_iter=n_iter, scoring='neg_mean_absolute_error',
                                    cv=cv_splits, random_state=RANDOM_STATE, n_jobs=2, verbose=0)
        search.fit(X_train, y_train)
        modelo = search.best_estimator_
        if nombre in ("LightGBM", "XGBoost"):
            modelo.set_params(n_estimators=n_arboles_optimo(nombre, modelo, X_train, y_train, cv_splits[0]))
            modelo.fit(X_train, y_train)
        duracion = time.perf_counter() - t_inicio

        resumen = []
        for test, (X_te, y_te) in tests.items():
            m = metricas(y_te, modelo.predict(X_te))
            filas.append({'Escenario': escenario, 'Modelo': nombre, 'Test': test, 'N_train': len(y_train),
                          'Tiempo_s': duracion, **m})
            resumen.append(f"{test.replace('Test ', '')}: {m['MAE']:.2f}")
        print(f"      * {nombre:<13} MAE  " + " | ".join(resumen) + f"  ({duracion:.0f} s)")
    return filas


# ==============================================================================
# 1. CARGA
# ==============================================================================
print("1. Cargando Hannum y GSE87571...")
hannum = cargar_hannum()
gse = cargar_gse87571(hannum)
print(f"   -> Hannum   train {len(hannum.idx_train)} | test {len(hannum.idx_test)}")
print(f"   -> GSE87571 train {len(gse.idx_train)} | test {len(gse.idx_test)}")


def armar_tests(columnas):
    def X(ds, idx):
        return np.hstack([ds.matriz[np.ix_(idx, columnas)], ds.sexo_binario[idx]])
    X_h, y_h = X(hannum, hannum.idx_test), hannum.edades[hannum.idx_test]
    X_g, y_g = X(gse, gse.idx_test), gse.edades[gse.idx_test]
    return {"Test Hannum": (X_h, y_h), "Test GSE87571": (X_g, y_g),
            "Test combinado": (np.vstack([X_h, X_g]), np.concatenate([y_h, y_g]))}


filas = []

# ==============================================================================
# 2. ESCENARIO A: DEFINITIVO (solo Hannum, exactamente como era antes de la fusión)
# ==============================================================================
print("\n2. Escenario 'Definitivo (solo Hannum)'...")
p = hannum.particion
cols_def = rankear_sondas(p, p.idx_train)[:N_SONDAS_TOP]
X_train_def, y_train_def, _, _ = armar_matrices_con_mixup(p, cols_def)
filas += entrenar_y_evaluar("Definitivo (solo Hannum)", X_train_def, y_train_def, splits_sin_fuga_con_mixup(p),
                            armar_tests(cols_def))

# ==============================================================================
# 3. FUSIÓN: SONDAS, FOLDS Y MIXUP POR DATASET
# ==============================================================================
print("\n3. Fusionando trains, seleccionando sondas y generando Mixup por dataset...")
columnas = rankear_sondas_fuentes([(hannum.matriz, hannum.edades, hannum.idx_train, False),
                                   (gse.matriz, gse.edades, gse.idx_train, False)])[:N_SONDAS_TOP]

datasets = [hannum, gse]
X_reales_ds = [np.hstack([ds.matriz[:, columnas], ds.sexo_binario]) for ds in datasets]

# Reales de train fusionados (primero Hannum, luego GSE87571) y su fold estratificado por dataset
X_reales = np.vstack([X[ds.idx_train] for X, ds in zip(X_reales_ds, datasets)])
y_reales = np.concatenate([ds.edades[ds.idx_train] for ds in datasets])
etiqueta_ds = np.concatenate([np.full(len(ds.idx_train), i) for i, ds in enumerate(datasets)])
fold_real = np.empty(len(y_reales), dtype=int)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for k, (_, pos_val) in enumerate(skf.split(X_reales, etiqueta_ds)):
    fold_real[pos_val] = k

# Mixup dentro de cada dataset (mismo sexo, mismo laboratorio)
X_synth, y_synth, folds_padres = [], [], []
desplazamiento_pos = 0
for i, (ds, X_ds) in enumerate(zip(datasets, X_reales_ds)):
    pos_en_train = {idx: desplazamiento_pos + j for j, idx in enumerate(ds.idx_train)}
    np.random.seed(RANDOM_STATE + i)
    plan = planificar_mixup_estratificado(ds.edades, ds.sexos, indices_disponibles=ds.idx_train)
    X_synth += [interpolar(X_ds[s['idx_1']], X_ds[s['idx_2']], s['lam']) for s in plan]
    y_synth += [s['edad'] for s in plan]
    folds_padres += [(fold_real[pos_en_train[s['idx_1']]], fold_real[pos_en_train[s['idx_2']]]) for s in plan]
    desplazamiento_pos += len(ds.idx_train)
    print(f"   -> {ds.nombre}: {len(plan)} sintéticos")
X_synth = np.array(X_synth, dtype=np.float32)
y_synth = np.array(y_synth, dtype=np.float32)
folds_padres = np.array(folds_padres)


def splits_fusion(con_mixup):
    splits = []
    for k in range(N_FOLDS):
        pos_val = np.where(fold_real == k)[0]
        pos_tr = np.where(fold_real != k)[0]
        if con_mixup:
            pos_synth = len(y_reales) + np.where((folds_padres != k).all(axis=1))[0]
            pos_tr = np.concatenate([pos_tr, pos_synth])
        splits.append((pos_tr, pos_val))
    return splits


tests_fusion = armar_tests(columnas)

print("\n4. Escenario 'Fusión sin Mixup'...")
filas += entrenar_y_evaluar("Fusión sin Mixup", X_reales, y_reales, splits_fusion(False), tests_fusion)

print("\n5. Escenario 'Fusión con Mixup'...")
filas += entrenar_y_evaluar("Fusión con Mixup", np.vstack([X_reales, X_synth]), np.concatenate([y_reales, y_synth]),
                            splits_fusion(True), tests_fusion)

df = pd.DataFrame(filas)
df.to_csv(ARCHIVO_CSV, index=False)

# ==============================================================================
# 4. RESUMEN
# ==============================================================================
for test in TESTS:
    tabla = df[df['Test'] == test].pivot(index='Escenario', columns='Modelo', values='MAE').loc[ESCENARIOS, MODELOS]
    print("\n" + "=" * 95)
    print(f"MAE (años) en {test}")
    print("=" * 95)
    print(tabla.round(2).to_string())

# ==============================================================================
# 5. FIGURA
# ==============================================================================
graficar(df)
print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")
plt.show()
