"""
PASO 4 - Curva de selección de sondas: ¿con cuántas sondas CpG mejora el modelo?

Para cada número de sondas k se mide el MAE por validación cruzada (5 folds sin
fuga) y se elige el k óptimo SOLO con esa validación. El MAE en test se calcula
como referencia, pero no participa en la elección.

Sin fuga de datos: en cada fold, el ranking de sondas (|Spearman ρ| con la edad) se
calcula solo con los pacientes de entrenamiento de ese fold (ver src/datos.py).

Los hiperparámetros de cada modelo son fijos (sin tuning por k) para que la curva
refleje solo el efecto del número de sondas.

Entrada : Datasets fusionados de datos.DATASETS (Hannum + GSE87571)
Salida  : resultados/seleccion_sondas/
          - curva_sondas.csv
          - 13_curva_optimizacion_sondas.png
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from config import CARPETA_RESULTADOS_SONDAS, configurar_estilo_graficos, crear_carpeta
from datos import (
    N_FOLDS,
    armar_matrices,
    preparar_particion,
    rankear_sondas,
    splits_sin_fuga,
)
from modelos import modelos_referencia

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
K_VALORES = [10, 20, 35, 50, 75, 100, 130, 160, 200, 250, 300, 400, 500, 750, 1000]

MODELOS = modelos_referencia()

configurar_estilo_graficos()
carpeta_salida = crear_carpeta(CARPETA_RESULTADOS_SONDAS)

# ==============================================================================
# 1. PARTICIÓN SIN FUGA Y RANKINGS DE SONDAS
# ==============================================================================
print("1. Cargando pacientes reales y separando train / test...")
p = preparar_particion()
cv_splits = splits_sin_fuga(p)
print(f"   -> Train: {len(p.idx_train)} pacientes | Test: {len(p.idx_test)} pacientes")

print("\n2. Calculando ranking de sondas por fold (solo pacientes de entrenamiento de cada fold)...")
fold_reales = p.fold_reales_train
rankings_fold = [rankear_sondas(p, p.idx_train[fold_reales != f]) for f in range(N_FOLDS)]
ranking_train = rankear_sondas(p, p.idx_train)

# ==============================================================================
# 2. CURVA MAE vs NÚMERO DE SONDAS
# ==============================================================================
print("\n3. Evaluando modelos para cada número de sondas k...")
filas = []

for k in K_VALORES:
    t_inicio = time.perf_counter()
    errores_cv = {nombre: [] for nombre in MODELOS}

    # Validación cruzada: cada fold con su propio ranking de sondas
    for f in range(N_FOLDS):
        X_train_f, y_train_f, _, _ = armar_matrices(p, rankings_fold[f][:k])
        pos_tr, pos_val = cv_splits[f]
        for nombre, modelo in MODELOS.items():
            m = clone(modelo).fit(X_train_f[pos_tr], y_train_f[pos_tr])
            errores_cv[nombre].append(mean_absolute_error(y_train_f[pos_val], m.predict(X_train_f[pos_val])))

    # Referencia en test (no se usa para elegir k)
    X_train, y_train, X_test, y_test = armar_matrices(p, ranking_train[:k])
    for nombre, modelo in MODELOS.items():
        m = clone(modelo).fit(X_train, y_train)
        filas.append({
            'k': k,
            'Modelo': nombre,
            'MAE_CV': np.mean(errores_cv[nombre]),
            'MAE_CV_SE': np.std(errores_cv[nombre], ddof=1) / np.sqrt(N_FOLDS),
            'MAE_Test': mean_absolute_error(y_test, m.predict(X_test)),
        })

    resumen_k = " | ".join(f"{n}: {np.mean(e):.2f}" for n, e in errores_cv.items())
    print(f"   -> k={k:>4}  MAE CV  {resumen_k}  ({time.perf_counter() - t_inicio:.1f} s)")

df_curva = pd.DataFrame(filas)
df_curva.to_csv(os.path.join(carpeta_salida, "curva_sondas.csv"), index=False)

# ==============================================================================
# 3. ELECCIÓN DEL k ÓPTIMO (SOLO CON VALIDACIÓN CRUZADA)
# ==============================================================================
# - k mínimo: el de menor MAE en CV.
# - k regla 1-SE: el menor k cuyo MAE en CV no supera (mínimo + 1 error estándar);
#   da un modelo más simple con un rendimiento estadísticamente equivalente.
resumen = []
for nombre, df_m in df_curva.groupby('Modelo', sort=False):
    fila_min = df_m.loc[df_m['MAE_CV'].idxmin()]
    umbral = fila_min['MAE_CV'] + fila_min['MAE_CV_SE']
    fila_1se = df_m[df_m['MAE_CV'] <= umbral].sort_values('k').iloc[0]
    resumen.append({
        'Modelo': nombre,
        'k_min_CV': int(fila_min['k']),
        'MAE_CV_min': round(fila_min['MAE_CV'], 2),
        'k_regla_1SE': int(fila_1se['k']),
        'MAE_CV_1SE': round(fila_1se['MAE_CV'], 2),
        'MAE_Test_en_k_1SE': round(fila_1se['MAE_Test'], 2),
    })
df_resumen = pd.DataFrame(resumen).sort_values('MAE_CV_min')

mejor = df_resumen.iloc[0]
k_recomendado = int(mejor['k_regla_1SE'])

print("\n" + "="*95)
print("SELECCIÓN DEL NÚMERO DE SONDAS (elegido solo con validación cruzada)")
print("="*95)
print(df_resumen.to_string(index=False))
print(f"\n-> Recomendación: k = {k_recomendado} sondas (mejor modelo en CV: {mejor['Modelo']}, regla 1-SE).")
print("   Para usarlo, cambia N_SONDAS_TOP en src/datos.py.")

# ==============================================================================
# 4. FIGURA: CURVA DE OPTIMIZACIÓN
# ==============================================================================
colores = dict(zip(MODELOS, ['#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c']))
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for nombre, df_m in df_curva.groupby('Modelo', sort=False):
    c = colores[nombre]
    axes[0].plot(df_m['k'], df_m['MAE_CV'], '-o', color=c, lw=2.2, ms=6, label=nombre)
    axes[0].fill_between(df_m['k'], df_m['MAE_CV'] - df_m['MAE_CV_SE'], df_m['MAE_CV'] + df_m['MAE_CV_SE'],
                         color=c, alpha=0.12)
    axes[1].plot(df_m['k'], df_m['MAE_Test'], '-o', color=c, lw=2.2, ms=6, label=nombre)

for ax in axes:
    ax.axvline(k_recomendado, color='red', linestyle='--', lw=2, label=f'Óptimo CV (k={k_recomendado})')
    ax.set_xlabel("Número de Sondas CpG Incluidas en el Modelo (k)", fontsize=11)
    ax.legend(loc='upper right')

axes[0].set_ylabel("Error Absoluto Medio - MAE (Años)", fontsize=11)
axes[0].set_title("Validación Cruzada sin Fuga (5 folds, media ± EE)\nUsada para elegir k", fontweight='bold')
axes[1].set_title("Test Real Independiente\nSolo referencia, no se usa para elegir k", fontweight='bold')

fig.suptitle("Curva de Optimización de Características: Selección Óptima de Sondas CpG",
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
ruta_figura = os.path.join(carpeta_salida, "13_curva_optimizacion_sondas.png")
fig.savefig(ruta_figura, bbox_inches='tight', dpi=300)

print(f"\nResultados guardados en: {carpeta_salida}")
plt.show()
