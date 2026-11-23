"""
PASO 6 - Curva de aprendizaje: ¿necesitamos más pacientes de entrenamiento?

Se entrena con fracciones crecientes de los pacientes de train y se mide el MAE en
validación cruzada (5 folds sin fuga). Con eso se calculan tres diagnósticos:

  1. Ganancia al duplicar los datos: se ajusta una ley de potencia
     MAE(n) = a * n^(-b) + c  y se extrapola a 2n. Si la ganancia es grande,
     conseguir más pacientes vale la pena. 'c' estima el error mínimo alcanzable
     con estas variables aunque hubiera infinitos pacientes.
  2. Pendiente final: años de MAE que se ganan por cada 100 pacientes extra en
     el último tramo de la curva.
  3. Brecha validación - entrenamiento: brecha grande = varianza (sobreajuste,
     más datos ayudan); brecha chica con error alto = sesgo (hacen falta mejores
     variables o modelos, no más pacientes).

Sin fuga: en cada fold y fracción, el ranking de sondas se calcula solo con el
subconjunto de pacientes de entrenamiento usado.

Entrada : Datasets fusionados de datos.DATASETS (Hannum + GSE87571)
Salida  : resultados/curva_aprendizaje/
          - curva_aprendizaje.csv
          - diagnostico_datos.csv
          - 14_curva_aprendizaje.png
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from config import CARPETA_RESULTADOS_APRENDIZAJE, configurar_estilo_graficos, crear_carpeta
from datos import (
    N_FOLDS,
    N_SONDAS_TOP,
    RANDOM_STATE,
    matriz_reales,
    preparar_particion,
    rankear_sondas,
)
from modelos import modelos_referencia

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
FRACCIONES = [0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
UMBRAL_GANANCIA = 0.3   # Años de MAE que se ganarían al duplicar los datos para considerar que vale la pena
UMBRAL_BRECHA = 2.0     # Años entre validación y entrenamiento para considerar alta varianza

MODELOS = modelos_referencia()

configurar_estilo_graficos()
carpeta_salida = crear_carpeta(CARPETA_RESULTADOS_APRENDIZAJE)

# ==============================================================================
# 1. PARTICIÓN SIN FUGA
# ==============================================================================
print("1. Cargando pacientes y separando train / test...")
p = preparar_particion()
fold_reales = p.fold_reales_train
print(f"   -> Train: {len(p.idx_train)} pacientes | Sondas por modelo: {N_SONDAS_TOP}")

# ==============================================================================
# 2. ENTRENAMIENTO CON FRACCIONES CRECIENTES DE PACIENTES
# ==============================================================================
print("\n2. Entrenando con fracciones crecientes de pacientes (5 folds)...")
filas = []

for f in range(N_FOLDS):
    t_inicio = time.perf_counter()
    idx_val = p.idx_train[fold_reales == f]
    # Orden aleatorio fijo por fold: los subconjuntos quedan anidados (cada fracción contiene a la anterior)
    orden = np.random.RandomState(RANDOM_STATE + f).permutation(p.idx_train[fold_reales != f])

    for frac in FRACCIONES:
        idx_sub = orden[:int(round(frac * len(orden)))]

        # Ranking de sondas solo con el subconjunto de entrenamiento
        columnas = rankear_sondas(p, idx_sub)[:N_SONDAS_TOP]
        X_reales = matriz_reales(p, columnas)
        X_sub, y_sub = X_reales[idx_sub], p.edades[idx_sub]
        X_val, y_val = X_reales[idx_val], p.edades[idx_val]

        for nombre, modelo in MODELOS.items():
            m = clone(modelo).fit(X_sub, y_sub)
            filas.append({
                'Fold': f,
                'Fraccion': frac,
                'N_pacientes': len(idx_sub),
                'Modelo': nombre,
                'MAE_Val': mean_absolute_error(y_val, m.predict(X_val)),
                'MAE_Train': mean_absolute_error(y_sub, m.predict(X_sub)),
            })

    print(f"   -> Fold {f + 1}/{N_FOLDS} completado ({time.perf_counter() - t_inicio:.1f} s)")

df_folds = pd.DataFrame(filas)
df_curva = (
    df_folds.groupby(['Modelo', 'Fraccion'], sort=False)
    .agg(N_pacientes=('N_pacientes', 'mean'),
         MAE_Val=('MAE_Val', 'mean'),
         MAE_Val_SE=('MAE_Val', lambda x: x.std(ddof=1) / np.sqrt(len(x))),
         MAE_Train=('MAE_Train', 'mean'))
    .reset_index()
)
df_curva.to_csv(os.path.join(carpeta_salida, "curva_aprendizaje.csv"), index=False)

# ==============================================================================
# 3. DIAGNÓSTICO: ¿MÁS DATOS AYUDARÍAN?
# ==============================================================================
def ley_potencia(n, a, b, c):
    return a * np.power(n, -b) + c


def ajustar_ley_potencia(n, mae):
    """Ajusta MAE(n) = a·n^(-b) + c. Devuelve los parámetros o None si no converge."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, _ = curve_fit(
                ley_potencia, n, mae,
                p0=(mae[0] * np.sqrt(n[0]), 0.5, mae.min() * 0.8),
                bounds=([0, 0.01, 0], [np.inf, 3.0, mae.min()]),
                maxfev=20000,
            )
        return params
    except (RuntimeError, ValueError):
        return None


diagnostico = []
ajustes = {}
for nombre, df_m in df_curva.groupby('Modelo', sort=False):
    n = df_m['N_pacientes'].values
    mae_val = df_m['MAE_Val'].values
    n_max = n[-1]

    params = ajustar_ley_potencia(n, mae_val)
    ajustes[nombre] = params
    if params is not None:
        ganancia_x2 = ley_potencia(n_max, *params) - ley_potencia(2 * n_max, *params)
        mae_asintota = params[2]
    else:
        ganancia_x2, mae_asintota = np.nan, np.nan

    # Pendiente del último tramo (desde la fracción 0.7 hasta 1.0)
    i_07 = FRACCIONES.index(0.7)
    pendiente_100 = (mae_val[i_07] - mae_val[-1]) / (n[-1] - n[i_07]) * 100
    brecha = mae_val[-1] - df_m['MAE_Train'].values[-1]

    diagnostico.append({
        'Modelo': nombre,
        'N_pacientes_max': int(n_max),
        'MAE_Val_actual': round(mae_val[-1], 2),
        'MAE_Train_actual': round(df_m['MAE_Train'].values[-1], 2),
        'Brecha_Val_Train': round(brecha, 2),
        'Ganancia_por_100_pacientes': round(pendiente_100, 2),
        'Ganancia_estimada_x2_datos': round(ganancia_x2, 2),
        'MAE_minimo_estimado': round(mae_asintota, 2),
    })

df_diag = pd.DataFrame(diagnostico)
df_diag.to_csv(os.path.join(carpeta_salida, "diagnostico_datos.csv"), index=False)

print("\n" + "="*110)
print("DIAGNÓSTICO: ¿NECESITAMOS MÁS DATOS DE ENTRENAMIENTO? (validación cruzada)")
print("="*110)
print(df_diag.to_string(index=False))

print("\nInterpretación:")
for _, fila in df_diag.iterrows():
    if np.isnan(fila['Ganancia_estimada_x2_datos']):
        datos_txt = "no se pudo ajustar la ley de potencia"
    elif fila['Ganancia_estimada_x2_datos'] >= UMBRAL_GANANCIA:
        datos_txt = f"MÁS PACIENTES AYUDARÍAN (~{fila['Ganancia_estimada_x2_datos']:.2f} años menos al duplicar)"
    else:
        datos_txt = f"curva casi saturada (~{fila['Ganancia_estimada_x2_datos']:.2f} años al duplicar)"

    if fila['Brecha_Val_Train'] >= UMBRAL_BRECHA:
        brecha_txt = "brecha alta -> varianza/sobreajuste"
    else:
        brecha_txt = "brecha baja -> el límite es el sesgo (variables/modelo)"
    print(f"   - {fila['Modelo']:<11}: {datos_txt}; {brecha_txt}.")

# ==============================================================================
# 4. FIGURA: CURVAS DE APRENDIZAJE
# ==============================================================================
color = '#1f77b4'
fig, axes = plt.subplots(1, len(MODELOS), figsize=(6 * len(MODELOS), 5.5), sharey=True)

for ax, nombre in zip(axes, MODELOS):
    df_m = df_curva[df_curva['Modelo'] == nombre]
    n = df_m['N_pacientes'].values
    ax.plot(n, df_m['MAE_Val'], '-o', color=color, lw=2.2, ms=6, label="Validación")
    ax.fill_between(n, df_m['MAE_Val'] - df_m['MAE_Val_SE'], df_m['MAE_Val'] + df_m['MAE_Val_SE'],
                    color=color, alpha=0.15)
    ax.plot(n, df_m['MAE_Train'], '--', color='#ff7f0e', lw=1.8, label="Entrenamiento")

    params = ajustes[nombre]
    if params is not None:
        n_ext = np.linspace(n[-1], 2 * n[-1], 50)
        ax.plot(n_ext, ley_potencia(n_ext, *params), ':', color=color, lw=2.2, label="Extrapolación (2× pacientes)")

    ax.set_title(nombre, fontweight='bold')
    ax.set_xlabel("Pacientes de entrenamiento")
    ax.legend(loc='upper right', fontsize=9)

axes[0].set_ylabel("Error Absoluto Medio - MAE (Años)")
fig.suptitle(f"Curva de Aprendizaje: ¿Más Pacientes Mejoran el Modelo? (k={N_SONDAS_TOP} sondas, CV 5 folds)",
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
ruta_figura = os.path.join(carpeta_salida, "14_curva_aprendizaje.png")
fig.savefig(ruta_figura, bbox_inches='tight', dpi=300)

print(f"\nResultados guardados en: {carpeta_salida}")
plt.show()
