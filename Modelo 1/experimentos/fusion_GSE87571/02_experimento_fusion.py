"""
EXPERIMENTO - Paso 2: ¿mejora el reloj epigenético si fusionamos Hannum con GSE87571?

Escenarios de entrenamiento (todos con las mismas reglas sin fuga):
  1. Solo Hannum                    -> equivalente al modelo definitivo
  2. Solo GSE87571
  3. Fusión (Hannum + GSE87571)
  4. Fusión + corrección de lote    -> resta a GSE87571 el desplazamiento de cada sonda
                                       respecto a Hannum, estimado con un modelo lineal
                                       beta ~ edad + sexo + lote (edad y sexo protegidos)

Cada escenario se evalúa en DOS test independientes, nunca usados para entrenar ni elegir nada:
  - Test Hannum    : los mismos 164 pacientes del test de Hannum del pipeline.
  - Test GSE87571  : 25% de GSE87571 (validación externa para 'Solo Hannum').

Sin fuga: imputación, ranking de sondas (top N_SONDAS_TOP por |Spearman ρ|) y corrección
de lote se calculan solo con las muestras de entrenamiento de cada escenario.
Se usan los modelos de referencia con hiperparámetros fijos (src/modelos.py) y sin Mixup
(la curva de aprendizaje mostró que no aporta), para que la comparación sea justa.

No modifica nada del proyecto principal: solo LEE la caché de Hannum.

Entrada : datos/ (generado por 01_convertir_GSE87571.py) + caché de Hannum
Salida  : resultados/resultados_fusion.csv
          resultados/15_fusion_GSE87571.png
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score

from comun import CARPETA_RESULTADOS, cargar_gse87571, cargar_hannum, rankear_sondas_fuentes
from config import configurar_estilo_graficos, crear_carpeta
from comun import N_SONDAS_EXPERIMENTO as N_SONDAS_TOP
from datos import BLOQUE_RANKING
from modelos import modelos_referencia

crear_carpeta(CARPETA_RESULTADOS)
configurar_estilo_graficos()

# ==============================================================================
# 1. CARGA DE AMBOS DATASETS
# ==============================================================================
print("1. Cargando Hannum (misma partición que el modelo definitivo)...")
hannum = cargar_hannum()
H, edad_H, sexo_H = hannum.matriz, hannum.edades, hannum.sexo_binario
h_tr, h_te = hannum.idx_train, hannum.idx_test
print(f"   -> Hannum: train {len(h_tr)} | test {len(h_te)} | edad {edad_H.min():.0f}-{edad_H.max():.0f}")

print("\n2. Cargando GSE87571 (imputación con medianas de su train)...")
gse = cargar_gse87571(hannum)
G, edad_G, sexo_G = gse.matriz, gse.edades, gse.sexo_binario
g_tr, g_te = gse.idx_train, gse.idx_test
print(f"   -> GSE87571: train {len(g_tr)} | test {len(g_te)} | "
      f"edad {edad_G[np.concatenate([g_tr, g_te])].min():.0f}-{edad_G[np.concatenate([g_tr, g_te])].max():.0f}")

# ==============================================================================
# 2. CORRECCIÓN DE LOTE (solo con muestras de entrenamiento)
# ==============================================================================
print("\n3. Estimando desplazamiento de lote por sonda (beta ~ edad + sexo + lote, solo train)...")
diseno = np.column_stack([
    np.ones(len(h_tr) + len(g_tr)),
    np.concatenate([edad_H[h_tr], edad_G[g_tr]]),
    np.concatenate([sexo_H[h_tr, 0], sexo_G[g_tr, 0]]),
    np.concatenate([np.zeros(len(h_tr)), np.ones(len(g_tr))]),
])
proyeccion_lote = np.linalg.pinv(diseno)[3]   # fila que estima el coeficiente de 'lote'
desplazamiento = np.empty(G.shape[1], dtype=np.float32)
for inicio in range(0, G.shape[1], BLOQUE_RANKING):
    fin = inicio + BLOQUE_RANKING
    Y = np.vstack([H[h_tr, inicio:fin], G[g_tr, inicio:fin]])
    desplazamiento[inicio:fin] = proyeccion_lote @ Y
print(f"   -> |desplazamiento| medio: {np.abs(desplazamiento).mean():.4f} | "
      f"percentil 95: {np.percentile(np.abs(desplazamiento), 95):.4f} (escala beta 0-1)")


# ==============================================================================
# 3. ESCENARIOS
# ==============================================================================
# Cada fuente: (matriz, edades, sexo, índices, aplicar_corrección)
ESCENARIOS = {
    "Solo Hannum": [(H, edad_H, sexo_H, h_tr, False)],
    "Solo GSE87571": [(G, edad_G, sexo_G, g_tr, False)],
    "Fusión": [(H, edad_H, sexo_H, h_tr, False), (G, edad_G, sexo_G, g_tr, False)],
    "Fusión + corrección lote": [(H, edad_H, sexo_H, h_tr, False), (G, edad_G, sexo_G, g_tr, True)],
}


def armar_X(M, sexo, idx, columnas, corr):
    X = M[np.ix_(idx, columnas)]
    if corr:
        X = X - desplazamiento[columnas]
    return np.hstack([X, sexo[idx]])


def metricas(y, pred):
    return {'MAE': mean_absolute_error(y, pred), 'R2': r2_score(y, pred), 'Sesgo': float(np.mean(pred - y))}


print(f"\n4. Entrenando y evaluando escenarios (top {N_SONDAS_TOP} sondas, modelos de referencia)...")
MODELOS = modelos_referencia()
filas = []
for escenario, fuentes in ESCENARIOS.items():
    t_inicio = time.perf_counter()
    columnas = rankear_sondas_fuentes([(M, e, idx, corr) for M, e, _, idx, corr in fuentes],
                                      desplazamiento)[:N_SONDAS_TOP]
    X_tr = np.vstack([armar_X(M, s, idx, columnas, corr) for M, _, s, idx, corr in fuentes])
    y_tr = np.concatenate([e[idx] for _, e, _, idx, _ in fuentes])

    corregir_test_gse = any(corr for *_, corr in fuentes)
    tests = {
        "Test Hannum": (armar_X(H, sexo_H, h_te, columnas, False), edad_H[h_te]),
        "Test GSE87571": (armar_X(G, sexo_G, g_te, columnas, corregir_test_gse), edad_G[g_te]),
    }
    for nombre, modelo in MODELOS.items():
        m = clone(modelo).fit(X_tr, y_tr)
        for test, (X_te, y_te) in tests.items():
            filas.append({'Escenario': escenario, 'Modelo': nombre, 'Test': test,
                          'N_train': len(y_tr), **metricas(y_te, m.predict(X_te))})
    print(f"   -> {escenario:<26} train={len(y_tr):>4}  ({time.perf_counter() - t_inicio:.0f} s)")

df = pd.DataFrame(filas)
df.to_csv(CARPETA_RESULTADOS / "resultados_fusion.csv", index=False)

# ==============================================================================
# 4. RESUMEN
# ==============================================================================
orden_esc = list(ESCENARIOS)
for test in ["Test Hannum", "Test GSE87571"]:
    tabla = (df[df['Test'] == test]
             .pivot(index='Escenario', columns='Modelo', values='MAE')
             .loc[orden_esc, list(MODELOS)].round(2))
    sesgo = (df[df['Test'] == test].groupby('Escenario')['Sesgo'].mean().loc[orden_esc].round(2))
    tabla['Sesgo medio (años)'] = sesgo
    print("\n" + "=" * 95)
    print(f"MAE (años) en {test}")
    print("=" * 95)
    print(tabla.to_string())

# ==============================================================================
# 5. FIGURA
# ==============================================================================
colores = ['#1f77b4', '#9467bd', '#ff7f0e', '#2ca02c']
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
ancho = 0.8 / len(orden_esc)
x = np.arange(len(MODELOS))
for ax, test in zip(axes, ["Test Hannum", "Test GSE87571"]):
    d = df[df['Test'] == test]
    for i, (esc, color) in enumerate(zip(orden_esc, colores)):
        valores = d[d['Escenario'] == esc].set_index('Modelo').loc[list(MODELOS), 'MAE']
        barras = ax.bar(x + (i - (len(orden_esc) - 1) / 2) * ancho, valores, ancho, label=esc, color=color,
                        edgecolor='black', alpha=0.85)
        ax.bar_label(barras, fmt='%.2f', fontsize=7.5, padding=2)
    ax.set_xticks(x, list(MODELOS))
    ax.set_title(test, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
axes[0].set_ylabel("Error Absoluto Medio - MAE (Años)")
fig.suptitle(f"Fusión Hannum + GSE87571: ¿Mejora el Reloj Epigenético? (k={N_SONDAS_TOP} sondas)",
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(CARPETA_RESULTADOS / "15_fusion_GSE87571.png", bbox_inches='tight', dpi=300)

print(f"\nResultados guardados en: {CARPETA_RESULTADOS}")
plt.show()
