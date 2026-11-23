# Modelo 1 — Reloj epigenético (Hannum GSE40279 + GSE87571)

Predicción de la edad cronológica a partir de perfiles de metilación del ADN en
sangre (Illumina 450K), entrenada con dos cohortes fusionadas. **Todos los datos
son pacientes reales**: el proyecto no usa datos sintéticos.

| Dataset | Muestras | Edades | Uso |
|---|---|---|---|
| GSE40279 (Hannum) | 656 | 19–101 | 75% train / 25% test |
| GSE87571 (Johansson) | 729 válidas de 732 | 14–94 | 75% train / 25% test |

La fusión se decidió con el experimento de `experimentos/fusion_GSE87571/`: mejora a
todos los modelos y corrige la mala generalización a otro laboratorio del modelo
entrenado solo con Hannum. Para volver a entrenar solo con Hannum, poner
`DATASETS = ("Hannum",)` en `src/datos.py`.

## Estructura

```
Modelo 1/
├── README.md
├── requirements.txt
├── src/                          Códigos (ejecutar en orden)
│   ├── config.py                 Rutas y estilo de gráficos compartidos
│   ├── datos.py                  Fusión de datasets, partición, folds sin fuga, DATASETS y N_SONDAS_TOP
│   ├── gse87571.py               Conversión y caché de GSE87571 (mismo orden de sondas que Hannum)
│   ├── modelos.py                Modelos de referencia con hiperparámetros fijos (04 y 06)
│   ├── 01_extraer_metadatos.py   GEO .soft.gz        -> Metadata_Pacientes.csv
│   ├── 02_consolidar_dataset.py  Chunks + Metadata   -> Metilacion_Completo.csv (+ Muestra_10)
│   ├── 03_analisis_exploratorio.py  EDA de cada dataset + comparación entre ellos
│   ├── 04_curva_sondas.py        ¿Con cuántas sondas mejora el modelo? (elige k por CV)
│   ├── 05_entrenamiento.py       5 modelos: ElasticNet, SVR, RF, LightGBM, XGBoost
│   ├── 06_curva_aprendizaje.py   ¿Hacen falta más pacientes?
│   └── 07_guardar_elasticnet.py  Entrena solo ElasticNet (~1 min) y lo guarda en modelos/
├── Datasets/                     Todos los datos (.csv y .soft.gz)
│   ├── cache_matriz_completa/    Hannum: matriz binaria de las 470.043 sondas CpG (se crea sola, ~1,2 GB)
│   ├── GSE87571/                 Archivos originales de GEO de GSE87571 (~2,8 GB)
│   └── cache_GSE87571/           GSE87571: matriz binaria en el orden de sondas de Hannum (se crea sola, ~1,4 GB)
├── resultados/                   Se crea al ejecutar
│   ├── eda/
│   │   ├── Hannum/               Figuras 1-3 (clustermap, co-metilación, PCA) y top30_biomarcadores.csv
│   │   ├── GSE87571/             Las mismas figuras y tabla para GSE87571
│   │   └── comparacion/          Figuras 4-7 (edades, efecto de laboratorio, señal de edad, edad por sexo) y comparacion_datasets.csv
│   ├── seleccion_sondas/         Figura 13 y curva_sondas.csv
│   ├── entrenamiento/            Figuras 11, 12 y 17, auditoria_alucinaciones_modelos.csv, auditoria_por_dataset.csv y metricas_matriz_confusion.csv
│   └── curva_aprendizaje/        Figura 14, curva_aprendizaje.csv y diagnostico_datos.csv
├── modelos/                      elasticnet_reloj_epigenetico.joblib y particion_pacientes_elasticnet.csv (paso 07)
├── experimentos/fusion_GSE87571/ Experimento que justificó la fusión y descartó el Mixup (figuras 15 y 16)
└── _respaldo_original/           Copia de los scripts antes de reorganizar
```

## Cómo ejecutar

Desde la carpeta del proyecto (las rutas son relativas a `src/config.py`,
así que funcionan desde cualquier carpeta):

```bash
python src/01_extraer_metadatos.py
python src/02_consolidar_dataset.py
python src/03_analisis_exploratorio.py
python src/04_curva_sondas.py
python src/05_entrenamiento.py
python src/06_curva_aprendizaje.py
python src/07_guardar_elasticnet.py
```

Si los datasets ya existen, se puede saltar directamente al paso 3, 4, 5 o 6.
Tras el paso 04, copiar el `k` recomendado en `N_SONDAS_TOP` de `src/datos.py`
(lo usan los pasos 05 y 06).

## Diagnóstico de datos (paso 06)

`06_curva_aprendizaje.py` entrena con fracciones crecientes de pacientes y calcula:

- **Ganancia estimada al duplicar los datos**: ajusta `MAE(n) = a·n^(-b) + c` y
  extrapola a 2n. `c` es el MAE mínimo alcanzable con estas variables.
- **Ganancia por cada 100 pacientes** en el último tramo de la curva.
- **Brecha validación − entrenamiento**: alta = sobreajuste (más datos ayudan);
  baja con error alto = sesgo (hacen falta mejores variables, no más pacientes).

## Métricas de prueba diagnóstica (paso 05)

Además del MAE y el R², `05_entrenamiento.py` evalúa el reloj como prueba
diagnóstica de **persona mayor (≥ 60 años, Ley 19.828)**: la edad real y la
predicha se clasifican con ese corte (`EDAD_CORTE`) y se arma la matriz de
confusión (figura 17). Reporta exactitud, sensibilidad, especificidad, VPP, VPN
y F1, cada proporción con su IC 95 % de Wilson (guía STARD), para el test
combinado y el de cada dataset (`metricas_matriz_confusion.csv`).

## Modelo guardado (paso 07)

`07_guardar_elasticnet.py` reproduce el ElasticNet del paso 05 (misma partición,
sondas, hiperparámetros y semilla; MAE test = 3,13) y guarda en `modelos/`:

- **`elasticnet_reloj_epigenetico.joblib`**: diccionario con el modelo entrenado
  (`modelo`), el orden de las 501 variables (`columnas`), las medianas para imputar
  faltantes (`medianas_imputacion`), hiperparámetros y métricas de test. El docstring
  del script trae un ejemplo para predecir pacientes nuevos.
- **`particion_pacientes_elasticnet.csv`**: ID de GEO de cada paciente, dataset, edad,
  sexo, si fue train o test, fold y predicción. En train la predicción es **fuera de
  pliegue** (el modelo no vio a ese paciente, y el ranking de sondas se recalcula sin
  él); es la que debe usarse para entrenar un modelo fusionado (stacking) sin fuga.
  Los pacientes de test no deben usarse para entrenar el modelo fusionado.

## Protocolo sin fuga de datos (pasos 04, 05 y 06)

Los pasos leen las **470.043 sondas CpG** del array de ambos datasets desde sus
cachés binarias, que se crean solas la primera vez (Hannum ~90 s desde el CSV
Completo; GSE87571 ~100 s desde los archivos de GEO) y se regeneran si los originales
cambian. Para usar solo las primeras n sondas, cambiar `N_SONDAS_CANDIDATAS` en
`src/datos.py`. Luego (`src/datos.py`):

1. Dentro de cada dataset se separa el 25% de los pacientes como test antes de
   cualquier otro paso; el test final es la unión de ambos.
2. La imputación por mediana usa solo el train de cada dataset; el ranking de sondas
   usa el train fusionado.
3. Validación cruzada de 5 folds estratificada por dataset.

Además:

- **04**: en cada fold el ranking de sondas se recalcula solo con los pacientes de
  entrenamiento de ese fold, y el `k` óptimo se elige solo por CV (regla 1-SE).
- **05**: LightGBM/XGBoost eligen el número de árboles por parada temprana en un
  fold de validación de train y luego se reentrenan con todo train. Reporta el test
  combinado y el de cada dataset por separado (`auditoria_por_dataset.csv`).

El test (pacientes no vistos) solo se usa para la evaluación final.

## Análisis exploratorio (paso 03)

Descriptivo, con todos los pacientes válidos y las 470.043 sondas (no participa en el
entrenamiento). Para cada dataset por separado: Spearman de cada sonda con la edad
(con corrección de Bonferroni), top 30 biomarcadores con Mann-Whitney por sexo,
clustermap, co-metilación y PCA. Luego compara ambos: distribución de edad y sexo,
PCA conjunto (sondas al azar = efecto de laboratorio; top sondas de edad = señal
biológica), la señal de edad compartida (ρ en Hannum frente a ρ en GSE87571) y la
edad por sexo con prueba de Mann-Whitney U (cada dataset y el fusionado).

## Aumentación con datos sintéticos (descartada)

Versiones anteriores agregaban pacientes sintéticos con Mixup (interpolación de dos
pacientes reales del mismo sexo). Se evaluó en tres versiones del modelo y nunca
mejoró el error (±0,1 años), así que se quitó del pipeline. La comparación queda
registrada en `experimentos/fusion_GSE87571/` (figura 16) y el script original en
`_respaldo_original/Aumentodataset.py`.

## Datos

| Archivo | Lo genera | Lo usa |
|---|---|---|
| `GSE40279_family.soft.gz` | descarga de GEO | 01 |
| `Metadata_Pacientes.csv` | 01 | 02 |
| `Dataset_Hannum_Metilacion_Chunks.csv` | script previo (no incluido) | 02 |
| `Dataset_Hannum_Metilacion_Completo.csv` | 02 | la caché de Hannum |
| `cache_matriz_completa/` (`.npy` + `pacientes.csv`) | 03-06 (automático) | 03, 04, 05, 06 |
| `GSE87571/` (series matrix, `matrix1of2`/`2of2`, xlsx) | descarga de GEO | la caché de GSE87571 |
| `cache_GSE87571/` (`.npy` + `pacientes.csv`) | 04-06 (automático) | 04, 05, 06 |
| `Dataset_Hannum_Muestra_10.csv` | 02 | inspección manual |
| `auditoria_10_lineas.csv`, `dataset_celulas_limpio.csv` | — | ningún script de este modelo (datos ImmPort SDY296) |

Para cambiar una ruta o un nombre de archivo, editar solo `src/config.py`.

## Equivalencia con los scripts antiguos

| Antes | Ahora |
|---|---|
| `Sexo.py` | `src/01_extraer_metadatos.py` |
| `consolidado.py` | `src/02_consolidar_dataset.py` |
| `Aumentodataset.py` | eliminado (datos sintéticos descartados) |
| `EDA.py` + `Graficos nuevo.py` | `src/03_analisis_exploratorio.py` (sin las figuras de validación del Mixup) |
| — (antes con valores fijos) | `src/04_curva_sondas.py` |
| `entrenamiento.py` | `src/05_entrenamiento.py` |
