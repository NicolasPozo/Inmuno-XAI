# Modelo 2 — Edad inmunológica (Stanford 10K Immunomes)

Predicción de la edad cronológica a partir del perfil inmune: 16 poblaciones
celulares (CyTOF) y 29 citoquinas en suero, más el sexo. **Todos los datos de
entrenamiento son pacientes reales**: el aumento con datos sintéticos se probó
y se descartó (etapa 1 de la documentación).

**Modelo final:** TabPFN v2 como un solo modelo, entrenado con 354 pacientes.

| Evaluación | MAE (años) | R² |
|---|---|---|
| Validación cruzada agrupada 5×5 (473 pacientes) | 6,34 | 0,848 |
| Test (119 pacientes) | 6,52 | 0,828 |
| Estudios no vistos (promedio de 5) | 13,5 | — |

La historia completa (qué se probó, por qué se eligió este modelo, explicabilidad
y limitaciones) está en `DOCUMENTACION_MODELO_FINAL.md`.

## Estructura

```
Modelo 2/
├── README.md
├── DOCUMENTACION_MODELO_FINAL.md  Documentación completa del modelo (12 etapas)
├── requirements.txt
├── src/                           Pipeline del modelo final (ejecutar en orden)
│   ├── config.py                  Rutas de todos los archivos del proyecto
│   ├── 01_ampliar_dataset.py      ID de paciente de cada fila del train k45 (+ train ampliado, descartado)
│   ├── 02_preparar_train_reales.py      Train solo con los 354 pacientes reales
│   ├── 03_reconstruir_dataset_completo.py  Los mismos pacientes con todas las variables del 10K
│   ├── 04_construir_fuentes_unimodales.py  pareados.csv (tabla de trabajo) y fuentes de una modalidad
│   ├── 05_comparar_modelos.py     Compara 7 modelos con la misma evaluación (5 clásicos, dos ramas, TabPFN; ~15 min)
│   ├── 06_modelo_final_tabpfn.py  Entrena, evalúa y guarda el modelo final (CV 5×5, test, estudios no vistos)
│   ├── 07_graficos_modelo_final.py      Edad real vs predicha y evolución del error
│   ├── 08_explicabilidad.py       Importancias, dependencia parcial, explicación por paciente, brecha (~45 min)
│   ├── 09_exportar_dashboard.py   JSON del dashboard
│   └── 10_figuras_informe.py      Figuras, cifras y tiempos del Experimento A para el informe de título
├── Datasets/
│   ├── df_train_aumentado_k45.csv, df_test_puro_k45.csv   Datos de partida (no se modifican)
│   ├── df_train_reales_k45.csv, df_*_completo.csv, pareados.csv   Generados por los pasos 02-04
│   ├── df_train_aumentado_k45_ampliado.csv, _fluprint.csv  Trains ampliados (experimentos descartados)
│   ├── 10k/                       Stanford 10K Immunomes: zip, archivos extraídos e IDs de pacientes
│   ├── fluprint/, soundlife/, immport/   Cohortes externas probadas en los experimentos
│   │                              (de Sound Life solo queda el archivo armonizado; los crudos se borraron)
│   └── fuentes_celulas/, fuentes_citoquinas/, fuentes_immport/   Datos de una sola modalidad
├── modelos/                       modelo_tabpfn.joblib y variables_modelo.csv (paso 06)
├── resultados/
│   ├── comparacion_modelos/       Métricas, predicciones y gráfico de los 7 modelos (paso 05)
│   ├── modelo_final/              Métricas, predicciones y gráficos (pasos 06 y 07)
│   ├── explicabilidad/            Resultados del paso 08; dashboard/ con la página y los JSON (paso 09)
│   └── informe/                   Figuras y cifras del informe de título (paso 10)
├── experimentos/                  Etapas que llevaron al modelo final, cada una con sus resultados/
│   ├── 01_modelo_inicial/         Etapas 1 y 4: los 5 modelos clásicos con cada train (Entrenamiento.py)
│   ├── 02_mas_pacientes/          Etapas 2-3: FluPRINT y curva de aprendizaje
│   ├── 03_banco_pruebas/          Etapa 5: ideas para bajar el error, Sound Life
│   ├── 04_dos_ramas/              Etapas 6-7: dos modelos (células / citoquinas) + fusión, firma del estudio
│   ├── 05_modelo_dos_ramas/       Etapa 8: el modelo final anterior (dos ramas) y la prueba con ImmPort
│   ├── 06_tabpfn/                 Etapas 9-11: TabPFN, TabM y RealMLP; TabPFN como un solo modelo
│   └── 07_numero_variables/       Etapa 12: cuántas variables necesita cada modelo (se mantienen las 45)
└── _respaldo_original/            Copia de los scripts antes de reorganizar (incluye Entrenamiento_original.py)
```

## Cómo ejecutar

Desde cualquier carpeta (las rutas se calculan desde `src/config.py`):

```bash
python -X utf8 src/01_ampliar_dataset.py
python -X utf8 src/02_preparar_train_reales.py
python -X utf8 src/03_reconstruir_dataset_completo.py
python -X utf8 src/04_construir_fuentes_unimodales.py
python -X utf8 src/05_comparar_modelos.py
python -X utf8 src/06_modelo_final_tabpfn.py
python -X utf8 src/07_graficos_modelo_final.py
python -X utf8 src/08_explicabilidad.py
python -X utf8 src/09_exportar_dashboard.py
python -X utf8 src/10_figuras_informe.py
```

- Si `Datasets/pareados.csv` ya existe, se puede empezar en el paso 05.
- El paso 10 genera las figuras del Experimento A que usa el informe de título
  (`Modelo 1/Informe`) y necesita los resultados del paso 05; con `--sin-tiempos`
  se salta la medición de tiempos (~2 min).
- El paso 05 muestra por qué se eligió TabPFN; los pasos siguientes no lo
  necesitan, así que se puede saltar para solo reentrenar el modelo final.
- Algunos pasos usan cosas de `experimentos/`, así que esa carpeta no debe borrarse:
  el 05 ejecuta parte de `05_modelo_dos_ramas/modelo_final_dos_ramas.py` para
  entrenar el modelo de dos ramas; el 06 compara con las predicciones guardadas de
  ese modelo; el 07 y el 09 leen resultados para el gráfico de evolución y el dashboard.
- TabPFN en CPU rechaza más de 1.000 filas por defecto; los scripts que lo
  necesitan ya activan `TABPFN_ALLOW_CPU_LARGE_DATASET=1` por su cuenta.

Los scripts de `experimentos/` se ejecutan igual, por ejemplo
`python -X utf8 experimentos/06_tabpfn/tabpfn_un_modelo.py`. Algunos dependen de
otros (por ejemplo, los de `06_tabpfn/` ejecutan la primera parte de
`05_modelo_dos_ramas/modelo_final_dos_ramas.py`); el encabezado de cada script
dice qué necesita.

## Rutas

Todas las rutas están en `src/config.py`:

- Los scripts de `src/` usan constantes (`ARCHIVO_PAREADOS`, `CARPETA_RES_MODELO_FINAL`, …).
- Los scripts de `experimentos/` piden cada archivo por su **nombre antiguo** con
  `ruta("pareados.csv")` o `ruta("resultados_dos_ramas", "x.csv")`; el diccionario
  `UBICACION` lo traduce a la carpeta nueva. Por eso sus comentarios todavía usan
  los nombres antiguos (tabla de abajo).

Para mover un archivo o carpeta, basta con cambiarlo en `src/config.py`.

## Equivalencia con la estructura antigua

| Antes (todo en la carpeta principal) | Ahora |
|---|---|
| `df_*.csv`, `pareados.csv` | `Datasets/` |
| `datos_10k/`, `datos_fluprint/`, `datos_soundlife/`, `datos_immport/` | `Datasets/10k/`, `fluprint/`, `soundlife/`, `immport/` |
| `fuentes_celulas/`, `fuentes_citoquinas/`, `fuentes_immport/` | `Datasets/` (mismo nombre) |
| `ampliar_dataset.py`, `preparar_train_reales.py`, `reconstruir_dataset_completo.py`, `construir_fuentes_unimodales.py` | `src/01` a `src/04` |
| `modelo_final_tabpfn.py`, `graficos_modelo_final_tabpfn.py`, `explicabilidad_tabpfn.py`, `exportar_dashboard.py` | `src/06` a `src/09` (`src/05_comparar_modelos.py` es nuevo) |
| `resultados_modelo_final_tabpfn/` | `resultados/modelo_final/` (el `.joblib` y `variables_modelo.csv` en `modelos/`) |
| `resultados_explicabilidad/` | `resultados/explicabilidad/` |
| `graficos_entrenamiento_multimodal*/` | `experimentos/01_modelo_inicial/resultados/` (mismo nombre) |
| `datos_10k/log_*.txt`, `datos_fluprint/log_entrenamiento_fluprint.txt` | `experimentos/01_modelo_inicial/resultados/logs/` |
| `graficos_curva_aprendizaje/` | `experimentos/02_mas_pacientes/resultados/` |
| `resultados_banco_pruebas/` | `experimentos/03_banco_pruebas/resultados/` |
| `resultados_dos_ramas/` | `experimentos/04_dos_ramas/resultados/` |
| `resultados_modelo_final/`, `_con_elisa/`, `_immport/` | `experimentos/05_modelo_dos_ramas/resultados/modelo_final/`, `modelo_final_con_elisa/`, `modelo_final_immport/` |
| `resultados_modelos_pesados/`, `resultados_tabpfn_un_modelo/` | `experimentos/06_tabpfn/resultados/modelos_pesados/`, `tabpfn_un_modelo/` |
| `Entrenamiento_original.py` | `_respaldo_original/` |

## Copia en XAI

`D:\Claude Code\Titulo\XAI` tiene una copia del modelo final, sus resultados, la
documentación y el dashboard (junto con el modelo epigenético de Modelo 1). Los
originales y los scripts están aquí.
