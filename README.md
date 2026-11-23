# Immuno-XAI

**Dos relojes de edad biológica explicables, a partir de la sangre.** Proyecto de título.

Un reloj estima la edad desde el **sistema inmune** y el otro desde la **metilación del ADN**. Los dos se entrenan y se validan por separado y **nunca se fusionan**: no existe un dataset público con ambos tipos de datos para las mismas personas. Cada predicción se explica con **SHAP**: cuántos años suma o resta cada variable.

| Reloj | Datos | Modelo | Error en el test (MAE) | R² test |
|---|---|---|---|---|
| **Inmunológico** | Stanford 10K Immunomes: 16 poblaciones celulares (CyTOF) + 29 citoquinas + sexo, 473 pacientes | TabPFN v2 | **6,52 años** | 0,828 |
| **Epigenético** | GEO GSE40279 (Hannum) + GSE87571: 500 sondas CpG + sexo, 1.385 perfiles | ElasticNet | **3,13 años** | 0,953 |

Del reloj inmunológico también se reporta la validación cruzada agrupada 5×5 (6,34 años) y la prueba con estudios no vistos (13,5 años), que muestra cuánto se degrada con datos de otro laboratorio.

## Estructura del repositorio

```
├── Modelo 1/   Reloj epigenético: extracción de GEO, análisis exploratorio, selección de sondas,
│               comparación de 5 algoritmos y ElasticNet final
├── Modelo 2/   Reloj inmunológico: preparación del 10K, 12 etapas de experimentos,
│               comparación de 7 modelos, TabPFN final y explicabilidad
├── XAI/        Integración: explicabilidad (SHAP y LIME), página web y servidor
├── DATOS.md    De dónde salen los datos, cómo descargarlos y dónde ponerlos
└── README.md
```

Cada carpeta tiene su propio `README.md` con los pasos para ejecutarla. La historia completa de cada modelo está en:

- `Modelo 2/DOCUMENTACION_MODELO_FINAL.md` (reloj inmunológico)
- `XAI/DOCUMENTACION_MODELO_EPIGENETICO.md` (reloj epigenético)
- `XAI/explicabilidad/README.md` (SHAP y LIME)
- `XAI/web/README.md` (página web)

## Qué incluye y qué no

**Incluye:** todo el código, la documentación, las métricas agregadas, los gráficos, el modelo epigenético entrenado y los datos que la página usa para el reloj epigenético.

**No incluye**, a propósito:

- **Los datasets.** Son de terceros y se descargan desde su fuente original. Ver [DATOS.md](DATOS.md).
- **Datos por paciente del Stanford 10K (ImmPort):** predicciones por paciente, valores SHAP o LIME por paciente y los pacientes de ejemplo. Los datos de ImmPort se obtienen con las condiciones de uso de ImmPort y no se redistribuyen.
- **El modelo inmunológico entrenado** (`modelo_tabpfn.joblib`). TabPFN guarda dentro del archivo a los pacientes de entrenamiento, así que publicarlo equivaldría a publicar sus datos. Se regenera con `Modelo 2/src/06_modelo_final_tabpfn.py`.

Los datos de metilación de GEO son públicos, así que los archivos por paciente del reloj epigenético sí están.

## Puesta en marcha rápida

Requiere Python 3.12 o superior.

**1. Página web (reloj epigenético completo)**

```bash
pip install -r XAI/web/requirements.txt
python -X utf8 XAI/web/servidor.py
```

Abre `http://localhost:8000`. El reloj epigenético funciona entero, incluida la carga de pacientes nuevos (hay un CSV de ejemplo en `XAI/web/publico/ejemplos/`). En el reloj inmunológico se ven las métricas y las explicaciones globales. Las tarjetas por paciente y la carga de pacientes nuevos necesitan los datos del 10K y el modelo entrenado: ver el punto 3.

**2. Reentrenar el reloj epigenético**

Descarga GSE40279 y GSE87571 (ver [DATOS.md](DATOS.md)) y sigue `Modelo 1/README.md`.

**3. Reentrenar el reloj inmunológico y completar la página**

1. Descarga el Stanford 10K (ver [DATOS.md](DATOS.md)).
2. Sigue `Modelo 2/README.md`.
3. Copia el modelo y los datos a `XAI`, como se explica en `XAI/README.md`.

## Tecnologías

Python · scikit-learn · TabPFN v2 · LightGBM · XGBoost · shap · lime · D3.js (página web) · servidor HTTP de la librería estándar de Python.

## Limitaciones

- El reloj inmunológico pierde precisión con datos de laboratorios que no vio: 13,5 años de error en estudios no vistos, por la «firma del estudio» en las citoquinas.
- SHAP explica cómo razona el modelo, no la biología. Un aporte grande no implica causalidad.
- Es un prototipo de investigación, no una herramienta clínica.
