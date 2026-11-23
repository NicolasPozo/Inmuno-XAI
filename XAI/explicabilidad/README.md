# Explicabilidad con SHAP y LIME (v3.1)

Tres scripts que usan las librerías `shap` y `lime` sobre los dos modelos. No entrenan nada: leen los modelos guardados en `modelos/`.

| Script | Qué hace | Tiempo en CPU |
|---|---|---|
| `shap_inmunologico.py` | Valores SHAP del modelo inmunológico (TabPFN) para los 119 pacientes de test. Genera los datos que muestra la página. | 1 hora y 5 minutos |
| `lime_inmunologico.py` | LIME para los mismos pacientes y comparación entre SHAP, LIME, oclusión y permutación. Es para la tesis; la página no lo usa. | 1 hora y cuarto |
| `shap_epigenetico_validacion.py` | Comprueba con `shap` el cálculo propio del modelo epigenético (coeficiente × valor estandarizado). | 2–3 minutos |
| `comun.py` | Lo que comparten los dos primeros: carga el modelo y los pacientes. | — |

## Correrlos

```bash
python -m pip install -r web/requirements.txt -r explicabilidad/requirements.txt
```

```bash
python -X utf8 explicabilidad/shap_inmunologico.py
```

```bash
python -X utf8 explicabilidad/lime_inmunologico.py
```

```bash
python -X utf8 explicabilidad/shap_epigenetico_validacion.py
```

- **Orden:** `lime_inmunologico.py` necesita los resultados de `shap_inmunologico.py`.
- **Datos:** los dos del modelo inmunológico leen `Modelo 2/Datasets/pareados.csv`; el del epigenético, las cachés de `Modelo 1/Datasets`.
- **Si se cortan:** los dos largos guardan el avance paciente por paciente (`resultados/explicabilidad/*_avance_*.csv`). Al repetirlos, siguen donde quedaron.

## Cómo se calcula el SHAP del modelo inmunológico

TabPFN no es lineal, así que su SHAP no tiene fórmula. Se estima con `shap.KernelExplainer`, que solo necesita las predicciones del modelo.

- **Referencia:** un paciente con las 46 variables en la mediana del train. Su edad predicha es la **base**: 50,74 años.
- **Valor SHAP de un biomarcador:** cuánto mueve la edad predicha tener el valor del paciente en vez del de referencia, promediado sobre combinaciones de los demás biomarcadores. Así se reparten también los efectos conjuntos.
- **Suma exacta:** base + los 46 valores SHAP = edad predicha del paciente.
- **Combinaciones:** 1.000 por paciente de test y 500 por paciente subido a la página web.
- **Biomarcadores vacíos:** se dejan vacíos, como los recibe el modelo.
- **Sin regularización** (`l1_reg=False`): por defecto `shap` deja solo 10 variables distintas de cero.

**Por qué una sola fila de referencia.** TabPFN tarda unos 0,03 segundos por fila. Con 1.000 combinaciones son unos 35 segundos por paciente; con 100 pacientes de referencia serían 100 veces más.

**Truco de velocidad.** TabPFN repite en cada llamada el trabajo sobre los 354 pacientes de train (unos 10 segundos). Los scripts usan una copia del modelo en modo `fit_with_cache`, que guarda ese trabajo. La copia se entrena en memoria con los mismos datos, se comprueba que predice lo mismo que el modelo guardado y no se guarda.

## Archivos que generan

| Archivo | Contenido |
|---|---|
| `resultados/explicabilidad/shap_inmunologico_test.csv` | Una fila por paciente de test: base, edad predicha y el valor SHAP de cada variable |
| `dashboard/datos/shap_inmune.json` | Lo mismo, una fila por paciente y variable, con el valor del biomarcador. Lo lee la página. |
| `dashboard/datos/shap_inmune_importancia.json` | Media de \|SHAP\| por variable |
| `dashboard/datos/pacientes_test.json` | Se le agrega el campo `base` |
| `resultados/explicabilidad/lime_inmunologico_test.csv` | Pesos de LIME por paciente y variable, con el ajuste del modelo local |
| `resultados/explicabilidad/comparacion_metodos_por_paciente.csv` | Acuerdo entre SHAP, LIME y oclusión en cada paciente |
| `resultados/explicabilidad/comparacion_metodos_global.csv` | Ranking de biomarcadores según SHAP, LIME, oclusión y permutación |
| `resultados/explicabilidad/comparacion_metodos_resumen.csv` | Las cifras de la comparación y de la estabilidad de cada método |
| `graficos/explicabilidad/comparacion_shap_lime.png` | Importancia global SHAP vs LIME y acuerdo por paciente |
| `resultados/explicabilidad/shap_epigenetico_validacion.csv` | Diferencias entre el cálculo propio y la librería `shap` |

## Qué resultó

| Medida | SHAP | LIME |
|---|---|---|
| Error al reconstruir la edad predicha | 0,0001 años (máximo) | 7,5 años (media) |
| Diferencia por variable entre dos corridas | 0,17 años | 1,19 años |
| Biomarcadores en común en el top 5 entre dos corridas | 4,7 de 5 | 3,9 de 5 |

- **Acuerdo entre SHAP y LIME:** Spearman de 0,85 en el ranking global (8 de los 10 primeros en común) y de 0,62 por paciente.
- **SHAP y oclusión** (el método anterior de la página): 0,97 global y 0,90 por paciente.
- **Conclusión:** SHAP es el que muestra la página. LIME queda como comparación: su recta local imita mal a TabPFN (R² de 0,38).

El detalle está en la sección 7.6 de `DOCUMENTACION_MODELO_FINAL.md`.

Si se regeneran los datos del modelo inmunológico (`Modelo 2/src/09_exportar_dashboard.py`), hay que volver a correr `shap_inmunologico.py`.
