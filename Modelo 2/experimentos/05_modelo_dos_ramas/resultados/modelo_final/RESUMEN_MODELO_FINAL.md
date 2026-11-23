# Modelo final – Immuno-XAI (Modelo 2)

> **Reemplazado.** Este era el modelo final anterior. El definitivo es **TabPFN v2 como un solo modelo** (MAE test 6,52; estudios no vistos 13,5). Ver `DOCUMENTACION_MODELO_FINAL.md` en la carpeta principal y `resultados/modelo_final/`.

Modelo de dos ramas con fusión, entrenado **solo con el 10K Immunomes**.
Lo genera `modelo_final_dos_ramas.py` (ejecución por defecto).

## Diseño

| Componente | Datos de entrenamiento | Algoritmo |
|---|---|---|
| Rama células | 354 pacientes pareados de train + 111 personas del 10K con CyTOF sin citoquinas | Ensamble de 4: LightGBM (prediciendo √edad), XGBoost, SVR y KNN |
| Rama citoquinas | 354 pacientes pareados de train | LightGBM |
| Fusión | Predicciones fuera de muestra de los pacientes pareados | Regresión lineal: edad = −15,21 + 0,75·células + 0,54·citoquinas |

- **Variables de células:** 24 poblaciones de CyTOF, 11 razones inmunes y el sexo.
- **Variables de citoquinas:** 46 citoquinas y el sexo.
- **Hiperparámetros:** se eligieron solo con el train, con validación cruzada agrupada por paciente.

## Métricas

### Validación cruzada agrupada 5×5 (473 pacientes)

| Modelo | MAE (años) | RMSE | R² | r | Acierto ±5 años |
|---|---|---|---|---|---|
| Rama células | 10,25 | 12,96 | 0,690 | 0,831 | 30,2 % |
| Rama citoquinas | 10,30 | 13,52 | 0,663 | 0,819 | 34,2 % |
| **Fusión** | **8,16** | **10,56** | **0,794** | **0,891** | **39,4 %** |

### Test (119 pacientes; modelo entrenado solo con train)

| Modelo | MAE (años) | RMSE | R² | r | Acierto ±5 años |
|---|---|---|---|---|---|
| Rama células | 10,91 | 13,56 | 0,637 | 0,804 | 22,7 % |
| Rama citoquinas | 9,85 | 12,35 | 0,699 | 0,854 | 34,5 % |
| **Fusión** | **8,69** | **10,89** | **0,766** | **0,883** | **36,1 %** |

## Limitación: estudios no vistos

El modelo se validó con pacientes de los mismos estudios del 10K, en train y en test. Para estimar su comportamiento en otro laboratorio, se entrenó dejando fuera un estudio completo y se evaluó en ese estudio:

| Estudio excluido | n | Edades | Células: MAE / r | Citoquinas: MAE / r | Fusión: MAE / r |
|---|---|---|---|---|---|
| SDY420 | 251 | 41–90 | 14,3 / 0,67 | 28,2 / −0,07 | 18,4 / 0,54 |
| SDY515 | 77 | 12–82 | 11,2 / 0,78 | 25,0 / 0,37 | 16,8 / 0,72 |
| SDY519 | 59 | 20–75 | 10,9 / 0,70 | 15,2 / −0,05 | 10,7 / 0,62 |
| SDY113 | 57 | 19–31 | 13,3 / 0,03 | 33,7 / −0,09 | 19,1 / −0,02 |
| SDY311 | 26 | 23–90 | 18,0 / 0,75 | 29,0 / 0,52 | 21,2 / 0,73 |
| **Promedio** | | | **13,5 / 0,59** | **26,2 / 0,14** | **17,2 / 0,52** |

- **Las citoquinas llevan una fuerte firma del estudio de origen.** Un clasificador adivina el estudio el 97 % de las veces solo con ellas; con las células, el 69 %. Además, los estudios tienen rangos de edad muy distintos.
- **Dentro del 10K, el modelo puede apoyarse en esa firma**, pero en un estudio nuevo la rama de citoquinas casi no ordena a los pacientes por edad (r medio 0,14). La rama de células sí conserva su señal (r medio 0,59).
- **Parte del error en estudios no vistos se debe al rango de edades.** Al excluir un estudio, el modelo pierde parte del rango; por ejemplo, sin SDY420 quedan pocos mayores de 60. Por eso r es más informativo que el MAE o el R² en esta prueba.
- **Hay una variante más robusta:** agregar a la rama de citoquinas el ELISA extra del 10K (`python modelo_final_dos_ramas.py con_elisa`). Reduce el error en estudios no vistos a 14,3 años (r 0,55), a cambio de un MAE algo peor dentro del 10K: 8,27 en validación cruzada y 8,87 en test. Sus resultados están en `../modelo_final_con_elisa/`.

## Archivos

- `modelo_dos_ramas.joblib`: ambas ramas entrenadas, la fusión y las listas de variables.
- `metricas_modelo_final.csv` y `metricas_estudios_no_vistos.csv`.
- `predicciones_test.csv`: predicción por paciente de cada rama y de la fusión, y brecha edad predicha − edad real.
- `test_real_vs_predicha.png`.
