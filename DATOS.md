# Datos del proyecto

Los datasets **no se incluyen** en el repositorio: son de terceros, pesan varios GB y, en el caso de ImmPort, sus condiciones de uso no permiten redistribuirlos. Esta guía explica de dónde salen, cómo descargarlos y dónde ponerlos para que los scripts los encuentren.

Las rutas se definen en `Modelo 1/src/config.py` y `Modelo 2/src/config.py`. Si guardas un archivo en otro lugar, basta con cambiar la ruta ahí.

| Reloj | Dataset | Fuente | Acceso | Tamaño aprox. |
|---|---|---|---|---|
| Epigenético | GSE40279 (Hannum et al., 2013) | NCBI GEO | Público, sin registro | ~3 GB (matriz beta) |
| Epigenético | GSE87571 (Johansson et al., 2013) | NCBI GEO | Público, sin registro | ~2,8 GB |
| Inmunológico | Stanford 10K Immunomes (Zalocusky et al., 2018) | ImmPort / repositorio del 10K | Requiere aceptar las condiciones de ImmPort | pocos MB (archivos normalizados) |

---

## 1. Reloj epigenético (`Modelo 1/`)

Ambas cohortes son sangre completa medida con el array **Illumina HumanMethylation450 (450K)**. Se usan las 1.385 muestras válidas: 656 de Hannum y 729 de GSE87571.

### GSE40279 (Hannum)

- **Página:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279
- **Archivos que se usan:**

| Archivo | Dónde ponerlo | Lo usa |
|---|---|---|
| `GSE40279_family.soft.gz` (sección *Download family → SOFT formatted family file*) | `Modelo 1/Datasets/` | `src/01_extraer_metadatos.py` (edad y sexo de cada muestra → `Metadata_Pacientes.csv`) |
| Matriz de valores beta de la serie, convertida a `Dataset_Hannum_Metilacion_Chunks.csv` | `Modelo 1/Datasets/` | `src/02_consolidar_dataset.py` |

`Dataset_Hannum_Metilacion_Chunks.csv` es la matriz beta de GEO traspuesta: **una fila por paciente**, con las columnas `ID_Paciente` (GSM…), `Edad` y una columna por sonda (`cg…`). Se armó por partes (*chunks*) para no llenar la memoria. El script que hizo esa conversión no forma parte del repositorio, pero cualquier lectura de la matriz que deje ese formato sirve. Desde ahí, `02_consolidar_dataset.py` agrega el sexo, y los pasos siguientes crean solos su caché binaria (`Datasets/cache_matriz_completa/`, ~1,2 GB).

### GSE87571

- **Página:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87571
- **Archivos** (sección *Supplementary file* y *Series Matrix*), en `Modelo 1/Datasets/GSE87571/`:
  - `GSE87571_series_matrix.txt.gz` (edad y sexo de cada muestra)
  - `GSE87571_matrix1of2.txt.gz` y `GSE87571_matrix2of2.txt.gz` (valores beta)
- `src/gse87571.py` los convierte la primera vez a una caché en el mismo orden de sondas que Hannum (`Datasets/cache_GSE87571/`, ~1,4 GB). Se descartan 3 muestras sin datos válidos.

### Ejecutar

Ver `Modelo 1/README.md` (pasos `01` a `07`). El paso `07` deja el modelo en `Modelo 1/modelos/`; el repositorio ya trae esa salida y la copia en `XAI/modelos/epigenetico_elasticnet/`.

---

## 2. Reloj inmunológico (`Modelo 2/`)

### Stanford 10K Immunomes

Reúne datos de varios estudios de Stanford depositados en **ImmPort** (SDY420, SDY515, SDY519, SDY113, SDY311, entre otros), ya normalizados entre estudios.

- **Repositorio del 10K:** https://github.com/buttelab/10kimmunomes_v2 (allí están los enlaces de descarga de cada tipo de dato)
- **ImmPort:** https://www.immport.org. Hay que registrarse y aceptar las condiciones de uso. No se pueden redistribuir los datos por paciente; por eso este repositorio no los trae.

Descarga los cuatro paquetes y descomprímelos en `Modelo 2/Datasets/10k/extraido/`, manteniendo su carpeta interna:

| Paquete | Archivo que usa el proyecto | Para qué |
|---|---|---|
| `CyTOF_PBMC.zip` | `extraido/CyTOF_PBMC/CyTOF_PBMC/cytof_pbmc_normalized.csv` | 16 poblaciones celulares (modelo final) |
| `Multiplex_ELISA.zip` | `extraido/Multiplex_ELISA/Multiplex_ELISA/Normalized_Files/multiplexELISA_normalized.csv` | 29 citoquinas (modelo final) |
| `Flow_Cytometry_PBMC.zip` | `extraido/Flow_Cytometry_PBMC/Flow_Cytometry_PBMC/Normalized_Files/flow_pbmc_normalized.csv` | Solo experimentos (etapa 7) |
| `Flow_Cytometry_Whole_Blood.zip` | (carpeta completa) | Solo experimentos |

Las células y las citoquinas se publican en archivos separados. El proyecto las une **por paciente y visita** y se queda con los **473 pacientes** que tienen ambas en la misma visita: 354 de entrenamiento y 119 de test.

### Archivos de partida

El pipeline de `Modelo 2/src/` parte de dos tablas por paciente armadas en una etapa previa desde los archivos normalizados del 10K:

- `Modelo 2/Datasets/df_train_aumentado_k45.csv`: train con los 45 biomarcadores; solo se usan las filas reales (`source_type = original`).
- `Modelo 2/Datasets/df_test_puro_k45.csv`: los 119 pacientes de test.

Contienen datos por paciente de ImmPort, así que **no se incluyen**. Con el 10K descargado se pueden reconstruir siguiendo la sección 3 de `Modelo 2/DOCUMENTACION_MODELO_FINAL.md` (datos y conjuntos). Las 45 variables están en `XAI/modelos/final_tabpfn/variables_modelo.csv`. Desde esas tablas, `src/04_construir_fuentes_unimodales.py` genera `pareados.csv`, la tabla de trabajo del modelo final, con los valores tomados directamente del 10K.

### Cohortes externas (solo experimentos descartados)

Se probaron para aumentar el número de pacientes o medir la robustez. El modelo final no las usa:

| Cohorte | Fuente | Carpeta | Etapa |
|---|---|---|---|
| FluPRINT (Tomic et al., 2019) | Stanford HIMC | `Modelo 2/Datasets/fluprint/` | 2 |
| Sound Life | Allen Institute for Immunology | `Modelo 2/Datasets/soundlife/` | 5 |
| ImmPort SDY1468, SDY1481, SDY1483, SDY1484 | https://www.immport.org | `Modelo 2/Datasets/immport/` | 8 |

### Ejecutar

Ver `Modelo 2/README.md` (pasos `01` a `10`). El paso `06` genera `modelos/modelo_tabpfn.joblib`. **No lo subas a un repositorio público:** contiene a los pacientes de entrenamiento.

---

## 3. Página web (`XAI/`)

La página lee sus datos de `XAI/dashboard/datos/*.json`:

- **Reloj epigenético:** los JSON vienen incluidos (datos públicos de GEO). Se regeneran con `XAI/dashboard/exportar_epigenetico.py`.
- **Reloj inmunológico:** vienen incluidos los JSON agregados (métricas, importancias, comparación de modelos, curvas). Los que tienen datos por paciente se generan después de entrenar:

| Archivo | Lo genera |
|---|---|
| `pacientes_test.json`, `aportes_paciente.json`, `brecha.json` | `Modelo 2/src/09_exportar_dashboard.py` (se copian desde `Modelo 2/resultados/explicabilidad/dashboard/`) |
| `shap_inmune.json` | `XAI/explicabilidad/shap_inmunologico.py` (~1 hora) |
| `simulador_inmune.json` | `XAI/dashboard/exportar_simulador_inmune.py` |

Además hay que copiar `Modelo 2/modelos/modelo_tabpfn.joblib` a `XAI/modelos/final_tabpfn/`. Para probar la carga de pacientes nuevos, crea un CSV con las columnas de `XAI/modelos/final_tabpfn/variables_modelo.csv`; el formato está en `XAI/web/README.md`. Puedes guardarlo como `XAI/web/publico/ejemplos/ejemplo_inmunologico.csv` para que la página lo ofrezca como ejemplo.

---

## Referencias

- Hannum G, et al. (2013). Genome-wide methylation profiles reveal quantitative views of human aging rates. *Molecular Cell*, 49(2), 359–367.
- Johansson Å, Enroth S, Gyllensten U. (2013). Continuous aging of the human DNA methylome throughout the human lifespan. *PLoS ONE*, 8(6), e67378.
- Zalocusky KA, et al. (2018). The 10,000 Immunomes Project: building a resource for human immunology. *Cell Reports*, 25(2), 513–522.
- Tomic A, et al. (2019). FluPRINT, a multidimensional analysis of the influenza vaccine imprint on the immune system. *Scientific Data*, 6, 214.
