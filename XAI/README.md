# XAI – Edad inmunológica (modelo final + dashboard)

Esta carpeta reúne el modelo ya entrenado, los resultados y el dashboard interactivo del proyecto Immuno-XAI. Es una **copia**: los originales y los scripts que los generan siguen en `D:\Claude Code\Titulo\Modelo 2`.

**Dashboard publicado (ambos modelos):** https://claude.ai/artifact/794jg3fRxv7WYTd2cLtEKS

El dashboard anterior, solo con el modelo inmunológico, fue borrado. El nuevo contiene todo lo que tenía, más el modelo epigenético.

## Contenido

| Carpeta / archivo | Qué es |
|---|---|
| `DOCUMENTACION_MODELO_FINAL.md` | Documentación completa: datos, evaluación, las 12 etapas, por qué se eligió el modelo, explicabilidad, limitaciones |
| `modelos/final_tabpfn/modelo_tabpfn.joblib` | **Modelo final.** TabPFN v2 entrenado con los 354 pacientes de train (45 biomarcadores + sexo) |
| `modelos/final_tabpfn/variables_modelo.csv` | Las 46 variables en el orden que espera el modelo |
| `modelos/predecir.py` | Script para predecir la edad de pacientes nuevos desde un CSV |
| `modelos/anterior_dos_ramas/modelo_dos_ramas.joblib` | Modelo anterior (dos ramas + fusión), solo como referencia |
| `dashboard/index.html` | Copia de la página del dashboard, con los dos modelos (solo funciona dentro del dashboard de Claude) |
| `dashboard/datos/*.json` | Los archivos de datos que lee el dashboard: 10 del modelo inmunológico, 13 del epigenético (`epi_*.json`) y 5 que solo usa la página web (`simulador_inmune`, `shap_inmune`, `shap_inmune_importancia`, `comparacion_modelos` y `curva_variables`) |
| `dashboard/exportar_epigenetico.py` | Genera los `epi_*.json` del modelo epigenético |
| `dashboard/mapa_dashboard.json` | Qué fuente del dashboard corresponde a cada archivo y su dirección dentro del artifact |
| `dashboard/Media/Loading screen.mp4` | Video de la animación de carga (se usan los primeros 9 s) |
| `web/` | Página web fuera de Claude: servidor Python, cálculo de pacientes nuevos y deploy (ver `web/README.md`) |
| `modelos/final_tabpfn/medianas_train.csv` | Mediana, media y desviación del train del modelo inmunológico. La mediana es la referencia de SHAP |
| `explicabilidad/` | **SHAP y LIME (v3.1):** scripts que usan las librerías `shap` y `lime` sobre los dos modelos (ver `explicabilidad/README.md`) |
| `resultados/modelo_final/` | Métricas (CV 5×5, test, estudios no vistos), predicciones del test y de la CV |
| `resultados/explicabilidad/` | Importancia por permutación, bloques, dependencia parcial, explicación individual, brecha de edad y, desde la v3.1, SHAP, LIME y la comparación entre métodos |
| `graficos/` | Evolución del error, edad real vs predicha y comparación de los 7 modelos con la misma evaluación y error según el número de variables |
| `DOCUMENTACION_MODELO_EPIGENETICO.md` | Documentación del **modelo epigenético** (metilación del ADN): datos, selección de sondas, comparación de modelos, métricas, explicabilidad, fusión y limitaciones |
| `modelos/epigenetico_elasticnet/modelo_elasticnet.joblib` | **Modelo epigenético final.** ElasticNet con 500 sondas CpG + sexo, entrenado con 1.038 pacientes (Hannum + GSE87571) |
| `modelos/epigenetico_elasticnet/variables_modelo.csv` | Las 501 variables en orden, con su mediana de imputación y su coeficiente |
| `modelos/predecir_epigenetico.py` | Script para predecir la edad epigenética de pacientes nuevos desde un CSV |
| `resultados/modelo_epigenetico/` | Métricas, predicciones del test y de la CV, comparación de 5 modelos y curvas |
| `graficos/epigenetico/` | Figuras del modelo epigenético |

## Modelo epigenético en una línea

ElasticNet con 500 sondas CpG. Test de 347 pacientes: MAE 3,13 años y R² 0,953 (Hannum: MAE 3,91, R² 0,882; GSE87571: MAE 2,44). Es una copia: los originales y los scripts están en `D:\Claude Code\Titulo\Modelo 1`.

```bash
python -X utf8 modelos/predecir_epigenetico.py pacientes.csv
```

- La entrada tiene una columna por sonda CpG (valores beta del array 450K o EPIC) y una columna `gender`.
- Las sondas faltantes se rellenan con la mediana del train.
- Requisitos: `scikit-learn` 1.9, `pandas`, `numpy` y `joblib`.

## Modelo final en una línea

TabPFN v2 como un solo modelo. Con datos del 10K: MAE 6,34 años en validación cruzada 5×5 y 6,52 en el test (R² 0,828); en estudios no vistos, 13,5 años.

## Usar el modelo

```bash
python -X utf8 modelos/predecir.py pacientes.csv
```

- La entrada necesita las 45 columnas de `variables_modelo.csv` (sin `sexo_Male`), en la escala de los archivos normalizados del 10K, y una columna `gender`.
- Requisitos: `tabpfn` 9.2 con los pesos v2, `torch`, `scikit-learn`, `pandas`, `numpy` y `joblib`.
- El modelo anterior (`modelo_dos_ramas.joblib`) necesita, para cargarse, que exista una función `log1p_positivo` en el script que lo abre (definida en `Modelo 2/experimentos/05_modelo_dos_ramas/modelo_final_dos_ramas.py`).

## El dashboard

Es un Artifact de tipo **Dashboard** de Claude. La página y sus datos no son archivos sueltos: viven en la base de datos del artifact.

- **La página** está en el documento `files/index.html` (campo `text`). La copia local es `dashboard/index.html`.
- **Cada gráfico lee una fuente** (`datasets/<id>`) que apunta a un archivo JSON subido al artifact. `mapa_dashboard.json` lista las fuentes con su archivo local y su dirección.
- **Dos modelos, por separado.** Arriba hay un selector: *Inmunológico* o *Epigenético*. Cada uno tiene sus propias pestañas y fuentes; no se fusionan.
- **Estilo.** El diseño se inspira en el estilo visual de ispidermedia.com:
  - una portada oscura con el título en Work Sans fina;
  - un solo color de acento violeta;
  - selectores y pestañas tipo píldora (las pestañas quedan fijas al bajar);
  - tarjetas redondeadas sobre fondo gris, con la primera cifra de cada resumen en violeta.

  Los colores del diseño se calculan a partir de las variables del dashboard, así que también funcionan en modo oscuro. Los gráficos siguen usando la paleta del dashboard.
- **Animaciones (v1.3).** Hay un reloj que gira en la portada, tarjetas y gráficos que aparecen al llegar a la pantalla, una pestaña activa que se desliza, y células y una hélice de ADN en la cabecera de cada modelo. Se apagan si el sistema pide reducir movimiento.
- **Versiones.** Las páginas se respaldan en `dashboard/versiones/` (v1.1 a v1.4, v2.1 «Diseños integrados», v2.2, v2.3, v2.4, v3.1 «SHAP y LIME», v3.2, v3.3 «Pipeline en la portada», v3.4 «Siete modelos» y v3.5 «Curvas de variables en Resumen»). `VERSIONES.md` explica qué cambia en cada una y cómo volver a una anterior. En Claude está publicada la v1.3; la v3.5 es la de la página web.

## La página web (fuera de Claude)

`web/` tiene un servidor Python que muestra el mismo dashboard en cualquier hosting, dividido en tres páginas: portada (`/`), reloj inmunológico (`/inmunologico`) y reloj epigenético (`/epigenetico`). También agrega el botón **Agregar un paciente**. Se sube un CSV y el servidor calcula la edad con los dos modelos reales; mientras tanto se ve la animación de carga con el video. `web/README.md` explica cómo correrla, qué debe traer el archivo y cómo desplegarla.

```bash
python -X utf8 web/servidor.py
```

**Modelo inmunológico** (10 fuentes, sin prefijo):
- Pestañas: Resumen, Qué mira el modelo y Explorar un paciente. En *Resumen*, desde la v3.4, la tarjeta «Los siete modelos con la misma evaluación» compara TabPFN con dos ramas, LightGBM, XGBoost, SVR, ElasticNet y Random Forest en la prueba elegida (validación cruzada, test o estudios no vistos). Desde la v3.5 también está «¿Hacen falta más o menos biomarcadores?», con el error según el número de biomarcadores de los 6 modelos (etapa 12): los k mejores de las 71 candidatas, con un anillo en 45 para los biomarcadores del modelo final. La pestaña *Brecha de edad* se quitó en la v2.4 (sigue en las versiones anteriores, en `dashboard/versiones/`). La brecha de cada paciente sigue apareciendo en *Explorar un paciente*.
- Controles: tipo de variable (Todas / Células / Citoquinas), biomarcador del perfil de dependencia y paciente de test.
- **SHAP (v3.1):** los valores se estiman con `shap.KernelExplainer` (`explicabilidad/shap_inmunologico.py`): referencia en la mediana del train (base 50,74 años) y 1.000 combinaciones por paciente. Base + los 46 aportes = edad predicha.
  - **Qué mira el modelo:** aporte medio de los 15 biomarcadores principales y un enjambre con un punto por paciente, coloreado por el valor del biomarcador (al hacer clic se abre ese paciente). Los que más aportan son Naive_CD8_T_cells (9,5 años en promedio), IL1R1 (5,4) y Effector_CD8_T_cells (5,3).
  - **Explorar un paciente:** cascada de la base a la edad predicha con los 12 biomarcadores de más aporte y el resto sumado. Reemplaza a las barras de oclusión (llevar cada biomarcador a la mediana), que no sumaban la edad predicha.
  - **Pacientes subidos:** el servidor les calcula SHAP con 500 combinaciones.
  - **Comparación con LIME (solo para la tesis):** los rankings globales de SHAP y LIME coinciden bastante (Spearman 0,85; 8 de los 10 primeros), pero por paciente menos (0,62). SHAP es más estable al repetirlo (0,17 años de diferencia por variable frente a 1,19 de LIME) y la recta local de LIME imita mal a TabPFN (R² de 0,38). Detalle en la sección 7.6 de `DOCUMENTACION_MODELO_FINAL.md`.
- **Simulador "qué pasaría si" (v2.4):** en *Explorar un paciente*, deslizadores para los 6 biomarcadores que, llevados solos a la mediana del train, más cambian la edad de ese paciente (del percentil 1 al 99 del train).
  - **Por qué está precalculado:** TabPFN tarda segundos por predicción, así que no calcula en vivo. `dashboard/exportar_simulador_inmune.py` precalcula con el modelo real cómo cambia la edad al mover cada biomarcador, uno a la vez, y lo guarda en `simulador_inmune.json`.
  - **En la página:** se interpola en cada curva y se mueve **un biomarcador a la vez** (al mover uno, los demás vuelven al valor del paciente). Así el resultado sale del modelo real.
  - **Por qué no se combinan:** el script probó sumar los cambios de los 6 movidos a la vez y compararlo con el cálculo exacto (`resultados/explicabilidad/simulador_aditividad.csv`). La suma se aleja 8,2 años en promedio (mediana 4,8; máximo 38,4), porque TabPFN combina variables.

**Modelo epigenético** (13 fuentes, prefijo `epi_`):
- **Resumen:** MAE, R² de Hannum, acierto ±5 años, tabla de métricas por cohorte, los 5 algoritmos comparados, curva de aprendizaje y curva de número de sondas. Un selector elige la cohorte del test (ambas, Hannum o GSE87571).
- **Qué mira el modelo:** las 20 sondas con más peso (coeficientes) y un bloque **SHAP global**. En un modelo lineal estandarizado, el aporte de una sonda (coeficiente × valor estandarizado) es exactamente su valor SHAP. El bloque tiene:
  - cuántas sondas hacen falta para explicar el aporte total (las 20 primeras explican solo el 21,9 %; la mitad necesita 68 y el 80 %, 156);
  - el aporte medio de las 20 principales en cada cohorte;
  - un enjambre con un punto por paciente, coloreado por metilación (al hacer clic se abre ese paciente);
  - un mapa de calor del aporte medio por grupo de edad.
  
  Debajo sigue la metilación de una sonda según la edad. La curva de número de sondas está en *Resumen* desde la v3.5 («¿Hacen falta más sondas?», junto a la curva de aprendizaje). La importancia media |SHAP| ordena las sondas igual que el coeficiente, por eso no se repite como ranking.
- **Explorar un paciente:** los 347 pacientes de test. Como el modelo es lineal, la explicación es exacta: base (54,76 años) + aporte de cada variable = edad predicha. La librería `shap` da los mismos valores (diferencia máxima de 0,0000000002 años; `explicabilidad/shap_epigenetico_validacion.py`).
  - **Cascada:** va de la base a la edad predicha sumando las 15 variables con más efecto y, al final, la suma del resto, con la edad real como referencia.
  - **Simulador "qué pasaría si":** deslizadores para la metilación de las 6 sondas que más pesan en ese paciente (del percentil 1 al 99 del train). La edad simulada se recalcula con la fórmula del modelo, aporte = coeficiente × (valor − media) / escala, usando las columnas `media`, `escala`, `p01` y `p99` de `epi_coeficientes.json`. Mueve una sonda a la vez sin tocar las demás, así que es una exploración del modelo, no una predicción biológica.
- **Brecha de edad:** se quitó en la v2.4. Usaba 1.385 pacientes con predicciones fuera de muestra (CV para train, modelo final para test), corregida con una recta común como en el inmunológico. La tabla de sondas usaba una correlación de Spearman **parcial (descontando la edad) dentro de cada cohorte**: juntar las cohortes mezclaba el efecto de laboratorio y hacía parecer asociadas a todas las sondas. Los datos (`epi_brecha*.json`) siguen en `dashboard/datos/`.

### Cambiar el dashboard

- **Cambiar un gráfico o un texto:** editar `files/index.html` en el artifact y mantener la copia local igual.
- **Cambiar los datos del modelo inmunológico:** regenerar los JSON con `Modelo 2/src/09_exportar_dashboard.py` (los deja en `Modelo 2/resultados/explicabilidad/dashboard/`), subir el archivo nuevo al artifact y actualizar la dirección (`url`), `name` y `updated` de su fuente.
- **Cambiar los datos del modelo epigenético:** `python -X utf8 dashboard/exportar_epigenetico.py` (1–2 minutos). Lee el modelo, `resultados/modelo_epigenetico/` y los valores beta de las 500 sondas desde `Modelo 1/Datasets` (las cachés `.npy`), y escribe `dashboard/datos/epi_*.json`. Antes de exportar comprueba que el modelo reproduce las predicciones de test guardadas.
- **Compartir:** desde el menú Compartir de la página. Quien abra un enlace público necesita sesión iniciada en Claude para ver los datos.

## Regenerar todo desde cero

Se hace desde `D:\Claude Code\Titulo\Modelo 2` (su `README.md` explica la estructura):

```bash
python -X utf8 src/06_modelo_final_tabpfn.py
python -X utf8 src/07_graficos_modelo_final.py
python -X utf8 src/08_explicabilidad.py
python -X utf8 src/09_exportar_dashboard.py
```

`src/08_explicabilidad.py` tarda unos 45 minutos en CPU. Los pasos 01 a 04 de `src/` preparan los datos; solo hacen falta si `Datasets/pareados.csv` no existe. El paso 05 (`src/05_comparar_modelos.py`) compara TabPFN con los otros modelos y no hace falta para regenerar el modelo final.
