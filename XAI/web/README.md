# Página web Immuno-XAI (fuera de Claude)

Es el mismo dashboard que está en Claude, pero corre en un servidor propio y está dividido en tres páginas:

| Dirección | Página |
|---|---|
| `/` | **Portada:** ocupa toda la pantalla, presenta el proyecto y lleva a cada reloj. Siempre es oscura y lo único que se mueve es el reloj. Al bajar aparece el contexto: qué es la edad biológica, los dos relojes, una tabla comparativa y qué hay en cada página. |
| `/inmunologico` | **Reloj inmunológico:** sus tres pestañas, con fondo animado de células. |
| `/epigenetico` | **Reloj epigenético:** sus tres pestañas, con fondo animado de ADN. |

Cada página de reloj tiene arriba una barra para volver a la portada o pasar al otro reloj. Las pestañas quedan en la dirección; por ejemplo, `/epigenetico#epi-paciente` abre directamente *Explorar un paciente*.

Además, en *Explorar un paciente* de los dos modelos está el botón **Agregar un paciente**: se sube un CSV, el servidor calcula la edad con el modelo real y la página muestra al paciente con los mismos gráficos que a los de test.

## Correrla en tu computador

```bash
python -X utf8 web/servidor.py
```

Luego abre http://127.0.0.1:8000.

- **Modelo epigenético:** queda listo al instante.
- **Modelo inmunológico (TabPFN):** se carga en segundo plano y tarda unos segundos.
- **Detener el servidor:** Ctrl+C.

## Cómo funciona

| Archivo | Qué hace |
|---|---|
| `servidor.py` | Servidor web hecho solo con la librería estándar de Python. Entrega la página, los datos y el video, y calcula la edad de los pacientes subidos. |
| `modelos_api.py` | Los cálculos de los dos modelos. Devuelve al paciente nuevo con el mismo formato que los archivos del dashboard. Para el inmunológico estima los valores SHAP con la librería `shap`. |
| `publico/portada.html` | La portada. Es una página propia: no carga el dashboard ni sus datos. |
| `publico/modelo.html` | La página de cada reloj. Envuelve al dashboard, define los colores en tema claro y oscuro, tiene la barra de arriba y el botón de tema, y según la dirección indica qué reloj mostrar (`window.EI_PAGINA`). |
| `publico/estaticos/dash-local.js` | Reemplaza las funciones que el dashboard de Claude le da a la página (`dash.data`, `dash.draw`, …). |
| `publico/estaticos/fondo.js` | Fondo animado de los bordes: células o ADN, según el modelo abierto. |
| `publico/ejemplos/` | Un CSV de ejemplo por modelo, con dos pacientes reales del test. |
| `preparar_recursos.py` | Se corrió una vez. Generó las medianas, medias y desviaciones del train inmunológico (`modelos/final_tabpfn/medianas_train.csv`) y los ejemplos, y comprobó que el cálculo coincide con el dashboard. |

La página del dashboard es **la misma** `dashboard/index.html`, sin copias:

- **En Claude** no existe el servidor, así que el botón no aparece y los dos modelos van en una sola página, con su portada.
- **Aquí** `publico/modelo.html` define `window.EI_API`, con lo que el botón se activa, y `window.EI_PAGINA`, con lo que se muestra un solo reloj y se oculta la portada interna.

**Rutas del servidor:**

| Ruta | Qué entrega |
|---|---|
| `/` | La portada (`publico/portada.html`) |
| `/inmunologico` y `/epigenetico` | La página de cada reloj (`publico/modelo.html`). Con barra final, redirige a la dirección sin barra. |
| `/dashboard.html` | `dashboard/index.html` |
| `/datos/…` | `dashboard/datos/` |
| `/media/…` | `dashboard/Media/` (el video de carga) |
| `/ejemplos/…` | Los CSV de ejemplo |
| `POST /api/inmunologico` y `POST /api/epigenetico` | Reciben el CSV en el cuerpo y devuelven `pacientes`, `aportes` y `avisos` en JSON. El inmunológico devuelve además `simulacion` (las curvas del simulador) y `oclusion` (el efecto de llevar cada biomarcador a la mediana, con el que se eligen los biomarcadores del simulador). |
| `GET /api/estado` | Indica si los modelos están listos. |

## Fondo animado

En los bordes vacíos, a los lados del contenido, hay una animación interactiva según el modelo abierto. Al cambiar de modelo, una se funde en la otra.

- **Modelo inmunológico:** células de partículas.
- **Modelo epigenético:** dos hélices de ADN, una a cada lado, con una hebra azul y la otra naranja.

Las partículas se apartan al pasar el mouse.

- **Código:** `publico/estaticos/fondo.js`.
- **Originales:** `dashboard/Media/animaciones/` (`celulas.html` y `Adn_animado_azul_naranja.html`; `Adn_animado.html` es la versión anterior, solo azul).

**Cambios respecto a las originales:**
- **Dónde se dibujan:** solo en los bordes, y se desvanecen al acercarse al contenido para no tapar el texto.
- **Tema oscuro:** brillan, igual que las originales.
- **Tema claro:** usan colores más oscuros y sin brillo.
- **Tamaño:** se adapta al ancho del borde.
- **Pantallas angostas:** en ventanas de menos de unos 1.400 px de ancho no hay borde libre, así que no se dibujan y no gastan recursos.
- **Movimiento reducido:** si el sistema lo pide, se muestra una imagen fija.

## Qué calcula cada modelo

- **Epigenético:** exacto, en menos de un segundo.
  - Las sondas que falten se rellenan con la mediana del train, igual que `modelos/predecir_epigenetico.py`.
  - La cascada es base + aporte de cada sonda.
- **Inmunológico:** alrededor de 40 segundos para un paciente, más unos 20 por cada paciente adicional, en un CPU de 6 núcleos. Son dos llamadas al modelo:
  - **Edad y explicación SHAP (v3.1):** los valores SHAP se estiman con la librería `shap` (KernelExplainer), con la mediana del train como referencia y 500 combinaciones de biomarcadores por paciente (`N_SHAP` en `modelos_api.py`). La base más los 46 aportes da exactamente la edad predicha. Los pacientes de test se calcularon igual, con 1.000 combinaciones (`explicabilidad/shap_inmunologico.py`).
  - **Simulador «qué pasaría si»:** las curvas de los 6 biomarcadores que, llevados solos a la mediana del train, más cambian la edad de cada paciente.
  - TabPFN tarda unos 10 segundos fijos por llamada y 0,03 por fila. Para pagar el costo fijo una sola vez, todas las filas que pide SHAP se predicen en una llamada (`shap_kernel` en `modelos_api.py`).
  - Mientras calcula, se repite la animación.

Con los pacientes de ejemplo, la edad y los aportes coinciden con el dashboard: la diferencia máxima es de 0,001 años.

## El archivo que se sube

- **Formato:** CSV separado por comas, o por punto y coma con coma decimal, como lo guarda Excel en español.
- **Una fila por paciente**, con un máximo de 20.
- **Columna `gender`:** obligatoria. Acepta Male/Female, M/F u Hombre/Mujer.
- **Columna `age`:** opcional. Es la edad real y sirve para calcular la brecha.
- **Columna `id`:** opcional. Es el nombre del paciente.
- **Biomarcadores o sondas:** con los mismos nombres que usa el modelo. Los ejemplos de `publico/ejemplos/` sirven de plantilla.

Los archivos no se guardan: el servidor los lee en memoria y responde. Los pacientes agregados duran solo mientras la página esté abierta.

## La animación de carga

- **Video:** `dashboard/Media/Loading screen.mp4`. Solo se usan los primeros 9 segundos, porque al final tiene un error.
- **Fondo gris:** se borra con filtros y mezcla CSS. En tema claro queda una silueta negra; en oscuro, una blanca.
- **Cuándo se cierra:** apenas termina el cálculo, después de mostrarse al menos 1,2 segundos para que no parpadee. Mientras tanto, el video se repite (sus primeros 9 segundos).
- **Barra de progreso (v2.4):** muestra el avance real que informa el servidor. La página pide el cálculo con `?progreso=1` y el servidor responde una línea JSON por evento: la etapa y el avance varias veces por segundo, y al final el resultado o el error.
  - **Etapas del inmunológico:** leer el archivo, calcular la edad y la explicación SHAP, y calcular el simulador. Si el modelo todavía está cargando, se agrega esa etapa.
  - **Etapas del epigenético:** leer el archivo y calcular la edad.
  - **Dentro de cada etapa:** TabPFN no informa su avance mientras predice, así que el avance se estima con las filas que predice en esa etapa (un costo fijo por llamada más un tiempo por fila, que se ajusta con cada cálculo). Así la estimación sirve igual para 1 paciente que para 20. La barra nunca llega al 100 % hasta que el cálculo termina de verdad.
  - **Texto bajo el video:** muestra la etapa y el porcentaje (por ejemplo, «Calculando el simulador · 74 %»).
- **Movimiento reducido:** si el sistema lo pide, se muestra un cuadro fijo.
- La v2.4 quitó el botón «Saltar animación» y el tope de 25 segundos.

La prueba aislada del video está en `dashboard/prueba_carga/index.html`.

## Deploy

Hace falta un hosting que corra **Python con PyTorch**, por ejemplo un VPS, Render, Railway o Fly.io. No sirve un hosting estático como GitHub Pages o Netlify, porque los modelos se calculan en el servidor.

1. **Subir solo estas carpetas** de `XAI`, con la misma estructura:

   | Carpeta o archivo | Tamaño aproximado |
   |---|---|
   | `web/` | — |
   | `dashboard/index.html` | — |
   | `dashboard/datos/` | ~5 MB |
   | `dashboard/Media/` | ~3 MB |
   | `modelos/final_tabpfn/` | ~47 MB |
   | `modelos/epigenetico_elasticnet/` | — |

2. **Instalar los paquetes:** `pip install -r web/requirements.txt`, mejor con torch para CPU, como indica el archivo. Desde la v3.1 incluye `shap`.
3. **Arrancar el servidor** con `HOST=0.0.0.0` y el puerto que dé el hosting en `PORT`:

   ```bash
   HOST=0.0.0.0 PORT=8000 python -X utf8 web/servidor.py
   ```

4. **Memoria:** mínimo unos 2 GB de RAM por TabPFN y torch.

**Antes de abrirla al público:**

- El servidor no tiene usuarios ni contraseña.
- Atiende un cálculo inmunológico a la vez; los demás esperan su turno.
- Para mucho tráfico conviene ponerlo detrás de un servidor como nginx o Caddy, con HTTPS.
- Si se van a subir datos de pacientes reales, revisa qué exige tu institución sobre datos de salud.
