"""
Configuración compartida del proyecto: rutas de archivos y estilo de gráficos.

Todas las rutas se calculan a partir de la ubicación de este archivo, por lo que
los scripts funcionan sin importar desde qué carpeta se ejecuten.
"""
import os
from pathlib import Path

# ==============================================================================
# CARPETAS DEL PROYECTO
# ==============================================================================
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_DATASETS = RAIZ_PROYECTO / "Datasets"
CARPETA_RESULTADOS = RAIZ_PROYECTO / "resultados"
CARPETA_RESULTADOS_EDA = CARPETA_RESULTADOS / "eda"
CARPETA_RESULTADOS_SONDAS = CARPETA_RESULTADOS / "seleccion_sondas"
CARPETA_RESULTADOS_APRENDIZAJE = CARPETA_RESULTADOS / "curva_aprendizaje"
CARPETA_RESULTADOS_ENTRENAMIENTO = CARPETA_RESULTADOS / "entrenamiento"
CARPETA_MODELOS = RAIZ_PROYECTO / "modelos"   # Modelos entrenados (.joblib) listos para reutilizar
# ==============================================================================
# ARCHIVOS DE DATOS (en orden del pipeline)
# ==============================================================================
# Datos crudos de GEO (GSE40279 - Hannum et al.)
ARCHIVO_GEO_SOFT = CARPETA_DATASETS / "GSE40279_family.soft.gz"

# Paso 1 -> Edad y Sexo de cada paciente
ARCHIVO_METADATA = CARPETA_DATASETS / "Metadata_Pacientes.csv"

# Matriz de metilación (ID_Paciente, Edad, cg...) generada previamente por chunks
ARCHIVO_METILACION_CHUNKS = CARPETA_DATASETS / "Dataset_Hannum_Metilacion_Chunks.csv"

# Paso 2 -> Matriz de metilación con la columna Sexo
ARCHIVO_METILACION_COMPLETO = CARPETA_DATASETS / "Dataset_Hannum_Metilacion_Completo.csv"
ARCHIVO_MUESTRA_10 = CARPETA_DATASETS / "Dataset_Hannum_Muestra_10.csv"

# Caché binaria de todas las sondas CpG (se crea sola desde el CSV Completo; pasos 03-06)
CARPETA_CACHE = CARPETA_DATASETS / "cache_matriz_completa"
ARCHIVO_CACHE_MATRIZ = CARPETA_CACHE / "matriz_cpg_float32.npy"
ARCHIVO_CACHE_SONDAS = CARPETA_CACHE / "sondas_cpg.npy"
ARCHIVO_CACHE_PACIENTES = CARPETA_CACHE / "pacientes.csv"

# Segundo dataset: GSE87571 (Johansson et al.), mismo array 450K, sangre completa, 14-94 años
CARPETA_GSE87571 = CARPETA_DATASETS / "GSE87571"                   # archivos originales de GEO
ARCHIVO_GSE87571_SERIE = CARPETA_GSE87571 / "GSE87571_series_matrix.txt.gz"
ARCHIVOS_GSE87571_BETAS = [CARPETA_GSE87571 / "GSE87571_matrix1of2.txt.gz",
                           CARPETA_GSE87571 / "GSE87571_matrix2of2.txt.gz"]
CARPETA_CACHE_GSE87571 = CARPETA_DATASETS / "cache_GSE87571"       # se crea sola (pasos 04-06)
ARCHIVO_CACHE_GSE87571_MATRIZ = CARPETA_CACHE_GSE87571 / "matriz_cpg_float32.npy"
ARCHIVO_CACHE_GSE87571_PACIENTES = CARPETA_CACHE_GSE87571 / "pacientes.csv"


def crear_carpeta(ruta):
    """Crea la carpeta (y sus padres) si no existe y la devuelve."""
    os.makedirs(ruta, exist_ok=True)
    return ruta


def configurar_estilo_graficos():
    """Estilo visual común para todas las figuras del proyecto."""
    import matplotlib.pyplot as plt

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 120
