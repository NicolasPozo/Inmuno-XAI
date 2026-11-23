"""
Configuración compartida del proyecto: rutas de datos, resultados y modelos.

Todas las rutas se calculan a partir de la ubicación de este archivo, por lo que
los scripts funcionan sin importar desde qué carpeta se ejecuten.
"""
from pathlib import Path

# ==============================================================================
# CARPETAS DEL PROYECTO
# ==============================================================================
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
CARPETA_DATASETS = RAIZ_PROYECTO / "Datasets"
CARPETA_RESULTADOS = RAIZ_PROYECTO / "resultados"
CARPETA_MODELOS = RAIZ_PROYECTO / "modelos"
CARPETA_EXPERIMENTOS = RAIZ_PROYECTO / "experimentos"

# ==============================================================================
# DATOS DE PARTIDA (no se modifican)
# ==============================================================================
# Train (con ruido y mixup) y test con los 45 biomarcadores seleccionados al inicio
ARCHIVO_TRAIN_AUMENTADO_K45 = CARPETA_DATASETS / "df_train_aumentado_k45.csv"
ARCHIVO_TEST_K45 = CARPETA_DATASETS / "df_test_puro_k45.csv"

# Stanford 10K Immunomes: archivos normalizados (github.com/buttelab/10kimmunomes_v2)
CARPETA_10K = CARPETA_DATASETS / "10k"
_EXTRAIDO_10K = CARPETA_10K / "extraido"
ARCHIVO_CYTOF_10K = _EXTRAIDO_10K / "CyTOF_PBMC" / "CyTOF_PBMC" / "cytof_pbmc_normalized.csv"
ARCHIVO_ELISA_10K = (_EXTRAIDO_10K / "Multiplex_ELISA" / "Multiplex_ELISA" / "Normalized_Files"
                     / "multiplexELISA_normalized.csv")
ARCHIVO_FLUJO_10K = (_EXTRAIDO_10K / "Flow_Cytometry_PBMC" / "Flow_Cytometry_PBMC" / "Normalized_Files"
                     / "flow_pbmc_normalized.csv")

# ==============================================================================
# DATOS GENERADOS POR EL PIPELINE (src/01 a src/04)
# ==============================================================================
# 01 -> ID de paciente de cada fila del train k45 (y el train ampliado, experimento descartado)
ARCHIVO_IDS_PACIENTES = CARPETA_10K / "ids_pacientes_actuales.csv"
ARCHIVO_PACIENTES_NUEVOS_10K = CARPETA_10K / "pacientes_nuevos_10k.csv"
ARCHIVO_TRAIN_AMPLIADO = CARPETA_DATASETS / "df_train_aumentado_k45_ampliado.csv"
# 02 -> train solo con pacientes reales
ARCHIVO_TRAIN_REALES_K45 = CARPETA_DATASETS / "df_train_reales_k45.csv"
# 03 -> mismos pacientes con todas las variables del 10K
ARCHIVO_TRAIN_REALES_COMPLETO = CARPETA_DATASETS / "df_train_reales_completo.csv"
ARCHIVO_TEST_COMPLETO = CARPETA_DATASETS / "df_test_completo.csv"
# 04 -> tabla de trabajo del modelo final y fuentes de una sola modalidad
ARCHIVO_PAREADOS = CARPETA_DATASETS / "pareados.csv"
CARPETA_FUENTES_CELULAS = CARPETA_DATASETS / "fuentes_celulas"
CARPETA_FUENTES_CITOQUINAS = CARPETA_DATASETS / "fuentes_citoquinas"

# ==============================================================================
# COMPARACIÓN DE MODELOS (src/05)
# ==============================================================================
CARPETA_RES_COMPARACION = CARPETA_RESULTADOS / "comparacion_modelos"

# ==============================================================================
# MODELO FINAL Y SUS RESULTADOS (src/06 a src/09)
# ==============================================================================
ARCHIVO_MODELO_FINAL = CARPETA_MODELOS / "modelo_tabpfn.joblib"
ARCHIVO_VARIABLES_MODELO = CARPETA_MODELOS / "variables_modelo.csv"
CARPETA_RES_MODELO_FINAL = CARPETA_RESULTADOS / "modelo_final"
CARPETA_RES_EXPLICABILIDAD = CARPETA_RESULTADOS / "explicabilidad"
CARPETA_DASHBOARD = CARPETA_RES_EXPLICABILIDAD / "dashboard"
# 10 -> figuras y cifras del informe de título (Experimento A)
CARPETA_RES_INFORME = CARPETA_RESULTADOS / "informe"

# ==============================================================================
# EXPERIMENTOS (etapas que llevaron al modelo final; ver README.md)
# ==============================================================================
EXP_MODELO_INICIAL = CARPETA_EXPERIMENTOS / "01_modelo_inicial"
EXP_MAS_PACIENTES = CARPETA_EXPERIMENTOS / "02_mas_pacientes"
EXP_BANCO_PRUEBAS = CARPETA_EXPERIMENTOS / "03_banco_pruebas"
EXP_DOS_RAMAS = CARPETA_EXPERIMENTOS / "04_dos_ramas"
EXP_MODELO_DOS_RAMAS = CARPETA_EXPERIMENTOS / "05_modelo_dos_ramas"
EXP_TABPFN = CARPETA_EXPERIMENTOS / "06_tabpfn"
EXP_NUMERO_VARIABLES = CARPETA_EXPERIMENTOS / "07_numero_variables"

CARPETA_RES_MODELO_INICIAL = EXP_MODELO_INICIAL / "resultados"
CARPETA_RES_BANCO_PRUEBAS = EXP_BANCO_PRUEBAS / "resultados"
CARPETA_RES_MODELO_DOS_RAMAS = EXP_MODELO_DOS_RAMAS / "resultados" / "modelo_final"
CARPETA_RES_MODELOS_PESADOS = EXP_TABPFN / "resultados" / "modelos_pesados"
CARPETA_RES_TABPFN_UN_MODELO = EXP_TABPFN / "resultados" / "tabpfn_un_modelo"
ARCHIVO_SCRIPT_DOS_RAMAS = EXP_MODELO_DOS_RAMAS / "modelo_final_dos_ramas.py"  # src/05 ejecuta parte de él

# ==============================================================================
# NOMBRES ANTIGUOS -> UBICACIÓN ACTUAL
# Antes todo estaba en una sola carpeta. Los scripts de experimentos/ siguen
# pidiendo cada archivo por su nombre antiguo con ruta(), y aquí se traduce.
# ==============================================================================
UBICACION = {
    # Datos
    "df_train_aumentado_k45.csv": ARCHIVO_TRAIN_AUMENTADO_K45,
    "df_test_puro_k45.csv": ARCHIVO_TEST_K45,
    "df_train_reales_k45.csv": ARCHIVO_TRAIN_REALES_K45,
    "df_train_reales_completo.csv": ARCHIVO_TRAIN_REALES_COMPLETO,
    "df_test_completo.csv": ARCHIVO_TEST_COMPLETO,
    "df_train_aumentado_k45_ampliado.csv": ARCHIVO_TRAIN_AMPLIADO,
    "df_train_aumentado_k45_fluprint.csv": CARPETA_DATASETS / "df_train_aumentado_k45_fluprint.csv",
    "pareados.csv": ARCHIVO_PAREADOS,
    "datos_10k": CARPETA_10K,
    "datos_fluprint": CARPETA_DATASETS / "fluprint",
    "datos_soundlife": CARPETA_DATASETS / "soundlife",
    "datos_immport": CARPETA_DATASETS / "immport",
    "fuentes_celulas": CARPETA_FUENTES_CELULAS,
    "fuentes_citoquinas": CARPETA_FUENTES_CITOQUINAS,
    "fuentes_immport": CARPETA_DATASETS / "fuentes_immport",
    # Resultados de los experimentos
    "graficos_curva_aprendizaje": EXP_MAS_PACIENTES / "resultados",
    "resultados_banco_pruebas": CARPETA_RES_BANCO_PRUEBAS,
    "resultados_dos_ramas": EXP_DOS_RAMAS / "resultados",
    "resultados_modelo_final": CARPETA_RES_MODELO_DOS_RAMAS,
    "resultados_modelo_final_con_elisa": EXP_MODELO_DOS_RAMAS / "resultados" / "modelo_final_con_elisa",
    "resultados_modelo_final_immport": EXP_MODELO_DOS_RAMAS / "resultados" / "modelo_final_immport",
    "resultados_modelos_pesados": CARPETA_RES_MODELOS_PESADOS,
    "resultados_tabpfn_un_modelo": CARPETA_RES_TABPFN_UN_MODELO,
    "resultados_modelo_final_tabpfn": CARPETA_RES_MODELO_FINAL,
    "resultados_explicabilidad": CARPETA_RES_EXPLICABILIDAD,
    # Scripts que otros scripts ejecutan en parte (exec)
    "modelo_final_dos_ramas.py": EXP_MODELO_DOS_RAMAS / "modelo_final_dos_ramas.py",
}


def ruta(nombre, *resto):
    """Ruta actual (texto) de un archivo o carpeta pedido por su nombre antiguo.

    ruta("pareados.csv") -> .../Datasets/pareados.csv
    ruta("resultados_dos_ramas", "x.csv") -> .../experimentos/04_dos_ramas/resultados/x.csv
    Las carpetas graficos_entrenamiento_multimodal* (una por cada train probado
    con Entrenamiento.py) van todas a experimentos/01_modelo_inicial/resultados/.
    """
    if nombre in UBICACION:
        base = UBICACION[nombre]
    elif nombre.startswith("graficos_entrenamiento_multimodal"):
        base = CARPETA_RES_MODELO_INICIAL / nombre
    else:
        raise KeyError(f"Ruta desconocida: '{nombre}'. Agrégala a UBICACION en src/config.py")
    return str(base.joinpath(*resto))
