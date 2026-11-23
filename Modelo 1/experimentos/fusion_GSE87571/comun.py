"""
Funciones compartidas por los scripts del experimento (02 y 03).

Carga Hannum SOLO (partición del modelo previo a la fusión), carga GSE87571 desde la
caché del pipeline (src/gse87571.py) con su propia partición train/test, e implementa
el ranking de sondas sobre varias fuentes de datos a la vez. Solo LEE el proyecto principal.

El Mixup (datos sintéticos) se quitó del pipeline definitivo; las funciones de este
módulo y mixup.py lo conservan solo para que el experimento siga reproduciendo sus
resultados (figuras 15 y 16), que se generaron con N_SONDAS_EXPERIMENTO = 130 sondas.
"""
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

CARPETA_EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA_EXP.parent.parent / "src"))
from config import ARCHIVO_CACHE_SONDAS  # noqa: E402
from datos import (  # noqa: E402
    BLOQUE_RANKING, MAX_FRACCION_NA, N_FOLDS, RANDOM_STATE, TEST_SIZE, matriz_reales, preparar_particion,
)
from gse87571 import cargar_gse87571 as cargar_cache_gse87571  # noqa: E402
from mixup import interpolar, planificar_mixup_estratificado  # noqa: E402  (mixup.py de este experimento)

CARPETA_RESULTADOS = CARPETA_EXP / "resultados"
N_SONDAS_EXPERIMENTO = 130   # k con el que se hizo el experimento (el pipeline definitivo usa otro)


@dataclass
class Dataset:
    nombre: str
    matriz: np.ndarray       # muestras x sondas (orden de sondas de Hannum), imputada
    edades: np.ndarray       # float32
    sexos: np.ndarray        # 'M' / 'F'
    sexo_binario: np.ndarray # columna Sexo_M
    idx_train: np.ndarray
    idx_test: np.ndarray
    particion: object = None  # Particion de src/datos.py (solo Hannum)


def cargar_hannum():
    """Hannum solo, con la misma partición e imputación que tenía el entrenamiento antes de la fusión."""
    p = preparar_particion(datasets=("Hannum",))
    return Dataset("Hannum", p.matriz, p.edades, p.sexos, p.sexo_binario, p.idx_train, p.idx_test, p)


def cargar_gse87571(hannum):
    """GSE87571 con partición 75/25 propia e imputación con medianas de su train."""
    G, pac = cargar_cache_gse87571(np.load(ARCHIVO_CACHE_SONDAS))
    G = np.array(G)  # copia en memoria (la caché se abre en solo lectura)
    validos = pac['Sexo'].isin(['M', 'F']) & pac['Edad'].notna() & (pac['Fraccion_NA'] <= MAX_FRACCION_NA)
    g_tr, g_te = train_test_split(np.where(validos)[0], test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Imputación con medianas del train de GSE87571 (respaldo: medianas del train de Hannum)
    filas_nan, cols_nan = np.where(np.isnan(G))
    if len(filas_nan):
        cols_unicas = np.unique(cols_nan)
        medianas = np.empty(G.shape[1], dtype=np.float32)
        for inicio in range(0, len(cols_unicas), BLOQUE_RANKING):
            cols = cols_unicas[inicio:inicio + BLOQUE_RANKING]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # sondas sin ningún dato en train
                med = np.nanmedian(G[np.ix_(g_tr, cols)], axis=0)
            sin_dato = np.isnan(med)
            if sin_dato.any():
                med[sin_dato] = np.median(hannum.matriz[np.ix_(hannum.idx_train, cols[sin_dato])], axis=0)
            medianas[cols] = med
        G[filas_nan, cols_nan] = medianas[cols_nan]

    return Dataset(
        "GSE87571", G,
        pac['Edad'].values.astype(np.float32),
        pac['Sexo'].values,
        (pac['Sexo'] == 'M').astype(np.float32).values.reshape(-1, 1),
        g_tr, g_te,
    )


def rankear_sondas_fuentes(fuentes, desplazamiento=None):
    """
    Índices de sondas ordenadas por |Spearman ρ| con la edad, calculado sobre la unión de
    las fuentes. Cada fuente es (matriz, edades, índices, restar_desplazamiento).
    """
    y = rankdata(np.concatenate([e[idx] for _, e, idx, _ in fuentes]))
    y = (y - y.mean()) / np.linalg.norm(y - y.mean())
    n_sondas = fuentes[0][0].shape[1]
    rho = np.zeros(n_sondas, dtype=np.float32)
    for inicio in range(0, n_sondas, BLOQUE_RANKING):
        fin = inicio + BLOQUE_RANKING
        bloque = np.vstack([M[idx, inicio:fin] - (desplazamiento[inicio:fin] if corr else 0)
                            for M, _, idx, corr in fuentes])
        rangos = rankdata(bloque, axis=0)
        rangos -= rangos.mean(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho[inicio:fin] = (y @ rangos) / np.linalg.norm(rangos, axis=0)
    return np.argsort(-np.abs(np.nan_to_num(rho, nan=0.0)), kind='stable')


# ==============================================================================
# Mixup del modelo "Definitivo (solo Hannum)" tal como era antes de quitarlo del pipeline
# ==============================================================================
def _plan_mixup_hannum(p):
    np.random.seed(RANDOM_STATE)
    return planificar_mixup_estratificado(p.edades, p.sexos, indices_disponibles=p.idx_train)


def armar_matrices_con_mixup(p, columnas):
    """X/y de train (reales + sintéticos Mixup) y de test (solo reales), como el antiguo src/datos.py."""
    X_reales = matriz_reales(p, columnas)
    plan = _plan_mixup_hannum(p)
    X_synth = np.array([interpolar(X_reales[s['idx_1']], X_reales[s['idx_2']], s['lam']) for s in plan],
                       dtype=np.float32).reshape(-1, X_reales.shape[1])
    y_synth = np.array([s['edad'] for s in plan], dtype=np.float32)
    X_train = np.vstack([X_reales[p.idx_train], X_synth])
    y_train = np.concatenate([p.edades[p.idx_train], y_synth])
    return X_train, y_train, X_reales[p.idx_test], p.edades[p.idx_test]


def splits_sin_fuga_con_mixup(p):
    """Folds que excluyen del entrenamiento los sintéticos con algún padre en el fold de validación."""
    plan = _plan_mixup_hannum(p)
    n_reales = len(p.idx_train)
    fold_reales = p.fold_reales_train
    folds_padres = np.array([(p.fold_paciente[s['idx_1']], p.fold_paciente[s['idx_2']]) for s in plan])
    splits = []
    for k in range(N_FOLDS):
        pos_synth = n_reales + np.where((folds_padres != k).all(axis=1))[0]
        splits.append((np.concatenate([np.where(fold_reales != k)[0], pos_synth]), np.where(fold_reales == k)[0]))
    return splits
