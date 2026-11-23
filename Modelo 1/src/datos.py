"""
Preparación de datos sin fuga, compartida por los pasos 04, 05 y 06.

Flujo (solo pacientes reales, sin datos sintéticos):
  1. Se cargan los pacientes de cada dataset de DATASETS con TODAS las sondas CpG del
     array (cachés binarias en Datasets/) y se fusionan en una sola matriz
     (primero Hannum, luego GSE87571).
  2. Dentro de cada dataset se separa el 25% como test antes de cualquier otro paso.
  3. Imputación por mediana calculada solo con el train de cada dataset.
  4. Cada paciente de train recibe un fold de validación cruzada (estratificado por dataset).
"""
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from config import (
    ARCHIVO_CACHE_MATRIZ,
    ARCHIVO_CACHE_PACIENTES,
    ARCHIVO_CACHE_SONDAS,
    ARCHIVO_METILACION_COMPLETO,
    CARPETA_CACHE,
    crear_carpeta,
)
from gse87571 import cargar_gse87571

DATASETS = ("Hannum", "GSE87571")  # Datasets fusionados; ("Hannum",) entrena solo con Hannum
TEST_SIZE = 0.25            # 25% de pacientes reales de cada dataset reservados exclusivamente para test
N_SONDAS_CANDIDATAS = None  # None = todas las sondas CpG del array; un entero = solo las primeras n
N_SONDAS_TOP = 500          # Sondas usadas por los modelos (k recomendado por la curva del paso 04)
N_FOLDS = 5
RANDOM_STATE = 42
BLOQUE_RANKING = 20000      # Sondas procesadas a la vez al calcular Spearman (controla la RAM)
MAX_FRACCION_NA = 0.05      # Muestras con más sondas faltantes se descartan


@dataclass
class Particion:
    matriz: np.ndarray          # Sondas CpG imputadas (float32), una fila por paciente real
    sondas: np.ndarray          # Nombre de cada columna de 'matriz'
    edades: np.ndarray          # Edad (float32)
    sexos: np.ndarray           # 'M' / 'F'
    sexo_binario: np.ndarray    # Columna Sexo_M (M=1, F=0)
    lote: np.ndarray            # Dataset de origen de cada fila (índice en 'nombres_lote')
    nombres_lote: tuple         # Nombre de cada dataset, p. ej. ('Hannum', 'GSE87571')
    ids: np.ndarray             # ID de GEO de cada paciente (GSM...)
    idx_train: np.ndarray       # Posiciones de pacientes reales de train
    idx_test: np.ndarray        # Posiciones de pacientes reales de test
    fold_paciente: np.ndarray   # Fold de validación de cada paciente (-1 = test)

    @property
    def fold_reales_train(self):
        return self.fold_paciente[self.idx_train]

    def por_dataset(self, indices):
        """{nombre del dataset: subconjunto de 'indices' que pertenece a ese dataset}."""
        return {nombre: indices[self.lote[indices] == i] for i, nombre in enumerate(self.nombres_lote)}


def _parsear_valores(texto, n_esperado):
    """Convierte 'v1,v2,...' a float32; los campos vacíos o no numéricos quedan como NaN."""
    valores = np.fromstring(texto, sep=',', dtype=np.float32)
    if valores.size == n_esperado:
        return valores
    # Ruta lenta solo si la fila tiene campos vacíos o texto
    return pd.to_numeric(pd.Series(texto.split(',')), errors='coerce').values.astype(np.float32)


def _crear_cache():
    """Lee el CSV completo línea a línea y guarda la matriz de sondas CpG en formato binario."""
    print(f"   -> Creando caché binaria desde '{ARCHIVO_METILACION_COMPLETO.name}' (solo la primera vez)...")
    t_inicio = time.perf_counter()
    crear_carpeta(CARPETA_CACHE)

    with open(ARCHIVO_METILACION_COMPLETO, 'r', encoding='utf-8') as f:
        nombres = f.readline().rstrip('\r\n').split(',')
        sondas_todas = np.array(nombres[3:])
        es_cpg = np.char.startswith(sondas_todas, 'cg')

        filas, pacientes = [], []
        for num_linea, linea in enumerate(f):
            partes = linea.rstrip('\r\n').split(',', 3)
            pacientes.append(partes[:3])
            filas.append(_parsear_valores(partes[3], len(sondas_todas))[es_cpg])
            if (num_linea + 1) % 100 == 0:
                print(f"      * {num_linea + 1} pacientes leídos...")

    np.save(ARCHIVO_CACHE_MATRIZ, np.vstack(filas))
    np.save(ARCHIVO_CACHE_SONDAS, sondas_todas[es_cpg])
    pd.DataFrame(pacientes, columns=nombres[:3]).to_csv(ARCHIVO_CACHE_PACIENTES, index=False)
    print(f"   -> Caché creada: {len(filas)} pacientes x {es_cpg.sum()} sondas CpG "
          f"({time.perf_counter() - t_inicio:.0f} s)")


def cargar_matriz_completa():
    """Devuelve (matriz float32, nombres de sondas, DataFrame de pacientes) desde la caché."""
    cache_existe = all(os.path.exists(r) for r in (ARCHIVO_CACHE_MATRIZ, ARCHIVO_CACHE_SONDAS, ARCHIVO_CACHE_PACIENTES))
    if not cache_existe or os.path.getmtime(ARCHIVO_CACHE_MATRIZ) < os.path.getmtime(ARCHIVO_METILACION_COMPLETO):
        _crear_cache()
    return (np.load(ARCHIVO_CACHE_MATRIZ, mmap_mode='r'),
            np.load(ARCHIVO_CACHE_SONDAS),
            pd.read_csv(ARCHIVO_CACHE_PACIENTES))


def _cargar_fuentes(datasets):
    """Lista de (nombre, matriz desde disco, pacientes con columna 'Valido') y nombres de sondas."""
    matriz_h, sondas, pac_h = cargar_matriz_completa()
    fuentes = [("Hannum", matriz_h, pac_h.assign(Valido=True))]
    if "GSE87571" in datasets:
        matriz_g, pac_g = cargar_gse87571(sondas)
        valido = pac_g['Sexo'].isin(['M', 'F']) & pac_g['Edad'].notna() & (pac_g['Fraccion_NA'] <= MAX_FRACCION_NA)
        fuentes.append(("GSE87571", matriz_g, pac_g.assign(Valido=valido)))
    return fuentes, sondas


def _imputar(matriz, inicio, fin, idx_train_ds, medianas_respaldo):
    """Rellena los NaN de las filas inicio:fin con la mediana de cada sonda en 'idx_train_ds'."""
    sub_filas, cols_nan = np.where(np.isnan(matriz[inicio:fin]))  # vista: no copia la matriz
    if not len(sub_filas):
        return
    cols_unicas = np.unique(cols_nan)
    medianas = np.empty(matriz.shape[1], dtype=np.float32)
    for bloque in range(0, len(cols_unicas), BLOQUE_RANKING):
        cols = cols_unicas[bloque:bloque + BLOQUE_RANKING]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # sondas sin ningún dato en train
            med = np.nanmedian(matriz[np.ix_(idx_train_ds, cols)], axis=0)
        sin_dato = np.isnan(med)
        if sin_dato.any() and medianas_respaldo is not None:
            med[sin_dato] = medianas_respaldo(cols[sin_dato])
        medianas[cols] = med
    matriz[inicio + sub_filas, cols_nan] = medianas[cols_nan]


def preparar_particion(datasets=DATASETS, n_sondas=N_SONDAS_CANDIDATAS):
    """Carga y fusiona los pacientes reales de 'datasets' y construye la partición train/test sin fuga."""
    fuentes, sondas = _cargar_fuentes(datasets)
    if n_sondas is not None:
        sondas = sondas[:n_sondas]

    # Matriz fusionada (se reserva una sola vez para no duplicar memoria)
    n_total = sum(len(pac) for _, _, pac in fuentes)
    matriz = np.empty((n_total, len(sondas)), dtype=np.float32)
    lote = np.empty(n_total, dtype=int)
    idx_train, idx_test, rangos = [], [], []
    inicio = 0
    for i, (_, M, pac) in enumerate(fuentes):
        fin = inicio + len(pac)
        matriz[inicio:fin] = M[:, :len(sondas)]
        lote[inicio:fin] = i
        # Partición dentro de cada dataset (solo muestras válidas)
        tr, te = train_test_split(np.where(pac['Valido'].values)[0], test_size=TEST_SIZE, random_state=RANDOM_STATE)
        idx_train.append(inicio + tr)
        idx_test.append(inicio + te)
        rangos.append((inicio, fin))
        inicio = fin

    # Imputación con el train de cada dataset (respaldo: medianas del train del primer dataset)
    for i, (ini, fin) in enumerate(rangos):
        respaldo = None if i == 0 else (lambda cols: np.median(matriz[np.ix_(idx_train[0], cols)], axis=0))
        _imputar(matriz, ini, fin, idx_train[i], respaldo)

    pacientes = pd.concat([pac for _, _, pac in fuentes], ignore_index=True)
    edades = pacientes['Edad'].values.astype(np.float32)
    sexos = pacientes['Sexo'].values
    idx_train, idx_test = np.concatenate(idx_train), np.concatenate(idx_test)

    # Fold de validación asignado a cada paciente real de train (-1 = test)
    fold_paciente = np.full(n_total, -1)
    if len(fuentes) == 1:
        divisiones = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(idx_train)
    else:
        divisiones = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(
            idx_train, lote[idx_train])
    for k, (_, pos_val) in enumerate(divisiones):
        fold_paciente[idx_train[pos_val]] = k

    return Particion(
        matriz=matriz,
        sondas=sondas,
        edades=edades,
        sexos=sexos,
        sexo_binario=(pacientes['Sexo'] == 'M').astype(np.float32).values.reshape(-1, 1),
        lote=lote,
        nombres_lote=tuple(nombre for nombre, _, _ in fuentes),
        ids=pacientes['ID_Paciente'].values,
        idx_train=idx_train,
        idx_test=idx_test,
        fold_paciente=fold_paciente,
    )


def correlaciones_spearman(p, indices, absoluto=True):
    """Spearman ρ (|ρ| si absoluto) de cada sonda con la edad, usando solo los pacientes 'indices'."""
    y = rankdata(p.edades[indices])
    y = (y - y.mean()) / np.linalg.norm(y - y.mean())

    rho = np.zeros(p.matriz.shape[1], dtype=np.float32)
    for inicio in range(0, p.matriz.shape[1], BLOQUE_RANKING):
        rangos = rankdata(p.matriz[indices, inicio:inicio + BLOQUE_RANKING], axis=0)
        rangos -= rangos.mean(axis=0)
        normas = np.linalg.norm(rangos, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rho[inicio:inicio + BLOQUE_RANKING] = (y @ rangos) / normas
    rho = np.nan_to_num(rho, nan=0.0)  # Sondas constantes -> 0
    return np.abs(rho) if absoluto else rho


def rankear_sondas(p, indices):
    """Índices de columna de las sondas ordenadas por |Spearman ρ| con la edad (mayor primero)."""
    return np.argsort(-correlaciones_spearman(p, indices), kind='stable')


def matriz_reales(p, columnas):
    """Matriz de todos los pacientes reales con las sondas 'columnas' (índices) + Sexo_M."""
    return np.hstack([p.matriz[:, columnas], p.sexo_binario])


def armar_matrices(p, columnas):
    """X/y de train y de test con las sondas 'columnas' (índices) + Sexo_M."""
    X_reales = matriz_reales(p, columnas)
    return X_reales[p.idx_train], p.edades[p.idx_train], X_reales[p.idx_test], p.edades[p.idx_test]


def splits_sin_fuga(p):
    """Folds de validación cruzada en coordenadas de X_train (filas de p.idx_train)."""
    fold_reales = p.fold_reales_train
    return [(np.where(fold_reales != k)[0], np.where(fold_reales == k)[0]) for k in range(N_FOLDS)]
