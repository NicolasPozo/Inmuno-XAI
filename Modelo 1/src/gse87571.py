"""
Carga de GSE87571 (Johansson et al.): 732 muestras de sangre completa, Illumina 450K
(GPL13534, el mismo array que Hannum), personas sanas de 14 a 94 años.

La primera vez convierte los archivos originales de GEO (Datasets/GSE87571/) a una caché
binaria (Datasets/cache_GSE87571/) con las sondas en el MISMO orden que Hannum. Si los
archivos originales cambian, la caché se regenera.

Formato de GEO: cada muestra tiene una columna beta ('X1') y otra con el p-valor de
detección ('X1.1'); las betas de sondas fallidas vienen como 'NA'.
"""
import gzip
import os
import time

import numpy as np
import pandas as pd

from config import (
    ARCHIVO_CACHE_GSE87571_MATRIZ,
    ARCHIVO_CACHE_GSE87571_PACIENTES,
    ARCHIVOS_GSE87571_BETAS,
    ARCHIVO_GSE87571_SERIE,
    CARPETA_CACHE_GSE87571,
    crear_carpeta,
)

SEXO = {'Male': 'M', 'Female': 'F'}


def _leer_metadatos():
    """Edad y sexo de cada muestra desde la serie de GEO, indexados por título ('X1', 'X2', ...)."""
    campos = {}
    with gzip.open(ARCHIVO_GSE87571_SERIE, 'rt', encoding='utf-8') as f:
        for linea in f:
            if linea.startswith('!series_matrix_table_begin'):
                break
            clave, _, resto = linea.rstrip('\n').partition('\t')
            valores = [v.strip('"') for v in resto.split('\t')]
            if clave == '!Sample_characteristics_ch1':
                nombre = valores[0].split(':')[0].strip()
                campos[nombre] = [v.split(':', 1)[1].strip() for v in valores]
            elif clave in ('!Sample_title', '!Sample_geo_accession'):
                campos[clave] = valores

    return pd.DataFrame({
        'ID_Paciente': campos['!Sample_geo_accession'],
        'Titulo': [t.split()[0] for t in campos['!Sample_title']],  # 'X1 genomic DNA...' -> 'X1'
        'Edad': pd.to_numeric(campos['age'], errors='coerce'),
        'Sexo': [SEXO.get(s, 'NA') for s in campos['gender']],
    }).set_index('Titulo')


def _crear_cache(sondas_referencia):
    print("   -> Creando caché de GSE87571 desde los archivos de GEO (solo la primera vez)...")
    t_inicio = time.perf_counter()
    crear_carpeta(CARPETA_CACHE_GSE87571)
    meta = _leer_metadatos()

    bloques, titulos = [], []
    for archivo in ARCHIVOS_GSE87571_BETAS:
        with gzip.open(archivo, 'rt', encoding='utf-8') as f:
            cabecera = f.readline().rstrip('\n').split('\t')
        cols_beta = [c for c in cabecera[1:] if not c.endswith('.1')]
        df = pd.read_csv(archivo, sep='\t', usecols=['ID_REF'] + cols_beta, index_col='ID_REF',
                         dtype={c: np.float32 for c in cols_beta}, na_values=['NA'])
        # Mismo orden de sondas que Hannum; las que falten quedan como NaN
        bloques.append(df.reindex(sondas_referencia).values.T.astype(np.float32))
        titulos += cols_beta
        print(f"      * {archivo.name}: {len(cols_beta)} muestras")
        del df

    matriz = np.vstack(bloques)
    del bloques
    pacientes = meta.loc[titulos].reset_index()
    pacientes['Fraccion_NA'] = np.isnan(matriz).mean(axis=1)

    np.save(ARCHIVO_CACHE_GSE87571_MATRIZ, matriz)
    pacientes[['ID_Paciente', 'Edad', 'Sexo', 'Titulo', 'Fraccion_NA']].to_csv(
        ARCHIVO_CACHE_GSE87571_PACIENTES, index=False)
    print(f"   -> Caché GSE87571 creada: {matriz.shape[0]} muestras x {matriz.shape[1]} sondas "
          f"({time.perf_counter() - t_inicio:.0f} s)")


def cargar_gse87571(sondas_referencia):
    """
    Devuelve (matriz float32 en modo solo lectura desde disco, DataFrame de pacientes).
    'sondas_referencia' fija el orden de columnas (las sondas de Hannum).
    """
    existe = all(os.path.exists(r) for r in (ARCHIVO_CACHE_GSE87571_MATRIZ, ARCHIVO_CACHE_GSE87571_PACIENTES))
    origen_mas_nuevo = existe and os.path.getmtime(ARCHIVO_CACHE_GSE87571_MATRIZ) < max(
        os.path.getmtime(a) for a in [ARCHIVO_GSE87571_SERIE] + ARCHIVOS_GSE87571_BETAS)
    if not existe or origen_mas_nuevo:
        _crear_cache(sondas_referencia)
    matriz = np.load(ARCHIVO_CACHE_GSE87571_MATRIZ, mmap_mode='r')
    if matriz.shape[1] != len(sondas_referencia):
        raise ValueError("La caché de GSE87571 no coincide con las sondas de Hannum; borra "
                         f"'{CARPETA_CACHE_GSE87571}' para regenerarla.")
    return matriz, pd.read_csv(ARCHIVO_CACHE_GSE87571_PACIENTES)
