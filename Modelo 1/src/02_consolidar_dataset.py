"""
PASO 2 - Unión de la matriz de metilación con el Sexo de cada paciente.

Entrada : Datasets/Metadata_Pacientes.csv
          Datasets/Dataset_Hannum_Metilacion_Chunks.csv
Salida  : Datasets/Dataset_Hannum_Metilacion_Completo.csv
          Datasets/Dataset_Hannum_Muestra_10.csv (primeros 10 pacientes)
"""
import pandas as pd

from config import (
    ARCHIVO_METADATA,
    ARCHIVO_METILACION_CHUNKS,
    ARCHIVO_METILACION_COMPLETO,
    ARCHIVO_MUESTRA_10,
)

# 1. CARGA DEL DICCIONARIO DE SEXO (CONSUMO MÍNIMO DE RAM)
print("1. Cargando metadatos de pacientes...")
df_meta = pd.read_csv(ARCHIVO_METADATA)

# Limpieza de identificadores y mapeo ID -> Sexo
df_meta['ID_Paciente'] = df_meta['ID_Paciente'].astype(str).str.strip()
mapa_sexo = dict(zip(df_meta['ID_Paciente'], df_meta['Sexo']))
print(f"   -> {len(mapa_sexo)} pacientes indexados para cruce rápido.")

# 2. PROCESAMIENTO POR STREAMING (LÍNEA POR LÍNEA)
print("\n2. Iniciando unión por streaming a disco...")

contador_filas = 0

with open(ARCHIVO_METILACION_CHUNKS, 'r', encoding='utf-8') as f_in, \
     open(ARCHIVO_METILACION_COMPLETO, 'w', encoding='utf-8') as f_out_full, \
     open(ARCHIVO_MUESTRA_10, 'w', encoding='utf-8') as f_out_sample:

    # Procesar encabezado
    linea_header = f_in.readline()
    if not linea_header:
        raise ValueError("El archivo de metilación está vacío.")

    # El encabezado original es: ID_Paciente,Edad,cg...
    # Insertamos la columna 'Sexo' inmediatamente después de 'Edad'
    partes_header = linea_header.rstrip('\r\n').split(',', 2)
    nuevo_header = f"{partes_header[0]},{partes_header[1]},Sexo,{partes_header[2]}\n"

    f_out_full.write(nuevo_header)
    f_out_sample.write(nuevo_header)

    # Procesar paciente por paciente
    for linea in f_in:
        contador_filas += 1

        # Dividimos únicamente en los 2 primeros cortes para no procesar las 470,000 columnas
        # partes: [ID_Paciente, Edad, "cg1,cg2,cg3..."]
        partes = linea.rstrip('\r\n').split(',', 2)

        id_paciente = partes[0].strip()
        sexo_paciente = str(mapa_sexo.get(id_paciente, "NA"))

        nueva_linea = f"{partes[0]},{partes[1]},{sexo_paciente},{partes[2]}\n"

        # Escribir en el archivo maestro
        f_out_full.write(nueva_linea)

        # Escribir solo los primeros 10 pacientes en el archivo ligero
        if contador_filas <= 10:
            f_out_sample.write(nueva_linea)

        if contador_filas % 50 == 0:
            print(f"   -> Procesados {contador_filas} pacientes...")

print("\n" + "="*45)
print("PROCESO COMPLETADO EXITOSAMENTE")
print(f"Total de pacientes procesados: {contador_filas}")
print(f"Archivo completo generado: {ARCHIVO_METILACION_COMPLETO}")
print(f"Muestra ligera generada:   {ARCHIVO_MUESTRA_10} (10 filas)")
print("="*45)
