"""
PASO 1 - Extracción de metadatos clínicos (Edad y Sexo) desde GEO.

Entrada : Datasets/GSE40279_family.soft.gz
Salida  : Datasets/Metadata_Pacientes.csv
"""
import GEOparse
import pandas as pd

from config import ARCHIVO_GEO_SOFT, ARCHIVO_METADATA

print("1. Leyendo metadatos del archivo local (esto tomará solo unos segundos)...")

# Leemos el archivo SOFT (el que ya tienes y usamos para la matriz)
gse = GEOparse.get_GEO(filepath=str(ARCHIVO_GEO_SOFT))
metadata = gse.phenotype_data

print("2. Buscando dinámicamente la variable de Sexo/Género...")
columna_sexo = None

# Recorremos los NOMBRES de las columnas para buscar la palabra "gender" o "sex"
for col in metadata.columns:
    if 'gender' in col.lower() or 'sex' in col.lower():
        columna_sexo = col
        break

if columna_sexo:
    print(f"-> ¡Variable encontrada en la columna '{columna_sexo}'!")

    # Nos quedamos con el ID del paciente, la edad (que estaba en 'title') y el sexo
    df_clinico = metadata[['title', columna_sexo]].copy()

    # LIMPIEZA DE DATOS (Data Cleaning)
    # 1. Extraemos la edad como número
    df_clinico['Edad'] = df_clinico['title'].str.extract(r'age\s+(\d+)y').astype(float)

    # 2. Limpiamos el texto del sexo (por si acaso viene como "gender: M" o solo "M")
    df_clinico['Sexo'] = df_clinico[columna_sexo].astype(str).str.split(':').str[-1].str.strip()

    # Renombramos el índice para que coincida con tu matriz gigante
    df_clinico.index.name = 'ID_Paciente'

    # Filtramos para dejar solo las columnas limpias
    df_clinico = df_clinico[['Edad', 'Sexo']]

    print("\n--- VISTA PREVIA DE LOS DATOS ---")
    print(df_clinico.head())

    # Guardamos este pequeño archivo
    df_clinico.to_csv(ARCHIVO_METADATA)

    print("\n=============================================")
    print("🎉 ¡METADATOS EXTRAÍDOS CON ÉXITO!")
    print(f"Archivo guardado: '{ARCHIVO_METADATA}'")
    print("Este archivo será tu base para las etiquetas del modelo de Machine Learning.")
    print("=============================================")
else:
    print("❌ No se encontró la variable de sexo a simple vista. Revisa las columnas:")
    print(metadata.columns.tolist())
