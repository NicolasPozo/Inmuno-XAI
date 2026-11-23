# ==============================================================================
# EXPORTA LOS DATOS DEL DASHBOARD (JSON, una tabla por archivo)
# Lee los resultados ya calculados; no entrena nada.
# Requiere: src/06_modelo_final_tabpfn.py y src/08_explicabilidad.py ejecutados.
# Salida: resultados/explicabilidad/dashboard/*.json
# ==============================================================================

import os
import json
import pandas as pd

from config import (
    CARPETA_DASHBOARD,
    CARPETA_RES_BANCO_PRUEBAS,
    CARPETA_RES_COMPARACION,
    CARPETA_RES_EXPLICABILIDAD,
    CARPETA_RES_MODELOS_PESADOS,
    CARPETA_RES_MODELO_DOS_RAMAS,
    CARPETA_RES_MODELO_FINAL,
    CARPETA_RES_MODELO_INICIAL,
    CARPETA_RES_TABPFN_UN_MODELO,
    EXP_NUMERO_VARIABLES,
)  # rutas del proyecto: src/config.py
salida = CARPETA_DASHBOARD
os.makedirs(salida, exist_ok=True)

def guardar(nombre, df):
    filas = json.loads(df.to_json(orient="records", force_ascii=False))
    with open(os.path.join(salida, f"{nombre}.json"), "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{nombre}.json: {len(filas)} filas")

# 1. Métricas del modelo final
met = pd.read_csv(os.path.join(CARPETA_RES_MODELO_FINAL, "metricas_modelo_final.csv"))
met["prueba"] = met["evaluacion"].map({"CV 5x5 (473 pacientes)": "cv", "Test (119 pacientes)": "test",
                                       "Estudios no vistos (promedio de 5)": "no_vistos"})
guardar("metricas", met[["prueba", "evaluacion", "MAE", "RMSE", "R2", "r", "acierto_5a_%"]]
        .rename(columns={"acierto_5a_%": "acierto5"}).round(3))

# 2. Evolución del error (cada valor leído de su archivo de resultados)
lgb_ini = pd.read_csv(os.path.join(CARPETA_RES_MODELO_INICIAL, "graficos_entrenamiento_multimodal", "comparativa_modelos_multimodal.csv"))
lgb_real = pd.read_csv(os.path.join(CARPETA_RES_MODELO_INICIAL, "graficos_entrenamiento_multimodal_reales_k45", "comparativa_modelos_multimodal.csv"))
banco = pd.read_csv(os.path.join(CARPETA_RES_BANCO_PRUEBAS, "evaluacion_final.csv"))
mf = pd.read_csv(os.path.join(CARPETA_RES_MODELO_DOS_RAMAS, "metricas_modelo_final.csv"))
fus = pd.read_csv(os.path.join(CARPETA_RES_MODELOS_PESADOS, "fusiones_modelos_pesados.csv"))
loso_tab = pd.read_csv(os.path.join(CARPETA_RES_TABPFN_UN_MODELO, "estudios_no_vistos_resumen.csv")).set_index("modelo")
col_test = [c for c in lgb_ini.columns if c.startswith("MAE Test")][0]
mae = lambda d, ev, mo: d[(d.evaluacion == ev) & (d.modelo == mo)]["MAE"].iloc[0]
evo = pd.DataFrame([
    {"orden": 1, "etapa": "LightGBM, train aumentado (inicio)",
     "MAE_test": lgb_ini.loc[lgb_ini.Modelo == "LightGBM", col_test].iloc[0], "MAE_no_vistos": None},
    {"orden": 2, "etapa": "LightGBM, solo pacientes reales",
     "MAE_test": lgb_real.loc[lgb_real.Modelo == "LightGBM", col_test].iloc[0], "MAE_no_vistos": None},
    {"orden": 3, "etapa": "Ensamble + corrección de sesgo (R8)",
     "MAE_test": banco.loc[banco.receta.str.startswith("R8"), "MAE_test_119"].iloc[0], "MAE_no_vistos": None},
    {"orden": 4, "etapa": "Dos ramas + fusión (modelo anterior)",
     "MAE_test": mae(mf, "Test (119 pacientes)", "Fusión"), "MAE_no_vistos": mae(mf, "Estudios no vistos (promedio de 5)", "Fusión")},
    {"orden": 5, "etapa": "TabPFN dos ramas",
     "MAE_test": fus[(fus.rama_celulas == "TabPFN v2") & (fus.rama_citoquinas == "TabPFN v2")]["MAE_test"].iloc[0],
     "MAE_no_vistos": loso_tab.loc["TabPFN dos ramas (fusión)", "MAE"]},
    {"orden": 6, "etapa": "TabPFN un modelo (final)",
     "MAE_test": met.loc[met.prueba == "test", "MAE"].iloc[0], "MAE_no_vistos": met.loc[met.prueba == "no_vistos", "MAE"].iloc[0]},
])
guardar("evolucion", evo.round(3))

# 2b. Los 7 modelos con la misma evaluación (src/05_comparar_modelos.py)
comp = pd.read_csv(os.path.join(CARPETA_RES_COMPARACION, "resumen_comparacion.csv"))
comp = comp.assign(orden=range(1, len(comp) + 1), final=comp["modelo"] == "TabPFN v2").rename(columns={
    "MAE_estudios_no_vistos": "MAE_no_vistos", "r_estudios_no_vistos": "r_no_vistos",
    "acierto_5a_test_%": "acierto5", "tabpfn_mejor_por_anos": "mejora_tabpfn"})
guardar("comparacion_modelos", comp[["orden", "modelo", "final", "MAE_CV", "MAE_CV_sd", "R2_CV", "MAE_test", "R2_test",
                                     "acierto5", "MAE_no_vistos", "r_no_vistos", "mejora_tabpfn", "IC95_bajo", "IC95_alto"]].round(3))

# 2c. Error según el número de biomarcadores (etapa 12; CV 5x3 solo con el train)
res_k = EXP_NUMERO_VARIABLES / "resultados"
curva_k = pd.concat([
    pd.read_csv(res_k / "curva_numero_variables.csv").assign(modelo="TabPFN v2"),
    pd.read_csv(res_k / "curva_numero_variables_clasicos.csv"),
], ignore_index=True)
curva_k["candidatas"] = curva_k["candidatas"].map({"Las 45 actuales": "actuales", "Candidatas (71)": "candidatas"})
curva_k["clave"] = curva_k["modelo"] + "|" + curva_k["candidatas"] + "|" + curva_k["k"].astype(str)
guardar("curva_variables", curva_k[["clave", "modelo", "candidatas", "k", "MAE_CV_train", "MAE_sd"]]
        .rename(columns={"MAE_CV_train": "MAE_CV"}).round(3))

# 3. Estudios no vistos por estudio
nv = pd.read_csv(os.path.join(CARPETA_RES_TABPFN_UN_MODELO, "estudios_no_vistos_tabpfn.csv"))
nombres_nv = {"Modelo final: Fusión": "Modelo anterior (dos ramas)", "TabPFN dos ramas (fusión)": "TabPFN dos ramas",
              "TabPFN un modelo (45)": "TabPFN un modelo (final)"}
nv = nv[nv.modelo.isin(nombres_nv)].assign(modelo=lambda d: d.modelo.map(nombres_nv))
nv["clave"] = nv["estudio_no_visto"] + "|" + nv["modelo"]
guardar("no_vistos", nv[["clave", "estudio_no_visto", "n", "edades", "modelo", "MAE", "r"]]
        .rename(columns={"estudio_no_visto": "estudio"}).round(3))

# 4-6. Importancia, bloques y perfiles
imp = pd.read_csv(os.path.join(CARPETA_RES_EXPLICABILIDAD, "importancia_permutacion.csv"))
imp.insert(0, "puesto", range(1, len(imp) + 1))
guardar("importancia", imp.round(3))
guardar("bloques", pd.read_csv(os.path.join(CARPETA_RES_EXPLICABILIDAD, "importancia_bloques.csv")).round(3))
pdp = pd.read_csv(os.path.join(CARPETA_RES_EXPLICABILIDAD, "dependencia_parcial.csv"))
pdp["clave"] = pdp["variable"] + "|" + pdp.groupby("variable").cumcount().astype(str)
guardar("perfiles", pdp[["clave", "variable", "valor", "edad_predicha_media"]].round(4))

# 7-8. Pacientes de test y explicación individual
br = pd.read_csv(os.path.join(CARPETA_RES_EXPLICABILIDAD, "brecha_edad_473.csv"))
pt = pd.read_csv(os.path.join(CARPETA_RES_MODELO_FINAL, "predicciones_test.csv")).merge(
    br[["subject_accession", "estudio", "sexo"]], on="subject_accession", how="left")
guardar("pacientes_test", pt.rename(columns={"subject_accession": "id"})[
    ["id", "edad_real", "edad_predicha", "brecha_edad", "sexo", "estudio"]].round(2))
ex = pd.read_csv(os.path.join(CARPETA_RES_EXPLICABILIDAD, "explicacion_individual_test.csv"))
larga = ex.melt(id_vars="subject_accession", var_name="variable", value_name="aporte_anos")
tipos = dict(zip(imp.variable, imp.tipo))
larga["tipo"] = larga["variable"].map(tipos)
larga["clave"] = larga["subject_accession"] + "|" + larga["variable"]
guardar("aportes_paciente", larga.rename(columns={"subject_accession": "id"})[
    ["clave", "id", "variable", "tipo", "aporte_anos"]].round(3))

# 9-10. Brecha de edad
br["fila"] = range(len(br))
guardar("brecha", br[["fila", "subject_accession", "conjunto", "estudio", "sexo", "edad_real", "edad_predicha",
                      "brecha", "brecha_corregida"]].rename(columns={"subject_accession": "id"}).round(2))
guardar("brecha_biomarcadores", pd.read_csv(os.path.join(CARPETA_RES_EXPLICABILIDAD, "brecha_vs_biomarcadores.csv"))
        [["variable", "tipo", "rho_con_brecha", "p", "q_BH", "rho_con_edad", "n"]].round(4))
