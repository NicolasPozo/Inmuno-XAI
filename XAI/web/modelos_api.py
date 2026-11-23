# ==============================================================================
# CÁLCULO DE LA EDAD DE PACIENTES NUEVOS (los dos modelos)
#
# Lo usan servidor.py (la página web) y preparar_recursos.py. Cada función
# recibe el texto de un CSV y devuelve filas con el mismo formato que los
# archivos del dashboard, así la página dibuja al paciente nuevo con los
# mismos gráficos que a los pacientes de test:
#
#   Epigenético (ElasticNet)  -> epi_pacientes_test + epi_aportes
#
#   Inmunológico (TabPFN v2)  -> pacientes_test + shap_inmune + simulador_inmune
#
# Explicación de cada modelo, igual que en el dashboard. En los dos,
# base + suma de aportes = edad predicha:
#   - Inmunológico: valores SHAP estimados con la librería shap (KernelExplainer),
#     con la mediana del train como referencia. Ver shap_kernel().
#   - Epigenético: aporte exacto = coeficiente x valor estandarizado (es el
#     valor SHAP del modelo lineal).
# ==============================================================================

import io
import os
import threading
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

XAI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS = os.path.join(XAI, "modelos")
MAX_PACIENTES = 20          # filas por archivo
N_APORTES_EPI = 15          # sondas que se muestran por paciente (el resto va sumado)
MAX_FALTANTES_INMUNE = 0.5  # sobre esta fracción de biomarcadores vacíos no se predice
AVISO_IMPUTADAS_EPI = 0.10  # sobre esta fracción de sondas rellenadas se avisa
ORIGEN = "Tu archivo"       # lo que se muestra como estudio / cohorte


class ErrorEntrada(ValueError):
    """Problema con el archivo del usuario (el mensaje se muestra en la página)."""


# ------------------------------------------------------------------------------
# Lectura del CSV
# ------------------------------------------------------------------------------
ALIAS = {"gender": "gender", "sexo": "gender", "sex": "gender",
         "age": "age", "edad": "age",
         "id": "id", "paciente": "id", "subject_accession": "id", "id_paciente": "id"}
HOMBRE = {"male", "m", "hombre", "h", "masculino"}
MUJER = {"female", "f", "mujer", "femenino"}


def leer_csv(texto: str) -> pd.DataFrame:
    """Lee el CSV (separado por coma o por punto y coma) y normaliza id, edad y sexo."""
    texto = texto.lstrip("﻿")
    if not texto.strip():
        raise ErrorEntrada("El archivo está vacío.")
    primera = texto.split("\n", 1)[0]
    punto_coma = primera.count(";") > primera.count(",")
    try:
        df = pd.read_csv(io.StringIO(texto), sep=";" if punto_coma else ",",
                         decimal="," if punto_coma else ".")
    except Exception as e:
        raise ErrorEntrada(f"No se pudo leer el archivo como CSV ({e}).")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")
    if len(df) == 0:
        raise ErrorEntrada("El archivo no tiene filas con datos.")
    if len(df) > MAX_PACIENTES:
        raise ErrorEntrada(f"El archivo tiene {len(df)} filas; el máximo es {MAX_PACIENTES} pacientes por archivo.")

    minus = {c.lower(): c for c in df.columns}
    renombrar = {}
    for alias, destino in ALIAS.items():
        if alias in minus and destino not in df.columns and destino not in renombrar.values():
            renombrar[minus[alias]] = destino
    df = df.rename(columns=renombrar).reset_index(drop=True)

    if "gender" not in df.columns:
        raise ErrorEntrada('Falta la columna "gender" con el sexo de cada paciente (Male / Female).')
    sexo = df["gender"].astype(str).str.strip().str.lower()
    raros = sorted(set(sexo[~sexo.isin(HOMBRE | MUJER)]))
    if raros:
        raise ErrorEntrada(f'La columna "gender" tiene valores que no se reconocen: {", ".join(raros[:5])}. '
                           "Usa Male / Female (o M / F).")
    df["_hombre"] = sexo.isin(HOMBRE).astype(float)
    df["_sexo"] = np.where(df["_hombre"] == 1, "Male", "Female")

    if "age" in df.columns:
        df["_edad"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["_edad"] = np.nan
    if "id" in df.columns:
        ids = df["id"].astype(str).str.strip()
        ids = ids.where(ids.ne("") & ids.ne("nan"), [f"Paciente {i + 1}" for i in range(len(df))])
    else:
        ids = pd.Series([f"Paciente {i + 1}" for i in range(len(df))])
    # Prefijo para que nunca choque con un paciente de test
    df["_id"] = [f"{ORIGEN} · {i}" for i in ids]
    if df["_id"].duplicated().any():
        raise ErrorEntrada('Hay pacientes con el mismo "id" en el archivo.')
    return df


def _num(v, dec):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), dec)


def _fila_paciente(df, i, pred, extra):
    edad = df["_edad"].iloc[i]
    return {"id": df["_id"].iloc[i], "edad_real": _num(edad, 2), "edad_predicha": _num(pred, 2),
            "brecha_edad": _num(pred - edad, 2) if not np.isnan(edad) else None,
            "sexo": df["_sexo"].iloc[i], **extra}


# ------------------------------------------------------------------------------
# Modelo epigenético (ElasticNet)
# ------------------------------------------------------------------------------
class Epigenetico:
    def __init__(self):
        p = joblib.load(os.path.join(MODELOS, "epigenetico_elasticnet", "modelo_elasticnet.joblib"))
        self.pipe, self.columnas = p["modelo"], list(p["columnas"])
        self.sondas = self.columnas[:-1]
        self.medianas = p["medianas_imputacion"]
        self.escalador = self.pipe.named_steps["scaler"]
        self.coef = self.pipe.named_steps["model"].coef_
        self.base = float(self.pipe.named_steps["model"].intercept_)
        self.nombres = [c if c != "Sexo_M" else "Sexo (hombre)" for c in self.columnas]

    def predecir(self, texto: str, etapa=lambda nombre, **datos: None) -> dict:
        etapa("Leyendo el archivo")
        df = leer_csv(texto)
        etapa("Calculando la edad")
        presentes = [s for s in self.sondas if s in df.columns]
        if not presentes:
            raise ErrorEntrada("El archivo no tiene ninguna de las 500 sondas CpG del modelo (columnas como cg16867657). "
                               "¿Es un archivo del modelo inmunológico?")
        X = df.reindex(columns=self.sondas).apply(pd.to_numeric, errors="coerce")
        imputadas = X.isna().sum(axis=1).values
        fuera = int(((X < 0) | (X > 1)).sum().sum())
        X = X.fillna(self.medianas)
        X["Sexo_M"] = df["_hombre"].values
        M = X[self.columnas].values.astype(float)

        aportes = self.escalador.transform(M) * self.coef
        pred = self.base + aportes.sum(axis=1)
        assert np.allclose(pred, self.pipe.predict(M), atol=1e-6)

        pacientes, filas, avisos = [], [], []
        for i in range(len(df)):
            pid = df["_id"].iloc[i]
            pacientes.append(_fila_paciente(df, i, pred[i], {"cohorte": ORIGEN, "base": round(self.base, 2),
                                                             "sondas_imputadas": int(imputadas[i])}))
            orden = np.argsort(-np.abs(aportes[i]))
            for j in orden[:N_APORTES_EPI]:
                filas.append({"clave": f"{pid}|{self.nombres[j]}", "id": pid, "variable": self.nombres[j],
                              "tipo": "Sexo" if self.columnas[j] == "Sexo_M" else "Sonda",
                              "beta": None if self.columnas[j] == "Sexo_M" else round(float(M[i, j]), 4),
                              "aporte_anos": round(float(aportes[i, j]), 4)})
            filas.append({"clave": f"{pid}|otras", "id": pid, "variable": f"Otras {len(self.columnas) - N_APORTES_EPI} variables",
                          "tipo": "Resto", "beta": None, "aporte_anos": round(float(aportes[i, orden[N_APORTES_EPI:]].sum()), 4)})
        muchas = [df["_id"].iloc[i].split(" · ", 1)[1] for i in range(len(df)) if imputadas[i] > AVISO_IMPUTADAS_EPI * len(self.sondas)]
        if muchas:
            avisos.append(f"Faltaba más del 10 % de las 500 sondas en {', '.join(muchas[:5])}{' y otros' if len(muchas) > 5 else ''} "
                          f"(hasta {int(imputadas.max())}). Se rellenaron con la mediana del train, así que su edad es menos precisa.")
        elif imputadas.any():
            avisos.append(f"Faltaban algunas sondas (hasta {int(imputadas.max())} de 500); se rellenaron con la mediana del train.")
        if fuera:
            avisos.append(f"Hay {fuera} valores fuera del rango 0–1. El modelo espera valores beta de metilación.")
        return {"modelo": "epigenetico", "pacientes": pacientes, "aportes": filas, "avisos": avisos}


# ------------------------------------------------------------------------------
# Modelo inmunológico (TabPFN v2): se carga en segundo plano porque tarda
# ------------------------------------------------------------------------------
CELULAS = {"B_cells", "CD16_neg_monocytes", "CD16_pos_monocytes", "CD4_T_cells", "CD8_T_cells",
           "Central_Memory_CD4_T_cells", "Central_Memory_CD8_T_cells", "Effector_CD4_T_cells", "Effector_CD8_T_cells",
           "Effector_Memory_CD4_T_cells", "Effector_Memory_CD8_T_cells", "Gamma_Delta_T_cells", "Lymphocytes",
           "Memory_B_cells", "Monocytes", "Naive_B_cells", "Naive_CD4_T_cells", "Naive_CD8_T_cells", "NK_cells",
           "NKT_cells", "Plasmablasts", "T_cells", "Transitional_B_cells", "Tregs"}
ARCHIVO_MEDIANAS_INMUNE = os.path.join(MODELOS, "final_tabpfn", "medianas_train.csv")
# Simulador "qué pasaría si": valores por los que se mueve cada biomarcador (percentiles del train)
ARCHIVO_PERCENTILES_INMUNE = os.path.join(MODELOS, "final_tabpfn", "percentiles_train.csv")
PERCENTILES_SIM = [1, 5, 10, 25, 50, 75, 90, 95, 99]
N_SIM = 6  # biomarcadores que se pueden mover por paciente


N_SHAP = 500  # combinaciones de biomarcadores por paciente subido (los de test se calcularon con más)


def shap_kernel(predecir, M, referencia, nsamples, semilla=0):
    """Valores SHAP de cada fila de M con la librería shap (KernelExplainer).

    La referencia es una sola fila (la mediana del train): el valor SHAP de una variable es cuánto
    mueve la edad predicha tener el valor del paciente en vez del de referencia, promediado sobre
    las combinaciones de las demás variables. Devuelve (base, valores), con
    base + valores[i].sum() = predicción de la fila i.

    shap llama al modelo una vez por paciente y TabPFN tiene un costo fijo por llamada. Por eso se
    hace en dos pasadas con la misma semilla: la primera solo anota qué filas pide shap, el modelo
    las predice todas en una llamada, y la segunda repite el cálculo con esas predicciones.
    """
    import shap  # se importa aquí: solo hace falta para el modelo inmunológico

    M = np.asarray(M, dtype=np.float32)
    referencia = np.asarray(referencia, dtype=np.float32)
    conocidas, pendientes = {}, {}

    def modelo(Z):
        Z = np.ascontiguousarray(Z, dtype=np.float32)
        salida = np.zeros(len(Z))
        for i, fila in enumerate(Z):
            clave = fila.tobytes()
            if clave in conocidas:
                salida[i] = conocidas[clave]
            else:
                pendientes[clave] = fila
        return salida

    for _ in range(3):  # la primera pasada anota; la segunda ya tiene todas las predicciones
        np.random.seed(semilla)
        explicador = shap.KernelExplainer(modelo, referencia[None, :].astype(np.float64))
        valores = np.vstack([explicador.shap_values(fila.astype(np.float64), nsamples=nsamples, l1_reg=False, silent=True)
                             for fila in M])
        if not pendientes:
            return float(np.ravel(explicador.expected_value)[0]), valores
        claves = list(pendientes)
        conocidas.update(zip(claves, predecir(np.vstack([pendientes[c] for c in claves]))))
        pendientes.clear()
    raise RuntimeError("El cálculo de SHAP no se pudo repetir con las mismas combinaciones.")


class Inmunologico:
    def __init__(self):
        self.listo = threading.Event()
        self.error = None
        self._candado = threading.Lock()  # TabPFN: una predicción a la vez
        threading.Thread(target=self._cargar, daemon=True).start()

    def _cargar(self):
        try:
            p = joblib.load(os.path.join(MODELOS, "final_tabpfn", "modelo_tabpfn.joblib"))
            self.modelo, self.variables = p["modelo"], list(p["variables"])  # 45 biomarcadores + sexo_Male
            train = pd.read_csv(ARCHIVO_MEDIANAS_INMUNE).set_index("variable").loc[self.variables]
            self.medianas = train["mediana"].values.astype(np.float32)  # la referencia de SHAP
            # Media y desviación del train: expresan cada valor en desviaciones estándar
            self.media, self.desv = train["media"].values, train["desviacion"].values
            self.nombres = self.variables[:-1] + ["Sexo"]
            self.tipos = [self.tipo(v) for v in self.nombres]
            self.percentiles = (pd.read_csv(ARCHIVO_PERCENTILES_INMUNE).set_index("variable")
                                if os.path.exists(ARCHIVO_PERCENTILES_INMUNE) else None)
        except Exception as e:  # se informa en cada petición
            self.error = f"No se pudo cargar el modelo inmunológico: {e}"
        finally:
            self.listo.set()

    @staticmethod
    def tipo(v):
        return "Sexo" if v == "Sexo" else ("Célula" if v in CELULAS else "Citoquina")

    def simulacion(self, M, ids, aportes, pred):
        """Simulador "qué pasaría si": para cada paciente, la edad predicha al mover cada uno de sus
        N_SIM biomarcadores con más efecto por los percentiles del train (uno a la vez, el resto fijo).
        Devuelve una fila por punto, con el cambio en años frente a su edad predicha (delta)."""
        if self.percentiles is None:
            return []
        bio = self.variables[:-1]
        tareas, bloques = [], []
        for i in range(len(M)):
            orden = [j for j in np.argsort(-np.abs(aportes[i])) if j < len(bio) and not np.isnan(M[i, j])][:N_SIM]
            for r, j in enumerate(orden):
                rejilla = self.percentiles.loc[bio[j]].values.astype(np.float32)
                rejilla = rejilla[np.abs(rejilla - M[i, j]) > 1e-6]  # el valor del paciente ya tiene delta 0
                Z = np.repeat(M[i:i + 1], len(rejilla), axis=0)
                Z[:, j] = rejilla
                bloques.append(Z)
                tareas.append((i, j, r, rejilla))
        if not bloques:
            return []
        with self._candado:
            out = self.modelo.predict(np.vstack(bloques))
        filas, pos = [], 0
        for i, j, r, rejilla in tareas:
            p = out[pos:pos + len(rejilla)]
            pos += len(rejilla)
            v0 = float(M[i, j])
            puntos = sorted([(float(g), float(pp - pred[i])) for g, pp in zip(rejilla, p)] + [(v0, 0.0)])
            for n, (v, d) in enumerate(puntos):
                filas.append({"clave": f"{ids[i]}|{bio[j]}|{n}", "id": ids[i], "variable": bio[j], "tipo": self.tipo(bio[j]),
                              "orden": r + 1, "valor_paciente": round(v0, 4), "valor": round(v, 4), "delta": round(d, 3)})
        return filas

    def predecir(self, texto: str, espera=180, etapa=lambda nombre, **datos: None) -> dict:
        """etapa(nombre, filas=..., proximas=...) avisa en qué parte del cálculo va y cuántas filas
        predice TabPFN en esa etapa y en las siguientes (para la barra de progreso de la página)."""
        etapa("Leyendo el archivo")
        df = leer_csv(texto)  # los errores del archivo se avisan sin esperar al modelo
        if not self.listo.is_set():
            etapa("Esperando a que cargue el modelo")
        if not self.listo.wait(espera):
            raise RuntimeError("El modelo inmunológico todavía se está cargando. Intenta de nuevo en un momento.")
        if self.error:
            raise RuntimeError(self.error)
        bio = self.variables[:-1]
        presentes = [v for v in bio if v in df.columns]
        if not presentes:
            raise ErrorEntrada("El archivo no tiene ninguno de los 45 biomarcadores del modelo (columnas como Naive_CD8_T_cells o IL6). "
                               "¿Es un archivo del modelo epigenético?")
        X = df.reindex(columns=bio).apply(pd.to_numeric, errors="coerce")
        faltantes = X.isna().sum(axis=1).values
        malos = [df["_id"].iloc[i] for i in range(len(df)) if faltantes[i] > MAX_FALTANTES_INMUNE * len(bio)]
        if malos:
            raise ErrorEntrada(f"A {', '.join(malos[:3])} le faltan más de la mitad de los 45 biomarcadores; "
                               "así la edad no sería confiable.")
        M = np.column_stack([X.values, df["_hombre"].values]).astype(np.float32)
        n, k = M.shape

        # Oclusión: por paciente, la fila original y una copia por variable con esa variable en la
        # mediana del train. Solo se usa para elegir los biomarcadores del simulador.
        bloques = []
        for i in range(n):
            Z = np.repeat(M[i:i + 1], k + 1, axis=0)
            Z[np.arange(1, k + 1), np.arange(k)] = self.medianas
            bloques.append(Z)
        oclusion = np.vstack(bloques)
        etapa("Calculando la edad y la explicación SHAP", filas=len(oclusion) + n * (N_SHAP + 1) + 1,
              proximas={"Calculando el simulador": n * N_SIM * len(PERCENTILES_SIM)})
        salida = {}

        def lote(Z):  # una sola llamada a TabPFN: las filas que pide SHAP y, detrás, las de la oclusión
            out = self.modelo.predict(np.vstack([Z, oclusion]))
            salida["oclusion"] = out[len(Z):].reshape(n, k + 1)
            return out[:len(Z)]

        with self._candado:
            base, shap = shap_kernel(lote, M, self.medianas, N_SHAP)
        pred = salida["oclusion"][:, 0]
        aportes = pred[:, None] - salida["oclusion"][:, 1:]  # + = su valor real lo hace parecer mayor
        shap[:, 0] += pred - (base + shap.sum(axis=1))       # diferencia de redondeo (milésimas de año)

        z = (M - self.media) / self.desv
        pacientes, filas, filas_oclusion, avisos = [], [], [], []
        for i in range(n):
            pid = df["_id"].iloc[i]
            pacientes.append(_fila_paciente(df, i, pred[i], {"estudio": ORIGEN, "base": round(base, 2),
                                                             "biomarcadores_faltantes": int(faltantes[i])}))
            for j, (v, tipo) in enumerate(zip(self.nombres, self.tipos)):
                vacio = bool(np.isnan(M[i, j]))
                filas.append({"clave": f"{pid}|{v}", "id": pid, "variable": v, "tipo": tipo,
                              "aporte_anos": round(float(shap[i, j]), 3),
                              "valor": None if vacio else round(float(M[i, j]), 4),
                              "z": None if vacio or tipo == "Sexo" else round(float(z[i, j]), 3)})
                filas_oclusion.append({"clave": f"{pid}|{v}", "id": pid, "variable": v, "tipo": tipo,
                                       "aporte_anos": round(float(aportes[i, j]), 3)})
        m = int((faltantes > 0).sum())
        if m:
            quien = "a tu paciente" if n == 1 else (f"a {m} de tus {n} pacientes" if m < n else f"a tus {m} pacientes")
            avisos.append(f"Le{'s' if m > 1 else ''} faltaban biomarcadores {quien} (hasta {int(faltantes.max())} de 45). "
                          "El modelo los trata como vacíos, igual que en el entrenamiento.")
        # Segunda llamada al modelo: las curvas del simulador "qué pasaría si"
        etapa("Calculando el simulador")
        simulacion = self.simulacion(M, list(df["_id"]), aportes, pred)
        return {"modelo": "inmunologico", "pacientes": pacientes, "aportes": filas, "oclusion": filas_oclusion,
                "simulacion": simulacion, "avisos": avisos}
