# ==============================================================================
# SERVIDOR DE LA PÁGINA IMMUNO-XAI (fuera de Claude)
#
#   python -X utf8 web/servidor.py          -> http://127.0.0.1:8000
#
# Entrega las páginas (/ portada, /inmunologico y /epigenetico), sus datos y el
# video, y calcula la edad de los pacientes que se suben:
#   POST /api/inmunologico   cuerpo: el CSV   -> pacientes + aportes (TabPFN)
#   POST /api/epigenetico    cuerpo: el CSV   -> pacientes + aportes (ElasticNet)
#   (con ?progreso=1 responde línea por línea: el avance del cálculo y al final el resultado)
#   GET  /api/estado                          -> si los modelos están listos
#
# Solo usa la librería estándar de Python (más los paquetes de los modelos).
# Variables de entorno para el deploy: HOST (por defecto 127.0.0.1; 0.0.0.0 en
# un servidor) y PORT (por defecto 8000).
# ==============================================================================

import json
import mimetypes
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import modelos_api as api  # noqa: E402

XAI = api.XAI
MAX_CUERPO = 5 * 1024 * 1024  # 5 MB por archivo

# Tres páginas: la portada y una por reloj (las dos usan modelo.html, que mira la dirección).
# La página del dashboard es la misma que se publica en Claude.
MODELO = os.path.join(AQUI, "publico", "modelo.html")
RUTAS = {
    "/": os.path.join(AQUI, "publico", "portada.html"),
    "/inmunologico": MODELO,
    "/epigenetico": MODELO,
    "/dashboard.html": os.path.join(XAI, "dashboard", "index.html"),
}
CARPETAS = {
    "/datos/": os.path.join(XAI, "dashboard", "datos"),
    "/media/": os.path.join(XAI, "dashboard", "Media"),
    "/ejemplos/": os.path.join(AQUI, "publico", "ejemplos"),
    "/estaticos/": os.path.join(AQUI, "publico", "estaticos"),
}
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("video/mp4", ".mp4")

print("Cargando el modelo epigenético...", flush=True)
EPI = api.Epigenetico()
print("Cargando el modelo inmunológico en segundo plano (tarda unos segundos)...", flush=True)
INM = api.Inmunologico()


# ------------------------------------------------------------------------------
# Progreso del cálculo (para la barra de la página)
# El cálculo avisa en qué etapa va. TabPFN no informa su avance mientras predice,
# así que dentro de cada etapa se estima:
#   - En las etapas de TabPFN, con las filas que predice: un costo fijo por
#     llamada más un tiempo por fila, que se ajusta con cada cálculo. Así la
#     estimación sirve igual para 1 paciente que para 20.
#   - En las demás, con lo que tardó esa etapa la vez anterior.
# La barra nunca llega al 100 % hasta que el cálculo termina.
# ------------------------------------------------------------------------------
DURACIONES = {  # segundos esperados por etapa; se actualizan con cada cálculo
    "Leyendo el archivo": 0.2,
    "Esperando a que cargue el modelo": 15.0,
    "Calculando la edad y la explicación SHAP": 30.0,
    "Calculando el simulador": 14.0,
    "Calculando la edad": 0.3,
}
TABPFN = {"fijo": 10.0, "por_fila": 0.035}  # segundos por llamada y por fila (por_fila se ajusta)
PLANES = {
    "/api/inmunologico": ["Leyendo el archivo", "Calculando la edad y la explicación SHAP", "Calculando el simulador"],
    "/api/epigenetico": ["Leyendo el archivo", "Calculando la edad"],
}


class Progreso:
    def __init__(self, plan):
        self.plan, self.actual, self.inicio = list(plan), plan[0], time.time()
        self.filas = {}  # etapa -> filas que predice TabPFN en ella, cuando se sabe
        self._candado = threading.Lock()

    def _esperado(self, etapa):
        if etapa in self.filas:
            return TABPFN["fijo"] + TABPFN["por_fila"] * self.filas[etapa]
        return max(DURACIONES.get(etapa, 1.0), 0.05)

    def _cerrar_etapa(self, ahora):
        real = ahora - self.inicio
        filas = self.filas.get(self.actual, 0)
        if filas >= 200:  # con pocas filas manda el costo fijo y no se aprende nada del tiempo por fila
            TABPFN["por_fila"] = 0.5 * TABPFN["por_fila"] + 0.5 * max(real - TABPFN["fijo"], 0.3 * real) / filas
        elif not filas:
            DURACIONES[self.actual] = 0.5 * DURACIONES.get(self.actual, real) + 0.5 * real

    def etapa(self, nombre, filas=None, proximas=None):
        """filas: las que predice TabPFN en esta etapa; proximas: {etapa: filas} de las que vienen."""
        with self._candado:
            ahora = time.time()
            if nombre != self.actual:
                self._cerrar_etapa(ahora)
                if nombre not in self.plan:
                    self.plan.insert(self.plan.index(self.actual) + 1, nombre)
                self.actual, self.inicio = nombre, ahora
            if filas:
                self.filas[nombre] = filas
            self.filas.update(proximas or {})

    def fin(self):
        with self._candado:
            self._cerrar_etapa(time.time())

    def fraccion(self):
        with self._candado:
            pesos = [self._esperado(e) for e in self.plan]
            i = self.plan.index(self.actual)
            dentro = min((time.time() - self.inicio) / pesos[i], 0.95) * pesos[i]
            return min((sum(pesos[:i]) + dentro) / sum(pesos), 0.97)


def archivo_de(ruta: str):
    """Ruta pedida -> archivo en disco, sin dejar salir de las carpetas permitidas."""
    if ruta in RUTAS:
        return RUTAS[ruta]
    for prefijo, carpeta in CARPETAS.items():
        if ruta.startswith(prefijo):
            destino = os.path.realpath(os.path.join(carpeta, unquote(ruta[len(prefijo):])))
            if destino.startswith(os.path.realpath(carpeta) + os.sep) and os.path.isfile(destino):
                return destino
    return None


class Manejador(BaseHTTPRequestHandler):
    server_version = "ImmunoXAI/1.0"

    def log_message(self, fmt, *args):
        sys.stdout.write(f"{self.log_date_time_string()}  {fmt % args}\n")
        sys.stdout.flush()

    def _json(self, codigo, datos):
        cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        ruta = urlparse(self.path).path
        if ruta == "/api/estado":
            return self._json(200, {"epigenetico": "listo",
                                    "inmunologico": "error" if INM.error else ("listo" if INM.listo.is_set() else "cargando")})
        sin_barra = ruta.rstrip("/")
        if ruta != "/" and sin_barra in RUTAS and sin_barra != ruta:
            # /inmunologico/ -> /inmunologico (con la barra, las rutas relativas de la página fallarían)
            self.send_response(301)
            self.send_header("Location", sin_barra)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        archivo = archivo_de(ruta)
        if not archivo:
            return self._json(404, {"error": "No encontrado"})
        tamano = os.path.getsize(archivo)
        tipo = mimetypes.guess_type(archivo)[0] or "application/octet-stream"
        if tipo.startswith("text/") or tipo in ("application/json", "application/javascript"):
            tipo += "; charset=utf-8"
        # Rangos de bytes: el navegador los usa para reproducir y rebobinar el video
        inicio, fin = 0, tamano - 1
        rango = self.headers.get("Range", "")
        if rango.startswith("bytes="):
            a, _, b = rango[6:].split(",")[0].partition("-")
            try:
                if a:
                    inicio, fin = int(a), (int(b) if b else tamano - 1)
                else:
                    inicio = max(0, tamano - int(b))
                fin = min(fin, tamano - 1)
            except ValueError:
                inicio, fin = 0, tamano - 1
            if inicio > fin:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{tamano}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {inicio}-{fin}/{tamano}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(fin - inicio + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(archivo, "rb") as f:
            f.seek(inicio)
            restante = fin - inicio + 1
            while restante > 0:
                trozo = f.read(min(1 << 16, restante))
                if not trozo:
                    break
                try:
                    self.wfile.write(trozo)
                except ConnectionError:
                    return  # el navegador canceló (pasa al rebobinar el video)
                restante -= len(trozo)

    def _calcular_con_progreso(self, ruta, modelo, texto):
        """Responde una línea JSON por evento: {"progreso", "etapa"} cada 0,3 s y al final {"resultado"} o {"error"}."""
        plan = list(PLANES[ruta])
        if modelo is INM and not INM.listo.is_set():
            plan.insert(1, "Esperando a que cargue el modelo")
        prog, salida, t0 = Progreso(plan), {}, time.time()

        def trabajar():
            try:
                res = modelo.predecir(texto, etapa=prog.etapa)
                prog.fin()
                res["segundos"] = round(time.time() - t0, 2)
                salida["resultado"] = res
            except api.ErrorEntrada as e:
                salida["error"] = (400, str(e))
            except RuntimeError as e:
                salida["error"] = (503, str(e))
            except Exception:
                traceback.print_exc()
                salida["error"] = (500, "Error inesperado al calcular la edad. Revisa la consola del servidor.")

        hilo = threading.Thread(target=trabajar, daemon=True)
        hilo.start()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")  # que un proxy (nginx) no junte las líneas
        self.end_headers()
        enviar = lambda obj: self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        try:
            while hilo.is_alive():
                enviar({"progreso": round(prog.fraccion(), 3), "etapa": prog.actual})
                hilo.join(0.3)
            if "resultado" in salida:
                enviar({"progreso": 1, "etapa": "Listo"})
                enviar({"resultado": salida["resultado"]})
                print(f"   {ruta}: {len(salida['resultado']['pacientes'])} paciente(s) en {salida['resultado']['segundos']} s", flush=True)
            else:
                codigo, mensaje = salida["error"]
                enviar({"error": mensaje, "codigo": codigo})
        except ConnectionError:
            return  # la página se cerró o canceló

    def do_POST(self):
        partes = urlparse(self.path)
        ruta = partes.path
        modelos = {"/api/inmunologico": INM, "/api/epigenetico": EPI}
        if ruta not in modelos:
            return self._json(404, {"error": "No encontrado"})
        largo = int(self.headers.get("Content-Length") or 0)
        if largo <= 0:
            return self._json(400, {"error": "No llegó ningún archivo."})
        if largo > MAX_CUERPO:
            return self._json(413, {"error": "El archivo es demasiado grande (máximo 5 MB)."})
        try:
            texto = self.rfile.read(largo).decode("utf-8-sig")
        except UnicodeDecodeError:
            return self._json(400, {"error": "El archivo no está en UTF-8. Guárdalo como «CSV UTF-8» e inténtalo de nuevo."})
        if "progreso=1" in partes.query:
            return self._calcular_con_progreso(ruta, modelos[ruta], texto)
        t0 = time.time()
        try:
            res = modelos[ruta].predecir(texto)
        except api.ErrorEntrada as e:
            return self._json(400, {"error": str(e)})
        except RuntimeError as e:
            return self._json(503, {"error": str(e)})
        except Exception:
            traceback.print_exc()
            return self._json(500, {"error": "Error inesperado al calcular la edad. Revisa la consola del servidor."})
        res["segundos"] = round(time.time() - t0, 2)
        print(f"   {ruta}: {len(res['pacientes'])} paciente(s) en {res['segundos']} s", flush=True)
        return self._json(200, res)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    puerto = int(os.environ.get("PORT", "8000"))
    servidor = ThreadingHTTPServer((host, puerto), Manejador)
    print(f"Listo: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{puerto}", flush=True)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
