# ==============================================================================
# Space de Hugging Face (Docker) para la página web de Immuno-XAI
#
# El Space no guarda el código: lo descarga del repositorio de GitHub al
# construirse. Para publicar cambios basta con subirlos a GitHub y reiniciar el
# Space (Settings → Factory rebuild).
#
# Igual que en GitHub, no incluye el modelo TabPFN ni datos por paciente de
# ImmPort: el reloj epigenético funciona entero y el inmunológico muestra sus
# métricas y explicaciones globales.
# ==============================================================================
FROM python:3.13-slim

# Solo lo que usa el reloj epigenético y el servidor (sin torch ni TabPFN)
RUN pip install --no-cache-dir scikit-learn==1.9.0 pandas==2.3.3 numpy==2.4.1 joblib==1.5.3

# Hugging Face corre el contenedor con el usuario 1000
RUN useradd -m -u 1000 usuario
WORKDIR /home/usuario

# Código desde GitHub. ADD vuelve a descargar el archivo si cambió la rama main.
ADD --chown=usuario https://github.com/NicolasPozo/Inmuno-XAI/archive/refs/heads/main.tar.gz /tmp/codigo.tar.gz
RUN tar -xzf /tmp/codigo.tar.gz --strip-components=1 "Inmuno-XAI-main/XAI" \
    && rm /tmp/codigo.tar.gz && chown -R usuario /home/usuario/XAI

USER usuario
ENV HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1 PYTHONUTF8=1
EXPOSE 7860
CMD ["python", "XAI/web/servidor.py"]
