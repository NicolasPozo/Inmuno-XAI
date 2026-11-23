---
title: Immuno-XAI
emoji: ⏱️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Dos relojes de edad biológica explicables con SHAP
---

# Immuno-XAI

Página web del proyecto de título **Immuno-XAI**: dos relojes de edad biológica, uno inmunológico y otro epigenético, que explican cada predicción con SHAP.

- **Autores:** Constanza Olivos Fernández y Nicolás Pozo Villagrán (UTEM).
- **Código y documentación:** https://github.com/NicolasPozo/Inmuno-XAI

## Qué funciona en esta página

- **Reloj epigenético:** completo. Se pueden ver los pacientes de test y subir pacientes nuevos en CSV.
- **Reloj inmunológico:** métricas, comparación de modelos y explicaciones globales. Las tarjetas por paciente y la carga de pacientes nuevos no están disponibles aquí, porque el modelo TabPFN guarda dentro a los pacientes de entrenamiento del Stanford 10K (ImmPort), cuyos datos no se pueden redistribuir.

Es un prototipo de investigación, no una herramienta clínica.
