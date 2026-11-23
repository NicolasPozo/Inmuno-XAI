# Experimento — Fusión Hannum (GSE40279) + GSE87571

Experimento que justificó incorporar GSE87571 al pipeline definitivo. Se conserva
como registro de la decisión (figuras 15 y 16). Solo lee el proyecto principal: los
datos de GSE87571 están en `Datasets/GSE87571/` y su conversión en `src/gse87571.py`.

## Dataset añadido

**GSE87571** (Johansson et al.): 732 muestras de sangre completa, Illumina 450K
(GPL13534, el mismo array que Hannum), personas sanas de 14 a 94 años, con edad y sexo.

## Estructura

```
fusion_GSE87571/
├── comun.py                               Carga Hannum solo + GSE87571, ranking sobre varias fuentes y k=130
├── mixup.py                               Aumentación Mixup (ya no está en el pipeline; solo para reproducir 03)
├── 02_experimento_fusion.py               Escenarios con modelos de referencia (rápido, ~3 min)
├── 03_entrenamiento_fusion_completo.py    Pipeline completo fusionado: Mixup + tuning (~30 min)
└── resultados/                            CSV y figuras 15 y 16
```

(El antiguo `01_convertir_GSE87571.py` pasó a `src/gse87571.py`.)

## Cómo ejecutar

```bash
python experimentos/fusion_GSE87571/02_experimento_fusion.py
python experimentos/fusion_GSE87571/03_entrenamiento_fusion_completo.py
```

`03 ... --solo-grafico` regenera la figura 16 desde el CSV sin reentrenar.

## Paso 02: escenarios con modelos de referencia

| Escenario | Entrena con |
|---|---|
| Solo Hannum | train de Hannum (modelo previo a la fusión) |
| Solo GSE87571 | train de GSE87571 |
| Fusión | ambos trains |
| Fusión + corrección lote | ambos, restando a GSE87571 el desplazamiento por sonda (modelo `beta ~ edad + sexo + lote`) |

Todos se evalúan en el **test de Hannum** (los mismos 164 pacientes) y en un **test de
GSE87571** (25%, validación externa). Modelos con hiperparámetros fijos, sin Mixup.

## Paso 03: pipeline completo fusionado

Partición train/test dentro de cada dataset → fusión de los trains → selección de
sondas sobre el train fusionado → Mixup **dentro de cada dataset** → tuning de los 5
modelos con CV sin fuga y parada temprana. Compara "Definitivo (solo Hannum)",
"Fusión sin Mixup" y "Fusión con Mixup".

Conclusión: la fusión mejora a todos los modelos; el Mixup no aporta (±0,1 años), por
lo que el pipeline definitivo (`src/05_entrenamiento.py`) quedó como **fusión sin
Mixup** (solo pacientes reales). Ambos scripts del experimento usan k=130 sondas
(`N_SONDAS_EXPERIMENTO` en `comun.py`), el valor con que se generaron las figuras 15 y 16.
