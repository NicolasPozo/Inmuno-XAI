"""
Aumentación Mixup estratificada por sexo (compartida por los pasos 03 y 05).

Cada muestra sintética es la interpolación lineal de dos pacientes reales del
mismo sexo:  x_nuevo = lam * x_i + (1 - lam) * x_j,  con lam ~ Beta(alpha, alpha).

Usa el generador global de NumPy, por lo que el resultado depende de la semilla
fijada con np.random.seed() antes de llamar a estas funciones.
"""
import numpy as np

ALPHA_BETA = 0.3             # Parámetro de la distribución Beta para lambda
PROPORCION_SINTETICOS = 0.5  # 50% extra de muestras por grupo


def planificar_mixup(edades, indices, n_sinteticos, sexo, alpha=ALPHA_BETA):
    """Elige pares de pacientes (de 'indices') y su peso de interpolación."""
    plan = []
    for idx in range(n_sinteticos):
        i, j = np.random.choice(indices, size=2, replace=False)
        lam = float(np.random.beta(alpha, alpha))
        edad_interp = lam * edades[i] + (1 - lam) * edades[j]
        plan.append({
            'id_nuevo': f"SYNTH_{sexo}_{idx+1}",
            'sexo': sexo,
            'edad': round(edad_interp, 2),
            'idx_1': i,
            'idx_2': j,
            'lam': lam
        })
    return plan


def planificar_mixup_estratificado(edades, sexos, indices_disponibles=None,
                                   proporcion=PROPORCION_SINTETICOS, alpha=ALPHA_BETA):
    """
    Plan Mixup separado por sexo (primero M, luego F).

    indices_disponibles: posiciones de los pacientes que pueden usarse como padres
    (por defecto todos). Permite generar sintéticos solo con pacientes de train.
    """
    sexos = np.asarray(sexos)
    if indices_disponibles is None:
        indices_disponibles = np.arange(len(sexos))
    indices_disponibles = np.asarray(indices_disponibles)

    indices_M = indices_disponibles[sexos[indices_disponibles] == 'M'].tolist()
    indices_F = indices_disponibles[sexos[indices_disponibles] == 'F'].tolist()

    n_sinteticos_M = int(len(indices_M) * proporcion)
    n_sinteticos_F = int(len(indices_F) * proporcion)

    return planificar_mixup(edades, indices_M, n_sinteticos_M, 'M', alpha) + \
           planificar_mixup(edades, indices_F, n_sinteticos_F, 'F', alpha)


def interpolar(v1, v2, lam):
    """Interpolación lineal de metilación (restringida a [0.0, 1.0])."""
    return np.clip(lam * v1 + (1.0 - lam) * v2, 0.0, 1.0)
