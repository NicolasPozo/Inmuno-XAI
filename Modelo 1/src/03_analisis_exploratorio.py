"""
PASO 3 - Análisis exploratorio (EDA) de cada dataset y comparación entre ellos.

Análisis descriptivo con TODOS los pacientes válidos (train + test) y las 470.043 sondas
CpG; no participa en el entrenamiento, así que no genera fuga.

  A. Por dataset (Hannum y GSE87571), el mismo análisis para cada uno:
     - Spearman de cada sonda con la edad (p-valor y corrección de Bonferroni).
     - Top 30 biomarcadores (15 que suben y 15 que bajan con la edad) + Mann-Whitney por sexo.
     - Figura 1: clustermap jerárquico | Figura 2: co-metilación | Figura 3: PCA.
  B. Comparación entre datasets:
     - Figura 4: distribución de edad y sexo.
     - Figura 5: PCA conjunto (efecto de laboratorio vs señal de edad).
     - Figura 6: señal de edad compartida (ρ en Hannum vs ρ en GSE87571).
     - Figura 7: edad por sexo con prueba de Mann-Whitney U (cada dataset y fusionado).
     - Figura 8: cinética de las 4 sondas cardinales (2 que suben y 2 que bajan con la edad).
     - comparacion_datasets.csv: resumen numérico.

Entrada : cachés de ambos datasets (Datasets/, se crean solas)
Salida  : resultados/eda/Hannum/, resultados/eda/GSE87571/, resultados/eda/comparacion/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import CARPETA_RESULTADOS_EDA, configurar_estilo_graficos, crear_carpeta
from datos import correlaciones_spearman, preparar_particion

N_TOP = 15                  # Sondas que suben y que bajan con la edad en el top de biomarcadores
N_SONDAS_PCA_LOTE = 20000   # Sondas al azar para medir el efecto de laboratorio en la figura 5
COLORES_SEXO = {'M': '#3498db', 'F': '#e74c3c'}
COLORES_DATASET = {'Hannum': '#1f77b4', 'GSE87571': '#ff7f0e'}

configurar_estilo_graficos()

# ==============================================================================
# 1. CARGA
# ==============================================================================
print("1. Cargando ambos datasets (todos los pacientes válidos)...")
p = preparar_particion()
pacientes = p.por_dataset(np.sort(np.concatenate([p.idx_train, p.idx_test])))
for nombre, idx in pacientes.items():
    print(f"   -> {nombre:<9} {len(idx)} pacientes")
print(f"   -> Sondas CpG: {len(p.sondas)}")


# ==============================================================================
# A. EDA POR DATASET
# ==============================================================================
def eda_dataset(nombre, idx):
    carpeta = crear_carpeta(CARPETA_RESULTADOS_EDA / nombre)
    edades, sexos, n = p.edades[idx], p.sexos[idx], len(idx)
    print(f"\n=== {nombre} ({n} pacientes) ===")

    # --- Bioestadística sobre todas las sondas ---
    rho = correlaciones_spearman(p, idx, absoluto=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-12, None))
    p_valor = 2 * stats.t.sf(np.abs(t), n - 2)
    umbral_bonf = 0.05 / len(rho)
    print(f"   Sondas asociadas a la edad: p<0.05 -> {np.sum(p_valor < 0.05):,} | "
          f"Bonferroni (p<{umbral_bonf:.1e}) -> {np.sum(p_valor < umbral_bonf):,} de {len(rho):,}")

    orden = np.argsort(rho)
    top = np.concatenate([orden[:N_TOP], orden[-N_TOP:]])  # bajan / suben con la edad
    X_top = p.matriz[np.ix_(idx, top)]
    es_m = sexos == 'M'
    p_mann = stats.mannwhitneyu(X_top[es_m], X_top[~es_m], axis=0, alternative='two-sided').pvalue
    tabla = pd.DataFrame({
        'Sonda_CpG': p.sondas[top], 'Spearman_rho': rho[top], 'p_spearman': p_valor[top],
        'Bonferroni_significativa': p_valor[top] < umbral_bonf, 'p_mann_whitney_sexo': p_mann,
    }).sort_values('Spearman_rho')
    tabla.to_csv(os.path.join(carpeta, "top30_biomarcadores.csv"), index=False)
    print(f"   Top 30: ρ entre {tabla['Spearman_rho'].min():+.2f} y {tabla['Spearman_rho'].max():+.2f} | "
          f"más fuerte: {tabla.iloc[tabla['Spearman_rho'].abs().argmax()]['Sonda_CpG']}")

    nombres_top = tabla['Sonda_CpG'].tolist()
    df_top = pd.DataFrame(p.matriz[np.ix_(idx, top)], columns=p.sondas[top])[nombres_top]

    # --- Figura 1: clustermap ---
    cmap_edad = sns.color_palette("viridis", as_cmap=True)
    norm_edad = plt.Normalize(edades.min(), edades.max())
    colores_fila = pd.DataFrame({
        'Sexo': pd.Series(sexos).map(COLORES_SEXO).fillna('#95a5a6'),
        'Edad': pd.Series(edades).map(lambda x: cmap_edad(norm_edad(x))),
    })
    g = sns.clustermap(df_top, row_colors=colores_fila, cmap="RdYlBu_r", center=0.5, figsize=(12, 10),
                       dendrogram_ratio=(0.15, 0.15), cbar_kws={'label': 'Nivel de Metilación (Valor Beta)'},
                       linewidths=0.0, yticklabels=False, xticklabels=True)
    g.ax_heatmap.set_xlabel("Top 30 Sondas CpG Asociadas a la Edad", fontsize=11, fontweight='bold')
    g.ax_heatmap.set_ylabel("Pacientes (Clustering Jerárquico)", fontsize=11, fontweight='bold')
    g.figure.suptitle(f"Clustermap de Metilación ({nombre}): Co-regulación y Fenotipo Clínico",
                      fontsize=13, fontweight='bold', y=1.02)
    g.savefig(os.path.join(carpeta, "1_clustermap_metilacion.png"), bbox_inches='tight', dpi=300)
    plt.close(g.figure)

    # --- Figura 2: co-metilación ---
    corr = df_top.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), cmap="vlag", center=0, square=True,
                linewidths=0.3, cbar_kws={"shrink": 0.8, "label": "Correlación de Spearman (ρ)"}, ax=ax)
    ax.set_title(f"Matriz de Co-metilación entre las Top 30 Sondas CpG ({nombre})", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "2_matriz_co_metilacion.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)

    # --- Figura 3: PCA ---
    pca = PCA(n_components=2)
    coords = pca.fit_transform(StandardScaler().fit_transform(df_top))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=edades, cmap='viridis', s=50, edgecolor='k', alpha=0.85)
    fig.colorbar(sc, ax=axes[0]).set_label('Edad (años)')
    var = pca.explained_variance_ratio_ * 100
    axes[0].set_title(f"PCA: Trayectoria Epigenética de la Edad ({nombre})\nPC1 ({var[0]:.1f}%) | PC2 ({var[1]:.1f}%)",
                      fontweight='bold')
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=sexos, palette=COLORES_SEXO, alpha=0.8, s=50, ax=axes[1])
    axes[1].set_title(f"PCA: Agrupamiento por Sexo ({nombre})", fontweight='bold')
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "3_pca_trayectoria.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)

    return rho, p_valor


resultados = {nombre: eda_dataset(nombre, idx) for nombre, idx in pacientes.items()}

# ==============================================================================
# B. COMPARACIÓN ENTRE DATASETS
# ==============================================================================
carpeta_cmp = crear_carpeta(CARPETA_RESULTADOS_EDA / "comparacion")
print("\n=== Comparación entre datasets ===")
nombres = list(pacientes)
todos = np.concatenate(list(pacientes.values()))
lote_todos = p.lote[todos]

# --- Figura 4: distribución de edad y sexo ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for nombre, idx in pacientes.items():
    sns.kdeplot(p.edades[idx], ax=axes[0], fill=True, alpha=0.3, lw=2, color=COLORES_DATASET[nombre], cut=0,
                label=f"{nombre} (n={len(idx)}, media {p.edades[idx].mean():.1f} años)")
axes[0].set_title("Distribución de Edad por Dataset", fontweight='bold')
axes[0].set_xlabel("Edad Cronológica (años)")
axes[0].set_ylabel("Densidad")
axes[0].legend()
df_edad = pd.DataFrame({'Edad': p.edades[todos], 'Sexo': p.sexos[todos],
                        'Dataset': [p.nombres_lote[l] for l in lote_todos]})
sns.violinplot(data=df_edad, x='Dataset', y='Edad', hue='Sexo', split=True, palette=COLORES_SEXO,
               inner='quartile', cut=0, ax=axes[1])
axes[1].set_title("Edad por Sexo en cada Dataset", fontweight='bold')
axes[1].set_ylabel("Edad Cronológica (años)")
fig.tight_layout()
fig.savefig(os.path.join(carpeta_cmp, "4_distribucion_edad_por_dataset.png"), bbox_inches='tight', dpi=300)
plt.close(fig)

# --- Figura 5: PCA conjunto: sondas al azar (lote) vs top sondas de edad (biología) ---
rho_conjunto = correlaciones_spearman(p, todos)
top_conjunto = np.argsort(-rho_conjunto)[:2 * N_TOP]
azar = np.sort(np.random.RandomState(0).choice(len(p.sondas), N_SONDAS_PCA_LOTE, replace=False))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, cols, titulo, color_por_edad in [
        (axes[0], azar, f"{N_SONDAS_PCA_LOTE:,} sondas al azar: efecto de laboratorio", False),
        (axes[1], top_conjunto, "Top 30 sondas de edad: señal biológica", True)]:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(StandardScaler().fit_transform(p.matriz[np.ix_(todos, cols)]))
    var = pca.explained_variance_ratio_ * 100
    for i, nombre in enumerate(nombres):
        m = lote_todos == i
        if color_por_edad:
            sc = ax.scatter(coords[m, 0], coords[m, 1], c=p.edades[todos][m], cmap='viridis', vmin=p.edades[todos].min(),
                            vmax=p.edades[todos].max(), s=35, alpha=0.8, edgecolor='k', linewidth=0.3,
                            marker=['o', '^'][i], label=nombre)
        else:
            ax.scatter(coords[m, 0], coords[m, 1], color=COLORES_DATASET[nombre], s=30, alpha=0.6, label=nombre)
    if color_por_edad:
        fig.colorbar(sc, ax=ax).set_label('Edad (años)')
    ax.set_title(f"PCA conjunto: {titulo}\nPC1 ({var[0]:.1f}%) | PC2 ({var[1]:.1f}%)", fontweight='bold')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(carpeta_cmp, "5_pca_lote_vs_edad.png"), bbox_inches='tight', dpi=300)
plt.close(fig)

# --- Figura 6: señal de edad compartida ---
rh, rg = resultados['Hannum'][0], resultados['GSE87571'][0]
corr_rho = np.corrcoef(rh, rg)[0, 1]
top_h, top_g = np.argsort(-np.abs(rh))[:2 * N_TOP], np.argsort(-np.abs(rg))[:2 * N_TOP]
compartidas_top10 = len(set(np.argsort(-np.abs(rh))[:10]) & set(np.argsort(-np.abs(rg))[:10]))
fig, ax = plt.subplots(figsize=(8, 7.5))
hb = ax.hexbin(rh, rg, gridsize=120, bins='log', cmap='Greys', mincnt=1)
fig.colorbar(hb, ax=ax).set_label('Número de sondas (escala log)')
ax.scatter(rh[top_h], rg[top_h], s=28, color=COLORES_DATASET['Hannum'], edgecolor='k', label='Top 30 Hannum')
ax.scatter(rh[top_g], rg[top_g], s=28, color=COLORES_DATASET['GSE87571'], edgecolor='k', marker='^', label='Top 30 GSE87571')
ax.plot([-1, 1], [-1, 1], 'r--', lw=1.2, label='Igualdad')
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(0, color='gray', lw=0.8)
ax.set_xlabel("Spearman ρ con la edad — Hannum")
ax.set_ylabel("Spearman ρ con la edad — GSE87571")
ax.set_title(f"Señal de Edad Compartida entre Datasets ({len(rh):,} sondas)\n"
             f"Correlación de ρ = {corr_rho:.2f} | {compartidas_top10}/10 sondas top en común", fontweight='bold')
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(carpeta_cmp, "6_senal_edad_compartida.png"), bbox_inches='tight', dpi=300)
plt.close(fig)

# --- Figura 7: edad por sexo con prueba de Mann-Whitney U (¿cohortes balanceadas?) ---
grupos = {**pacientes, 'Fusionado': todos}
p_mw_sexo = {}
fig, axes = plt.subplots(1, len(grupos), figsize=(6.5 * len(grupos), 6), sharey=True)
for ax, (nombre, idx) in zip(axes, grupos.items()):
    df_g = pd.DataFrame({'Edad': p.edades[idx], 'Sexo': pd.Series(p.sexos[idx]).map({'M': 'Hombre', 'F': 'Mujer'})})
    p_mw = stats.mannwhitneyu(df_g.loc[df_g['Sexo'] == 'Hombre', 'Edad'], df_g.loc[df_g['Sexo'] == 'Mujer', 'Edad'],
                              alternative='two-sided').pvalue
    p_mw_sexo[nombre] = p_mw
    orden_sexo = ['Hombre', 'Mujer']
    sns.boxplot(data=df_g, x='Sexo', y='Edad', order=orden_sexo, hue='Sexo', hue_order=orden_sexo,
                palette={'Hombre': '#4daf91', 'Mujer': '#9b97c9'}, width=0.5, showfliers=False, legend=False, ax=ax)
    sns.stripplot(data=df_g, x='Sexo', y='Edad', order=orden_sexo, color='black', alpha=0.35, size=3,
                  jitter=0.25, ax=ax)
    medianas = df_g.groupby('Sexo')['Edad'].median()
    ax.set_title(f"{nombre} (n={len(idx)})\nMann-Whitney U: p = {p_mw:.4f}", fontweight='bold')
    ax.set_xlabel("Sexo Biológico")
    ax.set_ylabel("Edad Cronológica (años)")
    texto = ("Distribución etaria balanceada entre sexos (p > 0,05)" if p_mw > 0.05
             else "Diferencia de edad significativa entre sexos (p ≤ 0,05)")
    ax.text(0.5, 0.02, f"{texto}\nMediana: H {medianas['Hombre']:.0f} | M {medianas['Mujer']:.0f} años",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
axes[0].set_ylim(p.edades[todos].min() - 20, p.edades[todos].max() + 5)  # espacio libre para el recuadro
fig.suptitle("Distribución Demográfica de la Cohorte por Sexo", fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(carpeta_cmp, "7_edad_por_sexo_mann_whitney.png"), bbox_inches='tight', dpi=300)
plt.close(fig)
print("   Edad por sexo (Mann-Whitney U): " + " | ".join(f"{n}: p = {v:.4f}" for n, v in p_mw_sexo.items()))

# --- Figura 8: cinética de las 4 sondas cardinales (2 que suben y 2 que bajan con la edad) ---
rho_signo = correlaciones_spearman(p, todos, absoluto=False)
orden_signo = np.argsort(rho_signo)
cardinales = [(orden_signo[-1], '#2ca02c'), (orden_signo[-2], '#2ca02c'), (orden_signo[0], '#d95f02'), (orden_signo[1], '#d95f02')]
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True)
for ax, (col, color_linea) in zip(axes.ravel(), cardinales):
    beta = p.matriz[todos, col]
    for i, nombre in enumerate(nombres):
        m = lote_todos == i
        ax.scatter(p.edades[todos][m], beta[m], s=9, alpha=0.45, color=COLORES_DATASET[nombre],
                   marker=['o', '^'][i], label=nombre)
    sns.regplot(x=p.edades[todos], y=beta, scatter=False, ci=95, color=color_linea, line_kws={'lw': 2.2}, ax=ax)
    rho_ds = " | ".join(f"{n}: ρ = {resultados[n][0][col]:+.2f}" for n in nombres)
    ax.set_title(f"{p.sondas[col]} (ρ conjunto = {rho_signo[col]:+.3f})\n{rho_ds}", fontweight='bold', fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Nivel de metilación (valor Beta)")
    ax.legend(loc='best', fontsize=8.5)
for ax in axes[1]:
    ax.set_xlabel("Edad Cronológica (años)")
fig.suptitle("Cinética de las Sondas CpG Cardinales del Envejecimiento (Hannum + GSE87571)",
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(carpeta_cmp, "8_cinetica_sondas_cardinales.png"), bbox_inches='tight', dpi=300)
plt.close(fig)
print("   Sondas cardinales: " + ", ".join(f"{p.sondas[c]} ({rho_signo[c]:+.3f})" for c, _ in cardinales))

# --- Resumen numérico ---
filas = []
for nombre, idx in pacientes.items():
    e, s = p.edades[idx], p.sexos[idx]
    rho, pv = resultados[nombre]
    filas.append({
        'Dataset': nombre, 'Pacientes': len(idx), 'Edad_media': round(float(e.mean()), 1),
        'Edad_DE': round(float(e.std()), 1),
        'Edad_min': int(e.min()), 'Edad_max': int(e.max()), 'Pct_hombres': round(100 * np.mean(s == 'M'), 1),
        'Pct_menores_30': round(100 * np.mean(e < 30), 1), 'Pct_80_o_mas': round(100 * np.mean(e >= 80), 1),
        'p_MannWhitney_edad_por_sexo': round(float(p_mw_sexo[nombre]), 4),
        'Beta_media': round(float(p.matriz[np.ix_(idx, azar)].mean()), 3),
        'Sondas_edad_Bonferroni': int(np.sum(pv < 0.05 / len(pv))),
        'Sonda_top': p.sondas[np.argmax(np.abs(rho))], 'Rho_sonda_top': round(float(rho[np.argmax(np.abs(rho))]), 3),
    })
resumen = pd.DataFrame(filas)
resumen.to_csv(os.path.join(carpeta_cmp, "comparacion_datasets.csv"), index=False)
print(resumen.to_string(index=False))
for k in (10, 100, 500):
    comp = len(set(np.argsort(-np.abs(rh))[:k]) & set(np.argsort(-np.abs(rg))[:k]))
    print(f"   Top-{k} sondas de edad compartidas: {comp} de {k}")
print(f"   Correlación de ρ entre datasets: {corr_rho:.3f}")

print("\n" + "=" * 50)
print(f"Archivos guardados en: {CARPETA_RESULTADOS_EDA}")
print("=" * 50)
