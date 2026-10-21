import type { NotebookContent } from '../../../src/content/types';

const NB = 'lab2_1';
const src = (cells: number[], extra: Partial<{ section: string; note: string }> = {}) => ({
  notebook: NB,
  cells,
  origin: 'original' as const,
  ...extra,
});

/** Nube 2D del notebook (celda 6) con PCA de 2 componentes ajustado (celda 8). */
const SETUP_TOY = `import numpy as np
from sklearn.decomposition import PCA
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
`;

const SETUP_DIGITS = `import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
digits = load_digits()
`;

const EXPERIMENT = `import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# celda 6 · 200 puntos 2D casi alineados
rng = np.random.RandomState({{seed}})
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

# celdas 8-15 · PCA completo (2 componentes) y ejes principales
pca = PCA(n_components=2)
pca.fit(X)
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', linewidth=2, shrinkA=0, shrinkB=0, color='k')
    ax.annotate('', v1, v0, arrowprops=arrowprops)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
ax[0].axis('equal'); ax[0].set_title('Ejes principales (celda 15)')

# celdas 18-20 · reducir a k_toy componentes y reconstruir
pca_k = PCA(n_components={{k_toy}})
pca_k.fit(X)
X_pca = pca_k.transform(X)
X_new = pca_k.inverse_transform(X_pca)
ax[1].scatter(X[:, 0], X[:, 1], alpha=0.2)
ax[1].scatter(X_new[:, 0], X_new[:, 1], alpha=0.8, color='r')
ax[1].axis('equal'); ax[1].set_title('Original (azul) y reconstrucción (rojo), celda 20')
plt.show()

# celdas 23-29 · digits: 64 features -> k_digits componentes
digits = load_digits()
pca_d = PCA({{k_digits}})
projected = pca_d.fit_transform(digits.data)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sc = ax[0].scatter(projected[:, 0], projected[:, 1], c=digits.target, edgecolor='none', alpha=0.5,
                   cmap=plt.get_cmap('jet', 10))     # celda 29 (plt.cm.get_cmap se eliminó en Matplotlib 3.11)
ax[0].set_xlabel('component 1'); ax[0].set_ylabel('component 2')
fig.colorbar(sc, ax=ax[0])

# celda 32 · varianza explicada acumulada
pca_all = PCA().fit(digits.data)
cum = np.cumsum(pca_all.explained_variance_ratio_)
ax[1].plot(cum)
ax[1].axhline({{umbral}}, color='r', ls='--')                       # (app) umbral elegido
n_umbral = int(np.argmax(cum >= {{umbral}})) + 1
ax[1].axvline(n_umbral - 1, color='r', ls=':')
ax[1].set_xlabel('number of components'); ax[1].set_ylabel('cumulative explained variance')
plt.show()

# (app) reconstrucción de los primeros dígitos con k_digits componentes
digits_rec = pca_d.inverse_transform(projected)
if {{show_rec}}:
    fig, ax = plt.subplots(2, 10, figsize=(10, 2.4))
    for i in range(10):
        ax[0, i].imshow(digits.data[i].reshape(8, 8), cmap='binary'); ax[0, i].axis('off')
        ax[1, i].imshow(digits_rec[i].reshape(8, 8), cmap='binary'); ax[1, i].axis('off')
    fig.suptitle('Arriba: originales · abajo: reconstruidos con %d componentes' % pca_d.n_components_)
    plt.show()

resultados = {
    'ev1': pca.explained_variance_[0], 'ev2': pca.explained_variance_[1],
    'evr1': pca.explained_variance_ratio_[0], 'evr2': pca.explained_variance_ratio_[1],
    'mse_toy': np.mean(np.sum((X - X_new) ** 2, axis=1)),
    'dim_toy': X_pca.shape[1],
    'var_digits_k': cum[pca_d.n_components_ - 1],
    'n_umbral': n_umbral,
    'mse_digits': np.mean(np.sum((digits.data - digits_rec) ** 2, axis=1)),
}
`;

export const lab2_1: NotebookContent = {
  id: NB,
  lab: 'lab2',
  title: 'Lab 2 · Parte 1 — PCA y reducción de dimensionalidad',
  short: 'Introducción a PCA',
  origin: 'original',
  context: {
    goal: `Entender **qué aprende PCA** y para qué sirve reducir dimensiones antes de los métodos de las partes 2 y 3.

El notebook (basado en Géron y VanderPlas) parte de una nube de 200 puntos 2D casi alineados: ajusta \`PCA\`, interpreta los **ejes principales** (\`components_\`) y la **varianza explicada** (\`explained_variance_\`, \`explained_variance_ratio_\`), proyecta a 1 dimensión y reconstruye con \`inverse_transform\` para ver qué información se pierde. Después aplica PCA a **digits** (imágenes 8×8 → 64 features) para visualizar en 2D y elegir el número de componentes con la varianza acumulada, y termina con **t-SNE**, un método no lineal solo para visualización.

En la asignatura este lab se llama «feature selection using PCA and MDA»: aquí PCA es **extracción** de nuevas features (combinaciones lineales), no selección de un subconjunto (ver la pestaña del Lab 2).

**Este notebook no tiene salidas guardadas**: todos los resultados que verás proceden de la ejecución en la app (y se han comprobado con tu entorno local).`,
    dataset: `1) **Nube sintética**: \`X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T\` con \`RandomState(1)\`: 200 muestras × 2 features, gaussiana correlacionada (una matriz aleatoria 2×2 mezcla dos normales estándar). Sin etiquetas.
2) **digits** (\`sklearn.datasets.load_digits\`, incluido en scikit-learn): 1797 imágenes de dígitos manuscritos 0–9 de 8×8 píxeles (valores 0–16). \`digits.data\` (1797, 64), \`digits.images\` (1797, 8, 8), \`digits.target\` (1797,).`,
    variables: [
      { name: 'X', meaning: 'Nube 2D sintética (filas = muestras).', shape: '(200, 2)' },
      { name: 'pca', meaning: 'Estimador PCA (se reutiliza el nombre con 2, 1 y todas las componentes).', shape: '—' },
      { name: 'pca.components_', meaning: 'Ejes principales: cada fila es un vector unitario en el espacio de features.', shape: '(n_components, n_features)' },
      { name: 'pca.explained_variance_', meaning: 'Varianza de los datos a lo largo de cada eje (ddof = 1).', shape: '(n_components,)' },
      { name: 'pca.explained_variance_ratio_', meaning: 'Fracción de la varianza total explicada por cada eje.', shape: '(n_components,)' },
      { name: 'pca.mean_', meaning: 'Media de cada feature, restada antes de proyectar.', shape: '(n_features,)' },
      { name: 'X_pca', meaning: 'Proyección (coordenadas) sobre la primera componente.', shape: '(200, 1)' },
      { name: 'X_new', meaning: 'Reconstrucción en 2D a partir de X_pca.', shape: '(200, 2)' },
      { name: 'digits.data / projected', meaning: 'Dígitos en 64 dimensiones / proyectados a 2.', shape: '(1797, 64) / (1797, 2)' },
      { name: 'X_reduced', meaning: 'Embedding 2D de t-SNE.', shape: '(1797, 2)' },
    ],
    classes: [{ id: '0-9', label: 'dígitos 0–9 (solo para colorear: PCA y t-SNE no usan etiquetas)' }],
    questions: [
      { id: 'Previo', text: 'La guía (apartado 3) pide leer y ejecutar este notebook para entender PCA antes de las partes 2 y 3; no tiene preguntas propias en el report.', blocks: ['b_fit', 'b_ncomp'], doc: 'lab2_guide' },
    ],
    notation: [
      { symbol: '$x\\in\\mathbb R^d$', meaning: 'vector de features de una muestra (d = 2 en la nube, 64 en digits)', code: 'X[i]' },
      { symbol: '$S_T$', meaning: 'matriz de dispersión total (covarianza) de los datos centrados', code: 'np.cov(X.T)' },
      { symbol: '$w_i$, $S_T w_i = \\lambda_i w_i$', meaning: 'eje principal i: autovector de $S_T$ (slides Lab2, p. 6)', code: 'pca.components_[i]' },
      { symbol: '$\\lambda_i$', meaning: 'autovalor: varianza a lo largo de $w_i$', code: 'pca.explained_variance_[i]' },
      { symbol: '$\\lambda_i / \\sum_j \\lambda_j$', meaning: 'proporción de varianza explicada', code: 'pca.explained_variance_ratio_[i]' },
      { symbol: '$y = W^T(x-\\bar x)$', meaning: "proyección sobre d' componentes", code: 'pca.transform(X)' },
      { symbol: '$\\hat x = \\bar x + W y$', meaning: 'reconstrucción', code: 'pca.inverse_transform(X_pca)' },
    ],
  },
  flow: [
    { stage: 'contexto', label: 'Contexto', blocks: ['b_intro', 'b_imports'], summary: 'Maldición de la dimensionalidad y proyección.' },
    { stage: 'datos', label: 'Datos', blocks: ['b_toy', 'b_digits_load'], summary: 'Nube 2D correlacionada y digits (64 features).' },
    { stage: 'exploracion', label: 'Exploración', blocks: ['b_digits_show'], summary: 'Ver las imágenes 8×8.' },
    { stage: 'metodo', label: 'Método', blocks: ['b_fit', 'b_reduce', 'b_digits_pca', 'b_tsne'], summary: 'PCA (fit / transform) y t-SNE.' },
    { stage: 'resultados', label: 'Resultados', blocks: ['b_attrs', 'b_vectors', 'b_inverse', 'b_digits_scatter', 'b_tsne_plot'], summary: 'Ejes, varianzas, proyecciones y embeddings.' },
    { stage: 'interpretacion', label: 'Interpretación', blocks: ['b_meaning', 'b_ncomp'], summary: '¿Cuántas componentes? ¿Qué se pierde?' },
  ],
  blocks: [
    {
      id: 'b_intro',
      title: 'Por qué reducir dimensiones: proyección',
      cells: [0, 1, 2],
      stage: 'contexto',
      problem: 'Muchos problemas tienen muchas features: entrenar es lento, hay que estimar muchos parámetros y no se puede visualizar (**maldición de la dimensionalidad**).',
      purpose: 'Presenta la idea de **proyección**: si los datos viven cerca de un subespacio de menor dimensión (features correlacionadas o casi constantes), se pueden describir con menos coordenadas. PCA busca el hiperplano más cercano a los datos.',
      inputs: 'Texto.',
      outputs: 'Motivación: acelerar el entrenamiento, reducir sobreajuste y visualizar en 2–3 dimensiones.',
      prereqs: ['maldicion-dimension', 'extraccion-features'],
      apis: [],
      lines: [],
      interpretation: 'El texto también avisa de que, si los datos se «retuercen» (variedades no lineales), la proyección lineal no basta y hacen falta métodos de *manifold learning* (fuera del alcance; t-SNE se ve al final solo para visualizar).',
      next: 'Importaciones y primer ejemplo 2D.',
      whatIf: [{ change: 'Datos sin correlación entre features (nube redonda)', effect: 'Todas las direcciones tendrían la misma varianza: PCA no podría comprimir sin perder mucha información.' }],
      check: {
        kind: 'mcq',
        id: 'lab2_1.check.intro',
        skills: ['concepto'],
        source: src([2]),
        prompt: 'Según la introducción, ¿cuándo es útil proyectar los datos sobre un subespacio?',
        options: [
          { text: 'Cuando las muestras están cerca de un subespacio de menor dimensión porque hay features correlacionadas o casi constantes', correct: true },
          { text: 'Siempre: proyectar nunca pierde información', feedback: 'Se pierde la varianza en las direcciones descartadas.' },
          { text: 'Solo cuando hay etiquetas de clase', feedback: 'PCA es no supervisado; no usa etiquetas.' },
        ],
        explanation: 'La proyección aprovecha la redundancia: si la nube es casi plana en alguna dirección, descartarla cuesta poca información.',
      },
    },
    {
      id: 'b_imports',
      title: 'Importaciones',
      cells: [3, 4],
      stage: 'contexto',
      problem: 'Disponer de NumPy, Matplotlib y Seaborn.',
      purpose: 'Fija los alias. `sns.set()` cambia el estilo por defecto de todas las figuras (fondo gris y paleta «deep»).',
      inputs: 'Nada.',
      outputs: 'Alias `np`, `plt`, `sns`.',
      prereqs: [],
      apis: [{ name: 'sns.set()', lib: 'seaborn', kind: 'función', does: 'Aplica el tema de seaborn a matplotlib (no dibuja nada).' }],
      lines: [{ cell: 4, lines: [3, 3], text: '`import seaborn as sns; sns.set()`: dos instrucciones en una línea separadas por `;`. Solo afecta al aspecto de las figuras.' }],
      interpretation: 'PCA se importa más adelante (celda 8), justo donde se usa.',
      next: 'Se generan los 200 puntos.',
      whatIf: [{ change: 'Quitar `sns.set()`', effect: 'Mismos resultados numéricos; cambia solo el estilo visual.' }],
    },
    {
      id: 'b_toy',
      title: 'Datos 2D correlacionados',
      cells: [5, 6],
      stage: 'datos',
      problem: 'Tener un ejemplo pequeño donde se vea la relación lineal entre dos variables.',
      purpose: 'Genera 200 puntos gaussianos y los mezcla con una matriz aleatoria 2×2: el resultado es una nube alargada (correlación fuerte), ideal para ver qué hace PCA.',
      inputs: 'Semilla 1 de `RandomState`.',
      outputs: '`X` con forma (200, 2) y la figura de dispersión.',
      prereqs: ['semilla', 'correlacion', 'scatter'],
      apis: [
        { name: 'np.random.RandomState', lib: 'numpy', kind: 'clase', does: 'Generador pseudoaleatorio «antiguo» (en Lab1 se usa `default_rng`).', args: [{ name: 'seed', value: '1', meaning: 'reproducible' }] },
        { name: 'rng.rand(2, 2)', lib: 'numpy', kind: 'método', does: 'Matriz 2×2 uniforme en [0, 1): mezcla lineal.' },
        { name: 'rng.randn(2, 200)', lib: 'numpy', kind: 'método', does: '2×200 normales estándar independientes.' },
        { name: 'np.dot(A, B).T', lib: 'numpy', kind: 'función', does: 'Producto matricial (2×2)·(2×200) = (2, 200); `.T` traspone a (200, 2).' },
        { name: "plt.axis('equal')", lib: 'matplotlib', kind: 'función', does: 'Misma escala en ambos ejes: imprescindible para ver direcciones y longitudes sin deformar.' },
      ],
      lines: [
        { cell: 6, lines: [2, 2], text: 'Cada columna de `rng.randn(2, 200)` es un punto incorrelado; multiplicar por la matriz 2×2 lo transforma linealmente → nube elíptica. `.T` deja filas = muestras, como espera scikit-learn.' },
        { cell: 6, lines: [4, 4], text: 'El `;` final evita que Jupyter imprima el valor devuelto por `plt.axis`.' },
      ],
      shapes: [{ expr: 'X', shape: '(200, 2)', meaning: '200 muestras, 2 features' }],
      interpretation: 'La figura muestra una relación casi lineal entre x e y: una sola dirección concentra casi toda la variación.',
      next: 'PCA cuantifica esa dirección (celda 8).',
      whatIf: [{ change: 'Olvidar el `.T`', effect: 'X tendría forma (2, 200): scikit-learn lo interpretaría como 2 muestras de 200 features.' }],
    },
    {
      id: 'b_fit',
      title: 'Ajustar PCA',
      cells: [7, 8, 9],
      stage: 'metodo',
      problem: 'Encontrar los ejes principales de la nube.',
      purpose: 'Aplica la Estimator API: crear `PCA(n_components=2)` y `fit(X)`. Es **no supervisado**: solo recibe X.',
      inputs: '`X` (200, 2).',
      outputs: 'El objeto `pca` con los atributos aprendidos.',
      prereqs: ['pca', 'estimator-api', 'covarianza', 'autovalor'],
      apis: [{
        name: 'PCA', lib: 'sklearn.decomposition', kind: 'clase', does: 'Análisis de componentes principales: centra los datos y calcula los ejes de máxima varianza (vía SVD).',
        args: [{ name: 'n_components', value: '2', meaning: 'número de componentes que se conservan (aquí todas)' }],
        learned: ['components_', 'explained_variance_', 'explained_variance_ratio_', 'mean_', 'n_components_', 'singular_values_'],
      }],
      lines: [
        { cell: 8, lines: [2, 3], text: '`fit(X)` sin `y`: PCA no usa etiquetas. En la celda, el valor devuelto (el propio objeto) se muestra como `PCA(n_components=2)`.' },
      ],
      interpretation: 'Matemáticamente, los ejes son los autovectores de la covarianza $S_T$ ordenados por autovalor (slides Lab2, p. 6). scikit-learn los obtiene con una SVD de los datos centrados, que es equivalente y más estable.',
      next: 'Se imprimen los atributos aprendidos (celdas 10–12).',
      whatIf: [{ change: '`PCA()` sin n_components', effect: 'Conserva min(n_samples, n_features) = 2 componentes: igual aquí.' }],
    },
    {
      id: 'b_attrs',
      title: 'Atributos aprendidos: componentes y varianza explicada',
      cells: [10, 11, 12, 13],
      stage: 'resultados',
      problem: 'Leer qué ha aprendido PCA.',
      purpose: 'Muestra `components_` (direcciones), `explained_variance_` (varianza a lo largo de cada dirección) y `explained_variance_ratio_` (proporción).',
      inputs: '`pca` ajustado.',
      outputs: 'components_ ≈ [[0.944, 0.329], [−0.329, 0.944]]; explained_variance_ ≈ [0.7625, 0.0185]; ratio ≈ [0.976, 0.024] (ejecución de la app, idéntica a tu entorno local).',
      prereqs: ['varianza-explicada', 'autovalor'],
      apis: [
        { name: 'pca.components_', lib: 'sklearn', kind: 'atributo', does: 'Matriz (n_components, n_features): fila i = eje principal i, vector unitario. Ordenados de mayor a menor varianza.' },
        { name: 'pca.explained_variance_', lib: 'sklearn', kind: 'atributo', does: 'Varianza de las proyecciones sobre cada eje, con n − 1 grados de libertad (= autovalores de la covarianza muestral).' },
        { name: 'pca.explained_variance_ratio_', lib: 'sklearn', kind: 'atributo', does: 'explained_variance_ / varianza total. Suma 1 si se conservan todas las componentes.' },
      ],
      lines: [
        { cell: 10, lines: [1, 1], text: 'Las dos filas son ortogonales y de norma 1. El signo de cada eje es arbitrario: (0.94, 0.33) y (−0.94, −0.33) describen la misma recta.' },
        { cell: 11, lines: [1, 1], text: 'Varianzas ≈ 0.76 y 0.018: la primera dirección tiene ~41 veces más varianza.' },
        { cell: 12, lines: [1, 1], text: 'Proporciones ≈ 0.976 y 0.024, el «98 % / 2 %» que comenta la celda 13.' },
      ],
      shapes: [
        { expr: 'pca.components_', shape: '(2, 2)', meaning: 'componentes × features' },
        { expr: 'pca.explained_variance_', shape: '(2,)', meaning: 'una varianza por componente' },
      ],
      interpretation: 'Casi toda la información (en el sentido de varianza) está en una dirección: la nube es casi una recta. Con una sola componente se conserva el 97.6 % de la varianza.',
      next: 'Se dibujan los ejes sobre los datos (celda 15).',
      whatIf: [{ change: 'Multiplicar la feature x por 100', effect: 'La primera componente se alinearía con x: PCA depende de la **escala** de las features (se ve en Lab4, feature scaling).' }],
    },
    {
      id: 'b_vectors',
      title: 'Dibujar los ejes principales',
      cells: [14, 15, 16],
      stage: 'resultados',
      problem: 'Ver las componentes como flechas sobre la nube.',
      purpose: 'Cada flecha sale de la media (`pca.mean_`), apunta en la dirección de `components_[i]` y mide $3\\sqrt{\\lambda_i}$ (tres desviaciones típicas).',
      inputs: '`X`, `pca`.',
      outputs: 'Figura con la nube y dos flechas perpendiculares.',
      prereqs: ['pca', 'desviacion'],
      apis: [
        { name: 'ax.annotate', lib: 'matplotlib', kind: 'método', does: 'Dibuja una flecha de `v0` a `v1` (texto vacío).', args: [{ name: 'arrowprops', meaning: 'estilo de la flecha' }] },
        { name: 'pca.mean_', lib: 'sklearn', kind: 'atributo', does: 'Media por feature (centro de la nube).' },
        { name: 'zip', lib: 'python', kind: 'función', does: 'Recorre a la vez varianzas y componentes.' },
      ],
      lines: [
        { cell: 15, lines: [1, 6], text: 'Función auxiliar: `ax or plt.gca()` usa los ejes actuales si no se pasan.' },
        { cell: 15, lines: [10, 12], text: 'Longitud ∝ $\\sqrt{\\text{varianza}}$ (desviación típica). La celda 14 dice «squared length» = varianza: es la longitud al cuadrado la que es proporcional a la varianza.' },
      ],
      interpretation: 'La flecha larga marca la dirección de máxima dispersión; la corta, la perpendicular con casi nada de varianza. Proyectar sobre la larga conserva casi todo.',
      next: 'Reducción a 1 componente (celda 18).',
      whatIf: [{ change: "Quitar `plt.axis('equal')`", effect: 'Las flechas dejarían de parecer perpendiculares: los ejes tendrían escalas distintas.' }],
    },
    {
      id: 'b_reduce',
      title: 'Reducir a 1 dimensión',
      cells: [17, 18],
      stage: 'metodo',
      problem: 'Describir cada punto con una sola coordenada.',
      purpose: 'PCA como reducción: se conserva solo la componente de mayor varianza; `transform` proyecta los datos centrados sobre ella.',
      inputs: '`X` (200, 2).',
      outputs: '`X_pca` (200, 1); se imprimen ambas formas.',
      prereqs: ['proyeccion', 'pca'],
      apis: [
        { name: 'PCA(n_components=1)', lib: 'sklearn.decomposition', kind: 'clase', does: 'Conserva una componente.' },
        { name: 'pca.transform(X)', lib: 'sklearn', kind: 'método', does: '$(X - \\bar x)\\,W$, con W = componentes traspuestas.', returns: 'array (n_samples, n_components)' },
      ],
      lines: [
        { cell: 18, lines: [1, 3], text: 'Se **reasigna** `pca`: el objeto de 2 componentes se pierde (las celdas 10–15 ya no se podrían repetir sin reajustar).' },
        { cell: 18, lines: [4, 5], text: 'Salida: `original shape: (200, 2)` y `transformed shape: (200, 1)`.' },
      ],
      shapes: [{ expr: 'X_pca', shape: '(200, 1)', meaning: 'una coordenada por muestra (sigue siendo 2D como array)' }],
      interpretation: 'La dimensión se reduce un 50 % conservando el 97.6 % de la varianza.',
      next: 'Se reconstruye para ver qué se ha perdido (celda 20).',
      whatIf: [{ change: '`n_components=3`', effect: '`ValueError`: no puede haber más componentes que min(n_samples, n_features) = 2.' }],
    },
    {
      id: 'b_inverse',
      title: 'Reconstrucción con inverse_transform',
      cells: [19, 20, 21],
      stage: 'resultados',
      problem: 'Visualizar la información descartada.',
      purpose: '`inverse_transform` lleva la coordenada 1D de vuelta al plano: $\\hat x = \\bar x + y\\,w_1$. Los puntos reconstruidos caen sobre la recta del primer eje.',
      inputs: '`X_pca`, `pca` (1 componente).',
      outputs: '`X_new` (200, 2) y la figura original (azul) frente a reconstruida (rojo).',
      prereqs: ['reconstruccion'],
      apis: [{ name: 'pca.inverse_transform', lib: 'sklearn', kind: 'método', does: 'Del espacio reducido al original.', returns: 'array (n_samples, n_features)' }],
      lines: [{ cell: 20, lines: [1, 1], text: 'No recupera X: la componente descartada se ha perdido. El error cuadrático medio de reconstrucción ≈ 0.0184 = varianza de la componente descartada × (n−1)/n.' }],
      shapes: [{ expr: 'X_new', shape: '(200, 2)', meaning: 'mismo espacio que X, pero sobre una recta' }],
      interpretation: 'La dispersión de los puntos azules alrededor de la recta roja es lo que se pierde; como es pequeña, la representación 1D es «suficientemente buena» (celda 21). PCA minimiza exactamente ese error de reconstrucción.',
      next: 'Se pasa a un caso de alta dimensión: digits.',
      whatIf: [{ change: 'Usar 2 componentes', effect: 'Reconstrucción perfecta (error ≈ 0): no se ha descartado nada.' }],
    },
    {
      id: 'b_digits_load',
      title: 'Cargar digits',
      cells: [22, 23],
      stage: 'datos',
      problem: 'Tener datos de alta dimensión donde la reducción sea realmente útil.',
      purpose: 'Carga 1797 imágenes 8×8 ya aplanadas en 64 features.',
      inputs: 'Dataset incluido en scikit-learn (no requiere descarga).',
      outputs: '`digits` (objeto Bunch) y la salida `(1797, 64)`.',
      prereqs: ['matriz-features'],
      apis: [{ name: 'load_digits', lib: 'sklearn.datasets', kind: 'función', does: 'Devuelve un Bunch con `data` (1797, 64), `images` (1797, 8, 8), `target` (1797,), `target_names`, `DESCR`.' }],
      lines: [{ cell: 23, lines: [3, 3], text: 'Última expresión de la celda: Jupyter muestra `(1797, 64)`.' }],
      shapes: [{ expr: 'digits.data', shape: '(1797, 64)', meaning: '64 píxeles (0–16) por imagen' }, { expr: 'digits.images', shape: '(1797, 8, 8)', meaning: 'las mismas imágenes sin aplanar' }],
      interpretation: 'Cada imagen es un punto en un espacio de 64 dimensiones: imposible de visualizar directamente.',
      next: 'Se visualizan las 64 primeras imágenes.',
      whatIf: [{ change: 'Usar `digits.images` en PCA', effect: 'Error: PCA espera un array 2D (n_samples, n_features), no 3D.' }],
    },
    {
      id: 'b_digits_show',
      title: 'Ver las imágenes',
      cells: [24, 25],
      stage: 'exploracion',
      problem: 'Saber qué representan las 64 features.',
      purpose: 'Rejilla 8×8 con las 64 primeras imágenes y su etiqueta en la esquina.',
      inputs: '`digits.images`, `digits.target`.',
      outputs: 'Figura con 64 subplots.',
      prereqs: [],
      apis: [
        { name: 'fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])', lib: 'matplotlib', kind: 'método', does: 'Subplot i de la rejilla, sin marcas en los ejes.' },
        { name: 'ax.imshow', lib: 'matplotlib', kind: 'método', does: 'Muestra una matriz como imagen.', args: [{ name: 'cmap', value: 'plt.cm.binary', meaning: 'blanco y negro' }, { name: 'interpolation', value: "'nearest'", meaning: 'píxeles sin suavizar' }] },
        { name: 'ax.text', lib: 'matplotlib', kind: 'método', does: 'Escribe la etiqueta verdadera.' },
      ],
      lines: [{ cell: 25, lines: [6, 11], text: 'Bucle sobre las 64 primeras muestras: `images[i]` (8, 8) y `target[i]`.' }],
      interpretation: 'Píxeles vecinos están muy correlacionados y los bordes casi siempre valen 0: hay mucha redundancia, que es lo que PCA aprovecha.',
      next: 'Proyección a 2 dimensiones.',
      whatIf: [],
    },
    {
      id: 'b_digits_pca',
      title: 'PCA de 64 a 2 dimensiones',
      cells: [26, 27],
      stage: 'metodo',
      problem: 'Representar cada imagen con 2 números.',
      purpose: '`fit_transform` ajusta y proyecta en un solo paso.',
      inputs: '`digits.data` (1797, 64).',
      outputs: '`projected` (1797, 2); se imprimen `(1797, 64)` y `(1797, 2)`.',
      prereqs: ['pca', 'proyeccion'],
      apis: [
        { name: 'PCA(2)', lib: 'sklearn.decomposition', kind: 'clase', does: 'Primer argumento posicional = n_components.' },
        { name: 'fit_transform', lib: 'sklearn', kind: 'método', does: 'Equivale a `fit(X)` seguido de `transform(X)`.' },
      ],
      lines: [{ cell: 27, lines: [2, 2], text: '`components_` pasa a tener forma (2, 64): cada componente es una «imagen» de 64 pesos.' }],
      shapes: [{ expr: 'projected', shape: '(1797, 2)', meaning: 'dos coordenadas por imagen' }, { expr: 'pca.components_', shape: '(2, 64)', meaning: 'dos direcciones en el espacio de píxeles' }],
      interpretation: 'Las dos primeras componentes solo explican ≈ 28.5 % de la varianza (14.9 % + 13.6 %): sirven para visualizar, no para representar fielmente.',
      next: 'Se dibuja la proyección coloreada por dígito.',
      whatIf: [{ change: '`fit_transform` sobre test en lugar de `transform`', effect: 'En un problema con train/test cambiaría la base: los datos de test quedarían en otro sistema de coordenadas (ver lab2_2).' }],
    },
    {
      id: 'b_digits_scatter',
      title: 'Dígitos proyectados en el plano',
      cells: [28, 29],
      stage: 'resultados',
      problem: 'Ver si los dígitos del mismo tipo quedan agrupados.',
      purpose: 'Dispersión de las dos componentes coloreada por `digits.target` (solo para mirar; PCA no ha usado las etiquetas).',
      inputs: '`projected`, `digits.target`.',
      outputs: 'Figura con barra de color.',
      prereqs: ['scatter'],
      apis: [
        { name: 'plt.scatter(..., c=digits.target, cmap=...)', lib: 'matplotlib', kind: 'función', does: 'Colorea cada punto según su etiqueta.' },
        { name: "plt.get_cmap('jet', 10)", lib: 'matplotlib', kind: 'función', does: 'Mapa de color discretizado en 10 niveles (uno por dígito).' },
        { name: 'plt.colorbar', lib: 'matplotlib', kind: 'función', does: 'Leyenda de colores.' },
      ],
      lines: [{ cell: 29, lines: [3, 3], text: '`plt.cm.get_cmap` está obsoleta desde Matplotlib 3.7 y se **eliminó en la 3.11**: en tu entorno (3.11.2) falla con `AttributeError`; en la app (3.10.8) solo avisa. La app ejecuta `plt.get_cmap(\'jet\', 10)`.' }],
      interpretation: 'Algunos dígitos (0, 6, 4) forman grupos reconocibles; otros se solapan mucho. Dos componentes con 28 % de la varianza no separan las 10 clases.',
      runCode: `plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();`,
      corrections: [{ cell: 29, kind: 'api-obsoleta', original: "cmap=plt.cm.get_cmap('jet', 10))", fixed: "cmap=plt.get_cmap('jet', 10))   # o matplotlib.colormaps['jet'].resampled(10)", why: '`matplotlib.cm.get_cmap` quedó obsoleta en la 3.7 y se eliminó en la 3.11 (tu entorno local). La app usa Matplotlib 3.10, donde aún funciona con un aviso, pero ejecuta la versión corregida.' }],
      next: '¿Qué significan las componentes y cuántas hacen falta?',
      whatIf: [{ change: 'Colorear sin `cmap` discreto', effect: 'Funciona, pero con un mapa continuo cuesta distinguir los 10 dígitos.' }],
    },
    {
      id: 'b_meaning',
      title: 'Qué significan las componentes',
      cells: [30],
      stage: 'interpretacion',
      problem: 'Dar sentido a las coordenadas de PCA.',
      purpose: 'PCA elige una base óptima: cada dato ≈ media + Σ (coeficiente × componente). Los coeficientes son la representación reducida.',
      inputs: 'Texto.',
      outputs: 'Idea de «funciones base» (se ilustra en un lab posterior, según el notebook).',
      prereqs: ['reconstruccion'],
      apis: [],
      lines: [],
      interpretation: 'En digits, cada componente es una imagen 8×8 de pesos; la reconstrucción con pocas componentes da dígitos borrosos (el experimento de la app lo muestra como ampliación).',
      next: 'Elección del número de componentes.',
      whatIf: [],
    },
    {
      id: 'b_ncomp',
      title: 'Elegir el número de componentes: varianza acumulada',
      cells: [31, 32, 33],
      stage: 'interpretacion',
      problem: 'Decidir cuántas componentes conservar.',
      purpose: 'Se ajusta PCA con todas las componentes y se dibuja `np.cumsum(explained_variance_ratio_)`: cuánta varianza se retiene con las N primeras.',
      inputs: '`digits.data`.',
      outputs: 'Curva creciente de 0 a 1 (64 puntos).',
      prereqs: ['varianza-explicada'],
      apis: [
        { name: 'PCA().fit', lib: 'sklearn', kind: 'método', does: 'Todas las componentes (64).' },
        { name: 'np.cumsum', lib: 'numpy', kind: 'función', does: 'Suma acumulada.' },
      ],
      lines: [{ cell: 32, lines: [2, 2], text: 'El eje x empieza en 0: el punto x = k corresponde a k + 1 componentes.' }],
      shapes: [{ expr: 'pca.explained_variance_ratio_', shape: '(64,)', meaning: 'una proporción por componente' }],
      interpretation: 'Valores reales: 10 componentes → 73.8 % («≈75 %» en la celda 33); hacen falta **21** para llegar al 90 % (con 20, 89.4 %, el «unas 20» del texto); 50 → 99.95 %. La curva mide redundancia: si sube muy rápido, pocas componentes bastan.',
      next: 'Comparación con un método no lineal: t-SNE.',
      whatIf: [{ change: 'Pedir el 95 %', effect: 'Hacen falta 29 componentes (ver el experimento con el umbral).' }],
    },
    {
      id: 'b_tsne',
      title: 't-SNE a 2 dimensiones',
      cells: [34, 35],
      stage: 'metodo',
      problem: 'Visualizar digits con un método no lineal.',
      purpose: 't-SNE busca un embedding 2D que conserve **vecindades locales** (puntos cercanos en 64D quedan cercanos en 2D).',
      inputs: '`digits.data`.',
      outputs: '`X_reduced` (1797, 2). Tarda ≈ 7 s en tu entorno local y ≈ 30 s en la app (Python en WebAssembly).',
      prereqs: ['tsne'],
      apis: [{
        name: 'TSNE', lib: 'sklearn.manifold', kind: 'clase', does: 'Embedding no lineal para visualización.',
        args: [
          { name: 'n_components', value: '2', meaning: 'dimensión del embedding' },
          { name: 'init', value: '"random"', meaning: 'posiciones iniciales aleatorias' },
          { name: 'learning_rate', value: '"auto"', meaning: 'paso ajustado al tamaño de los datos' },
          { name: 'random_state', value: '1', meaning: 'reproducible' },
        ],
        learned: ['embedding_', 'kl_divergence_'],
      }],
      lines: [{ cell: 35, lines: [5, 5], text: 'Solo existe `fit_transform`: t-SNE **no tiene `transform`** para datos nuevos, por eso no sirve como preprocesado de un clasificador entrenado en train y evaluado en test.' }],
      shapes: [{ expr: 'X_reduced', shape: '(1797, 2)', meaning: 'coordenadas del embedding' }],
      interpretation: 'A diferencia de PCA, los ejes de t-SNE no tienen significado ni varianza explicada, y las distancias entre grupos lejanos no son interpretables.',
      next: 'Se dibuja el embedding.',
      whatIf: [{ change: 'Cambiar `random_state`', effect: 'Grupos en otras posiciones y orientaciones; la estructura local se mantiene.' }],
    },
    {
      id: 'b_tsne_plot',
      title: 'Embedding de t-SNE',
      cells: [36],
      stage: 'resultados',
      problem: 'Comparar visualmente con la proyección PCA.',
      purpose: 'Misma figura que la celda 29 con el embedding de t-SNE.',
      inputs: '`X_reduced`, `digits.target`.',
      outputs: 'Figura 13×10 con 10 grupos bien separados.',
      prereqs: ['tsne'],
      apis: [{ name: 'plt.scatter / plt.colorbar', lib: 'matplotlib', kind: 'función', does: 'Igual que en la celda 29.' }],
      lines: [{ cell: 36, lines: [4, 4], text: 'De nuevo `plt.cm.get_cmap`: la app ejecuta `plt.get_cmap`.' }],
      interpretation: 't-SNE separa casi todos los dígitos en grupos compactos mientras que PCA 2D los mezclaba: es mejor **para ver** la estructura, pero no es una reducción lineal reutilizable ni conserva varianzas.',
      runCode: `plt.figure(figsize=(13, 10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10))
#plt.axis('off')
plt.colorbar();
plt.show()`,
      corrections: [{ cell: 36, kind: 'api-obsoleta', original: "cmap=plt.cm.get_cmap('jet', 10))", fixed: "cmap=plt.get_cmap('jet', 10))", why: 'Igual que en la celda 29.' }],
      whatIf: [{ change: 'Interpretar la distancia entre dos grupos lejanos', effect: 'Error conceptual: t-SNE no conserva distancias globales.' }],
    },
  ],
  experiments: [
    {
      id: 'lab2_1.exp.pca',
      title: 'PCA: componentes, reconstrucción y número de componentes',
      goal: 'Cambiar cuántas componentes se conservan en la nube 2D y en digits, y ver cómo cambian la varianza retenida, el error de reconstrucción y las figuras. El umbral de varianza responde a «¿cuántas componentes necesito?».',
      blocks: ['b_toy', 'b_attrs', 'b_vectors', 'b_reduce', 'b_inverse', 'b_digits_pca', 'b_digits_scatter', 'b_ncomp'],
      template: EXPERIMENT,
      seedParam: 'seed',
      params: [
        { id: 'seed', label: 'Semilla de la nube 2D', kind: 'datos', control: 'range', min: 0, max: 30, step: 1, default: '1', variable: 'np.random.RandomState(…)', computation: 'Otra matriz de mezcla 2×2 y otros puntos.', result: 'Cambian la orientación y lo alargada que es la nube: varía la proporción de varianza de la primera componente.', why: 'La forma de la nube depende de la matriz aleatoria `rng.rand(2, 2)`.' },
        { id: 'k_toy', label: 'Componentes en la nube 2D', kind: 'modelo', control: 'select', default: '1', options: [{ value: '1', label: '1 (celda 18)' }, { value: '2', label: '2 (todas)' }], variable: 'PCA(n_components=…)', computation: 'Número de columnas de `X_pca`.', result: 'Con 2 la reconstrucción coincide con X (error ≈ 0); con 1 los puntos rojos caen sobre una recta.', why: 'Lo que no se proyecta se pierde: con todas las componentes no se pierde nada.' },
        { id: 'k_digits', label: 'Componentes en digits', kind: 'modelo', control: 'select', default: '2', options: ['2', '3', '5', '10', '21', '40', '64'].map((v) => ({ value: v, label: v === '2' ? '2 (celda 27)' : v })), variable: 'PCA(…)', computation: 'Dimensión de `projected` y del espacio de reconstrucción.', result: 'Más componentes → más varianza retenida y menor error de reconstrucción (dígitos más nítidos). El scatter solo muestra las dos primeras.', why: 'Las componentes están ordenadas por varianza: añadir más siempre retiene más.' },
        { id: 'umbral', label: 'Umbral de varianza acumulada', kind: 'evaluacion', control: 'select', default: '0.9', options: ['0.5', '0.75', '0.9', '0.95', '0.99'].map((v) => ({ value: v, label: `${Number(v) * 100} %` })), variable: 'cum >= umbral', computation: 'Solo cambia el criterio con el que se lee la curva de la celda 32.', result: 'Número mínimo de componentes que alcanza el umbral (90 % → 21).', why: 'El criterio es una decisión del analista, no algo que aprenda PCA.', origin: 'ampliacion' },
        { id: 'show_rec', label: 'Mostrar dígitos reconstruidos', kind: 'visualizacion', control: 'toggle', default: 'True', variable: 'if …:', computation: 'Ninguno: añade una figura.', result: 'Las métricas no cambian.', why: 'Parámetro de visualización.', origin: 'ampliacion' },
      ],
      metrics: [
        { key: 'ev1', label: 'Varianza eje 1 (nube)', format: 'num', digits: 4 },
        { key: 'ev2', label: 'Varianza eje 2 (nube)', format: 'num', digits: 4 },
        { key: 'evr1', label: 'Proporción eje 1 (nube)', format: 'pct' },
        { key: 'evr2', label: 'Proporción eje 2 (nube)', format: 'pct' },
        { key: 'dim_toy', label: 'Dimensión de X_pca', format: 'int' },
        { key: 'mse_toy', label: 'Error de reconstrucción (nube)', format: 'num', digits: 5, better: 'low' },
        { key: 'var_digits_k', label: 'Varianza retenida en digits', format: 'pct', better: 'high' },
        { key: 'mse_digits', label: 'Error de reconstrucción (digits)', format: 'num', digits: 2, better: 'low' },
        { key: 'n_umbral', label: 'Componentes para el umbral', format: 'int' },
      ],
      observe: [
        'Con la semilla 1: ¿por qué la proporción de la primera componente es tan alta? Prueba otras semillas hasta encontrar una nube más redonda.',
        'Pon 2 componentes en la nube: ¿qué le pasa al error de reconstrucción?',
        'En digits, compara 2, 10 y 21 componentes: varianza retenida y nitidez de los dígitos reconstruidos.',
        'Cambia el umbral a 95 % y 99 %: ¿cuántas componentes hacen falta? ¿Merece la pena?',
      ],
      source: src([6, 8, 10, 11, 12, 15, 18, 20, 27, 29, 32], { note: 'Umbral y reconstrucción de dígitos son ampliaciones de la app. El notebook no tiene salidas guardadas.' }),
      fidelity: [
        { metrics: { ev1: 0.7625315008826108, ev2: 0.018477895513562492, evr1: 0.9763410074208769, mse_toy: 0.01838550603599476, var_digits_k: 0.28509364823699296, n_umbral: 21 }, tol: 1e-9, source: 'Ejecución local con tus versiones (numpy 2.5.3, scikit-learn 1.9.1)' },
        { params: { umbral: '0.95' }, metrics: { n_umbral: 29 }, tol: 0, source: 'Ejecución local con tus versiones' },
      ],
    },
  ],
  activities: [
    // ---- nivel 1
    {
      kind: 'mcq', id: 'lab2_1.read.components', skills: ['api', 'concepto'], blocks: ['b_attrs'], source: src([10]),
      prompt: '¿Qué contiene `pca.components_` tras `PCA(n_components=2).fit(X)` con X de forma (200, 2)?',
      options: [
        { text: 'Dos vectores unitarios (filas) que indican las direcciones de máxima varianza, ordenados', correct: true },
        { text: 'Las coordenadas de las 200 muestras en la nueva base', feedback: 'Eso lo devuelve `transform(X)`.' },
        { text: 'Las varianzas de cada feature original', feedback: 'Las varianzas de las componentes están en `explained_variance_`.' },
        { text: 'La media de cada feature', feedback: 'Es `mean_`.' },
      ],
      explanation: '`components_` tiene forma (n_components, n_features): cada fila es un eje principal (autovector de la covarianza).',
    },
    {
      kind: 'mcq', id: 'lab2_1.read.vectors', skills: ['codigo'], blocks: ['b_vectors'], source: src([15]),
      prompt: 'En `v = vector * 3 * np.sqrt(length)`, ¿qué representa la longitud de la flecha dibujada?',
      code: 'for length, vector in zip(pca.explained_variance_, pca.components_):\n    v = vector * 3 * np.sqrt(length)\n    draw_vector(pca.mean_, pca.mean_ + v)',
      options: [
        { text: 'Tres desviaciones típicas de los datos a lo largo de esa componente', correct: true },
        { text: 'Tres veces la varianza de la componente', feedback: 'Se toma la raíz: es una desviación típica.' },
        { text: 'La distancia media de los puntos a la media', feedback: 'No se calcula ninguna distancia media.' },
      ],
      explanation: '`explained_variance_` es una varianza; su raíz es la desviación típica. Por eso la longitud al cuadrado es proporcional a la varianza (lo que dice la celda 14).',
    },
    {
      kind: 'text', id: 'lab2_1.read.tsne', skills: ['concepto', 'metodo'], blocks: ['b_tsne', 'b_tsne_plot'], source: src([35, 36]),
      prompt: 'Explica dos diferencias entre PCA y t-SNE que importen si quisieras usarlos antes de un clasificador.',
      rubric: [
        'PCA es lineal y t-SNE no lineal.',
        'PCA tiene transform para proyectar datos nuevos (test); t-SNE solo fit_transform.',
        'PCA conserva varianza/estructura global (con varianza explicada interpretable); t-SNE conserva vecindades locales y sus ejes/distancias globales no son interpretables.',
        'Por eso t-SNE se usa para visualizar, no como preprocesado.',
      ],
      model: 'PCA es una proyección **lineal** aprendida en entrenamiento que después se aplica igual a cualquier dato nuevo con `transform`; además informa de cuánta varianza retiene cada componente. t-SNE es un embedding **no lineal** que solo existe para los puntos con los que se calculó (`fit_transform`), conserva vecindades locales y no distancias globales, y sus ejes no tienen significado. Por eso t-SNE es útil para **visualizar** (separa muy bien los dígitos), pero no como preprocesado de un clasificador que se evalúa en test.',
    },
    // ---- nivel 2
    {
      kind: 'predict', id: 'lab2_1.pred.xpca', skills: ['codigo'], blocks: ['b_reduce'], source: src([18]),
      prompt: '¿Qué forma tiene `X_pca`?', setup: SETUP_TOY,
      code: 'pca = PCA(n_components=1)\npca.fit(X)\nX_pca = pca.transform(X)', expr: 'X_pca.shape', answerKind: 'shape',
      explanation: '200 muestras y 1 componente: (200, 1). Es un array 2D aunque tenga una sola columna.',
    },
    {
      kind: 'predict', id: 'lab2_1.pred.comp_digits', skills: ['api'], blocks: ['b_digits_pca'], source: src([27]),
      prompt: 'Tras `pca = PCA(2); projected = pca.fit_transform(digits.data)`, ¿qué forma tiene `pca.components_`?', setup: SETUP_DIGITS,
      code: 'pca = PCA(2)\nprojected = pca.fit_transform(digits.data)', expr: 'pca.components_.shape', answerKind: 'shape',
      explanation: '(n_components, n_features) = (2, 64): cada componente es un vector en el espacio de 64 píxeles.',
    },
    {
      kind: 'predict', id: 'lab2_1.pred.ratio_sum', skills: ['concepto'], blocks: ['b_attrs'], source: src([12]),
      prompt: '¿Cuánto suma `pca.explained_variance_ratio_` para `PCA(n_components=2)` sobre la nube 2D? (2 decimales)', setup: SETUP_TOY,
      code: 'pca = PCA(n_components=2).fit(X)', expr: 'float(pca.explained_variance_ratio_.sum())', answerKind: 'numeric', tolerance: 0.005,
      explanation: 'Con todas las componentes (2 de 2 features) se explica el 100 % de la varianza: suma 1.',
    },
    {
      kind: 'predict', id: 'lab2_1.pred.n90', skills: ['interpretacion'], blocks: ['b_ncomp'], source: src([32, 33]),
      prompt: 'El texto de la celda 33 dice que hacen falta «unas 20 componentes» para retener el 90 % de la varianza de digits. ¿Cuál es el número mínimo exacto?', setup: SETUP_DIGITS,
      code: 'pca = PCA().fit(digits.data)\ncum = np.cumsum(pca.explained_variance_ratio_)\nn90 = int(np.argmax(cum >= 0.9)) + 1', expr: 'n90', answerKind: 'numeric', tolerance: 0,
      explanation: 'Con 20 componentes se retiene el 89.4 %; con 21, el 90.3 %. `argmax` devuelve el primer índice que cumple la condición (base 0), por eso se suma 1.',
    },
    // ---- nivel 3
    {
      kind: 'code', level: 3, id: 'lab2_1.fill.inverse', title: 'Completa la reducción y la reconstrucción', skills: ['api', 'codigo'], blocks: ['b_reduce', 'b_inverse'], source: src([18, 20]),
      practice: 'PCA.transform / inverse_transform',
      prompt: 'Rellena los huecos para proyectar la nube sobre su primera componente y reconstruirla en 2D.',
      available: '`np`, `PCA` y `X` (200, 2) de la celda 6.',
      produce: '`X_pca` (200, 1) y `X_new` (200, 2).',
      setup: SETUP_TOY,
      starter: 'pca = PCA(n_components=___)\npca.___(X)\nX_pca = pca.___(X)\nX_new = pca.___(X_pca)',
      solution: 'pca = PCA(n_components=1)\npca.fit(X)\nX_pca = pca.transform(X)\nX_new = pca.inverse_transform(X_pca)',
      alternatives: ['pca = PCA(1)\nX_pca = pca.fit_transform(X)\nX_new = pca.inverse_transform(X_pca)'],
      wrong: [{ code: 'pca = PCA(n_components=1)\npca.fit(X)\nX_pca = pca.transform(X)\nX_new = pca.inverse_transform(X)', expect: 'dimensiones' }],
      checks: "c.shape('X_pca', (200, 1))\nc.same('X_pca', rtol=1e-8, atol=1e-10)\nc.same('X_new', rtol=1e-8, atol=1e-10)",
      hints: ['Una sola componente: `n_components=1`.', 'Primero `fit`, después `transform`.', '`inverse_transform` recibe la proyección (200, 1), no los datos originales.'],
    },
    {
      kind: 'code', level: 3, id: 'lab2_1.fill.cumsum', title: 'Completa la curva de varianza acumulada', skills: ['api', 'codigo'], blocks: ['b_ncomp'], source: src([32]),
      prompt: 'Completa el código para obtener la varianza acumulada de digits con todas las componentes.',
      available: '`np`, `PCA`, `digits`.',
      produce: '`cum` (64,), creciente y terminada en 1.',
      setup: SETUP_DIGITS,
      starter: 'pca = PCA().___(digits.data)\ncum = np.___(pca.___)',
      solution: 'pca = PCA().fit(digits.data)\ncum = np.cumsum(pca.explained_variance_ratio_)',
      alternatives: ['pca = PCA(n_components=64).fit(digits.data)\ncum = pca.explained_variance_ratio_.cumsum()', 'pca = PCA().fit(digits.data)\nev = pca.explained_variance_\ncum = np.cumsum(ev) / ev.sum()'],
      wrong: [{ code: 'pca = PCA().fit(digits.data)\ncum = np.cumsum(pca.explained_variance_)', expect: 'razonamiento' }],
      checks: "c.shape('cum', (64,))\nc.same('cum', rtol=1e-8)",
      hints: ['La proporción de varianza está en `explained_variance_ratio_`.', '`np.cumsum` calcula la suma acumulada.'],
    },
    // ---- nivel 4
    {
      kind: 'code', level: 4, id: 'lab2_1.fix.cmap', title: 'Corrige la API obsoleta de colormaps', skills: ['api'], blocks: ['b_digits_scatter'], source: src([29], { note: 'API eliminada en Matplotlib 3.11' }),
      prompt: 'Este código de la celda 29 falla en tu entorno local (Matplotlib 3.11: `AttributeError`; en la app, 3.10, solo da un aviso de obsolescencia). Reescríbelo con la API actual para que dibuje la proyección coloreada por dígito con un mapa de 10 colores.',
      available: '`np`, `plt` (matplotlib.pyplot), `digits`, `projected` (1797, 2).',
      produce: 'Una figura de dispersión con barra de color.',
      setup: SETUP_DIGITS + 'import matplotlib.pyplot as plt\nprojected = PCA(2).fit_transform(digits.data)\n',
      starter: "plt.scatter(projected[:, 0], projected[:, 1],\n            c=digits.target, edgecolor='none', alpha=0.5,\n            cmap=plt.cm.get_cmap('jet', 10))\nplt.colorbar()\nplt.show()",
      solution: "plt.scatter(projected[:, 0], projected[:, 1],\n            c=digits.target, edgecolor='none', alpha=0.5,\n            cmap=plt.get_cmap('jet', 10))\nplt.colorbar()\nplt.show()",
      alternatives: ["import matplotlib\nplt.scatter(projected[:, 0], projected[:, 1], c=digits.target, alpha=0.5, cmap=matplotlib.colormaps['jet'].resampled(10))\nplt.colorbar()\nplt.show()"],
      wrong: [{ code: "plt.scatter(projected[:, 0], projected[:, 1], c=digits.target, cmap=plt.cm.get_cmap('jet', 10))\nplt.show()", expect: 'api' }],
      checks: "c.true('cm.get_cmap' not in c.run.get('_code', ''), 'Usa la API actual de colormaps', 'Sigue usando plt.cm.get_cmap, eliminada en Matplotlib 3.11', 'api')\nc.figures(1, min_axes=2, desc='Figura con la dispersión y la barra de color')",
      hints: ['En Matplotlib 3.11 el módulo `plt.cm` ya no tiene `get_cmap` (AttributeError).', 'Usa `plt.get_cmap(nombre, n)` o `matplotlib.colormaps[nombre].resampled(n)`.'],
      explanation: 'La barra de color es un segundo eje de la figura: por eso se comprueba que haya 2 ejes.',
    },
    {
      kind: 'code', level: 4, id: 'lab2_1.fix.center', title: 'Corrige la proyección manual', skills: ['concepto', 'codigo'], blocks: ['b_reduce', 'b_fit'], source: src([18]),
      prompt: 'Se quiere reproducir `pca.transform(X)` a mano con los atributos aprendidos, pero el resultado no coincide. Encuentra y corrige el error.',
      available: '`np`, `X` (200, 2) y `pca = PCA(n_components=1).fit(X)`.',
      produce: '`X_manual` igual a `pca.transform(X)` (200, 1).',
      setup: SETUP_TOY + 'pca = PCA(n_components=1).fit(X)\n',
      starter: 'X_manual = X @ pca.components_.T',
      solution: 'X_manual = (X - pca.mean_) @ pca.components_.T',
      alternatives: ['X_manual = np.dot(X - X.mean(axis=0), pca.components_.T)'],
      wrong: [{ code: 'X_manual = (X - pca.mean_) @ pca.components_', expect: 'dimensiones' }],
      checks: "c.shape('X_manual', (200, 1))\nc.same('X_manual', ref_name='X_manual', rtol=1e-8, atol=1e-10)",
      hints: ['PCA proyecta los datos **centrados**.', 'La media aprendida está en `pca.mean_`.'],
      explanation: '$y = (x - \\bar x)\\,W$ con W = `components_.T` (2, 1).',
    },
    // ---- nivel 5
    {
      kind: 'code', level: 5, id: 'lab2_1.mem.ncomp', title: 'Cuántas componentes para el 95 %', skills: ['escritura', 'interpretacion'], blocks: ['b_ncomp'], source: src([32, 33]),
      prompt: 'Sin mirar el notebook: carga digits, ajusta PCA con todas las componentes y calcula el **número mínimo** de componentes que retiene al menos el 95 % de la varianza.',
      available: 'Nada importado.',
      produce: '`n95` (entero).',
      setup: '',
      starter: '',
      solution: 'import numpy as np\nfrom sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\ndigits = load_digits()\ncum = np.cumsum(PCA().fit(digits.data).explained_variance_ratio_)\nn95 = int(np.argmax(cum >= 0.95)) + 1',
      alternatives: ['from sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\npca = PCA(n_components=0.95).fit(load_digits().data)\nn95 = pca.n_components_', 'import numpy as np\nfrom sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\nr = PCA().fit(load_digits().data).explained_variance_ratio_\nn95 = next(k for k in range(1, 65) if r[:k].sum() >= 0.95)'],
      wrong: [{ code: 'import numpy as np\nfrom sklearn.datasets import load_digits\nfrom sklearn.decomposition import PCA\ncum = np.cumsum(PCA().fit(load_digits().data).explained_variance_ratio_)\nn95 = int(np.argmax(cum >= 0.95))', expect: 'razonamiento' }],
      checks: "c.same('n95')",
      hints: ['La curva es `np.cumsum(explained_variance_ratio_)`.', '`np.argmax(condición)` da el primer índice True, empezando en 0.'],
      explanation: 'Resultado: 29. PCA también acepta `n_components=0.95` (fracción de varianza).',
    },
    {
      kind: 'code', level: 5, id: 'lab2_1.mem.eig', title: 'PCA a mano con la covarianza', skills: ['escritura', 'concepto'], blocks: ['b_attrs', 'b_fit'], source: { notebook: NB, cells: [10, 11], doc: 'lab2_slides', section: 'p. 6', origin: 'original' },
      prompt: 'Las slides definen las componentes como autovectores de $S_T$. Calcula los autovalores de la covarianza muestral de `X` ordenados de mayor a menor y comprueba que coinciden con `explained_variance_`.',
      available: '`np`, `X` (200, 2).',
      produce: '`autovalores` (2,), ordenados de mayor a menor.',
      setup: SETUP_TOY,
      starter: '',
      solution: 'C = np.cov(X.T)\nautovalores = np.sort(np.linalg.eigvalsh(C))[::-1]',
      alternatives: ['vals, vecs = np.linalg.eig(np.cov(X, rowvar=False))\nautovalores = np.sort(vals.real)[::-1]', 'Xc = X - X.mean(axis=0)\nautovalores = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / (len(X) - 1)))[::-1]'],
      wrong: [{ code: 'Xc = X - X.mean(axis=0)\nautovalores = np.sort(np.linalg.eigvalsh(Xc.T @ Xc / len(X)))[::-1]', expect: 'razonamiento' }],
      checks: "c.same('autovalores', rtol=1e-8, desc='Los autovalores (iguales a explained_variance_)')",
      hints: ['`np.cov` espera variables en filas: transpón X o usa `rowvar=False`.', '`np.linalg.eigvalsh` sirve para matrices simétricas.', '`explained_variance_` usa n − 1 en el denominador.'],
    },
    // ---- nivel 6
    {
      kind: 'code', level: 6, id: 'lab2_1.adapt.wine', title: 'Adapta: PCA en otro dataset (wine)', skills: ['metodo', 'interpretacion', 'escritura'], blocks: ['b_digits_pca', 'b_ncomp'], source: { origin: 'app', note: 'Dataset de transferencia incluido en scikit-learn' }, transfer: true,
      prompt: 'Aplica el mismo análisis a `load_wine()` (178 vinos, 13 medidas químicas en escalas muy distintas): proporción de varianza de las dos primeras componentes y número de componentes para el 99 %. Después piensa: ¿qué feature domina y por qué?',
      available: 'Nada importado.',
      produce: '`evr2` (array (2,)) con `explained_variance_ratio_` de `PCA(2)` y `n99` (entero).',
      setup: '',
      starter: '',
      solution: 'import numpy as np\nfrom sklearn.datasets import load_wine\nfrom sklearn.decomposition import PCA\nwine = load_wine()\nevr2 = PCA(2).fit(wine.data).explained_variance_ratio_\ncum = np.cumsum(PCA().fit(wine.data).explained_variance_ratio_)\nn99 = int(np.argmax(cum >= 0.99)) + 1',
      checks: "c.same('evr2', rtol=1e-6)\nc.same('n99')",
      hints: ['Mismo patrón que digits: `PCA(2)` y `np.cumsum`.', 'Mira `wine.data.std(axis=0)`: una feature (proline) tiene valores de cientos.'],
      explanation: 'La primera componente explica el 99.8 %: casi coincide con *proline*, la feature de mayor escala. PCA maximiza varianza en las unidades originales; sin escalar, la feature más grande domina (feature scaling se estudia en Lab4).',
    },
    // ---- interpretación
    {
      kind: 'mcq', id: 'lab2_1.int.misconception', skills: ['interpretacion', 'concepto'], blocks: ['b_digits_scatter', 'b_ncomp'], source: src([29, 33]),
      prompt: '«Como las dos primeras componentes de digits son las de mayor varianza, son también las que mejor separan los 10 dígitos». ¿Qué error contiene?',
      options: [
        { text: 'PCA maximiza la varianza sin usar las etiquetas: no garantiza separabilidad entre clases', correct: true },
        { text: 'Ninguno: varianza alta implica separación', feedback: 'Una dirección puede tener mucha varianza dentro de cada clase y no separar nada.' },
        { text: 'Las componentes no están ordenadas por varianza', feedback: 'Sí lo están.' },
      ],
      explanation: 'Es la advertencia de las slides de Lab2 (p. 6–7): PCA minimiza el error de reconstrucción, pero no garantiza separar clases. MDA sí usa las etiquetas (lab2_2).',
    },
    {
      kind: 'text', id: 'lab2_1.int.curve', skills: ['interpretacion'], blocks: ['b_ncomp'], source: src([32, 33]),
      figure: { experiment: 'lab2_1.exp.pca', params: { show_rec: 'False' }, figureIndex: 1 },
      prompt: 'Interpreta la curva de varianza acumulada de digits (figura de la derecha): ¿qué dice sobre la redundancia de los 64 píxeles y qué número de componentes elegirías para entrenar un clasificador?',
      rubric: [
        'Pocas componentes concentran mucha varianza (10 → ~74 %, 21 → 90 %, ~50 → casi 100 %).',
        'Hay mucha redundancia: píxeles correlacionados o casi constantes.',
        'Con 2 componentes solo se retiene ~28 %: útil para visualizar, no para representar.',
        'La elección es un compromiso (p. ej. 90–95 %), y debería validarse con el error de un clasificador en test.',
      ],
      model: 'La curva sube rápido: con 10 componentes ya se retiene ≈ 74 % de la varianza, con 21 el 90 % y con ~50 prácticamente el 100 %. Los 64 píxeles son muy redundantes (vecinos correlacionados y bordes casi siempre a 0). Dos componentes (≈ 28 %) solo sirven para visualizar. Para clasificar elegiría el número que alcanza un umbral razonable (90–95 %: 21–29 componentes) y lo confirmaría midiendo el error en test, como hace la parte 3 con las curvas de error frente a la dimensión.',
    },
    {
      kind: 'numeric', id: 'lab2_1.int.mse', skills: ['concepto', 'interpretacion'], blocks: ['b_inverse'], source: src([20, 21]),
      prompt: 'La varianza de la segunda componente de la nube es λ₂ = 0.018478 (con n − 1 = 199). El error cuadrático medio de reconstrucción con 1 componente, $\\frac1n\\sum\\|x_i-\\hat x_i\\|^2$, ¿cuánto vale? (4 decimales)',
      answer: 0.0184, tolerance: 0.0001,
      explanation: 'Lo perdido es exactamente la proyección sobre la componente descartada: error = λ₂·(n−1)/n = 0.018478·199/200 ≈ 0.01839.',
    },
    {
      kind: 'dynamic', id: 'lab2_1.dyn.cum', skills: ['interpretacion', 'api'], blocks: ['b_ncomp'], source: src([32], { note: 'Variante generada' }),
      answerKind: 'numeric',
      generator: `import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
rng = np.random.default_rng(SEED)
thr = float(rng.choice([0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]))
cum = np.cumsum(PCA().fit(load_digits().data).explained_variance_ratio_)
n = int(np.argmax(cum >= thr)) + 1
variante = {
  'prompt': f'¿Cuántas componentes principales necesitas, como mínimo, para retener al menos el {thr*100:.0f} % de la varianza de digits?',
  'answer': n, 'tolerance': 0,
  'explanation': f'Con {n-1} componentes se retiene {cum[n-2]*100:.1f} % y con {n}, {cum[n-1]*100:.1f} %.'
}`,
    },
    {
      kind: 'mcq', id: 'lab2_1.int.tsne_new', skills: ['metodo'], blocks: ['b_tsne'], source: src([35]),
      prompt: 'Quieres reducir digits a 2 dimensiones, entrenar LDA en train y evaluarlo en test. ¿Por qué no usar t-SNE como en la celda 35?',
      options: [
        { text: 'Porque t-SNE no tiene `transform`: no puede proyectar el test con la transformación aprendida en train', correct: true },
        { text: 'Porque t-SNE es más lento que LDA', feedback: 'Es más lento, pero el problema de fondo es otro.' },
        { text: 'Porque t-SNE necesita las etiquetas', feedback: 'No las usa.' },
      ],
      explanation: 'La transformación debe aprenderse con train y aplicarse igual a test (slides Lab2, p. 4); t-SNE recalcula el embedding cada vez.',
    },
    // ---------------------------------------------------------- cobertura: imports, datos 2D, digits
    {
      kind: 'mcq',
      id: 'lab2_1.cov.imports',
      skills: ['api'],
      blocks: ['b_imports'],
      source: src([4]),
      prompt: 'La celda 4 termina con `import seaborn as sns; sns.set()`. ¿Qué efecto tiene `sns.set()` en el resto del notebook?',
      options: [
        { text: 'Solo cambia el estilo de las figuras de matplotlib (fondo, rejilla, paleta)', correct: true },
        { text: 'Normaliza los datos antes de aplicar PCA', feedback: 'No toca ningún cálculo: es una opción de estilo.' },
        { text: 'Fija la semilla aleatoria', feedback: 'La semilla la fija `np.random.RandomState(1)` en la celda 6.' },
        { text: 'Es necesario para que PCA funcione', feedback: 'PCA es de scikit-learn y no depende de seaborn.' },
      ],
      explanation: 'Es un parámetro de **visualización**: si se elimina, las figuras cambian de aspecto pero `components_`, `explained_variance_` y las proyecciones son idénticos.',
    },
    {
      kind: 'predict',
      id: 'lab2_1.cov.toy',
      skills: ['codigo'],
      blocks: ['b_toy'],
      source: src([6]),
      prompt: 'En la celda 6, `X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T`. ¿Qué forma tiene `X`?',
      setup: 'import numpy as np\nrng = np.random.RandomState(1)',
      code: 'A = rng.rand(2, 2)\nZ = rng.randn(2, 200)\nX = np.dot(A, Z).T',
      expr: 'X.shape',
      answerKind: 'shape',
      explanation: '`A` (2×2) mezcla dos variables independientes `Z` (2×200): el producto es (2, 200) y la traspuesta deja 200 muestras × 2 features, el formato `[n_samples, n_features]` que espera `PCA.fit`. La mezcla lineal crea la correlación que PCA detecta.',
    },
    {
      kind: 'predict',
      id: 'lab2_1.cov.digits',
      skills: ['codigo', 'api'],
      blocks: ['b_digits_load'],
      source: src([23]),
      prompt: 'La celda 23 muestra `digits.data.shape` = (1797, 64). ¿Qué forma tiene `digits.images`?',
      setup: SETUP_DIGITS,
      code: 'shape_images = digits.images.shape',
      expr: 'shape_images',
      answerKind: 'shape',
      explanation: '(1797, 8, 8): las mismas imágenes de 8×8 píxeles. `data` es la versión aplanada (64 features por fila) que usan PCA y los clasificadores; `images` se usa para dibujarlas con `imshow`.',
    },
    {
      kind: 'code',
      level: 3,
      id: 'lab2_1.cov.imshow',
      title: 'Dibuja una imagen de digits y su etiqueta',
      skills: ['api', 'codigo'],
      blocks: ['b_digits_show'],
      source: src([25]),
      prompt: 'Como en la celda 25 pero para una sola imagen: dibuja la imagen de índice 10 en escala de grises binaria y guarda su etiqueta en `etiqueta`.',
      available: '`digits` (Bunch de `load_digits`), `np`. Importa `matplotlib.pyplot` si lo necesitas.',
      produce: 'Una figura con la imagen y `etiqueta` (entero).',
      setup: SETUP_DIGITS,
      starter: 'import matplotlib.pyplot as plt\nplt.imshow(digits.___[10], cmap=plt.cm.binary, interpolation="nearest")\netiqueta = digits.___[10]\nplt.show()',
      solution: 'import matplotlib.pyplot as plt\nplt.imshow(digits.images[10], cmap=plt.cm.binary, interpolation="nearest")\netiqueta = digits.target[10]\nplt.show()',
      alternatives: ['import matplotlib.pyplot as plt\nplt.imshow(digits.data[10].reshape(8, 8), cmap="binary")\netiqueta = int(digits.target[10])\nplt.show()'],
      wrong: [{ code: 'import matplotlib.pyplot as plt\nplt.imshow(digits.data[10], cmap=plt.cm.binary)\netiqueta = digits.target[10]\nplt.show()', expect: 'dimensiones' }],
      checks: "c.same('etiqueta')\nc.figures(1)",
      hints: ['`imshow` necesita una matriz 2D: usa `images`, o `data[i].reshape(8, 8)`.', 'La etiqueta de cada imagen está en `digits.target`.'],
    },
  ],
  recall: ['lab2_1.fill.inverse', 'lab2_1.fill.cumsum', 'lab2_1.fix.cmap', 'lab2_1.fix.center', 'lab2_1.mem.ncomp', 'lab2_1.mem.eig', 'lab2_1.adapt.wine'],
  practice: [
    {
      id: 'lab2_1.practice.pca',
      title: 'Explicar PCA de principio a fin',
      scenario: 'Te piden explicar en el examen, con código y sin el notebook, qué aprende PCA sobre la nube 2D, cómo se proyecta y reconstruye, y cómo elegir el número de componentes en digits.',
      steps: ['lab2_1.read.components', 'lab2_1.fix.center', 'lab2_1.fill.inverse', 'lab2_1.int.mse', 'lab2_1.mem.ncomp', 'lab2_1.int.misconception'],
      source: { notebook: NB, doc: 'lab2_slides', section: 'PCA', origin: 'original' },
    },
  ],
  notes: [
    'El notebook original no tiene salidas guardadas: los resultados mostrados proceden de la ejecución en la app; coinciden con tu entorno local (numpy 2.5.3, scikit-learn 1.9.1) hasta 1e-15.',
    't-SNE (celda 35) tarda ≈ 30 s en la app frente a ≈ 7 s en tu entorno local.',
  ],
};
