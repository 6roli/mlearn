import type { NotebookContent } from '../../../src/content/types';

const NB = 'lab2_2';
const src = (cells: number[], extra: Partial<{ section: string; note: string }> = {}) => ({
  notebook: NB,
  cells,
  origin: 'original' as const,
  ...extra,
});

/** Estado del notebook tras las celdas 2, 4, 6 y 11 (datos de train y test y funciones de clasificación). */
export const lab2_2Setup = (snr = 10) => `import numpy as np
import warnings
warnings.simplefilter('ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.decomposition import PCA
from numpy.random import default_rng
rng = default_rng(seed=5)
SNR = ${snr}
dist = 1
n_classes = 3
n_samples = 500
n_feat = 3
M_Means = dist * np.array([[1, 1, -1], [-1, 0, -1], [-1, -1, 0]])
energy = 0
for i in range(0, n_classes):
    energy = energy + np.dot(M_Means[i], M_Means[i])
energy = energy / n_classes
SNR = 10 ** (SNR / 10)
sig = energy / SNR
sig = sig / n_feat
M_covar = np.zeros(shape=(n_classes, n_feat, n_feat))
D = sig * np.array([[1, 1, 1], [0.1, 0.3, 0.6], [2, 0.01, 0.99]])
for i in range(0, n_classes):
    H = rng.normal(size=(n_feat, n_feat))
    Heval, Hevec = np.linalg.eig(np.matmul(H, H.T))
    M_covar[i, :, :] = np.real(np.matmul(Hevec.T, np.matmul(np.diag(D[i, :]), Hevec)))
X_train = np.empty((0, n_feat), float)
y_train = np.empty((0), int)
for i in range(0, n_classes):
    X_train = np.append(X_train, rng.multivariate_normal(mean=M_Means[i, :], cov=M_covar[i, :, :], size=n_samples), axis=0)
    y_train = np.append(y_train, i * np.ones((n_samples), dtype=int), axis=0)
rp = rng.permutation(len(X_train))
X_train = X_train[rp]
y_train = y_train[rp]
X_test = np.empty((0, n_feat), float)
y_test = np.empty((0), int)
for i in range(0, n_classes):
    X_test = np.append(X_test, rng.multivariate_normal(mean=M_Means[i, :], cov=M_covar[i, :, :], size=n_samples), axis=0)
    y_test = np.append(y_test, i * np.ones((n_samples), dtype=int), axis=0)
rp = rng.permutation(len(X_test))
X_test = X_test[rp]
y_test = y_test[rp]

def lda_classifier(X_train, y_train, X_test, y_test):
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)
    tr = 1. - accuracy_score(y_train, lda.predict(X_train))
    te = 1. - accuracy_score(y_test, lda.predict(X_test))
    cm = np.array([metrics.confusion_matrix(y_train, lda.predict(X_train)), metrics.confusion_matrix(y_test, lda.predict(X_test))])
    return lda, np.array([tr, te]), cm

def qda_classifier(X_train, y_train, X_test, y_test):
    qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)
    tr = 1. - accuracy_score(y_train, qda.predict(X_train))
    te = 1. - accuracy_score(y_test, qda.predict(X_test))
    cm = np.array([metrics.confusion_matrix(y_train, qda.predict(X_train)), metrics.confusion_matrix(y_test, qda.predict(X_test))])
    return qda, np.array([tr, te]), cm
`;

const EXPERIMENT = `import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.decomposition import PCA
from numpy.random import default_rng
rng = default_rng(seed={{seed}})

# celda 4 · medias, SNR y covarianzas aleatorias
SNR = {{SNR}}
dist = 1
n_classes = 3
n_samples = {{n_samples}}
n_feat = 3
M_Means = dist * np.array([[1, 1, -1],[-1, 0, -1], [-1, -1, 0]])
energy = 0
for i in range(0,n_classes):
  energy = energy + np.dot(M_Means[i],M_Means[i])
energy = energy / n_classes
SNR = 10 ** (SNR/10)
sig = energy / SNR
sig = sig / n_feat
M_covar = np.zeros(shape=(n_classes,n_feat,n_feat))
D = sig * np.array([[1, 1, 1],[0.1, 0.3, 0.6], [2, 0.01, 0.99]])
for i in range(0,n_classes):
  H = rng.normal(size=(n_feat,n_feat))
  Heval, Hevec = np.linalg.eig(np.matmul(H,H.T))
  M_covar[i,:,:] = np.real(np.matmul(Hevec.T, np.matmul(np.diag(D[i,:]), Hevec)))   # (app) np.real: eig puede devolver tipo complejo

# celda 6 · train y test de la misma distribución
def generar():
  X = np.empty((0,n_feat), float)
  y = np.empty((0), int)
  for i in range(0,n_classes):
    Xdata = rng.multivariate_normal(mean=M_Means[i,:], cov= M_covar[i,:,:], size= n_samples)
    X = np.append(X, Xdata, axis=0)
    y = np.append(y, i * np.ones((n_samples),dtype=int), axis=0)
  rp = rng.permutation(len(X))
  return X[rp], y[rp]
X_train, y_train = generar()
X_test, y_test = generar()

# celda 11 · entrenar y evaluar (error de train y de test)
def errores(clf, A, B):
  clf.fit(A, y_train)
  return [1. - accuracy_score(y_train, clf.predict(A)), 1. - accuracy_score(y_test, clf.predict(B))]

# celdas 13, 15, 19, 23 y 27 · todas las dimensiones con PCA y con MDA
resultados = {}
proy = {'3d': (X_train, X_test)}
for k in (2, 1):
  pca = PCA(n_components=k).fit(X_train)            # la proyección se aprende SOLO con train
  proy[f'pca{k}'] = (pca.transform(X_train), pca.transform(X_test))
  resultados[f'pca{k}_ratio'] = pca.explained_variance_ratio_.sum()
  mda = LinearDiscriminantAnalysis(n_components=k).fit(X_train, y_train)
  proy[f'mda{k}'] = (mda.transform(X_train), mda.transform(X_test))
  resultados[f'mda{k}_ratio'] = mda.explained_variance_ratio_.sum()
for nombre, (A, B) in proy.items():
  tr, te = errores(LinearDiscriminantAnalysis(solver="svd",store_covariance=True), A, B)
  resultados[f'lda_{nombre}_train'], resultados[f'lda_{nombre}_test'] = tr, te
  tr, te = errores(QuadraticDiscriminantAnalysis(store_covariance=True), A, B)
  resultados[f'qda_{nombre}_train'], resultados[f'qda_{nombre}_test'] = tr, te
for k, v in resultados.items():
  print(f'{k:16s} {v:.4f}')

# celdas 17, 21, 25 y 29 · figura de la proyección elegida (sobre train)
vista = {{vista}}
X, y = proy[vista][0], y_train
colors = ['tab:blue', 'tab:orange', 'tab:green']      # (app) colores de clase fijos
if X.shape[1] == 2:
  h = {{h}}
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  fig, ax = plt.subplots(figsize=(12,5.5), nrows=1, ncols=2)
  for a, clf, nom in ((ax[0], LinearDiscriminantAnalysis(), 'LDA'), (ax[1], QuadraticDiscriminantAnalysis(), 'QDA')):
    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    a.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.2)
    for idc in range(n_classes):
      a.scatter(X[y==idc,0], X[y==idc,1], color=colors[idc], label='class %d' %idc, alpha=0.6, s=10)
    a.set_title(f'{vista.upper()} - {nom} boundaries'); a.legend()
else:
  # (app) en 1D se dibuja un histograma por clase (el original dibuja X frente a X)
  fig, ax = plt.subplots(figsize=(8,4))
  for idc in range(n_classes):
    ax.hist(X[y==idc,0], bins=40, alpha=0.5, color=colors[idc], label='class %d' %idc)
  ax.set_title(f'{vista.upper()} - proyección 1D'); ax.legend()
plt.show()
`;

export const lab2_2: NotebookContent = {
  id: NB,
  lab: 'lab2',
  title: 'Lab 2 · Parte 2 — PCA frente a MDA en datos gaussianos sintéticos',
  short: 'PCA vs MDA (sintético)',
  origin: 'original',
  context: {
    goal: `Comparar dos formas de **reducir la dimensión** antes de clasificar: **PCA** (no supervisada, conserva varianza) y **MDA** (análisis discriminante de Fisher, supervisada, maximiza la separación entre clases). Con 3 clases gaussianas en 3D se entrenan LDA y QDA con 3, 2 y 1 dimensiones y se comparan los errores de **train y test**.

La guía llama a este proceso «feature selection by dimensionality reduction»: aquí no se elige un subconjunto de features, se construyen nuevas (combinaciones lineales) con una matriz W aprendida **solo con train**.

Preguntas del report (Q1–Q4): tablas de errores LC/QC para 3, 2 y 1 features con PCA y con MDA, a SNR = 10 dB y a 0 dB, y análisis de los scatter plots en 2D y 1D.

**Este notebook no tiene salidas guardadas**: los resultados proceden de la ejecución en la app.`,
    dataset: `Datos **generados**: 3 clases × 500 muestras (train) + otras 500 por clase (test), 3 features. Medias $(1,1,-1)$, $(-1,0,-1)$, $(-1,-1,0)$. Cada clase tiene una covarianza distinta, $C_i = V_i^T\\,\\mathrm{diag}(D_i)\\,V_i$, con autovalores $D_i \\propto \\sigma^2\\cdot$ (1, 1, 1), (0.1, 0.3, 0.6) y (2, 0.01, 0.99), y una rotación aleatoria $V_i$ (autovectores de $HH^T$ con $H$ normal). Semilla \`default_rng(seed=5)\`. La guía pide SNR = 10 dB y 0 dB.

**Aviso de reproducibilidad**: las muestras de la clase 0 (covarianza isótropa $\\sigma^2 I$) dependen de la librería de álgebra lineal: \`multivariate_normal\` factoriza la covarianza con SVD y, con autovalores repetidos, la base es arbitraria. Por eso los errores de tu entorno local y los de la app difieren en unas pocas muestras (las conclusiones son las mismas).`,
    variables: [
      { name: 'M_Means', meaning: 'Medias de las 3 clases.', shape: '(3, 3)' },
      { name: 'D', meaning: 'Autovalores (varianzas a lo largo de los ejes) de cada clase.', shape: '(3, 3)' },
      { name: 'M_covar', meaning: 'Covarianza de cada clase.', shape: '(3, 3, 3)' },
      { name: 'X_train, y_train / X_test, y_test', meaning: 'Train y test barajados.', shape: '(1500, 3), (1500,)' },
      { name: 'XX_train, XX_test', meaning: 'Copias de los datos originales (no se modifican después).', shape: '(1500, 3)' },
      { name: 'lda_error, qda_error', meaning: 'Array [error train, error test].', shape: '(2,)' },
      { name: 'lda_cmat, qda_cmat', meaning: 'Matrices de confusión de train y test apiladas.', shape: '(2, 3, 3)' },
      { name: 'X_train_pca2, X_train_pca1', meaning: 'Proyecciones PCA (2 y 1 componentes).', shape: '(1500, 2), (1500, 1)' },
      { name: 'X_train_mda2, X_train_mda1', meaning: 'Proyecciones MDA (2 y 1 direcciones discriminantes).', shape: '(1500, 2), (1500, 1)' },
    ],
    classes: [{ id: '0', label: 'clase 0' }, { id: '1', label: 'clase 1' }, { id: '2', label: 'clase 2' }],
    questions: [
      { id: 'Q1', text: 'Errores de train y test de LC y QC con 3, 2 y 1 features usando **PCA**, SNR = 10 dB; scatter 2D y 1D.', blocks: ['b_3d', 'b_pca2', 'b_pca1'], doc: 'lab2_guide' },
      { id: 'Q2', text: 'Lo mismo usando **MDA**, SNR = 10 dB, con la misma semilla.', blocks: ['b_mda2', 'b_mda1'], doc: 'lab2_guide' },
      { id: 'Q3', text: 'PCA con SNR = 0 dB.', blocks: ['b_params', 'b_pca2', 'b_pca1'], doc: 'lab2_guide' },
      { id: 'Q4', text: 'MDA con SNR = 0 dB.', blocks: ['b_params', 'b_mda2', 'b_mda1'], doc: 'lab2_guide' },
    ],
    notation: [
      { symbol: "$y = W^T x$, $W\\in\\mathbb R^{d\\times d'}$", meaning: "reducción lineal de d a d' features (slides Lab2, p. 4)", code: 'pca.transform(X) / mda.transform(X)' },
      { symbol: '$S_T = S_B + S_C$', meaning: 'dispersión total = entre clases + intra-clase (slide 5)', code: '—' },
      { symbol: 'PCA: $\\max \\sum_i w_i^T S_T w_i$', meaning: 'columnas de W = autovectores de $S_T$ de mayor autovalor', code: 'PCA(n_components)' },
      { symbol: 'MDA: $\\max |W^T S_B W| / |W^T S_C W|$', meaning: 'autovectores generalizados $S_B w = \\lambda S_C w$; $d\' \\le \\min(d, c-1)$', code: 'LinearDiscriminantAnalysis(n_components)' },
      { symbol: '$D_i$', meaning: 'varianzas de la clase i a lo largo de sus ejes', code: 'D[i]' },
    ],
  },
  flow: [
    { stage: 'contexto', label: 'Contexto', blocks: ['b_intro', 'b_imports'], summary: 'PCA vs MDA como reducción.' },
    { stage: 'datos', label: 'Datos', blocks: ['b_params', 'b_gen'], summary: '3 clases con covarianzas aleatorias; train y test.' },
    { stage: 'exploracion', label: 'Exploración', blocks: ['b_scatter3d', 'b_pairplot'], summary: 'Nubes en 3D y por parejas.' },
    { stage: 'metodo', label: 'Método', blocks: ['b_funcs', 'b_pca2', 'b_pca1', 'b_mda2', 'b_mda1'], summary: 'Clasificar en 3D, tras PCA y tras MDA.' },
    { stage: 'resultados', label: 'Resultados', blocks: ['b_3d', 'b_pca2_plot', 'b_pca1_plot', 'b_mda2_plot', 'b_mda1_plot'], summary: 'Errores y fronteras en cada dimensión.' },
    { stage: 'interpretacion', label: 'Interpretación', blocks: [], summary: 'Tablas Q1–Q4 en Experimentar e Interpretar.' },
  ],
  blocks: [
    {
      id: 'b_intro',
      title: 'Planteamiento: reducción por PCA y por MDA',
      cells: [0, 1],
      stage: 'contexto',
      problem: 'Saber cuántas features se pueden eliminar sin perder capacidad de clasificar, y con qué criterio.',
      purpose: 'Enmarca la parte 2: misma tarea (LDA y QDA) en 3, 2 y 1 dimensiones, con dos criterios para construir esas dimensiones.',
      inputs: 'Texto.',
      outputs: 'El plan del notebook.',
      prereqs: ['extraccion-features', 'pca', 'mda'],
      apis: [],
      lines: [],
      interpretation: 'El título dice «feature selection by dimensionality reduction»: en terminología estricta es **extracción** (se crean features nuevas), no selección de un subconjunto (eso se hace a mano en lab2_3).',
      next: 'Imports y datos.',
      whatIf: [],
      check: {
        kind: 'mcq', id: 'lab2_2.check.intro', skills: ['concepto'], source: src([1]),
        prompt: '¿Qué diferencia esencial hay entre PCA y MDA como reducción de dimensión?',
        options: [
          { text: 'PCA no usa las etiquetas (maximiza la varianza); MDA sí (maximiza la separación entre clases)', correct: true },
          { text: 'PCA es lineal y MDA es no lineal', feedback: 'Ambas son proyecciones lineales $y=W^Tx$.' },
          { text: 'MDA conserva más componentes que PCA', feedback: 'Al revés: MDA está limitado a min(d, c − 1).' },
        ],
        explanation: 'Slides Lab2, p. 6–8: PCA usa $S_T$; MDA usa $S_B$ y $S_C$, que requieren conocer la clase de cada muestra.',
      },
    },
    {
      id: 'b_imports',
      title: 'Importaciones y generador aleatorio',
      cells: [2],
      stage: 'contexto',
      problem: 'Disponer de las librerías y de una semilla.',
      purpose: 'Importa LDA, QDA, métricas y PCA, y crea `rng = default_rng(seed=5)`. Ese `rng` se usa primero para las rotaciones de las covarianzas y después para los datos: todo el notebook depende de su orden de uso.',
      inputs: 'Nada.',
      outputs: 'Alias y `rng`.',
      prereqs: ['semilla'],
      apis: [
        { name: 'PCA', lib: 'sklearn.decomposition', kind: 'clase', does: 'Reducción no supervisada.' },
        { name: 'LinearDiscriminantAnalysis', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Clasificador LDA y, con `transform`, reducción MDA.' },
        { name: 'from sklearn import metrics', lib: 'sklearn', kind: 'módulo', does: 'Se usa `metrics.confusion_matrix`.' },
      ],
      lines: [{ cell: 2, lines: [8, 9], text: '`roc_curve` y `scipy.stats as ss` se importan pero no se usan.' }],
      interpretation: 'No es necesario importar MDA aparte: en scikit-learn MDA es el método `transform` de `LinearDiscriminantAnalysis`.',
      next: 'Parámetros y covarianzas.',
      whatIf: [{ change: 'Crear `rng` de nuevo antes de generar el test', effect: 'Test y train saldrían de la misma secuencia aleatoria: serían idénticos.' }],
    },
    {
      id: 'b_params',
      title: 'SNR, medias y covarianzas aleatorias',
      cells: [3, 4],
      stage: 'datos',
      problem: 'Definir tres gaussianas con medias fijas y covarianzas distintas y orientadas al azar.',
      purpose: 'Igual que en Lab1, la SNR fija σ². Después, para cada clase, se elige una base ortonormal aleatoria (autovectores de $HH^T$) y se construye $C_i = V^T\\mathrm{diag}(D_i)V$: así cada clase tiene su forma y orientación.',
      inputs: 'SNR = 10 dB (la guía pide también 0 dB), semilla.',
      outputs: '`sig` = σ² ≈ 0.0778 (10 dB) o 0.778 (0 dB); `M_covar` (3, 3, 3).',
      prereqs: ['snr', 'covarianza', 'autovalor'],
      apis: [
        { name: 'rng.normal(size=(3,3))', lib: 'numpy.random.Generator', kind: 'método', does: 'Matriz aleatoria H.' },
        { name: 'np.linalg.eig', lib: 'numpy', kind: 'función', does: 'Autovalores y autovectores (columnas) de $HH^T$, simétrica definida positiva.', returns: '(autovalores, autovectores)' },
        { name: 'np.matmul', lib: 'numpy', kind: 'función', does: 'Producto matricial.' },
        { name: 'np.diag', lib: 'numpy', kind: 'función', does: 'Matriz diagonal con D[i].' },
      ],
      lines: [
        { cell: 4, lines: [10, 10], text: 'Medias a distancia ≈ 1.4–2.2 entre sí. `energy` = (3 + 2 + 2)/3 = 7/3.' },
        { cell: 4, lines: [19, 21], text: 'σ² = (7/3)/10/3 ≈ 0.0778 a 10 dB (0.778 a 0 dB).' },
        { cell: 4, lines: [24, 24], text: 'Varianzas por eje: clase 0 isótropa (1, 1, 1)·σ²; clase 1 pequeña (0.1, 0.3, 0.6)·σ²; clase 2 muy alargada (2, 0.01, 0.99)·σ².' },
        { cell: 4, lines: [27, 29], text: '`Hevec` tiene los autovectores en **columnas**; la fórmula usa `Hevec.T @ diag @ Hevec`, que también es una covarianza válida (rota con la base traspuesta). En NumPy 2.5 (tu entorno) `eig` devuelve tipo complejo y se descarta la parte imaginaria nula al asignar (aviso `ComplexWarning`).' },
      ],
      shapes: [{ expr: 'D', shape: '(3, 3)', meaning: 'fila = clase, columnas = varianzas por eje' }, { expr: 'M_covar', shape: '(3, 3, 3)', meaning: 'una covarianza por clase' }],
      interpretation: 'Como las covarianzas son distintas (Case 3 de Lab1), la frontera MAP óptima es cuadrática: QDA debería ganar a LDA, sobre todo con ruido alto (0 dB). La clase 0 es isótropa, así que su rotación no cambia nada.',
      corrections: [{ cell: 4, kind: 'concepto', original: 'Heval, Hevec = np.linalg.eig(np.matmul(H,H.T))', fixed: 'Heval, Hevec = np.linalg.eigh(np.matmul(H,H.T))   # matriz simétrica: autovalores reales y ordenados', why: '`eigh` es el método para matrices simétricas (resultado real y ordenado). La app conserva `eig` para reproducir el notebook y añade `np.real` al asignar.' }],
      next: 'Generación de train y test (celda 6).',
      whatIf: [
        { change: 'SNR = 0 dB', effect: 'σ² ×10: las nubes se solapan y las diferencias entre métodos aparecen (Q3–Q4).' },
        { change: 'Otra semilla', effect: 'Otras orientaciones de las covarianzas: cambia qué dirección elige PCA (la guía lo avisa).' },
      ],
    },
    {
      id: 'b_gen',
      title: 'Conjuntos de entrenamiento y test',
      cells: [5, 6],
      stage: 'datos',
      problem: 'Tener datos para entrenar y otros independientes para evaluar.',
      purpose: 'Mismo bucle de Lab1 para train y test; el barajado usa `rng.permutation` (reproducible). Se guardan copias `XX_train`, `XX_test`.',
      inputs: '`M_Means`, `M_covar`, `rng`.',
      outputs: '`X_train`, `y_train`, `X_test`, `y_test` (1500 muestras cada uno).',
      prereqs: ['train-test', 'gaussiana-multivariante'],
      apis: [
        { name: 'rng.multivariate_normal', lib: 'numpy.random.Generator', kind: 'método', does: 'Muestras gaussianas; internamente factoriza la covarianza con SVD.' },
        { name: 'rng.permutation', lib: 'numpy.random.Generator', kind: 'método', does: 'Permutación aleatoria reproducible.' },
      ],
      lines: [
        { cell: 6, lines: [11, 13], text: 'A diferencia de Lab1, se baraja con `rng`: el orden también es reproducible.' },
        { cell: 6, lines: [30, 31], text: '`XX_train = X_train` no copia el array: es otro nombre para el mismo objeto (con `.copy()` sería una copia real). Como nadie modifica X_train in situ, no causa problemas.' },
      ],
      shapes: [{ expr: 'X_train', shape: '(1500, 3)', meaning: '3 clases × 500' }],
      interpretation: 'Train y test siguen la misma distribución: las diferencias train/test medirán sobreajuste y variación muestral.',
      versionNotes: [{ cell: 6, text: 'Las muestras de la clase 0 (covarianza isótropa) cambian entre tu entorno local y la app: `multivariate_normal` usa una SVD cuya base es arbitraria cuando hay autovalores repetidos, y cada LAPACK elige una distinta. Las clases 1 y 2 coinciden. Los errores difieren en unas pocas muestras.' }],
      next: 'Exploración visual.',
      whatIf: [{ change: 'Ajustar PCA con X_test', effect: 'La proyección usaría información del test: evaluación sesgada (y distinta base para train y test).' }],
    },
    {
      id: 'b_scatter3d',
      title: 'Scatter 3D',
      cells: [7, 8],
      stage: 'exploracion',
      problem: 'Ver las tres nubes en el espacio original.',
      purpose: 'Comprobar formas: una esférica, una pequeña y una aplanada.',
      inputs: '`X_train`, `y_train`.',
      outputs: 'Figura 3D.',
      prereqs: ['scatter'],
      apis: [{ name: 'ax.scatter3D', lib: 'matplotlib', kind: 'método', does: 'Dispersión 3D.' }],
      lines: [],
      interpretation: 'A 10 dB las clases están bien separadas en 3D; la pregunta es si siguen separadas al proyectar.',
      next: 'Pairplot.',
      whatIf: [],
    },
    {
      id: 'b_pairplot',
      title: 'Pairplot de las features originales',
      cells: [9],
      stage: 'exploracion',
      problem: 'Ver las parejas de features.',
      purpose: 'Las proyecciones sobre los ejes originales son un caso particular de reducción (quedarse con 2 features): ninguna pareja tiene por qué ser la mejor.',
      inputs: '`X_train`, `y_train`.',
      outputs: 'Matriz 3×3 de gráficos.',
      prereqs: ['pairplot', 'dataframe'],
      apis: [{ name: 'sns.pairplot', lib: 'seaborn', kind: 'función', does: 'Matriz de dispersión.', args: [{ name: 'hue', value: "'class'", meaning: 'color por clase' }, { name: 'palette', value: "'deep'", meaning: 'paleta de seaborn' }] }],
      lines: [{ cell: 9, lines: [7, 7], text: '`Xyt.head()` no se muestra: no es la última expresión de la celda.' }],
      interpretation: 'Algunas parejas solapan clases que en 3D estaban separadas: perder dimensiones puede costar separabilidad.',
      next: 'Funciones de clasificación.',
      whatIf: [],
    },
    {
      id: 'b_funcs',
      title: 'Funciones para entrenar y evaluar LDA y QDA',
      cells: [10, 11],
      stage: 'metodo',
      problem: 'Repetir el mismo entrenamiento/evaluación con distintas representaciones sin copiar código.',
      purpose: 'Cada función entrena con train y devuelve el modelo, `[error train, error test]` y las dos matrices de confusión. Se llamará con X en 3D, PCA 2D/1D y MDA 2D/1D.',
      inputs: 'X_train, y_train, X_test, y_test (cualquier dimensión).',
      outputs: '(modelo, errores (2,), matrices (2, 3, 3)).',
      prereqs: ['lda', 'qda', 'matriz-confusion'],
      apis: [
        { name: 'LinearDiscriminantAnalysis(solver="svd", store_covariance=True)', lib: 'sklearn', kind: 'clase', does: 'Clasificador lineal.' },
        { name: 'QuadraticDiscriminantAnalysis(store_covariance=True)', lib: 'sklearn', kind: 'clase', does: 'Clasificador cuadrático.' },
        { name: 'metrics.confusion_matrix', lib: 'sklearn.metrics', kind: 'función', does: 'Filas = reales, columnas = predichas.' },
      ],
      lines: [
        { cell: 11, lines: [15, 16], text: 'Se empaquetan los dos errores y las dos matrices: `lda_error[0]` = train, `lda_error[1]` = test.' },
        { cell: 11, lines: [1, 1], text: 'Los parámetros se llaman igual que las variables globales: dentro de la función mandan los argumentos.' },
      ],
      shapes: [{ expr: 'lda_cmat', shape: '(2, 3, 3)', meaning: '[train, test] × reales × predichas' }],
      interpretation: 'Encapsular evita errores al repetir el experimento 5 veces y deja claro que **siempre** se entrena con train y se evalúa en ambos.',
      next: 'Clasificación en 3D (referencia).',
      whatIf: [{ change: 'Entrenar dentro de la función con X_test', effect: 'El «error de test» sería de entrenamiento: optimista.' }],
    },
    {
      id: 'b_3d',
      title: 'Referencia: LDA y QDA con las 3 features',
      cells: [12, 13],
      stage: 'resultados',
      problem: 'Saber el error sin reducir dimensión.',
      purpose: 'Línea base de las tablas Q1–Q4.',
      inputs: 'Datos 3D.',
      outputs: 'Ejecución de la app, 10 dB: LDA 0.00 % / 0.07 %, QDA 0.00 % / 0.00 % (train/test). 0 dB: LDA 11.4 % / 12.7 %, QDA 7.1 % / 6.1 %.',
      prereqs: ['lda', 'qda'],
      apis: [{ name: 'lda_classifier / qda_classifier', lib: 'notebook', kind: 'función', does: 'Definidas en la celda 11.' }],
      lines: [{ cell: 13, lines: [2, 2], text: 'Se sobrescribe `lda` con el modelo devuelto: las celdas de fronteras usan siempre el último modelo entrenado.' }],
      interpretation: 'A 0 dB QDA gana claramente: las covarianzas distintas hacen que la frontera óptima sea cuadrática.',
      next: 'PCA a 2D.',
      whatIf: [{ change: 'SNR = 0 dB', effect: 'Los errores suben a ≈ 6–13 % y la ventaja de QDA se hace visible.' }],
    },
    {
      id: 'b_pca2',
      title: 'PCA a 2 dimensiones y clasificación',
      cells: [14, 15],
      stage: 'metodo',
      problem: 'Quitar una dimensión conservando la máxima varianza.',
      purpose: 'Ajusta PCA **con train**, transforma train y test con la misma W y repite la clasificación.',
      inputs: '`X_train`, `X_test`.',
      outputs: '`X_train_pca2`, `X_test_pca2` (1500, 2); proporción de varianza ≈ [0.84, 0.14] (10 dB); errores LDA 0.20 % / 0.27 %, QDA 0.00 % / 0.07 %.',
      prereqs: ['pca', 'varianza-explicada'],
      apis: [
        { name: 'PCA(n_components=2).fit(X_train)', lib: 'sklearn', kind: 'método', does: 'Aprende media y componentes solo con train.' },
        { name: 'pca.transform(X_test)', lib: 'sklearn', kind: 'método', does: 'Aplica la misma W al test.' },
      ],
      lines: [
        { cell: 15, lines: [3, 5], text: 'Patrón clave de las slides (p. 4 y 10): `fit` con train, `transform` en train y en test.' },
        { cell: 15, lines: [9, 10], text: 'Proporción de varianza **total** retenida por cada componente; no dice nada de la separabilidad.' },
      ],
      shapes: [{ expr: 'X_train_pca2', shape: '(1500, 2)', meaning: 'dos componentes por muestra' }],
      interpretation: 'A 10 dB las clases siguen separadas en 2D: perder la tercera dirección (≈ 3 % de la varianza) casi no cuesta errores. A 0 dB sí empeora (LDA 15.2 % / 16.5 %).',
      next: 'Fronteras en el plano PCA.',
      whatIf: [{ change: '`pca.fit(X_test)` o `fit_transform` separado para test', effect: 'Train y test quedarían en bases distintas: el clasificador recibiría coordenadas incoherentes.' }],
    },
    {
      id: 'b_pca2_plot',
      title: 'Fronteras en el plano PCA',
      cells: [16, 17],
      stage: 'resultados',
      problem: 'Ver cómo separan LDA y QDA en 2D.',
      purpose: 'Clasifica una malla de puntos (`meshgrid`) con los últimos modelos `lda` y `qda` y la pinta con `contourf`, superponiendo las muestras de train.',
      inputs: '`X_train_pca2`, `y_train`, `lda`, `qda` (entrenados en PCA 2D).',
      outputs: 'Dos paneles: regiones lineales (LDA) y cuadráticas (QDA).',
      prereqs: ['frontera', 'meshgrid'],
      apis: [
        { name: 'np.meshgrid', lib: 'numpy', kind: 'función', does: 'Rejilla de coordenadas.' },
        { name: 'np.c_[xx.ravel(), yy.ravel()]', lib: 'numpy', kind: 'operador', does: 'Matriz (n_puntos, 2) para `predict`.' },
        { name: 'ax.contourf', lib: 'matplotlib', kind: 'método', does: 'Rellena las regiones por clase predicha.' },
      ],
      lines: [
        { cell: 17, lines: [5, 5], text: '`h = .01`: resolución de la malla (solo afecta al dibujo).' },
        { cell: 17, lines: [17, 19], text: 'Se predice cada punto de la malla y se devuelve a la forma de la rejilla.' },
      ],
      interpretation: 'LDA dibuja rectas; QDA curvas que se adaptan a la forma alargada de la clase 2.',
      next: 'PCA a 1 dimensión.',
      whatIf: [{ change: 'Reducir `h` a 0.001', effect: '100 veces más puntos: mucha más memoria y tiempo para el mismo resultado (lo mismo ocurrió en lab1_2).' }],
    },
    {
      id: 'b_pca1',
      title: 'PCA a 1 dimensión',
      cells: [18, 19],
      stage: 'metodo',
      problem: 'Quedarse solo con la dirección de máxima varianza.',
      purpose: 'Igual que la celda 15 con `n_components=1`.',
      inputs: 'Datos 3D.',
      outputs: 'Proporción ≈ 0.84 (10 dB); errores LDA 2.6 % / 4.0 %, QDA 2.3 % / 3.2 %. A 0 dB: ≈ 24–25 %.',
      prereqs: ['pca'],
      apis: [{ name: 'PCA(n_components=1)', lib: 'sklearn', kind: 'clase', does: 'Una componente.' }],
      lines: [{ cell: 19, lines: [9, 10], text: '**Errata**: el texto dice «first two components» pero se imprime una sola.' }],
      interpretation: 'Con 1 componente el error crece mucho: la dirección de máxima varianza no es la que mejor separa las 3 clases.',
      corrections: [{ cell: 19, kind: 'errata', original: "print('explained variance ratio (first two components): %s'", fixed: "print('explained variance ratio (first component): %s'", why: 'Etiqueta copiada de la celda 15 (también en las celdas 23 y 27 para MDA).' }],
      next: 'Gráfico 1D.',
      whatIf: [],
    },
    {
      id: 'b_pca1_plot',
      title: '«Scatter» de la proyección PCA 1D',
      cells: [20, 21],
      stage: 'resultados',
      problem: 'Ver cómo se solapan las clases en una dimensión.',
      purpose: 'El original dibuja `X[:,0]` frente a `X[:,0]`: todos los puntos caen sobre la diagonal y el solapamiento solo se ve por los colores.',
      inputs: '`X_train_pca1`, `y_train`.',
      outputs: 'Diagonal coloreada.',
      prereqs: ['proyeccion'],
      apis: [{ name: 'ax.scatter(X[idx,0], X[idx,0])', lib: 'matplotlib', kind: 'método', does: 'Proyección frente a sí misma.' }],
      lines: [
        { cell: 21, lines: [10, 12], text: 'Malla construida con la misma variable en ambos ejes; reutiliza `h` de la celda 17 (si no se ejecuta antes: `NameError`).' },
        { cell: 21, lines: [18, 18], text: '`X[idx,0], X[idx,0]`: el «scatter 1D» es la diagonal.' },
      ],
      interpretation: 'Para analizar el solapamiento en 1D es más claro un histograma por clase (el experimento de la app lo dibuja así).',
      corrections: [{ cell: 21, kind: 'concepto', original: 'ax.scatter(X[idx,0], X[idx,0], ...)', fixed: "ax.hist(X[idx,0], bins=40, alpha=0.5, label='class %d' % idc)", why: 'Un histograma por clase muestra directamente cuánto se solapan las proyecciones.' }],
      next: 'MDA a 2D.',
      whatIf: [],
    },
    {
      id: 'b_mda2',
      title: 'MDA a 2 dimensiones (LDA.transform)',
      cells: [22, 23],
      stage: 'metodo',
      problem: 'Reducir a 2 dimensiones maximizando la separación entre clases.',
      purpose: 'En scikit-learn, MDA es `LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train).transform(X)`. **Necesita y_train**.',
      inputs: '`X_train`, `y_train`, `X_test`.',
      outputs: '`X_train_mda2`, `X_test_mda2` (1500, 2); `explained_variance_ratio_` ≈ [0.86, 0.14]; errores LDA = los de 3D (0.00 % / 0.07 %), QDA 0.00 % / 0.00 %.',
      prereqs: ['mda', 'scatter-matrix'],
      apis: [{
        name: 'LinearDiscriminantAnalysis(n_components=2)', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Direcciones discriminantes de Fisher; `transform` proyecta sobre ellas.',
        args: [{ name: 'n_components', value: '2', meaning: 'como máximo min(n_features, n_classes − 1) = 2' }],
        learned: ['scalings_', 'explained_variance_ratio_', 'means_', 'xbar_'],
      }],
      lines: [
        { cell: 23, lines: [3, 3], text: 'A diferencia de PCA, `fit` recibe también `y_train`.' },
        { cell: 23, lines: [9, 10], text: 'En MDA, `explained_variance_ratio_` es la proporción de **separación entre clases** (autovalores de $S_C^{-1}S_B$), no de varianza total.' },
      ],
      shapes: [{ expr: 'X_train_mda2', shape: '(1500, 2)', meaning: 'dos direcciones discriminantes' }],
      interpretation: 'Con c = 3 clases, MDA solo puede dar 2 direcciones, y con ellas LDA obtiene **exactamente** los mismos errores que en 3D: toda la información que usa LDA (las medias en la métrica de la covarianza común) vive en ese subespacio de dimensión c − 1. QDA, en cambio, puede perder algo (a 0 dB: 6.1 % en 3D frente a 12.8 % en MDA 2D) porque la tercera dirección contenía diferencias de covarianza.',
      next: 'Fronteras en el plano MDA.',
      whatIf: [{ change: '`n_components=3`', effect: '`ValueError`: n_components no puede superar min(n_features, n_classes − 1) = 2.' }],
    },
    {
      id: 'b_mda2_plot',
      title: 'Fronteras en el plano MDA',
      cells: [24, 25],
      stage: 'resultados',
      problem: 'Ver la separación tras MDA.',
      purpose: 'Mismo código que la celda 17 sobre `X_train_mda2`.',
      inputs: '`X_train_mda2`, modelos entrenados en MDA 2D.',
      outputs: 'Dos paneles con regiones.',
      prereqs: ['frontera', 'meshgrid'],
      apis: [{ name: 'np.meshgrid / contourf', lib: 'numpy / matplotlib', kind: 'función', does: 'Como en la celda 17.' }],
      lines: [{ cell: 25, lines: [12, 15], text: 'Las coordenadas MDA están escaladas para que la dispersión intra-clase sea ≈ 1: la malla cubre un rango mayor que en PCA y tiene más puntos con el mismo `h`.' }],
      interpretation: 'Las tres clases aparecen bien separadas y casi esféricas: MDA «blanquea» la dispersión intra-clase.',
      next: 'MDA a 1D.',
      whatIf: [],
    },
    {
      id: 'b_mda1',
      title: 'MDA a 1 dimensión',
      cells: [26, 27],
      stage: 'metodo',
      problem: 'Quedarse con la mejor dirección discriminante.',
      purpose: 'Igual que la celda 23 con `n_components=1`.',
      inputs: 'Datos 3D y etiquetas.',
      outputs: '10 dB: LDA 0.33 % / 1.07 %, QDA 0.33 % / 0.67 % (mucho mejor que PCA 1D: 2.6 % / 4.0 %). 0 dB: ≈ 22 %.',
      prereqs: ['mda'],
      apis: [{ name: 'LinearDiscriminantAnalysis(n_components=1)', lib: 'sklearn', kind: 'clase', does: 'Una dirección discriminante.' }],
      lines: [{ cell: 27, lines: [9, 10], text: 'Misma errata «first two components».' }],
      interpretation: 'En 1D, MDA conserva mucha más capacidad de clasificar que PCA porque elige la dirección con más separación relativa entre clases, no la de más varianza.',
      next: 'Gráfico 1D.',
      whatIf: [],
    },
    {
      id: 'b_mda1_plot',
      title: '«Scatter» de la proyección MDA 1D',
      cells: [28, 29],
      stage: 'resultados',
      problem: 'Ver el solapamiento en la dirección discriminante.',
      purpose: 'Mismo gráfico diagonal que la celda 21.',
      inputs: '`X_train_mda1`.',
      outputs: 'Diagonal coloreada.',
      prereqs: ['proyeccion'],
      apis: [{ name: 'ax.scatter', lib: 'matplotlib', kind: 'método', does: 'Proyección frente a sí misma.' }],
      lines: [{ cell: 29, lines: [6, 6], text: 'Los colores cambian de orden (`g, b, r` frente a `r, g, b` en las demás figuras): la clase 0 sale verde aquí y roja en las otras. Cuidado al comparar figuras.' }],
      interpretation: 'Comparada con PCA 1D, las clases quedan más separadas a lo largo de la diagonal.',
      whatIf: [],
    },
  ],
  experiments: [
    {
      id: 'lab2_2.exp.reduce',
      title: 'Tabla Q1–Q4: LDA y QDA con 3, 2 y 1 dimensiones (PCA y MDA)',
      goal: 'Rellenar de una vez las cuatro tablas del report: para la SNR elegida se entrenan LDA y QDA en 3D, PCA 2D/1D y MDA 2D/1D, con la misma semilla para que todas las filas usen exactamente los mismos datos. La vista solo cambia la figura.',
      blocks: ['b_params', 'b_gen', 'b_3d', 'b_pca2', 'b_pca1', 'b_mda2', 'b_mda1', 'b_pca2_plot', 'b_mda2_plot'],
      template: EXPERIMENT,
      seedParam: 'seed',
      params: [
        { id: 'SNR', label: 'SNR (dB)', kind: 'datos', control: 'select', default: '10', options: [{ value: '10', label: '10 dB (Q1, Q2)' }, { value: '5', label: '5 dB' }, { value: '0', label: '0 dB (Q3, Q4)' }, { value: '-5', label: '−5 dB' }], variable: 'SNR', computation: 'σ² = energía / SNR / 3: escala todas las covarianzas.', result: 'A menor SNR, más error en todas las filas y más diferencia entre métodos (y entre LDA y QDA).', why: 'Con más ruido las clases se solapan y cada dimensión descartada pesa más.' },
        { id: 'seed', label: 'Semilla', kind: 'datos', control: 'range', min: 0, max: 40, step: 1, default: '5', variable: 'default_rng(seed=…)', computation: 'Otras rotaciones de las covarianzas y otras muestras.', result: 'Cambia qué dirección elige PCA y por tanto cuánto empeora con 1–2 componentes; MDA es más estable.', why: 'La guía avisa: la semilla cambia los autovectores de las covarianzas de clase.' },
        { id: 'n_samples', label: 'Muestras por clase', kind: 'datos', control: 'select', default: '500', options: [{ value: '30', label: '30' }, { value: '100', label: '100' }, { value: '500', label: '500 (original)' }, { value: '2000', label: '2000' }], variable: 'n_samples', computation: 'Tamaño de train y de test.', result: 'Con pocas muestras crece la diferencia train/test, sobre todo en QDA (estima 3 covarianzas).', why: 'Más parámetros con los mismos datos → más varianza de estimación.' },
        { id: 'vista', label: 'Figura', kind: 'visualizacion', control: 'select', default: "'pca2'", options: [{ value: "'pca2'", label: 'PCA 2D (celda 17)' }, { value: "'mda2'", label: 'MDA 2D (celda 25)' }, { value: "'pca1'", label: 'PCA 1D (celda 21)' }, { value: "'mda1'", label: 'MDA 1D (celda 29)' }], variable: 'vista', computation: 'Ninguno: elige qué proyección se dibuja.', result: 'Los errores de la tabla no cambian.', why: 'Es un parámetro de visualización.' },
        { id: 'h', label: 'Paso de la malla', kind: 'visualizacion', control: 'select', default: '.02', options: [{ value: '.01', label: '0.01 (original)' }, { value: '.02', label: '0.02' }, { value: '.05', label: '0.05' }], variable: 'h', computation: 'Resolución de la rejilla de fondo.', result: 'Solo cambia la nitidez de las regiones y el tiempo de dibujo.', why: 'La malla no interviene en el entrenamiento ni en los errores.' },
      ],
      metrics: [
        { key: 'lda_3d_train', label: 'LDA 3D · train', format: 'pct', better: 'low' },
        { key: 'lda_3d_test', label: 'LDA 3D · test', format: 'pct', better: 'low' },
        { key: 'qda_3d_train', label: 'QDA 3D · train', format: 'pct', better: 'low' },
        { key: 'qda_3d_test', label: 'QDA 3D · test', format: 'pct', better: 'low' },
        { key: 'lda_pca2_train', label: 'LDA PCA-2 · train', format: 'pct', better: 'low' },
        { key: 'lda_pca2_test', label: 'LDA PCA-2 · test', format: 'pct', better: 'low' },
        { key: 'qda_pca2_train', label: 'QDA PCA-2 · train', format: 'pct', better: 'low' },
        { key: 'qda_pca2_test', label: 'QDA PCA-2 · test', format: 'pct', better: 'low' },
        { key: 'lda_pca1_train', label: 'LDA PCA-1 · train', format: 'pct', better: 'low' },
        { key: 'lda_pca1_test', label: 'LDA PCA-1 · test', format: 'pct', better: 'low' },
        { key: 'qda_pca1_train', label: 'QDA PCA-1 · train', format: 'pct', better: 'low' },
        { key: 'qda_pca1_test', label: 'QDA PCA-1 · test', format: 'pct', better: 'low' },
        { key: 'lda_mda2_train', label: 'LDA MDA-2 · train', format: 'pct', better: 'low' },
        { key: 'lda_mda2_test', label: 'LDA MDA-2 · test', format: 'pct', better: 'low' },
        { key: 'qda_mda2_train', label: 'QDA MDA-2 · train', format: 'pct', better: 'low' },
        { key: 'qda_mda2_test', label: 'QDA MDA-2 · test', format: 'pct', better: 'low' },
        { key: 'lda_mda1_train', label: 'LDA MDA-1 · train', format: 'pct', better: 'low' },
        { key: 'lda_mda1_test', label: 'LDA MDA-1 · test', format: 'pct', better: 'low' },
        { key: 'qda_mda1_train', label: 'QDA MDA-1 · train', format: 'pct', better: 'low' },
        { key: 'qda_mda1_test', label: 'QDA MDA-1 · test', format: 'pct', better: 'low' },
        { key: 'pca2_ratio', label: 'Varianza retenida PCA-2', format: 'pct' },
        { key: 'pca1_ratio', label: 'Varianza retenida PCA-1', format: 'pct' },
        { key: 'mda1_ratio', label: 'Separación retenida MDA-1', format: 'pct' },
      ],
      observe: [
        'Ejecuta 10 dB y 0 dB y guarda ambas: son las tablas de Q1–Q4.',
        'Compara «LDA 3D» con «LDA MDA-2»: ¿por qué son idénticos? ¿Pasa lo mismo con QDA?',
        'En 1D, compara PCA y MDA: ¿qué método conserva más capacidad de clasificar? ¿Coincide con la varianza retenida?',
        'Cambia la semilla varias veces y observa cuánto varía el error de PCA-1 frente al de MDA-1.',
        'Cambia la figura entre PCA 2D y MDA 2D sin volver a cambiar nada más: la tabla no se mueve.',
      ],
      source: src([2, 4, 6, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29], { note: 'Las 5 configuraciones se calculan en un único bucle; histogramas 1D, colores fijos y `np.real` son añadidos de la app. Sin salidas guardadas en el notebook.' }),
      fidelity: [
        { metrics: { lda_3d_train: 0, lda_3d_test: 0.000666667, qda_3d_test: 0, lda_pca2_train: 0.002, lda_pca2_test: 0.002666667, lda_pca1_train: 0.026, lda_pca1_test: 0.04, qda_pca1_test: 0.032, lda_mda2_test: 0.000666667, lda_mda1_train: 0.003333333, lda_mda1_test: 0.010666667, qda_mda1_test: 0.006666667 }, tol: 1e-6, source: 'Ejecución celda a celda del notebook en la app, 10 dB (tu entorno local difiere en la clase 0; ver la nota de la celda 6)' },
        { params: { SNR: '0' }, metrics: { lda_3d_train: 0.114, lda_3d_test: 0.127333333, qda_3d_train: 0.070666667, qda_3d_test: 0.061333333, lda_pca2_test: 0.164666667, qda_pca2_test: 0.15, lda_mda2_test: 0.127333333, qda_mda2_test: 0.128, lda_pca1_test: 0.242, lda_mda1_test: 0.225333333 }, tol: 1e-6, source: 'Ejecución celda a celda del notebook en la app, 0 dB' },
      ],
    },
  ],
  activities: [
    // ---- nivel 1
    {
      kind: 'mcq', id: 'lab2_2.read.func', skills: ['codigo'], blocks: ['b_funcs'], source: src([11]),
      prompt: 'Tras `lda, lda_error, lda_cmat = lda_classifier(X_train_pca2, y_train, X_test_pca2, y_test)`, ¿qué es `lda_error[1]`?',
      options: [
        { text: 'El error de test del LDA entrenado sobre la proyección PCA 2D', correct: true },
        { text: 'El error de train', feedback: '`lda_error[0]` es el de train.' },
        { text: 'La accuracy de test', feedback: 'La función devuelve 1 − accuracy.' },
        { text: 'El error de la clase 1', feedback: 'Es un error global, no por clase.' },
      ],
      explanation: '`lda_error = np.array([lda_train_error, lda_test_error])`.',
    },
    {
      kind: 'text', id: 'lab2_2.read.fit_train', skills: ['concepto', 'codigo'], blocks: ['b_pca2'], source: { notebook: NB, cells: [15], doc: 'lab2_slides', section: 'p. 4', origin: 'original' },
      prompt: 'En la celda 15 se hace `pca.fit(X_train)` y después `pca.transform(X_train)` y `pca.transform(X_test)`. Explica por qué no se ajusta PCA con todos los datos o por separado con el test.',
      code: 'pca = PCA(n_components=2)\npca.fit(X_train)\nX_train_pca2 = pca.transform(X_train)\nX_test_pca2  = pca.transform(X_test)',
      rubric: [
        'La matriz de reducción W forma parte del modelo y debe aprenderse solo con train (slides p. 4).',
        'Usar el test para ajustarla filtra información del test y sesga la evaluación.',
        'Ajustar por separado con el test daría otra base: las coordenadas de test no serían comparables con las de train.',
        'El mismo pca.transform garantiza que train y test usan la misma media y los mismos ejes.',
      ],
      model: 'La proyección W (y la media que se resta) es parte del modelo: se aprende solo con train, igual que los parámetros del clasificador, y después se aplica sin cambios al test. Si se ajustara con todos los datos, el test influiría en el modelo y el error de test dejaría de ser una estimación honesta. Si se ajustara por separado con el test, los ejes (y hasta su signo) serían otros, así que el clasificador entrenado en las coordenadas de train recibiría coordenadas incoherentes.',
    },
    // ---- nivel 2
    {
      kind: 'predict', id: 'lab2_2.pred.shape_mda', skills: ['codigo'], blocks: ['b_mda2'], source: src([23]),
      prompt: '¿Qué forma tiene `X_test_mda2`?', setup: lab2_2Setup(10),
      code: 'mda = LinearDiscriminantAnalysis(n_components=2)\nmda.fit(X_train, y_train)\nX_test_mda2 = mda.transform(X_test)', expr: 'X_test_mda2.shape', answerKind: 'shape',
      explanation: '1500 muestras de test proyectadas sobre 2 direcciones discriminantes.',
    },
    {
      kind: 'predict', id: 'lab2_2.pred.mda3', skills: ['api', 'concepto'], blocks: ['b_mda2'], source: src([23]),
      prompt: '¿Qué ocurre al ejecutar `LinearDiscriminantAnalysis(n_components=3).fit(X_train, y_train)` con 3 features y 3 clases?', setup: lab2_2Setup(10),
      code: "try:\n    LinearDiscriminantAnalysis(n_components=3).fit(X_train, y_train)\n    res = 'funciona con 3 componentes'\nexcept ValueError:\n    res = 'ValueError'",
      expr: 'res', answerKind: 'mcq', options: ['ValueError', 'funciona con 3 componentes'],
      explanation: 'MDA admite como máximo min(d, c − 1) = min(3, 2) = 2 componentes: $S_B$ tiene rango c − 1.',
    },
    {
      kind: 'predict', id: 'lab2_2.pred.sig', skills: ['concepto'], blocks: ['b_params'], source: src([4]),
      prompt: 'Con las medias de la celda 4 y SNR = 10 dB, ¿cuánto vale `sig`? (4 decimales)',
      setup: 'import numpy as np\nM_Means = np.array([[1, 1, -1], [-1, 0, -1], [-1, -1, 0]])\nn_classes, n_feat = 3, 3\nSNR = 10',
      code: 'energy = 0\nfor i in range(0,n_classes):\n  energy = energy + np.dot(M_Means[i],M_Means[i])\nenergy = energy / n_classes\nSNR = 10 ** (SNR/10)\nsig = energy / SNR\nsig = sig / n_feat',
      expr: 'sig', answerKind: 'numeric', tolerance: 0.0005,
      explanation: 'energía = (3 + 2 + 2)/3 = 7/3; σ² = (7/3)/10/3 ≈ 0.0778.',
    },
    // ---- nivel 3
    {
      kind: 'code', level: 3, id: 'lab2_2.fill.pca', title: 'Completa la reducción PCA a 2D y la clasificación', skills: ['api', 'codigo'], blocks: ['b_pca2', 'b_funcs'], source: src([15]),
      prompt: 'Rellena los huecos: PCA a 2 componentes aprendida con train, proyección de train y test, y errores de LDA.',
      available: '`X_train`, `y_train`, `X_test`, `y_test` (10 dB), `PCA`, `lda_classifier` (celda 11).',
      produce: '`X_train_pca2`, `X_test_pca2` (1500, 2) y `lda_error` (2,).',
      setup: lab2_2Setup(10),
      starter: 'pca = PCA(n_components=___)\npca.fit(___)\nX_train_pca2 = pca.___(X_train)\nX_test_pca2 = pca.___(X_test)\nlda, lda_error, lda_cmat = lda_classifier(___, y_train, ___, y_test)',
      solution: 'pca = PCA(n_components=2)\npca.fit(X_train)\nX_train_pca2 = pca.transform(X_train)\nX_test_pca2 = pca.transform(X_test)\nlda, lda_error, lda_cmat = lda_classifier(X_train_pca2, y_train, X_test_pca2, y_test)',
      alternatives: ['pca = PCA(2)\nX_train_pca2 = pca.fit_transform(X_train)\nX_test_pca2 = pca.transform(X_test)\nlda, lda_error, lda_cmat = lda_classifier(X_train_pca2, y_train, X_test_pca2, y_test)'],
      wrong: [{ code: 'pca = PCA(2)\nX_train_pca2 = pca.fit_transform(X_train)\nX_test_pca2 = PCA(2).fit_transform(X_test)\nlda, lda_error, lda_cmat = lda_classifier(X_train_pca2, y_train, X_test_pca2, y_test)', expect: 'razonamiento' }],
      checks: "c.same('X_train_pca2', rtol=1e-8, atol=1e-10)\nc.same('X_test_pca2', rtol=1e-8, atol=1e-10, desc='La proyección del test (con la PCA de train)')\nc.same('lda_error', rtol=1e-9)",
      hints: ['`fit` solo con `X_train`.', 'Para el test, solo `transform`.', 'La función recibe primero los datos de train proyectados y después los de test.'],
    },
    {
      kind: 'code', level: 3, id: 'lab2_2.fill.mda', title: 'Completa la proyección MDA', skills: ['api'], blocks: ['b_mda2'], source: src([23]),
      practice: 'LinearDiscriminantAnalysis.transform',
      prompt: 'Rellena los huecos para proyectar train y test sobre 2 direcciones discriminantes.',
      available: '`X_train`, `y_train`, `X_test`, `LinearDiscriminantAnalysis`.',
      produce: '`X_train_mda2`, `X_test_mda2` (1500, 2).',
      setup: lab2_2Setup(10),
      starter: 'mda = LinearDiscriminantAnalysis(n_components=___)\nmda.fit(___, ___)\nX_train_mda2 = mda.___(X_train)\nX_test_mda2 = mda.___(X_test)',
      solution: 'mda = LinearDiscriminantAnalysis(n_components=2)\nmda.fit(X_train, y_train)\nX_train_mda2 = mda.transform(X_train)\nX_test_mda2 = mda.transform(X_test)',
      alternatives: ['mda = LinearDiscriminantAnalysis(n_components=2)\nX_train_mda2 = mda.fit_transform(X_train, y_train)\nX_test_mda2 = mda.transform(X_test)'],
      wrong: [{ code: 'mda = LinearDiscriminantAnalysis(n_components=2)\nmda.fit(X_train)\nX_train_mda2 = mda.transform(X_train)\nX_test_mda2 = mda.transform(X_test)', expect: 'api' }],
      checks: "c.same('X_train_mda2', rtol=1e-8, atol=1e-10)\nc.same('X_test_mda2', rtol=1e-8, atol=1e-10)",
      hints: ['Máximo min(d, c − 1) = 2 componentes.', 'MDA es supervisado: `fit` necesita también `y_train`.'],
    },
    // ---- nivel 4
    {
      kind: 'code', level: 4, id: 'lab2_2.fix.leak', title: 'Corrige la fuga de información del test', skills: ['concepto', 'metodo'], blocks: ['b_pca1', 'b_gen'], source: src([19]),
      prompt: 'Este código pretende reproducir la celda 19 (PCA 1D + QDA), pero ajusta la proyección con **todos** los datos. Corrígelo para que la evaluación sea honesta.',
      available: '`np`, `X_train`, `y_train`, `X_test`, `y_test` (10 dB), `PCA`, `qda_classifier`.',
      produce: '`X_train_pca1`, `X_test_pca1` (1500, 1) y `qda_error` (2,).',
      setup: lab2_2Setup(10),
      starter: 'pca = PCA(n_components=1)\npca.fit(np.vstack([X_train, X_test]))\nX_train_pca1 = pca.transform(X_train)\nX_test_pca1 = pca.transform(X_test)\nqda, qda_error, qda_cmat = qda_classifier(X_train_pca1, y_train, X_test_pca1, y_test)',
      solution: 'pca = PCA(n_components=1)\npca.fit(X_train)\nX_train_pca1 = pca.transform(X_train)\nX_test_pca1 = pca.transform(X_test)\nqda, qda_error, qda_cmat = qda_classifier(X_train_pca1, y_train, X_test_pca1, y_test)',
      checks: "c.same('X_train_pca1', rtol=1e-8, atol=1e-10)\nc.same('X_test_pca1', rtol=1e-8, atol=1e-10)\nc.same('qda_error', rtol=1e-9)",
      hints: ['¿Con qué datos se aprende la matriz W según las slides (p. 4)?', 'Solo `X_train` en `fit`.'],
      explanation: 'Aunque aquí el error cambie poco, ajustar con el test es un error metodológico: el test deja de ser independiente.',
    },
    {
      kind: 'code', level: 4, id: 'lab2_2.fix.ncomp', title: 'Corrige la llamada a MDA', skills: ['api', 'concepto'], blocks: ['b_mda2'], source: src([23]),
      prompt: 'Alguien intenta conservar las 3 dimensiones con MDA y obtiene un error. Corrige el código para usar el **máximo** número de direcciones discriminantes permitido y calcula los errores de LDA.',
      available: '`X_train`, `y_train`, `X_test`, `y_test`, `LinearDiscriminantAnalysis`, `lda_classifier`.',
      produce: '`X_train_mda`, `X_test_mda` y `lda_error` (2,).',
      setup: lab2_2Setup(10),
      starter: 'mda = LinearDiscriminantAnalysis(n_components=3)\nmda.fit(X_train, y_train)\nX_train_mda = mda.transform(X_train)\nX_test_mda = mda.transform(X_test)\nlda, lda_error, lda_cmat = lda_classifier(X_train_mda, y_train, X_test_mda, y_test)',
      solution: 'mda = LinearDiscriminantAnalysis(n_components=2)\nmda.fit(X_train, y_train)\nX_train_mda = mda.transform(X_train)\nX_test_mda = mda.transform(X_test)\nlda, lda_error, lda_cmat = lda_classifier(X_train_mda, y_train, X_test_mda, y_test)',
      alternatives: ['mda = LinearDiscriminantAnalysis().fit(X_train, y_train)\nX_train_mda = mda.transform(X_train)\nX_test_mda = mda.transform(X_test)\nlda, lda_error, lda_cmat = lda_classifier(X_train_mda, y_train, X_test_mda, y_test)'],
      checks: "c.shape('X_train_mda', (1500, 2))\nc.same('X_test_mda', rtol=1e-8, atol=1e-10)\nc.same('lda_error', rtol=1e-9)",
      hints: ['El mensaje del error dice cuál es el límite.', 'min(n_features, n_classes − 1).', 'Sin `n_components`, scikit-learn usa el máximo.'],
      explanation: 'El error de LDA coincide con el de 3D: con c − 1 direcciones, MDA conserva toda la información que usa LDA.',
    },
    // ---- nivel 5
    {
      kind: 'code', level: 5, id: 'lab2_2.mem.table', title: 'Escribe el bucle de la tabla Q2 (MDA)', skills: ['escritura', 'api'], blocks: ['b_mda2', 'b_mda1', 'b_funcs'], source: { notebook: NB, cells: [23, 27], doc: 'lab2_guide', section: 'Q2', origin: 'original' },
      prompt: 'Sin mirar: para MDA con 2 y 1 componentes, entrena LDA y QDA y guarda los errores de **test** en un diccionario `test_err` con claves `"lda2"`, `"qda2"`, `"lda1"`, `"qda1"`.',
      available: '`X_train`, `y_train`, `X_test`, `y_test` (10 dB). `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` y `accuracy_score` importados.',
      produce: '`test_err` (dict con 4 floats).',
      setup: lab2_2Setup(10),
      starter: '',
      solution: 'test_err = {}\nfor k in (2, 1):\n    mda = LinearDiscriminantAnalysis(n_components=k).fit(X_train, y_train)\n    A, B = mda.transform(X_train), mda.transform(X_test)\n    for nombre, clf in (("lda", LinearDiscriminantAnalysis()), ("qda", QuadraticDiscriminantAnalysis())):\n        clf.fit(A, y_train)\n        test_err[f"{nombre}{k}"] = 1 - accuracy_score(y_test, clf.predict(B))',
      alternatives: ['test_err = {}\nfor k in (2, 1):\n    mda = LinearDiscriminantAnalysis(n_components=k).fit(X_train, y_train)\n    A, B = mda.transform(X_train), mda.transform(X_test)\n    _, e, _ = lda_classifier(A, y_train, B, y_test)\n    test_err[f"lda{k}"] = e[1]\n    _, e, _ = qda_classifier(A, y_train, B, y_test)\n    test_err[f"qda{k}"] = e[1]'],
      checks: "c.true(isinstance(ns.get('test_err'), dict) and set(ns['test_err']) == {'lda2','qda2','lda1','qda1'}, 'Diccionario con las 4 claves', 'test_err debe ser un dict con claves lda2, qda2, lda1, qda1', 'ejecucion')\nfor k in ('lda2','qda2','lda1','qda1'):\n    c.true(abs(ns['test_err'][k] - ref['test_err'][k]) < 1e-9, f'{k} correcto', f'{k} no coincide', 'razonamiento')",
      hints: ['La proyección MDA se aprende con `fit(X_train, y_train)`.', 'Error = 1 − accuracy en test.'],
    },
    // ---- nivel 6
    {
      kind: 'code', level: 6, id: 'lab2_2.adapt.snr0', title: 'Adapta: ¿qué se pierde con MDA a 0 dB?', skills: ['metodo', 'interpretacion'], blocks: ['b_3d', 'b_mda2'], source: { notebook: NB, cells: [13, 23], doc: 'lab2_guide', section: 'Q4', origin: 'original' },
      prompt: 'Con los datos a **0 dB**, calcula el error de test de QDA en 3D y tras MDA a 2D, y la diferencia `perdida = err_mda2 - err_3d`. Después piensa por qué es positiva aunque LDA no pierda nada.',
      available: '`X_train`, `y_train`, `X_test`, `y_test` (0 dB), `qda_classifier`, `LinearDiscriminantAnalysis`.',
      produce: '`err_3d`, `err_mda2`, `perdida` (float).',
      setup: lab2_2Setup(0),
      starter: '',
      solution: '_, e3, _ = qda_classifier(X_train, y_train, X_test, y_test)\nmda = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)\n_, e2, _ = qda_classifier(mda.transform(X_train), y_train, mda.transform(X_test), y_test)\nerr_3d, err_mda2 = e3[1], e2[1]\nperdida = err_mda2 - err_3d',
      checks: "c.same('err_3d', rtol=1e-9)\nc.same('err_mda2', rtol=1e-9)\nc.same('perdida', rtol=1e-6, atol=1e-12)",
      hints: ['Usa las funciones de la celda 11 y quédate con el índice 1 (test).'],
      explanation: 'QDA pasa de ≈ 6.1 % a ≈ 12.8 %: MDA conserva las diferencias de **medias** (lo que usa LDA) pero descarta la tercera dirección, donde las clases tenían varianzas distintas que QDA aprovechaba.',
    },
    {
      kind: 'code', level: 6, id: 'lab2_2.adapt.wine', title: 'Transferencia: PCA frente a MDA en wine', skills: ['metodo', 'escritura'], blocks: ['b_pca2', 'b_mda2'], source: { origin: 'app', note: 'Dataset de transferencia incluido en scikit-learn' }, transfer: true,
      prompt: 'Con `load_wine()` (3 clases, 13 features), parte en train/test (70 %, `random_state=5`, estratificado), reduce a 2 dimensiones con PCA y con MDA aprendidas en train, y calcula el error de test de LDA en cada caso.',
      available: 'Nada importado (`train_test_split` está en `sklearn.model_selection`).',
      produce: '`err_pca`, `err_mda` (float).',
      setup: '',
      starter: '',
      solution: 'from sklearn.datasets import load_wine\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.metrics import accuracy_score\nw = load_wine()\nX_train, X_test, y_train, y_test = train_test_split(w.data, w.target, train_size=0.7, random_state=5, stratify=w.target)\npca = PCA(2).fit(X_train)\nlda = LinearDiscriminantAnalysis().fit(pca.transform(X_train), y_train)\nerr_pca = 1 - accuracy_score(y_test, lda.predict(pca.transform(X_test)))\nmda = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)\nlda = LinearDiscriminantAnalysis().fit(mda.transform(X_train), y_train)\nerr_mda = 1 - accuracy_score(y_test, lda.predict(mda.transform(X_test)))',
      checks: "c.same('err_pca', rtol=1e-9)\nc.same('err_mda', rtol=1e-9)",
      hints: ['Mismo patrón que la celda 15 y la celda 23.', '`train_test_split(X, y, train_size=0.7, random_state=5, stratify=y)` devuelve X_train, X_test, y_train, y_test.'],
      explanation: 'PCA 2D: ≈ 29.6 % de error; MDA 2D: 0 %. Sin escalar, la primera componente de PCA es casi la feature *proline* (la de mayor escala), que no separa bien las clases; MDA usa las etiquetas.',
    },
    // ---- interpretación
    {
      kind: 'mcq', id: 'lab2_2.int.equal', skills: ['interpretacion', 'concepto'], blocks: ['b_mda2', 'b_3d'], source: src([13, 23]),
      prompt: 'Los errores de LDA en 3D y tras MDA a 2D son **idénticos** (10 dB y 0 dB). ¿Por qué?',
      options: [
        { text: 'Con c = 3 clases, la información que usa LDA (medias en la métrica de la covarianza común) está en un subespacio de dimensión c − 1 = 2, que es el que elige MDA', correct: true },
        { text: 'Es casualidad de la semilla', feedback: 'Ocurre con cualquier semilla (pruébalo en el experimento).' },
        { text: 'Porque MDA conserva toda la varianza de los datos', feedback: 'MDA no conserva la varianza total; conserva la separación entre clases.' },
      ],
      explanation: 'LDA y MDA comparten el mismo modelo (covarianza común); LDA en el espacio de las c − 1 direcciones discriminantes es equivalente a LDA en el espacio completo.',
    },
    {
      kind: 'mcq', id: 'lab2_2.int.misconception', skills: ['interpretacion', 'metodo'], blocks: ['b_pca1', 'b_mda1'], source: src([19, 27]),
      prompt: '«PCA 1D retiene el 84 % de la varianza, así que solo pierde el 16 % de la información útil para clasificar». ¿Qué error contiene?',
      options: [
        { text: 'Varianza retenida no es separabilidad: la dirección descartada puede ser justo la que separa las clases', correct: true },
        { text: 'Ninguno: varianza e información para clasificar son lo mismo', feedback: 'PCA ignora las etiquetas.' },
        { text: 'El 84 % es de MDA, no de PCA', feedback: 'Es `pca.explained_variance_ratio_`.' },
      ],
      explanation: 'Con 1 componente PCA comete ≈ 4 % de error en test frente a ≈ 1 % de MDA 1D, que retiene «solo» la dirección más discriminante.',
    },
    {
      kind: 'text', id: 'lab2_2.int.q1', skills: ['interpretacion', 'metodo'], blocks: ['b_3d', 'b_pca2', 'b_pca1'], source: { notebook: NB, cells: [13, 15, 19], doc: 'lab2_guide', section: 'Q1', origin: 'original' },
      prompt: 'Q1 · Con la tabla PCA a 10 dB (app: 3D 0.00/0.07 % LDA y 0/0 % QDA; 2D 0.20/0.27 % y 0/0.07 %; 1D 2.6/4.0 % y 2.3/3.2 %, train/test), discute los resultados y los scatter plots.',
      rubric: [
        'Con 2 componentes apenas cambia el error: la dirección descartada tenía poca varianza y poca información.',
        'Con 1 componente el error crece claramente: las clases se solapan en la proyección.',
        'El test es algo peor que el train (generalización), más con menos dimensiones o en QDA.',
        'PCA no usa etiquetas: retiene varianza, no garantiza separabilidad.',
        'En el scatter 2D las clases siguen separadas; en 1D se solapan.',
      ],
      model: 'A 10 dB las clases están bien separadas en 3D (≈ 0 % de error). Con PCA a 2D el error apenas sube (0.2–0.3 % en LDA), porque la dirección descartada solo tenía ≈ 3 % de la varianza y las clases siguen separadas en el plano. Con 1 componente el error crece a 2.6 % (train) y 4.0 % (test) para LDA y 2.3/3.2 % para QDA: en la proyección sobre la dirección de máxima varianza las clases se solapan (se ve en el gráfico 1D). PCA elige direcciones por varianza total, sin mirar las etiquetas, así que no garantiza conservar la separabilidad. El test es algo peor que el train, como es normal.',
    },
    {
      kind: 'text', id: 'lab2_2.int.q4', skills: ['interpretacion', 'metodo'], blocks: ['b_mda2', 'b_mda1'], source: { notebook: NB, cells: [23, 27], doc: 'lab2_guide', section: 'Q3–Q4', origin: 'original' },
      prompt: 'Q3–Q4 · A 0 dB, compara PCA y MDA con 2 y 1 dimensiones (app, test: LDA 3D 12.7 %; PCA-2 16.5 %, MDA-2 12.7 %; PCA-1 24.2 %, MDA-1 22.5 %; QDA 3D 6.1 %, QDA MDA-2 12.8 %).',
      rubric: [
        'Con ruido alto todos los errores suben y las diferencias entre métodos se hacen visibles.',
        'MDA-2 mantiene el error de LDA en 3D; PCA-2 empeora.',
        'En 1D MDA es mejor que PCA, aunque ambos pierden mucho.',
        'QDA pierde con MDA-2 porque se descarta la dirección con diferencias de covarianza.',
        'Conclusión: para clasificar, MDA (supervisada) conserva mejor la información; PCA puede descartar direcciones discriminantes.',
      ],
      model: 'Con SNR = 0 dB las clases se solapan y todos los errores crecen. MDA con 2 direcciones conserva exactamente el error de LDA en 3D (12.7 %), mientras que PCA-2 lo empeora (16.5 %) porque descarta una dirección que, aunque tiene poca varianza, ayuda a separar. En 1D ambos pierden mucho, pero MDA (22.5 %) sigue por delante de PCA (24.2 %). QDA, que en 3D era el mejor (6.1 %) por aprovechar las covarianzas distintas, empeora con MDA-2 (12.8 %): MDA se basa en una covarianza común y descarta la información de forma de las clases. Para clasificar, la reducción supervisada es preferible; aun así, reducir por debajo de c − 1 tiene un coste.',
    },
    {
      kind: 'numeric', id: 'lab2_2.int.dmax', skills: ['concepto'], blocks: ['b_mda2'], source: { doc: 'lab2_slides', section: 'p. 8', origin: 'original' },
      prompt: 'En la parte 3 (Phoneme) hay d = 256 features y c = 5 clases. ¿Cuántas direcciones discriminantes puede dar MDA como máximo?',
      answer: 4, tolerance: 0,
      explanation: "d' ≤ min(d, c − 1) = min(256, 4) = 4 (es la $d_{max}$ de Q12). Las slides sugieren aplicar MDA por grupos de features si se necesitan más.",
    },
    {
      kind: 'dynamic', id: 'lab2_2.dyn.dmax', skills: ['concepto'], blocks: ['b_mda2'], source: { doc: 'lab2_slides', section: 'p. 8', origin: 'original', note: 'Variante generada' },
      answerKind: 'numeric',
      generator: `import numpy as np
rng = np.random.default_rng(SEED)
d = int(rng.choice([2, 3, 4, 10, 13, 30, 64, 256]))
c = int(rng.choice([2, 3, 4, 5, 10]))
variante = {
  'prompt': f'Un problema tiene d = {d} features y c = {c} clases. ¿Cuál es el número máximo de componentes de MDA (LinearDiscriminantAnalysis(n_components=…))? ¿Y de PCA, si hay 1000 muestras?  Responde el de MDA.',
  'answer': min(d, c - 1), 'tolerance': 0,
  'explanation': f'MDA: min(d, c − 1) = min({d}, {c - 1}) = {min(d, c - 1)}. PCA podría llegar a min(n_samples, d) = {min(1000, d)}.'
}`,
    },
    // ---------------------------------------------------------- cobertura: imports, gráficos de datos y proyecciones
    {
      kind: 'mcq',
      id: 'lab2_2.cov.imports',
      skills: ['api'],
      blocks: ['b_imports'],
      source: src([2, 11]),
      prompt: 'En la celda 2 se importa `from sklearn import metrics` además de `accuracy_score`. ¿Para qué se usa `metrics` en las funciones de la celda 11?',
      options: [
        { text: 'Para `metrics.confusion_matrix`, que no se importó por separado', correct: true },
        { text: 'Para calcular la accuracy', feedback: 'La accuracy se calcula con `accuracy_score`, importada directamente.' },
        { text: 'Para la curva ROC', feedback: '`roc_curve` se importa pero no se usa en este notebook.' },
        { text: 'Para PCA', feedback: 'PCA está en `sklearn.decomposition`.' },
      ],
      explanation: 'Dos formas de acceder a lo mismo: `from sklearn.metrics import confusion_matrix` o `from sklearn import metrics` y luego `metrics.confusion_matrix`. Sin el import de `metrics`, `lda_classifier` fallaría con `NameError`.',
    },
    {
      kind: 'predict',
      id: 'lab2_2.cov.counts',
      skills: ['codigo'],
      blocks: ['b_scatter3d'],
      source: src([6, 8]),
      prompt: 'En el scatter 3D (celda 8), la máscara `idx = y_train == idclass` selecciona los puntos de cada color. ¿Cuántos puntos tiene la clase 0?',
      setup: lab2_2Setup(10),
      code: 'idx = y_train == 0\nn0 = int(idx.sum())',
      expr: 'n0',
      answerKind: 'numeric',
      tolerance: 0,
      explanation: '`n_samples = 500` muestras por clase: el barajado cambia el orden, no cuántas hay de cada clase.',
    },
    {
      kind: 'predict',
      id: 'lab2_2.cov.pairplot',
      skills: ['codigo'],
      blocks: ['b_pairplot'],
      source: src([9]),
      prompt: '¿Qué forma tiene `Xyt`, el DataFrame que la celda 9 pasa a `sns.pairplot`?',
      setup: lab2_2Setup(10) + 'import pandas as pd\n',
      code: "Xt = pd.DataFrame(X_train,columns=['feat1','feat2','feat3'])\nyt = pd.DataFrame(y_train,columns=['class'])\nXyt = Xt.join(yt)",
      expr: 'Xyt.shape',
      answerKind: 'shape',
      explanation: '1500 muestras de train (3 clases × 500) y 4 columnas: las 3 features y la clase, que `hue="class"` usa para colorear. El pairplot tiene 3×3 paneles porque la columna de clase no se dibuja como variable.',
    },
    {
      kind: 'mcq',
      id: 'lab2_2.cov.pca2plot',
      skills: ['interpretacion', 'concepto'],
      blocks: ['b_pca2_plot'],
      source: src([17]),
      prompt: 'En la figura «PCA 2D – LDA/QDA boundaries» (celda 17), ¿qué representan los ejes?',
      options: [
        { text: 'Las coordenadas en las dos primeras componentes principales, aprendidas con X_train', correct: true },
        { text: 'Las features feat1 y feat2 originales', feedback: 'Se dibuja `X_train_pca2`: cada eje es una combinación lineal de las tres features.' },
        { text: 'Las dos direcciones que mejor separan las clases', feedback: 'Eso sería MDA. PCA busca máxima varianza, sin usar las etiquetas.' },
        { text: 'La probabilidad a posteriori de cada clase', feedback: 'Las regiones de color son las clases predichas; los ejes son coordenadas.' },
      ],
      explanation: 'Las regiones se calculan clasificando una malla en ese plano con los clasificadores entrenados sobre `X_train_pca2`. Si las clases se solapan aquí, es porque PCA ha conservado varianza que no separa.',
    },
    {
      kind: 'mcq',
      id: 'lab2_2.cov.pca1plot',
      skills: ['interpretacion', 'codigo'],
      blocks: ['b_pca1_plot'],
      source: src([21]),
      prompt: 'En el «scatter» PCA 1D (celda 21) todos los puntos quedan sobre una diagonal. ¿Por qué?',
      options: [
        { text: 'Porque se dibuja `X[idx,0]` frente a `X[idx,0]`: la misma coordenada en los dos ejes', correct: true },
        { text: 'Porque PCA 1D fuerza a que las clases queden alineadas', feedback: 'Es un efecto del código de dibujo, no de PCA.' },
        { text: 'Porque las features originales están correlacionadas', feedback: 'Con una sola componente no hay segunda coordenada que pueda correlacionarse.' },
        { text: 'Porque la malla `h` es demasiado gruesa', feedback: 'En esta celda la malla solo fija los límites de los ejes.' },
      ],
      explanation: 'Con un solo componente solo hay un eje con información; la diagonal no aporta nada. Para ver el solapamiento en 1D es mejor un histograma por clase de `X_train_pca1[:, 0]`.',
    },
    {
      kind: 'mcq',
      id: 'lab2_2.cov.mda2plot',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_mda2_plot'],
      source: src([25]),
      prompt: 'Comparando la figura MDA 2D (celda 25) con la de PCA 2D (celda 17) con los mismos datos, ¿qué esperas y por qué?',
      options: [
        { text: 'Clases más separadas en MDA, porque maximiza la dispersión entre clases frente a la intra-clase', correct: true },
        { text: 'Las mismas figuras rotadas, porque ambas son proyecciones lineales a 2D', feedback: 'Ambas son lineales, pero con criterios distintos: varianza total (PCA) frente a separabilidad (MDA).' },
        { text: 'Más varianza conservada en MDA', feedback: 'Es al revés: PCA es la proyección que conserva más varianza.' },
        { text: 'Peor separación en MDA porque usa menos componentes', feedback: 'Las dos usan 2 componentes, el máximo de MDA con c = 3 clases.' },
      ],
      explanation: 'MDA usa las etiquetas: sus ejes son las direcciones discriminantes. Por eso LDA en el plano MDA rinde igual que en 3D, mientras que en el plano PCA puede empeorar.',
    },
    {
      kind: 'mcq',
      id: 'lab2_2.cov.mda1plot',
      skills: ['codigo', 'interpretacion'],
      blocks: ['b_mda1_plot'],
      source: src([21, 29]),
      prompt: 'La celda 29 copia la 21 pero cambia `colors = [\'r\',\'g\',\'b\']` por `[\'g\',\'b\',\'r\']`. ¿Qué problema introduce al comparar las dos figuras?',
      options: [
        { text: 'La misma clase aparece con colores distintos en PCA 1D y MDA 1D, y se pueden confundir', correct: true },
        { text: 'Ninguno: los colores no afectan a nada', feedback: 'No afectan al cálculo, pero sí a la interpretación al comparar figuras.' },
        { text: 'Cambia el resultado de la proyección', feedback: 'Es un parámetro solo de visualización.' },
        { text: 'Hace que el código falle', feedback: 'Se ejecuta sin errores.' },
      ],
      explanation: 'Es una errata de visualización: conviene una paleta fija por clase en todo el notebook. Recuerda además que, como en la celda 21, el gráfico dibuja la proyección contra sí misma.',
    },
  ],
  recall: ['lab2_2.fill.pca', 'lab2_2.fill.mda', 'lab2_2.fix.leak', 'lab2_2.fix.ncomp', 'lab2_2.mem.table', 'lab2_2.adapt.snr0', 'lab2_2.adapt.wine'],
  practice: [
    {
      id: 'lab2_2.practice.report',
      title: 'Mini-report Q1–Q4',
      scenario: 'Prepara las tablas y la discusión de Q1–Q4 sin el notebook: reduce con PCA y con MDA, evalúa LDA y QDA en train y test, y explica qué se gana y qué se pierde en cada caso a 10 y 0 dB.',
      steps: ['lab2_2.fill.pca', 'lab2_2.fix.ncomp', 'lab2_2.mem.table', 'lab2_2.int.equal', 'lab2_2.adapt.snr0', 'lab2_2.int.q4'],
      source: { notebook: NB, doc: 'lab2_guide', section: 'Q1–Q4', origin: 'original' },
    },
  ],
  notes: [
    'El notebook original no tiene salidas guardadas: los valores mostrados proceden de la ejecución en la app.',
    'Tu entorno local da errores ligeramente distintos (unas pocas muestras) porque `multivariate_normal` genera de otra forma las muestras de la clase 0, cuya covarianza tiene autovalores repetidos. Las clases 1 y 2 coinciden exactamente.',
  ],
};
