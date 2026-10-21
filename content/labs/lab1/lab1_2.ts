import type { NotebookContent } from '../../../src/content/types';

const NB = 'lab1_2';
const src = (cells: number[], extra: Partial<{ section: string; note: string; doc: string }> = {}) => ({
  notebook: NB,
  cells,
  origin: 'original' as const,
  ...extra,
});

/** Estado del notebook tras las celdas 3–11 (datos de entrenamiento barajados) con SNR y ro configurables. */
export const lab1_2Setup = (snr = 10, ro = '[0.5, 0, -0.5, 0.8]', sig0 = 1) => `import numpy as np
from numpy.random import default_rng
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
rng = default_rng(seed=5)
SNR = ${snr}
dist = 1.
n_classes = 4
n_samples = 1000
n_feat = 2
ro = ${ro}
M_Means = 0.5 * dist * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
energy = 0
for i in range(0, n_classes):
    energy = energy + np.dot(M_Means[i], M_Means[i])
energy = energy / n_classes
SNR = 10 ** (SNR / 10)
sig = energy / SNR
M_covar = np.zeros(shape=(4, 2, 2))
sigma = sig * np.array([1, 1, 1, 1])
sigma[0] = sigma[0] * ${sig0}
for i in range(0, n_classes):
    M_covar[i, :, :] = sigma[i] / n_feat * np.array([[1, ro[i]], [ro[i], 1]])
xdata = np.empty((0, n_feat), float)
labels = np.empty((0), int)
for i in range(0, n_classes):
    ydata = rng.multivariate_normal(mean=M_Means[i, :], cov=M_covar[i, :, :], size=n_samples)
    xdata = np.append(xdata, ydata, axis=0)
    labels = np.append(labels, i * np.ones((n_samples), dtype=int), axis=0)
np.random.seed(0)
rp = np.random.permutation(len(xdata))
X_train = xdata[rp]
y_train = labels[rp]
`;

const EXPERIMENT = `import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.linalg import eigvals
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

rng = default_rng(seed={{seed}})

# celda 6 · parámetros de diseño
SNR = {{SNR}}
dist=1.
n_classes = 4
n_samples = 1000;
n_feat = 2

# celda 8 · caso de covarianzas (ro por clase)
ro = {{ro}}

# celda 9 · medias QPSK, energía, varianza del ruido y covarianzas
M_Means = 0.5 * dist * np.array([[1, 1],[1, -1], [-1, 1], [-1,-1]])
energy = 0
for i in range(0,n_classes):
  energy = energy + np.dot(M_Means[i],M_Means[i])
energy = energy / n_classes
SNR = 10 ** (SNR/10)
sig = energy / SNR
M_covar = np.zeros(shape=(4,2,2))
sigma = sig * np.array([1,1,1,1])
sigma[0] = sigma[0] * {{sig0}}        # guía Q11 / report Q12: sigma[0] *= 30 antes de construir M_covar
for i in range(0,n_classes):
  M_covar[i,:,:] = sigma[i] / n_feat * np.array([[1, ro[i]],[ro[i],1]])

# celda 11 · generación y barajado (np.random.permutation como el original: no consume rng)
xdata = np.empty((0,n_feat), float)
labels = np.empty((0), int)
for i in range(0,n_classes):
  ydata = rng.multivariate_normal(mean=M_Means[i,:], cov= M_covar[i,:,:], size= n_samples)
  ylabels = i * np.ones((n_samples),dtype=int)
  xdata = np.append(xdata, ydata, axis=0)
  labels = np.append(labels, ylabels, axis=0)
rp = np.random.permutation(len(xdata))
xdatas = xdata[rp]
xlabs  = labels[rp]

# celda 15 · LDA y QDA, evaluados sobre TRAIN (como en el notebook)
X_train = xdatas
y_train = xlabs
lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True)
ldamodel = lda.fit(X_train, y_train)
y_tpred_lda = ldamodel.predict(X_train)
linear_error = 1. - accuracy_score(y_train,y_tpred_lda)
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qdamodel = qda.fit(X_train, y_train)
y_tpred_qda = qdamodel.predict(X_train)
qda_error = 1. - accuracy_score(y_train,y_tpred_qda)
print('sigma:', sigma)
print('LDA error: %f' %linear_error)
print(confusion_matrix(y_train,y_tpred_lda))
print('QDA error: %f' %qda_error)
print(confusion_matrix(y_train,y_tpred_qda))

# celdas 19-20 · regiones de decisión (h solo cambia la resolución del dibujo)
h = {{h}}
col = ['tab:blue','tab:orange','tab:green','tab:red']
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
fig, ax = plt.subplots(figsize=(13,6.5), nrows=1, ncols=2)
for k, (model, name) in enumerate([(lda, 'LDA boundaries'), (qda, 'QDA boundaries')]):
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
  ax[k].contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.2)
  for idclass in range(0, n_classes):
    idx = xlabs==idclass
    ax[k].scatter(xdatas[idx,0], xdatas[idx,1], color = col[idclass], label='class %d' %idclass, alpha=0.4, s=8)
  ax[k].set_xlim(xx.min(), xx.max()); ax[k].set_ylim(yy.min(), yy.max())
  ax[k].set_title(name)
ax[0].legend()
plt.show()

# celda 17 · rectas w_i·x + b_i = 0 de LDA (opcional)
if {{lines}}:
  fig, ax = plt.subplots(figsize = (7,7))
  for l, c in zip(np.unique(y_train), col):
    plt.scatter(X_train[y_train==l,0], X_train[y_train==l,1], c=c, label='class %d' %l, alpha=0.3, s=8)
  x1 = np.array([np.min(X_train[:,0]), np.max(X_train[:,0])])
  for i, c in enumerate(col):
    b, w1, w2 = lda.intercept_[i], lda.coef_[i][0], lda.coef_[i][1]
    plt.plot(x1, -(b+x1*w1)/w2, c=c)
  plt.ylim(y_min, y_max)
  plt.title("Scatter plot with decision boundaries (celda 17)")
  plt.legend()
  plt.show()

# (app) autovalores teóricos (Q1: sigma_k^2/2 · (1 ± ro_k)) y muestrales (Q10, scipy.linalg.eigvals)
teo = [np.sort(sigma[k] / 2 * np.array([1 - ro[k], 1 + ro[k]])) for k in range(n_classes)]
emp = [np.sort(eigvals(np.cov(X_train[y_train == k].T, ddof=1)).real) for k in range(n_classes)]
resultados = {
  'sigma0': sigma[0],
  'err_lda': linear_error, 'err_qda': qda_error,
  'cm_lda': confusion_matrix(y_train, y_tpred_lda), 'cm_qda': confusion_matrix(y_train, y_tpred_qda),
  'eig_teo': np.array(teo), 'eig_emp': np.array(emp),
  'ratio4': float(np.sqrt(teo[3][1] / teo[3][0])),
}
`;

export const lab1_2: NotebookContent = {
  id: NB,
  lab: 'lab1',
  title: 'Lab 1 · Parte 2 — Autovalores de la covarianza y forma de los clusters (QPSK)',
  short: 'QPSK: Case 2 y Case 3',
  origin: 'original',
  context: {
    goal: `Ver cómo la **forma** de la matriz de covarianza (sus autovalores y autovectores) decide la forma de los clusters y, con ella, si la frontera MAP óptima es **lineal** o **cuadrática**.

Se usa una modulación **QPSK**: 4 clases (símbolos) en 2 dimensiones, con medias en $(\\pm 0.5, \\pm 0.5)$. Se comparan dos casos de teoría:
- **Case 2**: todas las clases comparten la misma covarianza, no diagonal ($\\rho$ común). La frontera óptima sigue siendo lineal → LDA está bien especificado.
- **Case 3**: cada clase tiene su propia $\\rho$ (y, al final, una varianza mucho mayor en la clase 0). La frontera óptima es cuadrática → QDA puede ganar a LDA.

A diferencia de la parte 1, **se evalúa sobre el conjunto de entrenamiento** (no hay test), y se dibujan las **regiones de decisión** con una malla de puntos.`,
    dataset: `Datos generados: 4 clases × 1000 muestras, 2 features. Medias $\\mu = 0.5\\,(\\pm1, \\pm1)$, energía media 0.5. Covarianza de la clase $k$: $C_k=\\frac{\\sigma_k^2}{2}\\begin{pmatrix}1&\\rho_k\\\\\\rho_k&1\\end{pmatrix}$ con $\\sigma^2 = 0.5/\\mathrm{SNR}_{lin}$ (= 0.05 a 10 dB). Semilla \`default_rng(seed=5)\`. El notebook guardado corresponde a **SNR = 10 dB, Case 3** (\`ro = [0.5, 0, -0.5, 0.8]\`).`,
    variables: [
      { name: 'ro', meaning: 'Lista de coeficientes de correlación, uno por clase (Case 2: iguales; Case 3: distintos).', shape: 'lista de 4' },
      { name: 'M_Means', meaning: 'Medias QPSK: filas (0.5,0.5), (0.5,−0.5), (−0.5,0.5), (−0.5,−0.5).', shape: '(4, 2)' },
      { name: 'sig', meaning: 'Varianza total σ² = energía / SNR lineal (aquí **no** se divide por n_feat en esta línea).', shape: 'escalar' },
      { name: 'sigma', meaning: 'σ² de cada clase; `sigma[0]` se multiplica por 30 en el último experimento.', shape: '(4,)' },
      { name: 'M_covar', meaning: 'Covarianzas por clase: (σ²/2)·[[1, ρ], [ρ, 1]].', shape: '(4, 2, 2)' },
      { name: 'xdatas, xlabs → X_train, y_train', meaning: 'Muestras barajadas y etiquetas 0–3.', shape: '(4000, 2), (4000,)' },
      { name: 'xx, yy, Z', meaning: 'Malla para dibujar regiones y su clase predicha.', shape: '(ny, nx)' },
    ],
    classes: [
      { id: '0', label: 'clase 0 · (0.5, 0.5)' },
      { id: '1', label: 'clase 1 · (0.5, −0.5)' },
      { id: '2', label: 'clase 2 · (−0.5, 0.5)' },
      { id: '3', label: 'clase 3 · (−0.5, −0.5)' },
    ],
    questions: [
      { id: 'Q1', text: 'Autovalores de $C=\\frac{\\sigma^2}{2}\\begin{pmatrix}1&\\rho\\\\\\rho&1\\end{pmatrix}$ en función de σ² y ρ.', blocks: ['b_covar'], doc: 'lab1_report' },
      { id: 'Q6', text: 'Case 2, ρ = 0, SNR = 10 dB: scatter, fronteras, matrices de confusión y errores de LC y QC.', blocks: ['b_cases', 'b_fit', 'b_regions_lda'], doc: 'lab1_guide' },
      { id: 'Q7', text: 'Case 2 con ρ = 0.5.', blocks: ['b_cases', 'b_fit'], doc: 'lab1_guide' },
      { id: 'Q8', text: 'Comparar Q6 y Q7 (el report dice «Q8 y Q9» por errata).', blocks: ['b_fit'], doc: 'lab1_report' },
      { id: 'Q9', text: 'Case 3 (ρ = 0.5, 0, −0.5, 0.8) a 5 y 10 dB: errores de LC y QC.', blocks: ['b_fit'], doc: 'lab1_guide' },
      { id: 'Q10 (report)', text: 'Tabla de autovalores teóricos y muestrales con `scipy.linalg.eigvals` (una SNR).', blocks: ['b_covar'], doc: 'lab1_report' },
      { id: 'Q11 (report) / Q10 (guía)', text: 'Relacionar la forma de los clusters con los autovalores a 5 y 10 dB.', blocks: ['b_scatter', 'b_regions_lda', 'b_regions_qda'], doc: 'lab1_guide' },
      { id: 'Q12 (report) / Q11 (guía)', text: '`sigma[0] *= 30` a 10 dB: errores, regiones y nuevo valor de sigma[0] y de la covarianza de la clase 1.', blocks: ['b_covar', 'b_regions_qda'], doc: 'lab1_guide' },
    ],
    notation: [
      { symbol: '$c=4$, $d=2$', meaning: 'clases (símbolos QPSK) y features', code: 'n_classes, n_feat' },
      { symbol: '$\\rho_k$', meaning: 'correlación entre las dos features en la clase k', code: 'ro[k]' },
      { symbol: '$\\sigma_k^2$', meaning: 'varianza total del ruido de la clase k (suma de las dos varianzas)', code: 'sigma[k]' },
      { symbol: '$C_k=\\frac{\\sigma_k^2}{2}\\begin{pmatrix}1&\\rho_k\\\\\\rho_k&1\\end{pmatrix}$', meaning: 'covarianza de la clase k', code: 'M_covar[k]' },
      { symbol: '$\\lambda_{1,2}=\\frac{\\sigma^2}{2}(1\\pm\\rho)$', meaning: 'autovalores: varianzas a lo largo de los ejes de la elipse', code: 'scipy.linalg.eigvals(C)' },
      { symbol: '$v=\\frac{1}{\\sqrt2}(1,\\pm1)$', meaning: 'autovectores: los ejes de la elipse siempre están a ±45°' },
      { symbol: '$g_i(x)=w_i^Tx+b_i$', meaning: 'discriminante lineal de la clase i en LDA', code: 'lda.coef_[i], lda.intercept_[i]' },
    ],
  },
  flow: [
    { stage: 'contexto', label: 'Contexto', blocks: ['b_intro', 'b_imports'], summary: 'QPSK, Case 2 y Case 3.' },
    { stage: 'datos', label: 'Datos', blocks: ['b_params', 'b_cases', 'b_covar', 'b_generate'], summary: 'SNR, ρ por clase y covarianzas → 4000 muestras.' },
    { stage: 'exploracion', label: 'Exploración', blocks: ['b_scatter'], summary: 'Forma de las nubes: círculos o elipses.' },
    { stage: 'preparacion', label: 'Preparación', blocks: [], summary: 'El barajado ya se hace en la celda 11; se usa X_train = xdatas.' },
    { stage: 'metodo', label: 'Método', blocks: ['b_fit'], summary: 'LDA y QDA sobre entrenamiento.' },
    { stage: 'resultados', label: 'Resultados', blocks: ['b_lines', 'b_regions_lda', 'b_regions_qda'], summary: 'Rectas de LDA y regiones de decisión.' },
    { stage: 'interpretacion', label: 'Interpretación', blocks: [], summary: 'Q6–Q12 en Interpretar y Práctica.' },
  ],
  blocks: [
    {
      id: 'b_intro',
      title: 'Título y planteamiento: QPSK, Case 2 y Case 3',
      cells: [0, 1, 2],
      stage: 'contexto',
      problem: 'Plantear un problema de 4 clases en 2D donde se puede cambiar la forma de las covarianzas.',
      purpose: 'Define qué se va a comparar: covarianza **común pero arbitraria** (Case 2) frente a covarianzas **distintas** por clase (Case 3).',
      inputs: 'Texto.',
      outputs: 'El modelo $f(x\\mid\\omega_c)=\\mathcal N(\\mu_c, C_c)$, $c = 1..4$.',
      prereqs: ['map', 'covarianza', 'qpsk'],
      apis: [],
      lines: [],
      interpretation: 'En Case 2 los términos cuadráticos $x^TC^{-1}x$ son iguales para todas las clases y se cancelan al comparar: fronteras **rectas**. En Case 3 no se cancelan: fronteras **curvas** (cónicas).',
      next: 'Imports y semilla; después se eligen SNR y ρ.',
      whatIf: [{ change: 'Usar ρ = 0 en todas las clases', effect: 'Se vuelve al Case 1 (covarianza isótropa común): nubes circulares y fronteras perpendiculares a la recta entre medias.' }],
      check: {
        kind: 'mcq',
        id: 'lab1_2.check.intro',
        skills: ['concepto'],
        source: src([2]),
        prompt: '¿En qué caso la frontera MAP óptima entre clases **no** es lineal?',
        options: [
          { text: 'Case 3: cada clase tiene una covarianza distinta', correct: true },
          { text: 'Case 2: covarianza común con ρ = 0.5', feedback: 'Con covarianza común el término cuadrático se cancela: sigue siendo lineal aunque las nubes sean elipses.' },
          { text: 'Case 1: covarianza σ²I común', feedback: 'Es el caso más simple: frontera lineal.' },
        ],
        explanation: 'Lo que decide la linealidad es si las covarianzas son **iguales** entre clases, no si son diagonales.',
      },
    },
    {
      id: 'b_imports',
      title: 'Imports y generador con semilla',
      cells: [3],
      stage: 'contexto',
      problem: 'Tener las librerías y un generador reproducible.',
      purpose: 'Igual que en la parte 1, pero aquí la semilla se fija en la primera celda: todo lo que se genere después sale del mismo `rng`.',
      inputs: 'Nada.',
      outputs: '`np`, `pd`, `sns`, `ss`, `plt`, `rng`.',
      prereqs: ['semilla'],
      apis: [{ name: 'default_rng', lib: 'numpy.random', kind: 'función', does: 'Generator PCG64 con semilla.', args: [{ name: 'seed', value: '5', meaning: 'reproducibilidad' }], returns: 'Generator' }],
      lines: [{ cell: 3, lines: [6, 8], text: 'La semilla se fija una sola vez: si ejecutas dos veces la celda 11 sin volver a esta, obtendrás **otras** muestras (el estado de `rng` ha avanzado).' }],
      interpretation: 'Para comparar casos (ρ, SNR) en igualdad de condiciones hay que volver a ejecutar desde esta celda.',
      next: 'Parámetros de diseño.',
      whatIf: [{ change: 'Re-ejecutar solo las celdas 6–11 tras cambiar ρ', effect: 'Las muestras cambiarían también por el estado del generador, no solo por ρ: la comparación quedaría contaminada.' }],
    },
    {
      id: 'b_params',
      title: 'SNR y parámetros de diseño',
      cells: [4, 5, 6],
      stage: 'datos',
      problem: 'Elegir el ruido y el tamaño del problema.',
      purpose: 'La guía pide SNR = 10 dB para Case 2 y 5 y 10 dB para Case 3. El resto queda fijo.',
      inputs: 'Constantes.',
      outputs: '`SNR` (dB), `dist`, `n_classes = 4`, `n_samples = 1000` por clase, `n_feat = 2`.',
      prereqs: ['snr', 'decibelio'],
      apis: [],
      lines: [
        { cell: 6, lines: [2, 3], text: 'Para cambiar de SNR se comenta/descomenta: el notebook guardado usa 10 dB.' },
        { cell: 6, lines: [7, 9], text: '4 clases × 1000 muestras = 4000 filas; 2 features → se puede dibujar todo el espacio.' },
      ],
      interpretation: 'Con d = 2 las regiones de decisión se pueden pintar completas: es el propósito de este notebook.',
      next: 'Elección del caso (ρ por clase).',
      whatIf: [{ change: 'SNR de 10 a 5 dB', effect: 'σ² se multiplica por √10 ≈ 3.16; las elipses crecen y se solapan: el error de train sube de ≈0.15 % a ≈6 %.' }],
    },
    {
      id: 'b_cases',
      title: 'Elección del caso: correlación ρ por clase',
      cells: [7, 8],
      stage: 'datos',
      problem: 'Decidir la forma de las covarianzas.',
      purpose: 'Con la misma ρ en las cuatro clases se obtiene Case 2; con ρ distintas, Case 3. El notebook guardado tiene activo Case 3.',
      inputs: 'Nada.',
      outputs: '`ro` (lista de 4).',
      prereqs: ['correlacion', 'covarianza'],
      apis: [],
      lines: [
        { cell: 8, lines: [1, 2], text: 'Case 2: ρ común (0 para Q6, 0.5 para Q7).' },
        { cell: 8, lines: [3, 3], text: 'Case 3 (guía): símbolo 1 ρ = +0.5, símbolo 2 ρ = 0, símbolo 3 ρ = −0.5, símbolo 4 ρ = +0.8. En el código las clases se numeran 0–3.' },
      ],
      interpretation: 'ρ > 0 inclina la elipse a lo largo de (1, 1); ρ < 0, a lo largo de (1, −1); ρ = 0, círculo. Cuanto mayor |ρ|, más alargada.',
      next: 'Con ρ y σ² se construyen las covarianzas.',
      whatIf: [
        { change: 'ρ = 1', effect: 'La covarianza sería singular (autovalor 0): todos los puntos caerían sobre una recta y `multivariate_normal` avisaría.' },
        { change: '|ρ| > 1', effect: 'La matriz no sería semidefinida positiva: no es una covarianza válida.' },
      ],
    },
    {
      id: 'b_covar',
      title: 'Medias QPSK, varianza y covarianzas',
      cells: [9],
      stage: 'datos',
      problem: 'Construir los parámetros de las cuatro gaussianas.',
      purpose: 'Calcula la energía (0.5), σ² = energía/SNR y $C_k=\\frac{\\sigma_k^2}{2}[[1,\\rho_k],[\\rho_k,1]]$. Es la celda que hay que tocar para el último experimento (`sigma[0] *= 30`).',
      inputs: '`SNR`, `dist`, `ro`, `n_feat`.',
      outputs: '`M_Means` (4, 2), `sig`, `sigma` (4,), `M_covar` (4, 2, 2). Imprime sigma y M_covar.',
      prereqs: ['covarianza', 'autovalor', 'snr'],
      apis: [{ name: 'np.dot', lib: 'numpy', kind: 'función', does: '‖μᵢ‖² de cada media.' }, { name: 'np.zeros', lib: 'numpy', kind: 'función', does: 'Reserva (4, 2, 2).' }],
      lines: [
        { cell: 9, lines: [2, 2], text: 'Cuatro símbolos QPSK en (±0.5, ±0.5): cada uno tiene ‖μ‖² = 0.5.' },
        { cell: 9, lines: [11, 12], text: 'σ² **total** = 0.5 / SNR lineal = 0.05 a 10 dB.' },
        { cell: 9, lines: [13, 13], text: 'Comentado respecto a la parte 1: aquí el reparto entre las 2 features se hace dentro de M_covar (`/ n_feat`, línea 18).' },
        { cell: 9, lines: [16, 16], text: 'Una σ² por clase. Para Q12 hay que añadir `sigma[0] = sigma[0] * 30` **aquí**, antes del bucle.' },
        { cell: 9, lines: [18, 18], text: '$C_k = \\frac{\\sigma_k^2}{2}\\begin{pmatrix}1&\\rho_k\\\\\\rho_k&1\\end{pmatrix}$: varianza 0.025 por feature y covarianza 0.025·ρ_k.' },
      ],
      shapes: [{ expr: 'M_Means', shape: '(4, 2)', meaning: 'una media por fila' }, { expr: 'M_covar', shape: '(4, 2, 2)', meaning: 'una covarianza por clase' }],
      interpretation: 'Autovalores (Q1): $\\det(C-\\lambda I)=0 \\Rightarrow \\lambda=\\frac{\\sigma^2}{2}(1\\pm\\rho)$, con autovectores $(1,\\pm1)/\\sqrt2$. A 10 dB: ρ = 0 → (0.025, 0.025); ρ = ±0.5 → (0.0125, 0.0375); ρ = 0.8 → (0.005, 0.045). La **traza** (suma de autovalores) es siempre σ² = 0.05: ρ no añade ruido, lo **redistribuye** entre direcciones. Los semiejes de las elipses de igual densidad son proporcionales a $\\sqrt{\\lambda}$.',
      next: 'Se generan las muestras con estas medias y covarianzas.',
      whatIf: [
        { change: '`sigma[0] *= 30` antes del bucle', effect: 'C_0 = 30·C_0 = [[0.75, 0.375], [0.375, 0.75]], autovalores 0.375 y 1.125: la clase 0 se dispersa sobre las demás y LDA (covarianza común) falla mucho más que QDA (12.7 % frente a 4.9 % en train).' },
        { change: 'Hacer `sigma[0] *= 30` después del bucle', effect: 'No tendría ningún efecto: M_covar ya estaría construida.' },
      ],
    },
    {
      id: 'b_generate',
      title: 'Generación y barajado',
      cells: [10, 11],
      stage: 'datos',
      problem: 'Obtener 1000 muestras de cada símbolo y mezclarlas.',
      purpose: 'Mismo bucle que la parte 1, ahora con 4 clases, y barajado en la misma celda.',
      inputs: '`M_Means`, `M_covar`, `rng`.',
      outputs: '`xdatas` (4000, 2), `xlabs` (4000,). Imprime las formas.',
      prereqs: ['gaussiana-multivariante', 'matriz-features'],
      apis: [
        { name: 'rng.multivariate_normal', lib: 'numpy.random.Generator', kind: 'método', does: 'Muestras de la gaussiana de cada clase.', returns: '(1000, 2)' },
        { name: 'np.random.permutation', lib: 'numpy', kind: 'función', does: 'Permutación con el generador global (sin semilla).' },
        { name: 'np.shape', lib: 'numpy', kind: 'función', does: 'Forma de un array.' },
      ],
      lines: [
        { cell: 11, lines: [6, 10], text: 'Bucle de 4 clases: tras él `xdata` es (4000, 2) y `labels` (4000,) ordenados por clase.' },
        { cell: 11, lines: [13, 15], text: 'Permutación **sin semilla**: el orden cambia en cada ejecución, pero los errores no (LDA y QDA no dependen del orden).' },
        { cell: 11, lines: [17, 18], text: 'Salida guardada: (4000, 2) y (4000,).' },
      ],
      shapes: [{ expr: 'xdatas', shape: '(4000, 2)', meaning: 'muestras barajadas' }, { expr: 'xlabs', shape: '(4000,)', meaning: 'etiquetas 0–3' }],
      interpretation: 'No hay conjunto de test: todo lo que sigue se evalúa sobre estos mismos datos (errores de entrenamiento, optimistas).',
      corrections: [{ cell: 11, kind: 'concepto', original: 'rp = np.random.permutation(len(xdata))', fixed: 'rp = rng.permutation(len(xdata))', why: 'El barajado no es reproducible porque usa el generador global sin semilla. No cambia las métricas; la app conserva el original para reproducir tus salidas.' }],
      next: 'Scatter plot para ver la forma de las nubes.',
      whatIf: [{ change: 'Añadir un test generado con el mismo `rng`', effect: 'Permitiría medir el error real; con 1000 muestras por clase saldría muy parecido al de train.' }],
    },
    {
      id: 'b_scatter',
      title: 'Scatter 2D de las cuatro clases',
      cells: [12, 13],
      stage: 'exploracion',
      problem: 'Ver la forma y orientación de cada nube.',
      purpose: 'Relacionar visualmente la forma de los clusters con ρ y los autovalores (Q11 del report).',
      inputs: '`xdatas`, `xlabs`.',
      outputs: 'Figura de dispersión.',
      prereqs: ['scatter', 'autovalor'],
      apis: [{ name: 'plt.scatter', lib: 'matplotlib', kind: 'función', does: 'Puntos de cada clase.', args: [{ name: 'alpha', value: '0.5', meaning: 'transparencia para ver solapamientos' }] }],
      lines: [{ cell: 13, lines: [5, 5], text: 'Orden de colores **distinto** del de las celdas 17–20 (rojo y verde intercambiados): la clase 2 es roja aquí y verde después.' }],
      interpretation: 'Case 3 a 10 dB: clase 0 (ρ = 0.5) elipse a lo largo de (1,1); clase 1 (ρ = 0) círculo; clase 2 (ρ = −0.5) elipse a lo largo de (1,−1); clase 3 (ρ = 0.8) muy alargada (razón de semiejes $\\sqrt{0.045/0.005}=3$).',
      corrections: [{ cell: 13, kind: 'errata', original: "col = ['tab:blue','tab:orange','tab:red','tab:green']", fixed: "col = ['tab:blue','tab:orange','tab:green','tab:red']", why: 'Así coincide con las figuras de fronteras y regiones (celdas 17, 19, 20) y con la paleta fija de la app.' }],
      next: 'Entrenar LDA y QDA.',
      whatIf: [{ change: 'Bajar a 5 dB', effect: 'Mismas orientaciones, nubes √(√10) ≈ 1.78 veces más anchas y más solapadas.' }],
    },
    {
      id: 'b_fit',
      title: 'LDA y QDA evaluados en entrenamiento',
      cells: [14, 15],
      stage: 'metodo',
      problem: 'Medir qué clasificador se adapta mejor a cada caso.',
      purpose: 'Entrena los dos modelos y calcula error y matriz de confusión **sobre train**. Con estas cifras se responden Q6, Q7, Q9 y Q12.',
      inputs: '`xdatas`, `xlabs`.',
      outputs: 'Priors, medias y covarianzas estimadas; error y matriz 4×4 de cada modelo. Guardado (Case 3, 10 dB): LDA 0.0015, QDA 0.0005.',
      prereqs: ['lda', 'qda', 'matriz-confusion', 'accuracy'],
      apis: [
        { name: 'LinearDiscriminantAnalysis', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Covarianza común estimada (promedio de las de clase).', args: [{ name: 'solver', value: '"svd"', meaning: 'algoritmo' }, { name: 'store_covariance', value: 'True', meaning: 'guarda covariance_' }], learned: ['priors_', 'means_', 'covariance_', 'coef_', 'intercept_'] },
        { name: 'QuadraticDiscriminantAnalysis', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Una covarianza por clase.', learned: ['priors_', 'means_', 'covariance_'] },
        { name: 'confusion_matrix', lib: 'sklearn.metrics', kind: 'función', does: 'Matriz 4×4: fila = real, columna = predicha.' },
      ],
      lines: [
        { cell: 15, lines: [7, 8], text: 'X_train y y_train son los **mismos** datos que luego se evalúan: error de entrenamiento.' },
        { cell: 15, lines: [15, 17], text: 'Aquí sí se imprime `means_` (en la parte 1 había una errata). La covarianza común de LDA ≈ promedio de las 4 de clase: con Case 3, una matriz con correlación ≈ (0.5 + 0 − 0.5 + 0.8)/4 · 0.025 ≈ 0.005.' },
        { cell: 15, lines: [20, 23], text: 'Error y matriz 4×4: casi todo en la diagonal a 10 dB.' },
        { cell: 15, lines: [27, 33], text: 'QDA estima 4 covarianzas: recupera la correlación de cada clase (≈0.0125, 0, −0.0125, 0.02 fuera de la diagonal).' },
      ],
      shapes: [{ expr: 'ldamodel.means_', shape: '(4, 2)', meaning: 'medias estimadas' }, { expr: 'ldamodel.covariance_', shape: '(2, 2)', meaning: 'covarianza común' }, { expr: 'qdamodel.covariance_', shape: 'lista de 4 × (2, 2)', meaning: 'una por clase' }, { expr: 'confusion_matrix(...)', shape: '(4, 4)', meaning: 'reales × predichas' }],
      interpretation: 'Case 2 (ρ común): LDA ≈ QDA, porque la frontera óptima es lineal (report: 0.225 % / 0.200 % con ρ = 0 y 0 % / 0 % con ρ = 0.5). Case 3: QDA mejora porque modela cada orientación (10 dB: 6 frente a 2 errores; 5 dB: 6.3 % frente a 5.9 %). Con `sigma[0] *= 30` la diferencia es grande (12.7 % frente a 4.9 %). Son errores de **entrenamiento**: QDA, con más parámetros, siempre tiene ventaja aquí y no se mide sobreajuste.',
      versionNotes: [{ cell: 15, text: 'Las covarianzas de QDA impresas difieren en un factor 1000/999 entre scikit-learn 1.9 (tu entorno: divide por n) y 1.8 (la app: n − 1). Errores y matrices de confusión coinciden.' }],
      next: 'Dibujar fronteras (celda 17) y regiones (celdas 19–20).',
      whatIf: [
        { change: 'ρ = 0.5 común (Case 2)', effect: 'Errores 0 / 0 en train: con covarianza común correlacionada, LDA usa la distancia de Mahalanobis y separa perfectamente esta realización.' },
        { change: 'Evaluar en un test independiente', effect: 'Errores ligeramente mayores; la ventaja de QDA en Case 3 se mantendría porque hay 1000 muestras por clase.' },
      ],
    },
    {
      id: 'b_lines',
      title: 'Rectas de LDA (coef_ e intercept_)',
      cells: [16, 17],
      stage: 'resultados',
      problem: 'Dibujar las fronteras lineales a partir de los parámetros aprendidos.',
      purpose: 'Muestra que LDA aprende un discriminante lineal por clase, $g_i(x) = w_i^Tx + b_i$, y dibuja cada recta $g_i(x)=0$.',
      inputs: '`lda.coef_` (4, 2), `lda.intercept_` (4,), datos.',
      outputs: 'Scatter con 4 rectas.',
      prereqs: ['frontera', 'lda'],
      apis: [
        { name: 'lda.coef_', lib: 'sklearn', kind: 'atributo', does: 'Pesos $w_i$ de cada clase.', returns: '(n_classes, n_features)' },
        { name: 'lda.intercept_', lib: 'sklearn', kind: 'atributo', does: 'Términos independientes $b_i$.', returns: '(n_classes,)' },
      ],
      lines: [
        { cell: 17, lines: [14, 14], text: 'Dos valores de x₁ (mínimo y máximo) bastan para dibujar una recta.' },
        { cell: 17, lines: [18, 19], text: 'De $b + w_1x_1 + w_2x_2 = 0$ se despeja $x_2 = -(b + w_1x_1)/w_2$.' },
      ],
      interpretation: 'El markdown habla de «one vs. all», pero **ninguna** de estas rectas es exactamente una frontera entre dos clases: la frontera entre i y j es $g_i(x) = g_j(x)$, es decir, $(w_i - w_j)^Tx + (b_i - b_j) = 0$. Las regiones reales son las de la celda 19 (con `predict`).',
      corrections: [{ cell: 17, kind: 'concepto', original: 'y1 = -(b+x1*w1)/w2   # recta g_i(x) = 0', fixed: 'w = lda.coef_[i] - lda.coef_[j]; b = lda.intercept_[i] - lda.intercept_[j]\ny1 = -(b + x1*w[0]) / w[1]   # frontera entre las clases i y j', why: 'La frontera MAP entre dos clases es donde sus discriminantes se igualan. Las rectas originales se conservan, pero no deben interpretarse como fronteras.' }],
      next: 'Regiones de decisión con una malla.',
      whatIf: [{ change: 'Comparar con la figura de regiones', effect: 'Verás que los límites de color no coinciden con estas rectas.' }],
    },
    {
      id: 'b_regions_lda',
      title: 'Regiones de decisión de LDA (malla)',
      cells: [18, 19],
      stage: 'resultados',
      origin: 'modificado',
      problem: 'Ver qué clase predice el modelo en cada punto del plano.',
      purpose: 'Clasifica todos los puntos de una rejilla y los colorea: muestra las **fronteras reales** del clasificador.',
      inputs: '`lda`, `X_train`, paso `h`.',
      outputs: 'Mapa de regiones + muestras.',
      prereqs: ['meshgrid', 'frontera'],
      apis: [
        { name: 'np.meshgrid', lib: 'numpy', kind: 'función', does: 'Rejilla 2D a partir de dos vectores.', returns: 'xx, yy con forma (ny, nx)' },
        { name: 'np.arange', lib: 'numpy', kind: 'función', does: 'Valores equiespaciados con paso h.' },
        { name: 'np.c_[xx.ravel(), yy.ravel()]', lib: 'numpy', kind: 'operador', does: 'Convierte la rejilla en una matriz (nx·ny, 2) que acepta `predict`.' },
        { name: 'Z.reshape(xx.shape)', lib: 'numpy', kind: 'método', does: 'Devuelve las predicciones a forma de rejilla.' },
        { name: 'plt.contourf', lib: 'matplotlib', kind: 'función', does: 'Rellena regiones según el valor de Z.' },
      ],
      lines: [
        { cell: 19, lines: [5, 5], text: '**Modificado por ti**: `h = .001` en el original. Con .001 la malla tiene ≈ 4·10⁷ puntos y agota la memoria; .02 da ≈ 10⁵. Solo cambia la resolución del dibujo.' },
        { cell: 19, lines: [15, 18], text: 'Margen de 1 unidad alrededor de los datos y rejilla con paso h.' },
        { cell: 19, lines: [22, 25], text: 'Se predice cada punto y se recupera la forma de la malla.' },
        { cell: 19, lines: [31, 33], text: 'Usa `xdatas`/`xlabs` (los mismos datos que X_train/y_train).' },
      ],
      shapes: [{ expr: 'np.c_[xx.ravel(), yy.ravel()]', shape: '(nx·ny, 2)', meaning: 'un punto de la malla por fila' }, { expr: 'Z', shape: '(ny, nx)', meaning: 'clase predicha en cada nodo' }],
      interpretation: 'Las fronteras de LDA son **segmentos rectos**. En Case 3 no pueden seguir las distintas orientaciones de las elipses (por ejemplo, la clase 3, muy alargada, invade regiones vecinas).',
      corrections: [{ cell: 19, kind: 'modificado', original: 'h = .001 # step size in the mesh', fixed: 'h = .02 # Ajuste de memoria: original .001; solo cambia la resolución del dibujo', why: 'Cambio tuyo para que la celda no agote la memoria. No afecta al entrenamiento ni a los errores.' }],
      next: 'Lo mismo con QDA.',
      whatIf: [{ change: 'h = 0.1', effect: 'Fronteras pixeladas pero mismo modelo; h no es un hiperparámetro.' }],
    },
    {
      id: 'b_regions_qda',
      title: 'Regiones de decisión de QDA (malla)',
      cells: [20],
      stage: 'resultados',
      origin: 'modificado',
      problem: 'Comparar las regiones cuadráticas con las lineales.',
      purpose: 'Misma figura con `qda.predict`. Es la que se usa para Q11 y Q12.',
      inputs: '`qda`, malla.',
      outputs: 'Mapa de regiones de QDA.',
      prereqs: ['qda', 'meshgrid'],
      apis: [{ name: 'qda.predict', lib: 'sklearn', kind: 'método', does: 'Clase de máximo posterior con una covarianza por clase.' }],
      lines: [
        { cell: 20, lines: [5, 5], text: 'Mismo cambio de resolución (h = .02).' },
        { cell: 20, lines: [22, 22], text: 'Única diferencia con la celda 19: `qda` en lugar de `lda`.' },
      ],
      interpretation: 'Las fronteras de QDA se curvan: se adaptan a la orientación de cada elipse. Con `sigma[0] *= 30` la región de la clase 0 envuelve a las demás como una cónica: las otras clases quedan como «islas» más compactas. La salida guardada de esta celda muestra la figura y un error vacío (ejecución interrumpida en tu sesión).',
      corrections: [{ cell: 20, kind: 'modificado', original: 'h = .001 # step size in the mesh', fixed: 'h = .02', why: 'Igual que en la celda 19.' }],
      whatIf: [{ change: 'Case 2 con ρ = 0.5', effect: 'Las fronteras de QDA son casi rectas: estima covarianzas casi iguales.' }],
    },
  ],
  experiments: [
    {
      id: 'lab1_2.exp.cases',
      title: 'Forma de las covarianzas, SNR y fronteras de LDA/QDA',
      goal: 'Reproducir Q6–Q12 cambiando **un factor cada vez**: el caso (ρ por clase), la SNR o la varianza de la clase 0. Las regiones de decisión, los errores de entrenamiento y los autovalores teóricos y muestrales se recalculan a la vez.',
      blocks: ['b_cases', 'b_covar', 'b_fit', 'b_regions_lda', 'b_regions_qda'],
      template: EXPERIMENT,
      seedParam: 'seed',
      params: [
        { id: 'ro', label: 'Caso (ρ por clase)', kind: 'datos', control: 'select', default: '[0.5, 0, -0.5, 0.8]', options: [{ value: '[0, 0, 0, 0]', label: 'Case 2 · ρ = 0 (Q6)' }, { value: '[0.5, 0.5, 0.5, 0.5]', label: 'Case 2 · ρ = 0.5 (Q7)' }, { value: '[0.5, 0, -0.5, 0.8]', label: 'Case 3 (guardado)' }, { value: '[0.9, -0.9, -0.9, 0.9]', label: 'Case 3 extremo (app)' }], variable: 'ro', computation: 'Cambia el término fuera de la diagonal de cada `M_covar[k]` y, por tanto, la orientación y el alargamiento de cada elipse.', result: 'Con ρ común (Case 2) LDA ≈ QDA; con ρ distintas (Case 3) QDA aprende fronteras curvas y comete menos errores.', why: 'Si las covarianzas son iguales, el término cuadrático del discriminante es el mismo para todas las clases y se cancela.' },
        { id: 'SNR', label: 'SNR (dB)', kind: 'datos', control: 'select', default: '10', options: [{ value: '15', label: '15 dB' }, { value: '10', label: '10 dB (guardado)' }, { value: '5', label: '5 dB (Q9)' }, { value: '0', label: '0 dB' }], variable: 'SNR', computation: 'σ² = 0.5 / 10^(SNR/10): escala **todas** las covarianzas por igual.', result: 'Bajar la SNR agranda todas las elipses (mismas orientaciones) y aumenta el error.', why: 'Los autovalores son proporcionales a σ²: a 5 dB se multiplican por √10 ≈ 3.16 respecto a 10 dB.' },
        { id: 'sig0', label: 'Factor de sigma[0]', kind: 'datos', control: 'select', default: '1', options: [{ value: '1', label: '× 1' }, { value: '30', label: '× 30 (Q12)' }], variable: 'sigma[0] = sigma[0] * …', computation: 'Multiplica la covarianza de la clase 0 antes de construir M_covar.', result: 'La clase 0 se dispersa sobre las otras; LDA, con una covarianza común, se equivoca mucho más que QDA.', why: 'LDA promedia una covarianza enorme con tres pequeñas y no representa ninguna bien; QDA modela cada escala.' },
        { id: 'seed', label: 'Semilla', kind: 'datos', control: 'range', min: 0, max: 50, step: 1, default: '5', variable: 'default_rng(seed=…)', computation: 'Otras muestras del mismo modelo.', result: 'Pequeñas variaciones en el número de errores (de unas pocas muestras a 10 dB).', why: 'Variabilidad muestral: con pocos errores, diferencias de 1–4 muestras no son significativas.' },
        { id: 'h', label: 'Paso de la malla h', kind: 'visualizacion', control: 'select', default: '.02', options: [{ value: '.1', label: '0.1' }, { value: '.05', label: '0.05' }, { value: '.02', label: '0.02 (tu versión)' }, { value: '.01', label: '0.01' }], variable: 'h', computation: 'Solo cuántos puntos de la malla se clasifican para pintar.', result: 'Fronteras más o menos pixeladas; errores idénticos.', why: 'El modelo ya está entrenado; la malla solo sirve para dibujarlo.' },
        { id: 'lines', label: 'Dibujar rectas de la celda 17', kind: 'visualizacion', control: 'toggle', default: 'False', variable: 'if …:', computation: 'Añade la figura de las rectas $w_i^Tx+b_i=0$.', result: 'Puedes comparar las rectas con las regiones reales.', why: 'Las rectas no son fronteras entre pares de clases.' },
      ],
      metrics: [
        { key: 'sigma0', label: 'sigma[0] (varianza total clase 0)', format: 'num', digits: 4 },
        { key: 'err_lda', label: 'Error LDA (train)', format: 'pct', digits: 3, better: 'low' },
        { key: 'err_qda', label: 'Error QDA (train)', format: 'pct', digits: 3, better: 'low' },
        { key: 'ratio4', label: 'Razón de semiejes clase 3 √(λmax/λmin)', format: 'num', digits: 3 },
        { key: 'eig_teo', label: 'Autovalores teóricos (fila = clase)', format: 'matrix' },
        { key: 'eig_emp', label: 'Autovalores muestrales (eigvals)', format: 'matrix' },
        { key: 'cm_lda', label: 'Confusión LDA (train)', format: 'matrix' },
        { key: 'cm_qda', label: 'Confusión QDA (train)', format: 'matrix' },
      ],
      observe: [
        'Pasa de ρ = 0 a ρ = 0.5 (Case 2): la traza de C no cambia, pero ¿qué les pasa a los autovalores y a los errores?',
        'Con Case 3, compara 10 y 5 dB: ¿cambian las orientaciones de las elipses o solo su tamaño?',
        'Pon sigma[0] × 30: ¿qué forma toma la región de la clase 0 en QDA? ¿Por qué LDA no puede reproducirla?',
        'Cambia h: ¿cambia algún número de la tabla?',
      ],
      source: src([6, 8, 9, 11, 15, 17, 19, 20], { note: 'El cálculo de autovalores y la opción «Case 3 extremo» son añadidos de la app; las dos figuras de regiones se dibujan en una sola fila.' }),
      fidelity: [
        { metrics: { err_lda: 0.0015, err_qda: 0.0005, sigma0: 0.05 }, tol: 1e-9, source: 'Salidas guardadas de las celdas 9 y 15 (SNR = 10 dB, Case 3)' },
        { params: { SNR: '5' }, metrics: { err_lda: 0.06275, err_qda: 0.059 }, tol: 1e-9, source: 'Report resuelto de Lab1, Q9 (5 dB)' },
        { params: { sig0: '30' }, metrics: { err_lda: 0.127, err_qda: 0.04875, sigma0: 1.5 }, tol: 1e-9, source: 'Report resuelto de Lab1, Q12' },
        { params: { ro: '[0, 0, 0, 0]' }, metrics: { err_lda: 0.00225, err_qda: 0.002 }, tol: 1e-9, source: 'Report resuelto de Lab1, Q6' },
        { params: { ro: '[0.5, 0.5, 0.5, 0.5]' }, metrics: { err_lda: 0, err_qda: 0 }, tol: 1e-9, source: 'Report resuelto de Lab1, Q7' },
      ],
    },
  ],
  activities: [
    // ---------------------------------------------------------- nivel 1
    {
      kind: 'mcq',
      id: 'lab1_2.read.covar',
      skills: ['codigo', 'concepto'],
      blocks: ['b_covar'],
      source: src([9]),
      prompt: '¿Qué representa `M_covar[3]` con `ro = [0.5, 0, -0.5, 0.8]` y σ² = 0.05?',
      code: 'M_covar[i,:,:] = sigma[i] / n_feat * np.array([[1, ro[i]],[ro[i],1]])',
      options: [
        { text: 'La covarianza de la clase 3: varianzas 0.025 y covarianza 0.02', correct: true },
        { text: 'La covarianza de la clase 4 con varianzas 0.05', feedback: 'Las clases se numeran desde 0 y la varianza por feature es σ²/2.' },
        { text: 'La matriz de correlación de la clase 3', feedback: 'La correlación tendría unos en la diagonal; aquí está multiplicada por σ²/2.' },
        { text: 'La covarianza común de todas las clases', feedback: 'En Case 3 cada clase tiene la suya.' },
      ],
      explanation: '`sigma[3]/2 = 0.025` en la diagonal y `0.025·0.8 = 0.02` fuera. Coincide con la salida guardada de la celda 9.',
    },
    {
      kind: 'mcq',
      id: 'lab1_2.read.mesh',
      skills: ['codigo', 'api'],
      blocks: ['b_regions_lda'],
      source: src([19]),
      prompt: '¿Para qué sirve `np.c_[xx.ravel(), yy.ravel()]` en la celda 19?',
      options: [
        { text: 'Convertir la rejilla en una matriz (n_puntos, 2) para que `predict` la acepte', correct: true },
        { text: 'Entrenar el modelo con más puntos', feedback: 'No se entrena nada: el modelo ya está ajustado.' },
        { text: 'Calcular las fronteras de forma analítica', feedback: 'Las fronteras se ven porque cambia la clase predicha entre puntos vecinos.' },
        { text: 'Concatenar X_train con la malla', feedback: 'Solo une las coordenadas x e y de la malla.' },
      ],
      explanation: '`ravel` aplana cada rejilla (ny, nx) a un vector y `np.c_` los pone como columnas: el formato `[n_samples, n_features]` de scikit-learn.',
    },
    {
      kind: 'text',
      id: 'lab1_2.read.lines',
      skills: ['codigo', 'concepto'],
      blocks: ['b_lines'],
      source: src([17]),
      figure: { notebook: NB, cell: 17, output: 0 },
      prompt: 'La celda 17 dibuja, para cada clase i, la recta $-(b_i + w_{i1}x_1)/w_{i2}$. ¿Qué representa realmente cada recta y por qué no es la frontera entre dos clases?',
      rubric: [
        'Cada recta es el conjunto de puntos donde el discriminante lineal de la clase i vale 0: $w_i^Tx + b_i = 0$.',
        'La decisión MAP compara discriminantes: se elige la clase con mayor $g_i(x)$.',
        'La frontera entre i y j es $g_i(x) = g_j(x)$, es decir, $(w_i - w_j)^Tx + (b_i - b_j) = 0$.',
        'Las regiones reales se ven con predict sobre una malla (celda 19).',
      ],
      model: 'Cada recta corresponde a $g_i(x) = w_i^Tx + b_i = 0$ con los parámetros `coef_[i]` e `intercept_[i]`. Pero el clasificador no mira el signo de un discriminante aislado, sino cuál es el mayor; la frontera entre las clases i y j está donde $g_i(x) = g_j(x)$, la recta $(w_i-w_j)^Tx + (b_i-b_j) = 0$. Por eso las rectas de la celda 17 no delimitan las regiones; las fronteras reales son los límites de color de la celda 19.',
    },
    // ---------------------------------------------------------- nivel 2
    {
      kind: 'predict',
      id: 'lab1_2.pred.shape_grid',
      skills: ['codigo'],
      blocks: ['b_regions_lda'],
      source: src([19]),
      prompt: 'Con `X_train` de la celda 15 y `h = .02`, ¿cuántas **columnas** tiene `np.c_[xx.ravel(), yy.ravel()]`?',
      setup: lab1_2Setup(),
      code: "h = .02\nX = X_train\nx_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1\ny_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1\nxx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))\ngrid = np.c_[xx.ravel(), yy.ravel()]",
      expr: 'grid.shape[1]',
      answerKind: 'numeric',
      tolerance: 0,
      explanation: '2 columnas: una por feature (x₁, x₂). El número de filas es nx·ny (≈ 1.2·10⁵ con h = .02).',
    },
    {
      kind: 'predict',
      id: 'lab1_2.pred.sigma',
      skills: ['concepto', 'codigo'],
      blocks: ['b_covar'],
      source: src([9]),
      prompt: 'A **5 dB**, ¿cuánto vale `sig` en la celda 9? (4 decimales)',
      setup: 'import numpy as np\nM_Means = 0.5 * np.array([[1, 1],[1, -1], [-1, 1], [-1,-1]])\nn_classes = 4\nSNR = 5',
      code: 'energy = 0\nfor i in range(0,n_classes):\n  energy = energy + np.dot(M_Means[i],M_Means[i])\nenergy = energy / n_classes\nSNR = 10 ** (SNR/10)\nsig = energy / SNR',
      expr: 'sig',
      answerKind: 'numeric',
      tolerance: 0.0005,
      explanation: 'Energía 0.5; 10^0.5 = 3.162 → σ² = 0.1581. En esta parte no se divide por n_feat en esa línea.',
    },
    {
      kind: 'predict',
      id: 'lab1_2.pred.lda_cov',
      skills: ['api'],
      blocks: ['b_fit'],
      source: src([15]),
      prompt: '¿Qué forma tiene `ldamodel.covariance_`? Escríbela como tupla.',
      setup: lab1_2Setup(),
      code: 'ldamodel = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)',
      expr: 'ldamodel.covariance_.shape',
      answerKind: 'shape',
      explanation: 'LDA estima **una** covarianza común (2, 2), aunque haya 4 clases. QDA guardaría una lista de 4 matrices (2, 2).',
    },
    // ---------------------------------------------------------- nivel 3
    {
      kind: 'code',
      level: 3,
      id: 'lab1_2.fill.covar',
      title: 'Completa la construcción de las covarianzas',
      skills: ['codigo'],
      blocks: ['b_covar'],
      source: src([9]),
      prompt: 'Rellena los huecos para construir `M_covar` con $C_k=\\frac{\\sigma_k^2}{2}\\begin{pmatrix}1&\\rho_k\\\\\\rho_k&1\\end{pmatrix}$.',
      available: '`np`, `sigma` (4,) = [0.05]*4, `ro = [0.5, 0, -0.5, 0.8]`, `n_feat = 2`, `n_classes = 4`.',
      produce: '`M_covar` (4, 2, 2).',
      setup: 'import numpy as np\nsigma = 0.05 * np.array([1, 1, 1, 1])\nro = [0.5, 0, -0.5, 0.8]\nn_feat = 2\nn_classes = 4',
      starter: 'M_covar = np.zeros(shape=(___, ___, ___))\nfor i in range(0, n_classes):\n  M_covar[i,:,:] = sigma[i] / ___ * np.array([[1, ___],[___, 1]])',
      solution: 'M_covar = np.zeros(shape=(4, 2, 2))\nfor i in range(0, n_classes):\n  M_covar[i,:,:] = sigma[i] / n_feat * np.array([[1, ro[i]],[ro[i], 1]])',
      alternatives: ['M_covar = np.array([s / 2 * np.array([[1, r], [r, 1]]) for s, r in zip(sigma, ro)])'],
      wrong: [{ code: 'M_covar = np.zeros(shape=(4, 2, 2))\nfor i in range(0, n_classes):\n  M_covar[i,:,:] = sigma[i] * np.array([[1, ro[i]],[ro[i], 1]])', expect: 'razonamiento' }],
      checks: "c.shape('M_covar', (4, 2, 2))\nc.same('M_covar', rtol=1e-12)",
      hints: ['La forma es (n_classes, n_feat, n_feat).', 'La varianza total se reparte entre las 2 features: divide por `n_feat`.', 'La correlación de la clase i es `ro[i]` y va fuera de la diagonal.'],
    },
    {
      kind: 'code',
      level: 3,
      id: 'lab1_2.fill.regions',
      title: 'Completa el cálculo de las regiones de decisión',
      skills: ['api', 'codigo'],
      blocks: ['b_regions_lda'],
      source: src([19]),
      practice: 'np.meshgrid, np.c_, reshape',
      prompt: 'Completa el código para obtener `Z`, la clase que predice QDA en cada nodo de la malla, con la misma forma que `xx`.',
      available: '`np`, `X_train` (4000, 2), `qda` entrenado (Case 3, 10 dB).',
      produce: '`xx`, `yy` (rejilla con paso h = 0.05 y margen 1) y `Z` con forma `xx.shape`.',
      setup: lab1_2Setup() + 'qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)\n',
      starter: 'h = 0.05\nx_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1\ny_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1\nxx, yy = np.___(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))\nZ = qda.predict(np.c_[___, ___])\nZ = Z.___(___)',
      solution: 'h = 0.05\nx_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1\ny_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1\nxx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))\nZ = qda.predict(np.c_[xx.ravel(), yy.ravel()])\nZ = Z.reshape(xx.shape)',
      alternatives: ['h = 0.05\nx_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1\ny_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1\nxx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))\nZ = qda.predict(np.column_stack([xx.flatten(), yy.flatten()])).reshape(xx.shape)'],
      wrong: [{ code: 'h = 0.05\nx_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1\ny_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1\nxx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))\nZ = qda.predict(np.c_[xx.ravel(), yy.ravel()])', expect: 'dimensiones' }],
      checks: "c.same('xx')\nc.shape('Z', ref_name='Z')\nc.same('Z')",
      hints: ['La rejilla se crea con `np.meshgrid`.', '`predict` necesita una matriz (n_puntos, 2): aplana cada rejilla con `.ravel()`.', 'Recupera la forma de la malla con `Z.reshape(xx.shape)`.'],
    },
    // ---------------------------------------------------------- nivel 4
    {
      kind: 'code',
      level: 4,
      id: 'lab1_2.fix.sigma30',
      title: 'Corrige dónde se aplica sigma[0] *= 30',
      skills: ['codigo', 'concepto'],
      blocks: ['b_covar'],
      source: src([9], { doc: 'lab1_guide', section: 'Q11 guía / Q12 report' }),
      prompt: 'Este código intenta multiplicar la covarianza de la clase 0 por 30 (Q12), pero no tiene ningún efecto sobre los datos. Corrígelo.',
      available: '`np`, `sig = 0.05`, `ro`, `n_feat`, `n_classes`.',
      produce: '`sigma` (4,) y `M_covar` (4, 2, 2) con la clase 0 multiplicada por 30.',
      setup: 'import numpy as np\nsig = 0.05\nro = [0.5, 0, -0.5, 0.8]\nn_feat = 2\nn_classes = 4',
      starter: 'M_covar = np.zeros(shape=(4,2,2))\nsigma = sig * np.array([1,1,1,1])\nfor i in range(0,n_classes):\n  M_covar[i,:,:] = sigma[i] / n_feat * np.array([[1, ro[i]],[ro[i],1]])\nsigma[0] *= 30',
      solution: 'M_covar = np.zeros(shape=(4,2,2))\nsigma = sig * np.array([1,1,1,1])\nsigma[0] *= 30\nfor i in range(0,n_classes):\n  M_covar[i,:,:] = sigma[i] / n_feat * np.array([[1, ro[i]],[ro[i],1]])',
      alternatives: ['M_covar = np.zeros(shape=(4,2,2))\nsigma = sig * np.array([30.,1,1,1])\nfor i in range(0,n_classes):\n  M_covar[i,:,:] = sigma[i] / n_feat * np.array([[1, ro[i]],[ro[i],1]])'],
      checks: "c.same('sigma', rtol=1e-12)\nc.same('M_covar', rtol=1e-12)",
      hints: ['Las muestras se generan con `M_covar`, no con `sigma`.', '¿Qué pasa con `M_covar[0]` si cambias `sigma[0]` después del bucle?'],
      explanation: 'Nuevo `sigma[0]` = 1.5; $C_0$ = [[0.75, 0.375], [0.375, 0.75]], autovalores 0.375 y 1.125 (report Q12).',
    },
    {
      kind: 'code',
      level: 4,
      id: 'lab1_2.fix.pairwise',
      title: 'Corrige la frontera entre dos clases',
      skills: ['concepto', 'api'],
      blocks: ['b_lines'],
      source: src([17]),
      prompt: 'El código calcula la recta de la celda 17 para la clase 0 y la presenta como «frontera entre las clases 0 y 1». Corrígelo para obtener la **verdadera** frontera LDA entre 0 y 1 como $x_2 = a + m\\,x_1$.',
      available: '`lda` entrenado (Case 3, 10 dB) con `coef_` (4, 2) e `intercept_` (4,).',
      produce: '`m` (pendiente) y `a` (ordenada en el origen) de la frontera entre las clases 0 y 1.',
      setup: lab1_2Setup() + 'lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)\n',
      starter: 'b, w1, w2 = lda.intercept_[0], lda.coef_[0][0], lda.coef_[0][1]\nm = -w1 / w2\na = -b / w2',
      solution: 'w = lda.coef_[0] - lda.coef_[1]\nb = lda.intercept_[0] - lda.intercept_[1]\nm = -w[0] / w[1]\na = -b / w[1]',
      alternatives: ['dw = lda.coef_[1] - lda.coef_[0]\ndb = lda.intercept_[1] - lda.intercept_[0]\nm = -dw[0] / dw[1]\na = -db / dw[1]'],
      checks: "c.same('m', rtol=1e-8, desc='La pendiente')\nc.same('a', rtol=1e-8, desc='La ordenada en el origen')",
      hints: ['La frontera entre i y j está donde $g_i(x) = g_j(x)$.', 'Resta los pesos y los términos independientes de las dos clases y despeja x₂.'],
      explanation: 'Las medias 0 = (0.5, 0.5) y 1 = (0.5, −0.5) difieren en x₂: la frontera es casi horizontal (pendiente pequeña, ordenada ≈ 0).',
    },
    {
      kind: 'mcq',
      id: 'lab1_2.misconception.h',
      skills: ['concepto', 'interpretacion'],
      blocks: ['b_regions_lda'],
      source: src([19]),
      prompt: '«Al pasar `h` de .001 a .02 he cambiado el clasificador, por eso mis errores no coinciden con los del report». ¿Qué error contiene esta afirmación?',
      options: [
        { text: 'h solo controla la resolución de la malla de dibujo; no interviene en el entrenamiento ni en el error', correct: true },
        { text: 'Ninguno: h es un hiperparámetro de LDA', feedback: 'LDA no tiene ningún argumento h; h aparece después del `fit`.' },
        { text: 'h cambia el número de muestras de entrenamiento', feedback: 'Las muestras salen de la celda 11.' },
      ],
      explanation: 'El error se calcula en la celda 15, antes de crear la malla. Si los errores no coinciden, la causa estará en la SNR, en ρ o en el estado del generador.',
    },
    // ---------------------------------------------------------- nivel 5
    {
      kind: 'code',
      level: 5,
      id: 'lab1_2.mem.eig',
      title: 'Autovalores teóricos y muestrales (Q10 del report)',
      skills: ['escritura', 'api', 'concepto'],
      blocks: ['b_covar', 'b_fit'],
      source: src([9, 15], { doc: 'lab1_report', section: 'Q10' }),
      practice: 'scipy.linalg.eigvals',
      prompt: 'Para cada clase k, calcula los autovalores **teóricos** con la fórmula de Q1 y los de la **covarianza muestral** de sus datos de entrenamiento con `scipy.linalg.eigvals`, ambos ordenados de menor a mayor.',
      available: '`np`, `X_train` (4000, 2), `y_train`, `sigma` (4,), `ro` (Case 3, 10 dB).',
      produce: '`teo` y `emp`, arrays (4, 2): fila k = autovalores ordenados de la clase k.',
      setup: lab1_2Setup(),
      starter: '',
      solution: 'from scipy.linalg import eigvals\nteo = np.array([np.sort(sigma[k] / 2 * np.array([1 - ro[k], 1 + ro[k]])) for k in range(4)])\nemp = np.array([np.sort(eigvals(np.cov(X_train[y_train == k].T, ddof=1)).real) for k in range(4)])',
      alternatives: ['import scipy.linalg\nteo = np.zeros((4, 2)); emp = np.zeros((4, 2))\nfor k in range(4):\n    lam = sigma[k] / 2\n    teo[k] = sorted([lam * (1 - abs(ro[k])), lam * (1 + abs(ro[k]))])\n    C = np.cov(X_train[y_train == k], rowvar=False)\n    emp[k] = np.sort(np.real(scipy.linalg.eigvals(C)))'],
      wrong: [{ code: 'from scipy.linalg import eigvals\nteo = np.array([np.sort(sigma[k] * np.array([1 - ro[k], 1 + ro[k]])) for k in range(4)])\nemp = np.array([np.sort(eigvals(np.cov(X_train[y_train == k].T)).real) for k in range(4)])', expect: 'razonamiento' }],
      checks: "c.same('teo', rtol=1e-10, desc='Autovalores teóricos')\nc.same('emp', rtol=1e-8, desc='Autovalores muestrales')",
      hints: ['Q1: $\\lambda = \\frac{\\sigma^2}{2}(1 \\pm \\rho)$.', '`np.cov` espera variables en filas: transpón o usa `rowvar=False`.', '`eigvals` devuelve complejos: quédate con `.real` y ordena.'],
      explanation: 'A 10 dB: teóricos (0.0125, 0.0375), (0.025, 0.025), (0.0125, 0.0375), (0.005, 0.045); los muestrales difieren en el tercer decimal por el tamaño finito de la muestra.',
    },
    {
      kind: 'code',
      level: 5,
      id: 'lab1_2.mem.compare',
      title: 'Case 2 vs Case 3 sin ayuda',
      skills: ['escritura', 'metodo'],
      blocks: ['b_cases', 'b_covar', 'b_generate', 'b_fit'],
      source: src([6, 8, 9, 11, 15]),
      prompt: 'Genera los datos QPSK a **10 dB** para Case 2 con ρ = 0.5 en todas las clases (semilla 5, 1000 muestras por clase, clases 0–3 en orden, sin barajar) y calcula el error de entrenamiento de LDA y QDA.',
      available: 'Nada importado.',
      produce: '`X` (4000, 2), `y` (4000,), `err_lda`, `err_qda`.',
      setup: '',
      starter: '',
      solution: 'import numpy as np\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nrng = np.random.default_rng(seed=5)\nM = 0.5 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])\nsig = 0.5 / 10 ** (10 / 10)\nC = sig / 2 * np.array([[1, 0.5], [0.5, 1]])\nX = np.vstack([rng.multivariate_normal(M[k], C, 1000) for k in range(4)])\ny = np.repeat(np.arange(4), 1000)\nerr_lda = 1 - LinearDiscriminantAnalysis().fit(X, y).score(X, y)\nerr_qda = 1 - QuadraticDiscriminantAnalysis().fit(X, y).score(X, y)',
      checks: "c.shape('X', (4000, 2))\nc.same('X', rtol=1e-8, desc='Las muestras (misma semilla y orden)')\nc.same('y')\nc.same('err_lda', rtol=1e-9)\nc.same('err_qda', rtol=1e-9)",
      hints: ['σ² = energía / SNR lineal con energía 0.5.', 'Covarianza común: (σ²/2)·[[1, 0.5], [0.5, 1]].', '`.score(X, y)` devuelve la accuracy.'],
    },
    // ---------------------------------------------------------- nivel 6
    {
      kind: 'code',
      level: 6,
      id: 'lab1_2.adapt.test',
      title: 'Adapta: evalúa el Case 3 en un test independiente',
      skills: ['metodo', 'escritura'],
      blocks: ['b_fit', 'b_generate'],
      source: src([11, 15]),
      transfer: true,
      prompt: 'El notebook evalúa sobre entrenamiento. Genera un **test** de 1000 muestras por clase con el mismo `rng` (continúa su estado, clases 0–3 en orden) y calcula el error de test de LDA y QDA entrenados con `X_train`.',
      available: '`np`, `rng` (estado tras generar train), `M_Means`, `M_covar` (Case 3, 10 dB), `X_train`, `y_train`, clases de scikit-learn importadas.',
      produce: '`X_test` (4000, 2), `y_test`, `err_lda_test`, `err_qda_test`.',
      setup: lab1_2Setup(),
      starter: '',
      solution: 'X_test = np.vstack([rng.multivariate_normal(M_Means[k], M_covar[k], 1000) for k in range(4)])\ny_test = np.repeat(np.arange(4), 1000)\nlda = LinearDiscriminantAnalysis().fit(X_train, y_train)\nqda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)\nerr_lda_test = 1 - accuracy_score(y_test, lda.predict(X_test))\nerr_qda_test = 1 - accuracy_score(y_test, qda.predict(X_test))',
      checks: "c.same('X_test', rtol=1e-8)\nc.same('err_lda_test', rtol=1e-9)\nc.same('err_qda_test', rtol=1e-9)",
      hints: ['Reutiliza el mismo bucle de la celda 11 con `rng`.', 'No vuelvas a entrenar con el test.'],
      explanation: 'Con 1000 muestras por clase, el error de test es del mismo orden que el de train: el sobreajuste de QDA es pequeño.',
    },
    // ---------------------------------------------------------- interpretación
    {
      kind: 'mcq',
      id: 'lab1_2.int.scatter',
      skills: ['interpretacion'],
      blocks: ['b_scatter'],
      figure: { notebook: NB, cell: 13, output: 0 },
      source: src([13]),
      prompt: 'Figura guardada (Case 3, 10 dB; ¡ojo con los colores de esta celda!). ¿Qué clase corresponde a la nube **más alargada**, y por qué?',
      options: [
        { text: 'La clase 3 (−0.5, −0.5), con ρ = 0.8: razón de semiejes 3', correct: true },
        { text: 'La clase 0, porque tiene ρ = 0.5', feedback: 'Es elíptica, pero menos: razón √3 ≈ 1.73.' },
        { text: 'La clase 1, porque tiene ρ = 0', feedback: 'ρ = 0 da un círculo.' },
        { text: 'Todas igual de alargadas, porque tienen la misma σ²', feedback: 'Misma traza, distinta distribución entre autovalores.' },
      ],
      explanation: 'Razón de semiejes $\\sqrt{\\lambda_{max}/\\lambda_{min}}=\\sqrt{(1+\\rho)/(1-\\rho)}$: 3 para ρ = 0.8. En esta figura la clase 3 es **verde** (en las siguientes es roja).',
    },
    {
      kind: 'mcq',
      id: 'lab1_2.int.regions',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_regions_lda', 'b_regions_qda'],
      figure: { experiment: 'lab1_2.exp.cases', params: { sig0: '30' }, figureIndex: 0 },
      source: src([19, 20], { note: 'Figura generada con sigma[0] × 30' }),
      prompt: 'Con `sigma[0] *= 30` (figura generada), ¿qué explica mejor la diferencia entre las regiones de LDA (izquierda) y QDA (derecha)?',
      options: [
        { text: 'QDA puede rodear a las clases compactas con una región curva de la clase dispersa; LDA solo puede usar rectas con una covarianza promedio', correct: true },
        { text: 'QDA usa más datos de entrenamiento', feedback: 'Ambos usan los mismos 4000 puntos.' },
        { text: 'LDA tiene peor resolución de malla', feedback: 'La malla es la misma para los dos.' },
        { text: 'Los priors de la clase 0 son mayores', feedback: 'Siguen siendo 0.25 para todas.' },
      ],
      explanation: 'Report Q12: LDA 12.7 % y QDA 4.875 % de error en train; QDA acierta 841/1000 de la clase 0 frente a 505 de LDA.',
    },
    {
      kind: 'text',
      id: 'lab1_2.int.q8',
      skills: ['interpretacion', 'concepto'],
      blocks: ['b_fit', 'b_covar'],
      source: { notebook: NB, cells: [15], doc: 'lab1_report', section: 'Q8', origin: 'original' },
      prompt: 'Q8 · Compara Case 2 con ρ = 0 (LDA 0.225 %, QDA 0.200 %) y con ρ = 0.5 (0 % y 0 %), ambos a 10 dB. ¿Por qué mejora el error si la traza de la covarianza es la misma?',
      rubric: [
        'La traza (σ² = 0.05) no cambia: ρ redistribuye la varianza entre direcciones.',
        'Autovalores: (0.025, 0.025) → (0.0125, 0.0375); las nubes pasan de círculos a elipses a lo largo de (1, 1).',
        'Las medias vecinas están separadas en direcciones donde ahora hay menos varianza, lo que reduce el solapamiento en esas direcciones.',
        'LDA y QDA rinden igual porque la covarianza es común (frontera óptima lineal).',
        'Son errores de train en una realización: no implica que toda correlación reduzca el error.',
      ],
      model: 'Con ρ = 0.5 la varianza total sigue siendo 0.05, pero se reparte: 0.0375 a lo largo de (1, 1) y 0.0125 a lo largo de (1, −1). Los símbolos vecinos de QPSK están separados horizontal y verticalmente; la dispersión relevante para cada frontera cambia y en esta realización ninguna muestra cae en la región equivocada. Como la covarianza es común, la frontera óptima sigue siendo lineal y LDA (que usa la distancia de Mahalanobis con esa covarianza) rinde igual que QDA. Son errores de entrenamiento de una sola semilla: no puede generalizarse que cualquier correlación mejore el error.',
    },
    {
      kind: 'numeric',
      id: 'lab1_2.int.eig08',
      skills: ['concepto'],
      blocks: ['b_covar'],
      source: src([9], { doc: 'lab1_report', section: 'Q1' }),
      prompt: 'A 10 dB (σ² = 0.05), ¿cuál es el autovalor **mayor** de la covarianza de la clase con ρ = 0.8?',
      answer: 0.045,
      tolerance: 0.0005,
      explanation: '$\\frac{0.05}{2}(1 + 0.8) = 0.045$; el menor es 0.005.',
    },
    {
      kind: 'dynamic',
      id: 'lab1_2.dyn.eig',
      skills: ['concepto'],
      blocks: ['b_covar'],
      source: src([9], { note: 'Variante generada' }),
      answerKind: 'numeric',
      generator: `import numpy as np
rng = np.random.default_rng(SEED)
snr = int(rng.choice([0, 3, 5, 7, 10, 13]))
rho = float(rng.choice([-0.8, -0.6, -0.3, 0.2, 0.4, 0.7, 0.9]))
s2 = 0.5 / 10**(snr/10)
cual = str(rng.choice(['mayor', 'menor']))
val = s2 / 2 * (1 + abs(rho)) if cual == 'mayor' else s2 / 2 * (1 - abs(rho))
variante = {
  'prompt': f'QPSK con energía media 0.5, SNR = {snr} dB y ρ = {rho}. ¿Cuál es el autovalor **{cual}** de la covarianza C = (σ²/2)·[[1, ρ], [ρ, 1]]? (5 decimales)',
  'answer': round(val, 7), 'tolerance': 5e-5,
  'explanation': f'σ² = 0.5/10^({snr}/10) = {s2:.5f}; λ = σ²/2·(1 ± |ρ|) = {s2/2:.5f}·(1 ± {abs(rho)}) → {val:.5f}.'
}`,
    },
    {
      kind: 'mcq',
      id: 'lab1_2.int.train',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_fit'],
      source: src([15]),
      prompt: 'En Case 3 a 10 dB, LDA comete 6 errores y QDA 2, **en entrenamiento**. ¿Qué conclusión no está justificada?',
      options: [
        { text: 'QDA tendrá exactamente un tercio de los errores de LDA en datos nuevos', correct: true, feedback: 'Correcto: es la conclusión injustificada. Con 6 y 2 errores en train no se puede cuantificar el error en test.' },
        { text: 'QDA se adapta mejor a las orientaciones distintas de cada clase' },
        { text: 'A 10 dB ambos modelos clasifican casi todo bien' },
      ],
      explanation: 'Los errores de entrenamiento son optimistas y la diferencia es de 4 muestras en una realización. Para cuantificar haría falta un test (ejercicio de adaptación) o varias semillas.',
    },
    // ---------------------------------------------------------- cobertura: imports y parámetros
    {
      kind: 'mcq',
      id: 'lab1_2.cov.imports',
      skills: ['concepto', 'codigo'],
      blocks: ['b_imports'],
      source: src([3, 11]),
      prompt: 'La celda 3 crea `rng = default_rng(seed=5)`, pero la celda 11 baraja con `np.random.permutation`. ¿Qué es reproducible entre ejecuciones?',
      options: [
        { text: 'Las muestras generadas y los errores; no el orden de las filas tras barajar', correct: true },
        { text: 'Todo, porque la semilla se fija al principio', feedback: '`default_rng(5)` solo controla `rng`; `np.random.permutation` usa el generador global, sin semilla.' },
        { text: 'Nada, porque hay una parte aleatoria sin semilla', feedback: 'Los datos sí se reproducen; LDA/QDA no dependen del orden de las filas.' },
        { text: 'Solo el orden de las filas', feedback: 'Es al revés.' },
      ],
      explanation: 'Errores y matrices de confusión coinciden en cada ejecución porque los datos (salidos de `rng`) son los mismos; solo cambia el orden de las filas, que no afecta al entrenamiento.',
    },
    {
      kind: 'predict',
      id: 'lab1_2.cov.sig',
      skills: ['concepto', 'codigo'],
      blocks: ['b_params'],
      source: src([6, 9]),
      prompt: 'Con las medias QPSK $(\\pm0.5, \\pm0.5)$ y **SNR = 10 dB**, ¿cuánto vale `sig` en la celda 9? (aquí la línea `sig = sig / n_feat` está comentada). 3 decimales.',
      setup: 'import numpy as np\nSNR = 10\nn_classes = 4\nM_Means = 0.5 * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])',
      code: 'energy = 0\nfor i in range(0,n_classes):\n  energy = energy + np.dot(M_Means[i],M_Means[i])\nenergy = energy / n_classes\nSNR = 10 ** (SNR/10)\nsig = energy / SNR',
      expr: 'float(sig)',
      answerKind: 'numeric',
      tolerance: 0.0005,
      explanation: 'Cada media tiene energía 0.25 + 0.25 = 0.5; SNR lineal = 10 → σ² = 0.05 (coincide con `print(sigma)` de la celda 9). El reparto entre las 2 features se hace al construir `sigma[i] / n_feat * [[1, ρ], [ρ, 1]]`.',
    },
  ],
  recall: ['lab1_2.fill.covar', 'lab1_2.fill.regions', 'lab1_2.fix.sigma30', 'lab1_2.fix.pairwise', 'lab1_2.mem.eig', 'lab1_2.mem.compare', 'lab1_2.adapt.test'],
  practice: [
    {
      id: 'lab1_2.practice.report',
      title: 'Mini-report: forma de las covarianzas (Q1, Q8, Q10, Q12)',
      scenario: 'Prepara sin el notebook la parte 2 del report: construye las covarianzas, coloca bien el cambio de `sigma[0]`, calcula la tabla de autovalores, obtén las regiones de decisión y justifica las diferencias entre LDA y QDA.',
      steps: ['lab1_2.int.eig08', 'lab1_2.fill.covar', 'lab1_2.fix.sigma30', 'lab1_2.mem.eig', 'lab1_2.fill.regions', 'lab1_2.int.regions', 'lab1_2.int.q8'],
      source: { notebook: NB, doc: 'lab1_report', section: 'Q1, Q8, Q10, Q12', origin: 'original' },
    },
  ],
  notes: [
    'La numeración de preguntas difiere entre la guía (Q1–Q11) y el report (Q1–Q12): el report añade la tabla de autovalores como Q10. Se citan ambas.',
    'Las métricas de esta parte son de **entrenamiento**, como en el notebook y en el report resuelto.',
  ],
};
