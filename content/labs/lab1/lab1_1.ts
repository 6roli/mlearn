import type { NotebookContent } from '../../../src/content/types';

const NB = 'lab1_1';
const src = (cells: number[], extra: Partial<{ section: string; note: string }> = {}) => ({
  notebook: NB,
  cells,
  origin: 'original' as const,
  ...extra,
});

/** Estado del notebook hasta el conjunto de test (celdas 3–14, 18, 34) con SNR configurable. */
export const lab1_1Setup = (snr = 3) => `import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from numpy.random import default_rng
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
SNR = ${snr}
dist = 1.
n_classes = 2
n_samples = 1000
n_feat = 3
M_Means = 0.5 * dist * np.array([[1, 1, 1], [-1, -1, -1]]) / np.sqrt(n_feat)
energy = 0
for i in range(0, n_classes):
    energy = energy + np.dot(M_Means[i], M_Means[i])
energy = energy / n_classes
SNR = 10 ** (SNR / 10)
sig = energy / SNR
sig = sig / n_feat
M_covar = np.zeros(shape=(2, 3, 3))
sigma = sig * np.array([1, 1])
for i in range(0, n_classes):
    M_covar[i, :, :] = sigma[i] * np.eye(N=n_feat, M=n_feat)
rng = default_rng(seed=5)
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
X_test = np.empty((0, n_feat), float)
y_test = np.empty((0), int)
for i in range(0, n_classes):
    ydata = rng.multivariate_normal(mean=M_Means[i, :], cov=M_covar[i, :, :], size=n_samples)
    X_test = np.append(X_test, ydata, axis=0)
    y_test = np.append(y_test, i * np.ones((n_samples), dtype=int), axis=0)
rp = np.random.permutation(len(X_test))
X_test = X_test[rp]
y_test = y_test[rp]
`;

const SETUP_PARAMS = `import numpy as np
dist = 1.
n_classes = 2
n_samples = 1000
n_feat = 3
M_Means = 0.5 * dist * np.array([[1, 1, 1], [-1, -1, -1]]) / np.sqrt(n_feat)
`;

const EXPERIMENT = `import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay, roc_auc_score
from scipy.stats import norm

# celdas 6-7 · parámetros de diseño
SNR = {{SNR}}
dist = {{dist}}
n_classes = 2
n_samples = {{n_samples}}
n_feat = 3
M_Means = 0.5 * dist * np.array([[1, 1, 1],[-1,-1,-1]]) / np.sqrt(n_feat)

# celdas 8-10 · energía, varianza del ruido y covarianzas (Case 1: C = sigma^2 I)
energy = 0
for i in range(0,n_classes):
  energy = energy + np.dot(M_Means[i],M_Means[i])
energy = energy / n_classes
SNR_dB = SNR                      # (app) guardamos el valor en dB antes de sobrescribirlo
SNR = 10 ** (SNR/10)
sig = energy / SNR
sig = sig / n_feat
M_covar = np.zeros(shape=(2,3,3))
sigma = sig * np.array([1,1])
for i in range(0,n_classes):
  M_covar[i,:,:] = sigma[i] * np.eye(N=n_feat,M=n_feat)

# celdas 13-14 y 18 · conjunto de entrenamiento
rng = default_rng(seed={{seed}})
xdata = np.empty((0,n_feat), float)
labels = np.empty((0), int)
for i in range(0,n_classes):
  ydata = rng.multivariate_normal(mean=M_Means[i,:], cov= M_covar[i,:,:], size= n_samples)
  ylabels = i * np.ones((n_samples),dtype=int)
  xdata = np.append(xdata, ydata, axis=0)
  labels = np.append(labels, ylabels, axis=0)
rp = np.random.permutation(len(xdata))   # como el original: no consume rng
X_train = xdata[rp]
y_train = labels[rp]

# celda 27 · entrenamiento de LDA y QDA
lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True, priors={{priors}})
ldamodel = lda.fit(X_train, y_train)
qda = QuadraticDiscriminantAnalysis(store_covariance=True, priors={{priors}})
qdamodel = qda.fit(X_train, y_train)
err_lda_train = 1. - accuracy_score(y_train, ldamodel.predict(X_train))
err_qda_train = 1. - accuracy_score(y_train, qdamodel.predict(X_train))

# celda 34 · conjunto de test independiente (mismo rng, misma distribución)
X_test = np.empty((0,n_feat), float)
y_test = np.empty((0), int)
for i in range(0,n_classes):
  ydata = rng.multivariate_normal(mean=M_Means[i,:], cov= M_covar[i,:,:], size= n_samples)
  ylabels = i * np.ones((n_samples),dtype=int)
  X_test = np.append(X_test, ydata, axis=0)
  y_test = np.append(y_test, ylabels, axis=0)
rp = np.random.permutation(len(X_test))
X_test = X_test[rp]
y_test  = y_test[rp]

# celda 36 · error y matriz de confusión en test
y_test_lda = ldamodel.predict(X_test)
linear_error = 1. - accuracy_score(y_test,y_test_lda)
y_test_qda = qdamodel.predict(X_test)
qda_error = 1. - accuracy_score(y_test,y_test_qda)
print('sigma (varianza por feature):', sigma)
print('LDA error: %f' %linear_error)
print('LDA confusion matrix:')
print(confusion_matrix(y_test,y_test_lda))
print('QDA error: %f' %qda_error)
print('QDA confusion matrix:')
print(confusion_matrix(y_test,y_test_qda))

# celda 37 · curvas ROC
X_roc, y_roc = {{roc_set}}
fig, ax = plt.subplots(figsize=(10,4), nrows=1, ncols=2)
ldaroc = RocCurveDisplay.from_estimator(ldamodel,X_roc,y_roc, ax=ax[0])
ax[0].set_title('Linear discriminant classifier')
qdaroc = RocCurveDisplay.from_estimator(qdamodel,X_roc,y_roc, ax=ax[1])
ax[1].set_title('Quadratic discriminant classifier')
plt.show()

# celda 20 · pairplot (solo visualización)
if {{pairplot}}:
  xdl = pd.DataFrame(X_train, columns=['feat1','feat2','feat3']).join(pd.DataFrame(y_train, columns=['class']))
  sns.pairplot(xdl, hue='class', height=2);
  plt.show()

# celdas 31 y 39 · distancia de Mahalanobis (función original, en test)
def mahalanobis(x=None, data=None, cov=None):
  x_minus_mu = x - np.mean(data)
  if not cov:
    cov = np.cov(data.T)
  inv_covmat = sp.linalg.inv(cov)
  left_term = np.dot(x_minus_mu, inv_covmat)
  mahal = np.dot(left_term, x_minus_mu.T)
  return mahal.diagonal()
xdata0 = X_test[y_test==0]
xdata1 = X_test[y_test==1]
mdist01 = np.mean(mahalanobis(x = xdata0, data = xdata1))
mdist10 = np.mean(mahalanobis(x = xdata1, data = xdata0))

# (app) referencias teóricas y versión corregida (media por variable)
def mahalanobis_corr(x, data):
  d = x - data.mean(axis=0)
  return np.einsum('ij,jk,ik->i', d, np.linalg.inv(np.cov(data.T)), d)
resultados = {
  'sigma2': sigma[0],
  'err_lda_train': err_lda_train, 'err_qda_train': err_qda_train,
  'err_lda_test': linear_error, 'err_qda_test': qda_error,
  'bayes': norm.cdf(-dist / (2 * np.sqrt(sigma[0]))),
  'cm_lda': confusion_matrix(y_test, y_test_lda), 'cm_qda': confusion_matrix(y_test, y_test_qda),
  'auc_lda': roc_auc_score(y_roc, ldamodel.predict_proba(X_roc)[:, 1]),
  'auc_qda': roc_auc_score(y_roc, qdamodel.predict_proba(X_roc)[:, 1]),
  'md01': mdist01, 'md10': mdist10,
  'md01_corr': np.mean(mahalanobis_corr(xdata0, xdata1)), 'md10_corr': np.mean(mahalanobis_corr(xdata1, xdata0)),
  'md_teorica': n_feat + dist**2 / sigma[0],
}
`;

export const lab1_1: NotebookContent = {
  id: NB,
  lab: 'lab1',
  title: 'Lab 1 · Parte 1 — LDA y QDA sobre datos gaussianos (Case 1)',
  short: 'MAP: LDA y QDA, Case 1',
  origin: 'original',
  context: {
    goal: `Construir un problema de clasificación **sintético** cuya respuesta óptima conocemos y comprobar cómo se comportan los clasificadores **MAP** lineal (LDA) y cuadrático (QDA) cuando cambia la relación señal-ruido (**SNR**).

El notebook genera dos clases gaussianas con la **misma** covarianza isotrópica $C=\\sigma^2 I$ (el *Case 1* de teoría). Con ese modelo la frontera MAP óptima es un **plano**, así que LDA está bien especificado y QDA no debería aportar ventaja. Después mide error, matriz de confusión, curva ROC y distancia de Mahalanobis, primero en entrenamiento y luego en un **test independiente**.

Preguntas que responde (guía y report de Lab1, Q2–Q5): cómo crece el error al bajar la SNR, qué muestran las matrices de confusión y las ROC a −3 y −10 dB, y por qué la distancia de Mahalanobis entre clases depende del orden de los argumentos.`,
    dataset: `Datos **generados** con \`rng.multivariate_normal\`: 2 clases × 1000 muestras, 3 features. Medias $\\mu_0 = +\\tfrac{0.5}{\\sqrt 3}(1,1,1)$ y $\\mu_1 = -\\tfrac{0.5}{\\sqrt 3}(1,1,1)$ (distancia entre medias = \`dist\` = 1). Covarianza $C_0=C_1=\\sigma^2 I_3$ con $\\sigma^2$ deducida de la SNR. Semilla \`default_rng(seed=5)\`. El notebook guardado se ejecutó con **SNR = 3 dB**; la guía pide repetirlo con 3, 0, −3 y −10 dB.`,
    variables: [
      { name: 'SNR', meaning: 'Relación señal-ruido: primero en dB (entrada), después sobrescrita en escala lineal.', shape: 'escalar' },
      { name: 'M_Means', meaning: 'Matriz de medias: fila i = vector de medias de la clase i.', shape: '(2, 3)' },
      { name: 'energy', meaning: 'Energía media de las medias: promedio de ‖μᵢ‖².', shape: 'escalar' },
      { name: 'sig / sigma', meaning: 'Varianza del ruido por feature (σ²), una por clase en sigma.', shape: 'escalar / (2,)' },
      { name: 'M_covar', meaning: 'Covarianzas de las dos clases apiladas.', shape: '(2, 3, 3)' },
      { name: 'xdata, labels', meaning: 'Muestras y etiquetas de entrenamiento en orden de clase.', shape: '(2000, 3), (2000,)' },
      { name: 'xdatas, xlabs → X_train, y_train', meaning: 'Las mismas muestras barajadas: matriz de features y vector objetivo.', shape: '(2000, 3), (2000,)' },
      { name: 'X_test, y_test', meaning: 'Conjunto de test independiente generado con el mismo modelo.', shape: '(2000, 3), (2000,)' },
      { name: 'ldamodel, qdamodel', meaning: 'Clasificadores entrenados (estimadores de scikit-learn).', shape: '—' },
    ],
    classes: [
      { id: '0', label: 'clase 0 (media positiva)' },
      { id: '1', label: 'clase 1 (media negativa)' },
    ],
    questions: [
      { id: 'Q1', text: 'Autovalores de $C=\\frac{\\sigma^2}{2}\\begin{pmatrix}1&\\rho\\\\\\rho&1\\end{pmatrix}$ (se usa en la parte 2).', blocks: [], doc: 'lab1_guide' },
      { id: 'Q2', text: 'Tabla de errores de LC y QC en test para SNR = 3, 0, −3, −10 dB.', blocks: ['b_test_eval'], doc: 'lab1_guide' },
      { id: 'Q3', text: 'Matrices de confusión en test para −10 y −3 dB.', blocks: ['b_test_eval'], doc: 'lab1_guide' },
      { id: 'Q4', text: 'Curvas ROC en test para −10 y −3 dB.', blocks: ['b_roc_test'], doc: 'lab1_guide' },
      { id: 'Q5', text: 'Distancia de Mahalanobis entre clases en test; por qué depende del orden.', blocks: ['b_mahal_test'], doc: 'lab1_guide' },
    ],
    notation: [
      { symbol: '$c$, $d$', meaning: 'número de clases y de features', code: 'n_classes = 2, n_feat = 3' },
      { symbol: '$\\mu_i$', meaning: 'vector de medias de la clase i', code: 'M_Means[i]' },
      { symbol: '$C_i = \\sigma^2 I$', meaning: 'matriz de covarianza de la clase i (Case 1)', code: 'M_covar[i]' },
      { symbol: '$\\sigma^2$', meaning: 'varianza del ruido en cada feature', code: 'sig, sigma[i]' },
      { symbol: '$\\mathrm{SNR}=10\\log_{10}\\frac{\\overline{\\|\\mu\\|^2}}{\\sigma^2}$', meaning: 'relación señal-ruido en dB (guía, p. 2)', code: 'SNR' },
      { symbol: '$P(y=k\\mid x)$', meaning: 'probabilidad a posteriori de la clase k', code: 'ldamodel.predict_proba(X)' },
      { symbol: '$P(y=k)$', meaning: 'probabilidad a priori', code: 'ldamodel.priors_' },
      { symbol: '$D_M^2(x,\\omega)=(x-\\mu)^T C^{-1}(x-\\mu)$', meaning: 'distancia de Mahalanobis al cuadrado (slides Lab1, p. 11)', code: 'mahalanobis()' },
    ],
  },
  flow: [
    { stage: 'contexto', label: 'Contexto', blocks: ['b_intro', 'b_imports'], summary: 'Case 1 y librerías.' },
    { stage: 'datos', label: 'Datos', blocks: ['b_snr', 'b_design', 'b_energy', 'b_sigma', 'b_covar', 'b_rng', 'b_generate'], summary: 'De la SNR a σ² y a 2000 muestras gaussianas.' },
    { stage: 'exploracion', label: 'Exploración', blocks: ['b_hist', 'b_pairplot', 'b_3d'], summary: '¿Son gaussianas? ¿Cuánto se solapan?' },
    { stage: 'preparacion', label: 'Preparación', blocks: ['b_shuffle'], summary: 'Barajar y organizar X, y.' },
    { stage: 'metodo', label: 'Método', blocks: ['b_sklearn', 'b_theory', 'b_fit'], summary: 'Estimator API, regla MAP, LDA y QDA.' },
    { stage: 'resultados', label: 'Resultados', blocks: ['b_roc_train', 'b_mahal_train', 'b_test_gen', 'b_test_eval', 'b_roc_test', 'b_mahal_test'], summary: 'Error, confusión, ROC y Mahalanobis en train y test.' },
    { stage: 'interpretacion', label: 'Interpretación', blocks: [], summary: 'Se trabaja en las pestañas Interpretar y Práctica (Q2–Q5).' },
  ],
  blocks: [
    {
      id: 'b_intro',
      title: 'Título y planteamiento: Case 1',
      cells: [0, 1, 2],
      stage: 'contexto',
      problem: 'Fijar el problema: clasificar con el criterio **MAP** (máximo a posteriori) datos gaussianos de 2 clases y 3 features independientes.',
      purpose: 'Define el modelo generador que se va a programar: misma covarianza para las dos clases, $C=\\sigma^2 I$ (*Case 1*). Todo el notebook depende de este supuesto.',
      inputs: 'Ninguno (texto).',
      outputs: 'El modelo $f(x\\mid\\omega_c)=\\mathcal N(\\mu_c,\\sigma^2 I)$, $c=1,2$.',
      prereqs: ['map', 'gaussiana-multivariante', 'covarianza'],
      apis: [],
      lines: [],
      interpretation: '«Features estadísticamente independientes» se traduce en una covarianza **diagonal**; «la misma covarianza» significa que $C_0=C_1$. Con esas dos condiciones la regla MAP se reduce a comparar distancias euclídeas a las medias (corregidas por los priors): la frontera es un plano perpendicular a $\\mu_0-\\mu_1$.',
      next: 'Las celdas siguientes traducen el modelo a código: medias (`M_Means`), varianza (`sig`) y covarianzas (`M_covar`).',
      whatIf: [
        { change: 'Si las covarianzas fueran distintas entre clases (Case 3)', effect: 'La frontera MAP óptima sería cuadrática y QDA debería superar a LDA. Es lo que estudia la parte 2 (lab1_2).' },
      ],
      check: {
        kind: 'mcq',
        id: 'lab1_1.check.intro',
        skills: ['concepto'],
        source: src([2]),
        prompt: 'En el Case 1, ¿qué forma tiene la frontera de decisión MAP entre las dos clases?',
        options: [
          { text: 'Un plano (lineal)', correct: true, feedback: 'Con $C_0=C_1=\\sigma^2I$ los términos cuadráticos se cancelan.' },
          { text: 'Una superficie cuadrática', feedback: 'Eso ocurre cuando las covarianzas son distintas (Case 3).' },
          { text: 'Depende del número de muestras', feedback: 'La frontera óptima depende del modelo, no de cuántas muestras generes.' },
        ],
        explanation: 'Al restar los discriminantes $h_0(x)-h_1(x)$ desaparece $x^T C^{-1} x$ porque $C$ es común: queda una función lineal de $x$.',
      },
    },
    {
      id: 'b_imports',
      title: 'Importación de librerías',
      cells: [3],
      stage: 'contexto',
      problem: 'Tener disponibles las herramientas numéricas, de datos y de gráficos.',
      purpose: 'Define los alias (`np`, `pd`, `sns`, `ss`, `sp`, `plt`) que usa todo el notebook. Sin esta celda, cualquier otra fallaría con `NameError`.',
      inputs: 'Nada.',
      outputs: 'Módulos importados en el espacio de nombres.',
      prereqs: [],
      apis: [
        { name: 'import numpy as np', lib: 'numpy', kind: 'módulo', does: 'Arrays y álgebra lineal (np.dot, np.eye, np.append…).' },
        { name: 'import pandas as pd', lib: 'pandas', kind: 'módulo', does: 'DataFrame para la tabla de muestras (celda 18).' },
        { name: 'import seaborn as sns', lib: 'seaborn', kind: 'módulo', does: 'histplot y pairplot.' },
        { name: 'import scipy as sp', lib: 'scipy', kind: 'módulo', does: 'sp.linalg.inv en la función de Mahalanobis.' },
        { name: 'import matplotlib.pyplot as plt', lib: 'matplotlib', kind: 'módulo', does: 'Figuras y subplots.' },
      ],
      lines: [
        { cell: 3, lines: [4, 5], text: '`scipy.stats as ss` no se usa en este notebook; `scipy as sp` sí (inversa de la covarianza). En SciPy reciente `sp.linalg` funciona tras `import scipy`, pero lo más robusto es `import scipy.linalg`.' },
      ],
      interpretation: 'Los imports no producen salida; su papel es fijar los nombres. En el examen conviene recordar los alias estándar porque el código de los labs los da por supuestos.',
      next: 'La siguiente celda fija la SNR, el único parámetro que la guía pide variar.',
      whatIf: [{ change: 'Quitar `import scipy as sp`', effect: '`mahalanobis()` fallaría con `NameError: name \'sp\' is not defined` (error de API/import).' }],
      corrections: [],
    },
    {
      id: 'b_snr',
      title: 'Entrada: SNR en dB',
      cells: [4, 5, 6],
      stage: 'datos',
      problem: 'Elegir cuánto ruido tendrán los datos.',
      purpose: 'La SNR es el **parámetro experimental** de la parte 1: la guía pide ejecutar con 3, 0, −3 y −10 dB y comparar. Todo lo demás (medias, número de muestras) queda fijo para aislar su efecto.',
      inputs: 'Un número en decibelios.',
      outputs: '`SNR` (en dB, de momento).',
      prereqs: ['snr', 'decibelio'],
      apis: [],
      lines: [
        { cell: 6, lines: [1, 1], text: 'El comentario dice «(3, 0, -3, 10)», pero la guía pide **−10** dB, no 10.' },
        { cell: 6, lines: [2, 2], text: '`SNR = 3` → este es el valor con el que se guardó el notebook. Cambiarlo regenera todo el experimento.' },
      ],
      interpretation: 'En dB, +3 dB ≈ señal el doble de potente que el ruido; 0 dB = iguales; −3 dB ≈ ruido el doble; −10 dB = ruido 10 veces más potente. Cuanto menor la SNR, más varianza y más solapamiento entre clases.',
      next: 'La SNR se convierte a escala lineal y se usa para calcular σ² en la celda 9.',
      whatIf: [
        { change: 'Bajar la SNR de 3 a −10 dB', effect: 'σ² se multiplica por $10^{1.3}\\approx 20$; las nubes se solapan y el error de test pasa de ≈1 % a ≈29 % (ver experimento).' },
      ],
    },
    {
      id: 'b_design',
      title: 'Parámetros de diseño y vectores de medias',
      cells: [7],
      stage: 'datos',
      problem: 'Definir cuántas clases, muestras y features hay y dónde están los centros de las clases.',
      purpose: 'Fija la **geometría** del problema: dos medias opuestas sobre la diagonal (1,1,1), separadas exactamente `dist`. Normalizar por $\\sqrt{d}$ hace que la distancia no dependa del número de features.',
      inputs: 'Constantes.',
      outputs: '`dist`, `n_classes`, `n_samples`, `n_feat`, `M_Means` (2×3).',
      prereqs: ['media', 'norma'],
      apis: [{ name: 'np.array', lib: 'numpy', kind: 'función', does: 'Crea el array 2×3 de medias a partir de listas anidadas.' }, { name: 'np.sqrt', lib: 'numpy', kind: 'función', does: 'Raíz cuadrada (normalización por √d).' }],
      lines: [
        { cell: 7, lines: [3, 3], text: '`n_samples` es por **clase**: en total habrá 2000 muestras. El `;` final es inofensivo en Python.' },
        { cell: 7, lines: [6, 6], text: 'Cada fila es una media: $\\mu_0=\\frac{0.5}{\\sqrt3}(1,1,1)\\approx(0.289,0.289,0.289)$ y $\\mu_1=-\\mu_0$. Así $\\|\\mu_0-\\mu_1\\| = $ `dist` $=1$.' },
      ],
      shapes: [{ expr: 'M_Means', shape: '(2, 3)', meaning: 'filas = clases, columnas = features' }],
      interpretation: 'La separación entre clases está fija (1); lo que cambia con la SNR es el ruido. Por eso el experimento mide «separación en unidades de ruido».',
      prev: 'Usa `np` de los imports.',
      next: 'Con `M_Means` se calcula la energía media (celda 8).',
      whatIf: [
        { change: 'Duplicar `dist`', effect: 'Con la misma SNR, σ² también se cuadruplica (porque la energía crece con dist²), así que la separación relativa no cambia: la SNR, no `dist`, controla la dificultad.' },
        { change: 'Quitar `/ np.sqrt(n_feat)`', effect: 'La distancia entre medias pasaría a ser $\\sqrt3$; la energía y σ² se escalarían igual, de nuevo sin cambiar la dificultad a SNR fija.' },
      ],
    },
    {
      id: 'b_energy',
      title: 'Energía media de las medias',
      cells: [8],
      stage: 'datos',
      problem: 'Medir la «potencia de la señal» para poder definir la SNR.',
      purpose: 'La SNR se define como cociente entre la energía media de las medias y la varianza del ruido; aquí se calcula el numerador.',
      inputs: '`M_Means`, `n_classes`.',
      outputs: '`energy` = $\\frac1c\\sum_i \\mu_i^T\\mu_i$ = 0.25.',
      prereqs: ['producto-escalar'],
      apis: [{ name: 'np.dot', lib: 'numpy', kind: 'función', does: 'Producto escalar de dos vectores 1D: $\\mu_i^T\\mu_i=\\|\\mu_i\\|^2$.', returns: 'escalar' }],
      lines: [
        { cell: 8, lines: [3, 4], text: 'Bucle sobre clases acumulando $\\|\\mu_i\\|^2$. Con las medias del notebook cada término vale $3\\cdot(0.5/\\sqrt3)^2 = 0.25$.' },
        { cell: 8, lines: [5, 5], text: 'Promedio sobre clases → `energy = 0.25`.' },
      ],
      interpretation: 'Es la energía de la «señal» limpia. Equivale a `np.mean(np.sum(M_Means**2, axis=1))`.',
      next: 'La celda 9 usa `energy` y la SNR para obtener σ².',
      whatIf: [{ change: 'Olvidar dividir por `n_classes`', effect: 'energy = 0.5 y σ² se duplicaría: la SNR real sería 3 dB menor que la indicada.' }],
    },
    {
      id: 'b_sigma',
      title: 'De la SNR a la varianza del ruido',
      cells: [9],
      stage: 'datos',
      problem: 'Obtener σ² para que los datos tengan exactamente la SNR pedida.',
      purpose: 'Invierte la definición $\\mathrm{SNR}_{dB}=10\\log_{10}(E/\\sigma^2_{tot})$: primero pasa de dB a lineal y después reparte la varianza total entre las `n_feat` features.',
      inputs: '`SNR` (dB), `energy`, `n_feat`.',
      outputs: '`SNR` (ahora **lineal**), `sig` = σ² por feature.',
      prereqs: ['snr', 'decibelio'],
      apis: [],
      lines: [
        { cell: 9, lines: [2, 2], text: '$10^{\\mathrm{SNR}/10}$: conversión de dB a escala lineal. **Sobrescribe** `SNR`: si vuelves a ejecutar esta celda sin la celda 6, convertirías dos veces.' },
        { cell: 9, lines: [3, 3], text: 'Varianza total del ruido = energía / SNR lineal.' },
        { cell: 9, lines: [4, 4], text: 'Se reparte por igual entre las 3 features: σ² por feature. Para SNR = 3 dB: 0.25/1.995/3 ≈ **0.04177** (coincide con la salida de la celda 11).' },
      ],
      shapes: [{ expr: 'sig', shape: 'escalar', meaning: 'varianza (no desviación típica) por feature' }],
      interpretation: '`sig` es una **varianza**. La desviación típica del ruido por feature es $\\sqrt{0.0418}\\approx0.204$, comparable a la semiseparación entre medias por feature (0.289): por eso a 3 dB el error es pequeño pero no nulo.',
      prev: 'Usa la SNR (celda 6) y la energía (celda 8).',
      next: 'σ² se coloca en la diagonal de las covarianzas (celda 10).',
      whatIf: [
        { change: 'Ejecutar dos veces esta celda', effect: 'La segunda vez se aplica $10^{(1.995)/10}$ sobre la SNR ya lineal: σ² sale mal sin ningún error visible (error de razonamiento típico de los notebooks).' },
        { change: 'Quitar la línea 4', effect: 'Cada feature tendría la varianza total: la SNR efectiva bajaría ≈4.8 dB (factor 3).' },
      ],
    },
    {
      id: 'b_covar',
      title: 'Matrices de covarianza (Case 1)',
      cells: [10, 11],
      stage: 'datos',
      problem: 'Construir $C_0$ y $C_1$.',
      purpose: 'Materializa el supuesto del Case 1: matrices diagonales iguales, $C_i=\\sigma^2 I_3$. Se guardan apiladas en un array 3D para poder indexar `M_covar[i]`.',
      inputs: '`sig`, `n_classes`, `n_feat`.',
      outputs: '`sigma` (2,), `M_covar` (2, 3, 3). La celda 11 las imprime.',
      prereqs: ['covarianza', 'matriz-identidad'],
      apis: [
        { name: 'np.zeros', lib: 'numpy', kind: 'función', does: 'Reserva el array (2,3,3) de ceros.', args: [{ name: 'shape', value: '(2,3,3)', meaning: '(clases, features, features)' }] },
        { name: 'np.eye', lib: 'numpy', kind: 'función', does: 'Matriz identidad.', args: [{ name: 'N, M', value: '3, 3', meaning: 'filas y columnas' }] },
      ],
      lines: [
        { cell: 10, lines: [2, 2], text: 'Array 3D: `M_covar[i,:,:]` es la covarianza de la clase i. El 2 está escrito a mano (sería más robusto `n_classes`).' },
        { cell: 10, lines: [3, 3], text: '`sigma` guarda una varianza **por clase** (aquí iguales). En la parte 2 se modifica `sigma[0]` para romper la igualdad.' },
        { cell: 10, lines: [5, 5], text: '$C_i = \\sigma_i^2 I$: diagonal → features independientes; mismos valores → ruido isótropo (nubes esféricas).' },
      ],
      shapes: [
        { expr: 'sigma', shape: '(2,)', meaning: 'σ² de cada clase' },
        { expr: 'M_covar', shape: '(2, 3, 3)', meaning: 'una matriz 3×3 por clase' },
      ],
      interpretation: 'La salida de la celda 11 debe mostrar dos matrices diagonales idénticas con 0.0417656 en la diagonal (SNR = 3 dB).',
      next: 'Medias y covarianzas son los parámetros de `multivariate_normal` (celda 14).',
      whatIf: [{ change: 'Usar `sigma = sig * np.array([1, 5])`', effect: 'Covarianzas distintas (Case 3): QDA empezaría a ganar a LDA.' }],
    },
    {
      id: 'b_rng',
      title: 'Generador aleatorio con semilla',
      cells: [12, 13],
      stage: 'datos',
      problem: 'Que cada ejecución produzca exactamente las mismas muestras.',
      purpose: 'Con semilla fija, los resultados del notebook (y los del report) son **reproducibles**: si cambias la SNR, la diferencia en el error se debe a la SNR y no al azar.',
      inputs: 'La semilla 5.',
      outputs: '`rng`, un `numpy.random.Generator`.',
      prereqs: ['semilla'],
      apis: [{ name: 'default_rng', lib: 'numpy.random', kind: 'función', does: 'Crea un Generator (PCG64) inicializado con la semilla.', args: [{ name: 'seed', value: '5', meaning: 'estado inicial reproducible' }], returns: 'Generator' }],
      lines: [{ cell: 13, lines: [3, 3], text: 'El estado de `rng` avanza cada vez que se usa: entrenamiento y test (celda 34) salen del **mismo** flujo, por eso son distintos pero reproducibles.' }],
      interpretation: 'La semilla no hace los datos «mejores»: solo los hace repetibles. Para estimar la variabilidad del error habría que repetir con varias semillas.',
      next: 'La celda 14 usa `rng.multivariate_normal`.',
      whatIf: [{ change: 'Cambiar `seed=5` por otra', effect: 'Errores ligeramente distintos (±0.2 % a 3 dB): es la variación muestral. El experimento tiene un botón «Nueva muestra» para verlo.' }],
    },
    {
      id: 'b_generate',
      title: 'Generación del conjunto de entrenamiento',
      cells: [14],
      stage: 'datos',
      problem: 'Obtener muestras etiquetadas de cada clase según su gaussiana.',
      purpose: 'Crea la **matriz de features** y el **vector de etiquetas** que usarán los clasificadores: 1000 muestras de $\\mathcal N(\\mu_0,C_0)$ con etiqueta 0 y 1000 de $\\mathcal N(\\mu_1,C_1)$ con etiqueta 1.',
      inputs: '`M_Means`, `M_covar`, `n_samples`, `rng`.',
      outputs: '`xdata` (2000, 3) y `labels` (2000,), ordenados por clase.',
      prereqs: ['gaussiana-multivariante', 'matriz-features'],
      apis: [
        { name: 'rng.multivariate_normal', lib: 'numpy.random.Generator', kind: 'método', does: 'Muestras de una gaussiana multivariante.', args: [{ name: 'mean', value: 'M_Means[i,:]', meaning: 'vector (3,)' }, { name: 'cov', value: 'M_covar[i,:,:]', meaning: 'matriz (3,3) simétrica semidefinida positiva' }, { name: 'size', value: 'n_samples', meaning: 'número de muestras' }], returns: 'array (n_samples, 3)' },
        { name: 'np.ones', lib: 'numpy', kind: 'función', does: 'Vector de unos; multiplicado por i da las etiquetas de la clase.', args: [{ name: 'dtype', value: 'int', meaning: 'etiquetas enteras' }] },
        { name: 'np.append', lib: 'numpy', kind: 'función', does: 'Concatena creando un array nuevo.', args: [{ name: 'axis', value: '0', meaning: 'apila filas (muestras)' }] },
        { name: 'np.empty', lib: 'numpy', kind: 'función', does: 'Array inicial con 0 filas y 3 columnas para poder concatenar.' },
      ],
      lines: [
        { cell: 14, lines: [4, 5], text: 'Arrays vacíos con la forma correcta: (0, 3) para features y (0,) para etiquetas. Sin esa forma, `np.append(..., axis=0)` fallaría por dimensiones.' },
        { cell: 14, lines: [7, 7], text: '1000 muestras de la clase i: array (1000, 3).' },
        { cell: 14, lines: [8, 8], text: 'Etiquetas de la clase i: `[i, i, …, i]` (1000,).' },
        { cell: 14, lines: [9, 10], text: 'Se apilan las filas. Tras el bucle: `xdata.shape == (2000, 3)`; las primeras 1000 filas son clase 0.' },
      ],
      shapes: [
        { expr: 'ydata', shape: '(1000, 3)', meaning: 'muestras de una clase' },
        { expr: 'xdata', shape: '(2000, 3)', meaning: '[n_samples totales, n_features]' },
        { expr: 'labels', shape: '(2000,)', meaning: 'vector 1D de 0 y 1' },
      ],
      interpretation: 'Es el formato que espera scikit-learn: filas = muestras, columnas = features; etiquetas en un vector 1D de la misma longitud.',
      prev: 'Necesita medias, covarianzas y `rng`.',
      next: 'Se explora (histogramas, celda 16) y se baraja (celda 18).',
      whatIf: [
        { change: 'Usar `axis=1` en `np.append`', effect: 'Intentaría pegar columnas: `ValueError` de dimensiones (0 filas frente a 1000).' },
        { change: 'Generar con `rng.normal` por feature', effect: 'Con covarianza diagonal sería equivalente en distribución, pero no en la secuencia aleatoria: los números concretos cambiarían.' },
      ],
    },
    {
      id: 'b_hist',
      title: 'Histogramas por clase y feature',
      cells: [15, 16],
      stage: 'exploracion',
      problem: 'Comprobar visualmente que cada feature de cada clase es gaussiana y ver su centro.',
      purpose: 'Es EDA de Lab0 aplicada a datos sintéticos: valida que la generación hace lo esperado antes de clasificar.',
      inputs: '`xdata`, `labels`.',
      outputs: 'Figura 3×2: fila = feature, columna = clase.',
      prereqs: ['histograma', 'kde'],
      apis: [
        { name: 'plt.subplots', lib: 'matplotlib', kind: 'función', does: 'Crea figura y rejilla de ejes.', args: [{ name: 'nrows, ncols', value: '3, 2', meaning: 'features × clases' }] },
        { name: 'sns.histplot', lib: 'seaborn', kind: 'función', does: 'Histograma de conteos.', args: [{ name: 'kde', value: 'True', meaning: 'superpone una estimación de densidad por kernel' }] },
      ],
      lines: [
        { cell: 16, lines: [11, 12], text: 'Máscara booleana `labels == idclass` y selección de una columna: `xd` tiene forma (1000,).' },
        { cell: 16, lines: [13, 13], text: 'Índice del subplot (1..6) en orden fila-columna.' },
        { cell: 16, lines: [16, 16], text: '`sns.distplot` (comentado) está **obsoleto**; `histplot(kde=True)` es su sustituto. La curva es una KDE, **no** un ajuste normal.' },
      ],
      interpretation: 'Cada histograma es campaniforme y centrado en ≈ +0.29 (clase 0) o −0.29 (clase 1). Con SNR baja las campanas se ensanchan y se solapan más.',
      next: 'Se pasa a ver las features conjuntamente (pairplot y 3D).',
      whatIf: [{ change: 'Bajar la SNR', effect: 'Las campanas se ensanchan; las medias no cambian.' }],
    },
    {
      id: 'b_shuffle',
      title: 'Barajado y DataFrame',
      cells: [17, 18],
      stage: 'preparacion',
      problem: 'Quitar el orden por clases y tener una tabla con nombres de columna.',
      purpose: 'Barajar evita que el orden de las filas esconda la estructura (y es imprescindible si después se parte en train/test por posición). El DataFrame `xdl` se usa para `pairplot` con `hue`.',
      inputs: '`xdata`, `labels`.',
      outputs: '`xdatas`, `xlabs` (barajados, mismo orden), `xdl` (DataFrame 2000×4).',
      prereqs: ['dataframe'],
      apis: [
        { name: 'np.random.permutation', lib: 'numpy', kind: 'función', does: 'Permutación aleatoria de 0..n−1 (usa el generador **global**, sin semilla).', returns: 'array de índices (2000,)' },
        { name: 'pd.DataFrame', lib: 'pandas', kind: 'clase', does: 'Tabla con columnas con nombre.', args: [{ name: 'columns', value: "['feat1','feat2','feat3']", meaning: 'nombres de columna' }] },
        { name: 'DataFrame.join', lib: 'pandas', kind: 'método', does: 'Une por índice dos DataFrames (features + clase).' },
        { name: 'DataFrame.head', lib: 'pandas', kind: 'método', does: 'Primeras 5 filas.' },
      ],
      lines: [
        { cell: 18, lines: [2, 4], text: 'El **mismo** vector `rp` reordena muestras y etiquetas: la correspondencia fila-etiqueta se mantiene.' },
        { cell: 18, lines: [7, 9], text: 'Dos DataFrames y `join` por índice → columnas feat1, feat2, feat3, class.' },
      ],
      shapes: [{ expr: 'xdl', shape: '(2000, 4)', meaning: '3 features + clase' }],
      interpretation: 'La tabla impresa (`head`) cambia en cada ejecución porque la permutación no tiene semilla; los **errores no cambian**, porque LDA/QDA no dependen del orden de las muestras.',
      corrections: [{ cell: 18, kind: 'concepto', original: 'rp = np.random.permutation(len(xdata))', fixed: 'rp = rng.permutation(len(xdata))', why: 'El barajado usa el generador global sin semilla, así que `xdl.head()` no es reproducible. Usar `rng` lo haría reproducible, pero cambiaría el estado de `rng` y, por tanto, los datos de test de la celda 34 respecto al notebook guardado. La app conserva el original para reproducir tus resultados.' }],
      next: 'Con `xdl` se dibuja el pairplot; `xdatas`/`xlabs` serán `X_train`/`y_train`.',
      whatIf: [{ change: 'Barajar solo `xdata`', effect: 'Las etiquetas quedarían desalineadas: el clasificador aprendería ruido (error ≈ 50 %).' }],
    },
    {
      id: 'b_pairplot',
      title: 'Scatter 2D: pairplot',
      cells: [19, 20],
      stage: 'exploracion',
      problem: 'Ver todas las parejas de features coloreadas por clase.',
      purpose: 'Permite estimar a ojo el **solapamiento** (y por tanto el error esperable) y comprobar que las nubes son redondas (covarianza isótropa).',
      inputs: '`xdl`.',
      outputs: 'Matriz de gráficos 3×3: dispersión fuera de la diagonal, densidades por clase en la diagonal.',
      prereqs: ['scatter'],
      apis: [{ name: 'sns.pairplot', lib: 'seaborn', kind: 'función', does: 'Matriz de dispersión de todas las parejas de columnas numéricas.', args: [{ name: 'hue', value: "'class'", meaning: 'colorea por la columna clase' }] }],
      lines: [{ cell: 20, lines: [2, 2], text: 'El `;` evita que el notebook imprima el objeto `PairGrid` devuelto.' }],
      interpretation: 'Nubes circulares (no elípticas) confirman $C=\\sigma^2 I$. La recta que mejor las separa es perpendicular a la diagonal: la frontera óptima es un plano.',
      next: 'Vista 3D complementaria.',
      whatIf: [{ change: 'SNR = −10 dB', effect: 'Las dos nubes se superponen casi por completo en cada pareja.' }],
    },
    {
      id: 'b_3d',
      title: 'Scatter 3D',
      cells: [21, 22],
      stage: 'exploracion',
      problem: 'Ver las tres features a la vez.',
      purpose: 'Con d = 3 aún se puede ver el espacio completo; la frontera MAP es un plano que corta la recta entre medias.',
      inputs: '`xdatas`, `xlabs`.',
      outputs: 'Figura 3D.',
      prereqs: ['scatter'],
      apis: [
        { name: 'plt.axes(projection="3d")', lib: 'matplotlib', kind: 'función', does: 'Ejes tridimensionales.' },
        { name: 'ax.scatter3D', lib: 'matplotlib', kind: 'método', does: 'Dispersión 3D.' },
      ],
      lines: [{ cell: 22, lines: [7, 8], text: 'Máscara por clase y tres columnas (x, y, z).' }],
      interpretation: 'En Colab la figura no es interactiva; sirve como apoyo visual.',
      next: 'Empieza el bloque de scikit-learn.',
      whatIf: [],
    },
    {
      id: 'b_sklearn',
      title: 'Scikit-learn: representación de datos y Estimator API',
      cells: [23],
      stage: 'metodo',
      problem: 'Conocer el contrato de scikit-learn antes de usarlo.',
      purpose: 'Explica la **matriz de features** `[n_samples, n_features]`, el **vector objetivo** `[n_samples]` y los 5 pasos de la Estimator API (importar, instanciar con hiperparámetros, organizar X e y, `fit`, `predict`/`transform`).',
      inputs: 'Texto.',
      outputs: 'El patrón que siguen todos los labs.',
      prereqs: ['estimator-api'],
      apis: [
        { name: 'estimator.fit(X, y)', lib: 'sklearn', kind: 'método', does: 'Aprende los parámetros a partir de los datos; devuelve el propio estimador.' },
        { name: 'estimator.predict(X)', lib: 'sklearn', kind: 'método', does: 'Etiqueta predicha para cada fila de X.' },
      ],
      lines: [],
      interpretation: 'Clave para el examen: distinguir **argumentos** (hiperparámetros al crear el objeto, p. ej. `solver`), **valores devueltos** (`predict` devuelve un array) y **atributos aprendidos** (terminan en `_`, existen solo tras `fit`: `priors_`, `means_`, `covariance_`).',
      next: 'Teoría de LDA/QDA y después el código.',
      whatIf: [],
    },
    {
      id: 'b_theory',
      title: 'Teoría: regla de Bayes, QDA y LDA',
      cells: [24, 25],
      stage: 'metodo',
      problem: 'Justificar por qué LDA y QDA son clasificadores MAP.',
      purpose: 'Ambos modelan $P(x\\mid y=k)$ con una gaussiana y eligen la clase de máximo **log-posterior**. QDA estima una covarianza por clase; LDA una común, lo que elimina el término cuadrático.',
      inputs: 'Texto y fórmulas.',
      outputs: '$\\log P(y=k\\mid x) = -\\tfrac12\\log|\\Sigma_k| - \\tfrac12 (x-\\mu_k)^T\\Sigma_k^{-1}(x-\\mu_k) + \\log P(y=k) + C$.',
      prereqs: ['bayes', 'map', 'mahalanobis'],
      apis: [],
      lines: [],
      interpretation: 'En LDA el término $(x-\\mu_k)^T\\Sigma^{-1}(x-\\mu_k)$ es la **distancia de Mahalanobis** al cuadrado: LDA asigna x a la media más cercana en Mahalanobis, corregida por $\\log P(y=k)$. Sin hiperparámetros que ajustar y con solución cerrada.',
      next: 'La celda 27 entrena ambos modelos.',
      whatIf: [{ change: 'Priors desiguales', effect: 'El término $\\log P(y=k)$ desplaza la frontera hacia la clase menos probable (ver parámetro «priors» del experimento).' }],
    },
    {
      id: 'b_fit',
      title: 'Entrenar LDA y QDA y evaluar en entrenamiento',
      cells: [26, 27],
      stage: 'metodo',
      problem: 'Ajustar los dos clasificadores y medir su error sobre los datos con los que aprendieron.',
      purpose: 'Aplica la Estimator API: instancia, `fit`, `predict`. Imprime los parámetros estimados (priors, covarianza) y calcula **error = 1 − accuracy** y la **matriz de confusión**.',
      inputs: '`xdatas`, `xlabs` (como `X_train`, `y_train`).',
      outputs: '`ldamodel`, `qdamodel`, predicciones, error y matriz de confusión de entrenamiento.',
      prereqs: ['lda', 'qda', 'accuracy', 'matriz-confusion'],
      apis: [
        { name: 'LinearDiscriminantAnalysis', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Clasificador gaussiano con covarianza común (frontera lineal).', args: [{ name: 'solver', value: '"svd"', meaning: 'método numérico; no necesita invertir la covarianza' }, { name: 'store_covariance', value: 'True', meaning: 'guarda `covariance_` (con svd no se guarda por defecto)' }], learned: ['priors_', 'means_', 'covariance_', 'coef_', 'intercept_', 'classes_'] },
        { name: 'QuadraticDiscriminantAnalysis', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Clasificador gaussiano con una covarianza por clase (frontera cuadrática).', args: [{ name: 'store_covariance', value: 'True', meaning: 'guarda la lista `covariance_`' }], learned: ['priors_', 'means_', 'covariance_'] },
        { name: 'accuracy_score(y_true, y_pred)', lib: 'sklearn.metrics', kind: 'función', does: 'Proporción de aciertos.', returns: 'float en [0, 1]' },
        { name: 'confusion_matrix(y_true, y_pred)', lib: 'sklearn.metrics', kind: 'función', does: 'Conteos: fila = clase real, columna = clase predicha.', returns: 'array (c, c)' },
      ],
      lines: [
        { cell: 27, lines: [2, 7], text: 'Imports de scikit-learn. `roc_curve` se importa pero no se usa.' },
        { cell: 27, lines: [10, 11], text: 'Renombrado a la convención de scikit-learn: `X_train` (2000, 3), `y_train` (2000,).' },
        { cell: 27, lines: [14, 16], text: 'Pasos 2, 4 y 5 de la Estimator API. `fit` devuelve el mismo objeto: `ldamodel is lda` es `True`.' },
        { cell: 27, lines: [18, 20], text: '**Errata**: la línea 19 imprime `priors_` con la etiqueta «means». Las medias están en `means_` (2, 3).' },
        { cell: 27, lines: [23, 23], text: 'Error de clasificación = 1 − accuracy. Con SNR = 3 dB: 0.0085 (17 errores de 2000).' },
        { cell: 27, lines: [26, 26], text: 'Matriz [[990, 10], [7, 993]]: 10 muestras de clase 0 clasificadas como 1 y 7 de clase 1 como 0.' },
        { cell: 27, lines: [30, 32], text: 'Mismo patrón para QDA. Sus `covariance_` son dos matrices (una por clase).' },
      ],
      shapes: [
        { expr: 'ldamodel.priors_', shape: '(2,)', meaning: 'frecuencias relativas: [0.5, 0.5]' },
        { expr: 'ldamodel.means_', shape: '(2, 3)', meaning: 'medias estimadas por clase' },
        { expr: 'ldamodel.covariance_', shape: '(3, 3)', meaning: 'covarianza común estimada (≈ σ²I)' },
        { expr: 'qdamodel.covariance_', shape: 'lista de 2 arrays (3, 3)', meaning: 'una covarianza por clase' },
        { expr: 'confusion_matrix(...)', shape: '(2, 2)', meaning: 'reales × predichas' },
      ],
      interpretation: 'La covarianza estimada por LDA ≈ 0.0418·I (diagonal ≈ 0.041–0.043, fuera ≈ ±0.002): estima bien el modelo. LDA y QDA cometen los **mismos** 17 errores en train: con covarianzas iguales QDA no tiene nada extra que aprovechar. Es un error **de entrenamiento** (optimista); el test (celda 36) lo confirmará.',
      prev: 'Necesita `xdatas`, `xlabs` (celda 18).',
      next: 'ROC y Mahalanobis en train; después, test independiente.',
      versionNotes: [{ cell: 27, text: 'Las covarianzas de QDA impresas difieren en un factor 1000/999: scikit-learn 1.9 (tu entorno, salida guardada) usa el estimador de máxima verosimilitud (divide por n) y 1.8 (la app) el insesgado (divide por n − 1). Errores, predicciones y matrices de confusión son idénticos.' }],
      corrections: [{ cell: 27, kind: 'errata', original: "print('LDA model means:',ldamodel.priors_)", fixed: "print('LDA model means:',ldamodel.means_)", why: 'Se imprime el atributo equivocado (también en la línea 35 para QDA). No afecta a los errores.' }],
      whatIf: [
        { change: 'Quitar `store_covariance=True` en LDA', effect: 'Con `solver="svd"`, `ldamodel.covariance_` no existiría: la línea 20 daría `AttributeError`.' },
        { change: 'Cambiar `solver` a "lsqr"', effect: 'Mismas predicciones (mismo modelo, otro algoritmo numérico).' },
        { change: 'Evaluar con `accuracy_score` sin el `1. -`', effect: 'Obtendrías la tasa de acierto (≈0.99), no el error.' },
      ],
    },
    {
      id: 'b_roc_train',
      title: 'Curvas ROC en entrenamiento',
      cells: [28, 29],
      stage: 'resultados',
      problem: 'Evaluar el clasificador para **todos** los umbrales, no solo el de la decisión MAP.',
      purpose: 'La ROC dibuja TPR frente a FPR variando el umbral sobre $P(y=1\\mid x)$; el AUC resume la capacidad de separar las clases independientemente del umbral.',
      inputs: '`ldamodel`, `qdamodel`, `X_train`, `y_train`.',
      outputs: 'Dos figuras ROC con AUC en la leyenda.',
      prereqs: ['roc', 'auc', 'tpr-fpr'],
      apis: [{ name: 'RocCurveDisplay.from_estimator', lib: 'sklearn.metrics', kind: 'método', does: 'Calcula y dibuja la ROC de un estimador entrenado usando `predict_proba` (o `decision_function`).', args: [{ name: 'estimator, X, y', meaning: 'modelo y datos de evaluación' }, { name: 'ax', value: 'ax[0]', meaning: 'subplot donde dibujar' }], returns: 'objeto display con `roc_auc`' }],
      lines: [{ cell: 29, lines: [2, 2], text: 'Clase positiva = 1 (la etiqueta mayor). Se evalúa sobre **train**: optimista.' }],
      interpretation: 'Curva pegada a la esquina superior izquierda y AUC ≈ 1 → clases casi separables a 3 dB. La diagonal (AUC = 0.5) sería un clasificador al azar.',
      next: 'Distancia de Mahalanobis entre clases (train).',
      whatIf: [{ change: 'Bajar la SNR', effect: 'La curva se acerca a la diagonal y el AUC baja (≈0.96 a −3 dB, ≈0.79 a −10 dB).' }],
    },
    {
      id: 'b_mahal_train',
      title: 'Distancia de Mahalanobis entre clases (train)',
      cells: [30, 31],
      stage: 'resultados',
      problem: 'Cuantificar la separación entre clases en «unidades de dispersión».',
      purpose: 'Para cada muestra de una clase calcula $D_M^2$ respecto a la media y covarianza **estimadas de la otra** clase y promedia. Es una medida de separabilidad que se compara con el error.',
      inputs: '`X_train`, `y_train`.',
      outputs: '`mdist01` ≈ 26.89 y `mdist10` ≈ 26.73 (SNR = 3 dB).',
      prereqs: ['mahalanobis', 'covarianza'],
      apis: [
        { name: 'np.cov(data.T)', lib: 'numpy', kind: 'función', does: 'Covarianza muestral; espera variables en filas, por eso se transpone.', returns: '(3, 3)' },
        { name: 'sp.linalg.inv', lib: 'scipy.linalg', kind: 'función', does: 'Inversa de la covarianza.' },
        { name: 'np.dot', lib: 'numpy', kind: 'función', does: 'Producto matricial.' },
        { name: 'ndarray.diagonal', lib: 'numpy', kind: 'método', does: 'Diagonal de la matriz n×n.' },
      ],
      lines: [
        { cell: 31, lines: [2, 2], text: '**Errata**: `np.mean(data)` es la media de **todos** los elementos (un escalar), no el vector de medias por feature. Aquí apenas importa porque las tres medias de cada clase son iguales por diseño.' },
        { cell: 31, lines: [3, 4], text: '`if not cov` funciona con `None`, pero fallaría si se pasa una matriz («truth value of an array is ambiguous»); lo correcto es `if cov is None`.' },
        { cell: 31, lines: [6, 8], text: 'Calcula la matriz n×n $(X-\\mu)C^{-1}(X-\\mu)^T$ y se queda con su diagonal: correcto pero ineficiente (1000×1000).' },
        { cell: 31, lines: [13, 14], text: 'Promedio de $D_M^2$ de las muestras de clase 0 respecto a la clase 1.' },
      ],
      shapes: [{ expr: 'md01', shape: '(1000,)', meaning: 'una distancia por muestra de la clase 0' }],
      interpretation: 'Es una distancia **al cuadrado** (no se toma raíz). Valor esperado poblacional: $d + \\|\\Delta\\mu\\|^2/\\sigma^2 = 3 + 1/0.0418 \\approx 26.9$. El 3 procede de la dispersión propia de la clase. Los dos órdenes difieren porque cada uno usa la media y la covarianza **estimadas** de una clase distinta: no es simétrica.',
      corrections: [{ cell: 31, kind: 'errata', original: 'x_minus_mu = x - np.mean(data)', fixed: 'x_minus_mu = x - np.mean(data, axis=0)', why: 'La media debe calcularse por columna (feature). Con medias de features distintas, el resultado original sería incorrecto.' }],
      next: 'Se genera un test independiente para evaluar sin sesgo.',
      whatIf: [{ change: 'Usar la versión corregida (`axis=0`)', effect: 'Valores casi idénticos aquí (26.48 / 26.82 en test a 3 dB); en un problema con medias de features distintas el cambio sería grande.' }],
    },
    {
      id: 'b_test_gen',
      title: 'Generación de un test independiente',
      cells: [32, 33, 34],
      stage: 'resultados',
      problem: 'Evaluar el modelo con datos que **no** ha visto.',
      purpose: 'El error de entrenamiento es optimista. Un test generado con el **mismo modelo** pero muestras nuevas da una estimación sin sesgo del error real.',
      inputs: '`rng` (su estado continúa), `M_Means`, `M_covar`.',
      outputs: '`X_test` (2000, 3), `y_test` (2000,) barajados.',
      prereqs: ['train-test'],
      apis: [{ name: 'rng.multivariate_normal', lib: 'numpy.random.Generator', kind: 'método', does: 'Igual que en la celda 14.' }],
      lines: [
        { cell: 34, lines: [6, 10], text: 'Mismo bucle que la celda 14 → mismas distribuciones, muestras distintas (el estado de `rng` ha avanzado).' },
        { cell: 34, lines: [13, 15], text: 'Barajado con la misma permutación para X e y.' },
      ],
      interpretation: 'Como train y test proceden de la misma distribución y hay muchas muestras, se espera un error de test parecido al de train (el notebook lo anticipa en la celda 32).',
      next: 'Se evalúan los modelos ya entrenados sobre este test.',
      whatIf: [{ change: 'Reentrenar con X_test', effect: 'Dejaría de ser un test: la evaluación volvería a ser optimista.' }],
    },
    {
      id: 'b_test_eval',
      title: 'Error y matrices de confusión en test',
      cells: [35, 36],
      stage: 'resultados',
      problem: 'Medir el rendimiento real de LDA y QDA (respuesta a Q2 y Q3).',
      purpose: 'Usa los modelos entrenados con `X_train` para predecir `X_test`; calcula error y matriz de confusión. Repitiendo para cada SNR se rellena la tabla de Q2.',
      inputs: '`ldamodel`, `qdamodel`, `X_test`, `y_test`.',
      outputs: 'LDA: 0.0095, [[992, 8], [11, 989]]; QDA: 0.0100, [[992, 8], [12, 988]] (SNR = 3 dB).',
      prereqs: ['matriz-confusion', 'train-test'],
      apis: [{ name: 'predict', lib: 'sklearn', kind: 'método', does: 'Etiquetas predichas para X_test (sin reentrenar).' }],
      lines: [
        { cell: 36, lines: [4, 5], text: 'Solo `predict` (el modelo ya está entrenado). `linear_error` sobrescribe el error de train de la celda 27.' },
        { cell: 36, lines: [11, 12], text: 'Igual con QDA.' },
      ],
      interpretation: 'Error de test ≈ error de train (0.95 % vs 0.85 %): no hay sobreajuste apreciable con 3 features y 2000 muestras. QDA es ligeramente peor por estimar más parámetros (dos covarianzas) sin necesitarlo; una diferencia de 1 muestra **no** es significativa. Error de Bayes teórico a 3 dB: $\\Phi(-\\frac{1}{2\\sigma})\\approx 0.72\\,\\%$.',
      next: 'ROC y Mahalanobis en test.',
      whatIf: [{ change: 'SNR = −10 dB', effect: 'Errores ≈ 29 %; la matriz de confusión pierde peso en la diagonal ([[708, 292], [286, 714]] para LDA).' }],
    },
    {
      id: 'b_roc_test',
      title: 'Curvas ROC en test',
      cells: [37],
      stage: 'resultados',
      problem: 'Comparar la capacidad discriminante en datos nuevos (Q4).',
      purpose: 'Misma figura que en train pero con `X_test`; es la que pide el report.',
      inputs: 'Modelos y test.',
      outputs: 'Dos ROC con su AUC.',
      prereqs: ['roc', 'auc'],
      apis: [{ name: 'RocCurveDisplay.from_estimator', lib: 'sklearn.metrics', kind: 'método', does: 'ROC sobre el conjunto dado.' }],
      lines: [{ cell: 37, lines: [3, 5], text: 'Mismo código que la celda 29 cambiando `X_train, y_train` por `X_test, y_test`.' }],
      interpretation: 'LDA y QDA tienen ROC prácticamente superpuestas: en el Case 1 son casi equivalentes. Al bajar la SNR, ambas se acercan a la diagonal. El error de la celda 36 es **un punto** de la curva (umbral 0.5 con priors iguales); el AUC resume todos.',
      next: 'Mahalanobis en test (Q5).',
      whatIf: [{ change: 'Calcular la ROC con `predict` en vez de probabilidades', effect: 'Solo habría un umbral: la «curva» tendría un único punto intermedio.' }],
    },
    {
      id: 'b_mahal_test',
      title: 'Distancia de Mahalanobis en test (Q5)',
      cells: [38, 39],
      stage: 'resultados',
      problem: 'Repetir la medida de separabilidad con el test, para cada SNR.',
      purpose: 'Rellenar Q5: comparar cómo baja la distancia al bajar la SNR y explicar la asimetría.',
      inputs: '`X_test`, `y_test`, función `mahalanobis` (celda 31).',
      outputs: '26.46 (0→1) y 26.83 (1→0) a 3 dB.',
      prereqs: ['mahalanobis'],
      apis: [],
      lines: [{ cell: 39, lines: [4, 10], text: 'Dos órdenes: muestras de 0 respecto a la clase 1 y viceversa.' }],
      interpretation: 'Con SNR = 0, −3, −10 dB los valores esperados son ≈ 15, 9.0 y 4.2 ($3 + 12\\cdot10^{\\mathrm{SNR}/10}$). Distancia menor ↔ más solapamiento ↔ más error. La asimetría se debe a que cada orden usa la media y la covarianza estimadas de una clase diferente.',
      prev: 'Reutiliza la función definida en la celda 31.',
      whatIf: [{ change: 'Usar una covarianza común (la de LDA) para ambos órdenes y medir entre centroides', effect: 'La medida sería simétrica.' }],
    },
  ],
  experiments: [
    {
      id: 'lab1_1.exp.snr',
      title: 'SNR, solapamiento y rendimiento de LDA/QDA',
      goal: 'Reproducir la tabla de Q2–Q5 cambiando **solo** la SNR, con semilla y número de muestras fijos. Después, explorar cómo afectan los priors, el tamaño de la muestra y el conjunto de evaluación.',
      blocks: ['b_snr', 'b_sigma', 'b_generate', 'b_fit', 'b_test_eval', 'b_roc_test', 'b_mahal_test'],
      template: EXPERIMENT,
      seedParam: 'seed',
      params: [
        { id: 'SNR', label: 'SNR (dB)', kind: 'datos', control: 'select', default: '3', options: [{ value: '10', label: '10 dB' }, { value: '3', label: '3 dB (guardado)' }, { value: '0', label: '0 dB' }, { value: '-3', label: '−3 dB' }, { value: '-10', label: '−10 dB' }, { value: '-15', label: '−15 dB' }], variable: 'SNR', computation: 'Cambia la conversión `10 ** (SNR/10)` y, con ella, `sig` = σ² por feature y las covarianzas `M_covar`.', result: 'Más ruido → nubes más anchas → más solapamiento → más error, AUC menor y Mahalanobis menor.', why: 'La separación entre medias es fija; la SNR decide cuántas desviaciones típicas de ruido caben en esa separación.' },
        { id: 'seed', label: 'Semilla', kind: 'datos', control: 'range', min: 0, max: 50, step: 1, default: '5', variable: 'default_rng(seed=…)', computation: 'Otra secuencia aleatoria: otras muestras de las mismas gaussianas.', result: 'Los errores varían ligeramente alrededor del mismo valor.', why: 'Es la variabilidad muestral: el modelo es el mismo; cambian los datos concretos.' },
        { id: 'n_samples', label: 'Muestras por clase', kind: 'datos', control: 'select', default: '1000', options: [{ value: '20', label: '20' }, { value: '50', label: '50' }, { value: '200', label: '200' }, { value: '1000', label: '1000 (original)' }, { value: '5000', label: '5000' }], variable: 'n_samples', computation: 'Tamaño de train **y** de test.', result: 'Con pocas muestras, el error de train se vuelve más optimista que el de test y QDA (más parámetros) se resiente más que LDA.', why: 'QDA estima 2 covarianzas (12 parámetros) frente a 1 en LDA (6): necesita más datos para estimarlas bien.' },
        { id: 'dist', label: 'Distancia entre medias', kind: 'datos', control: 'select', default: '1.', options: [{ value: '0.5', label: '0.5' }, { value: '1.', label: '1 (original)' }, { value: '2.', label: '2' }], variable: 'dist', computation: 'Escala las medias y, a SNR fija, también σ² (porque la energía crece con dist²).', result: 'Errores prácticamente iguales: a SNR fija la dificultad no cambia.', why: 'La SNR ya expresa la separación relativa a la dispersión.' },
        { id: 'priors', label: 'Priors (LDA y QDA)', kind: 'modelo', control: 'select', default: 'None', options: [{ value: 'None', label: 'Frecuencias (por defecto)' }, { value: '[0.5, 0.5]', label: '[0.5, 0.5]' }, { value: '[0.8, 0.2]', label: '[0.8, 0.2]' }, { value: '[0.2, 0.8]', label: '[0.2, 0.8]' }], variable: 'priors=…', computation: 'Cambia $\\log P(y=k)$ en el discriminante: desplaza la frontera MAP hacia la clase menos probable.', result: 'La matriz de confusión se desequilibra (más predicciones de la clase favorecida); el error total sube porque las clases reales están equilibradas. El AUC no cambia.', why: 'El AUC recorre todos los umbrales; los priors solo eligen otro punto de la ROC.', origin: 'ampliacion' },
        { id: 'roc_set', label: 'Conjunto de la ROC', kind: 'evaluacion', control: 'select', default: 'X_test, y_test', options: [{ value: 'X_test, y_test', label: 'Test (celda 37)' }, { value: 'X_train, y_train', label: 'Train (celda 29)' }], variable: 'X_roc, y_roc', computation: 'Datos sobre los que se calculan ROC y AUC.', result: 'Con 1000 muestras por clase apenas cambia; con 20, el AUC de train es más optimista.', why: 'Evaluar sobre los datos de entrenamiento mide memoria además de generalización.' },
        { id: 'pairplot', label: 'Dibujar pairplot', kind: 'visualizacion', control: 'toggle', default: 'False', variable: 'if …:', computation: 'Ninguno: solo añade una figura.', result: 'Las métricas no cambian.', why: 'Los parámetros de visualización no tocan datos ni modelos.' },
      ],
      metrics: [
        { key: 'sigma2', label: 'σ² por feature', format: 'num', digits: 5 },
        { key: 'err_lda_train', label: 'Error LDA (train)', format: 'pct', better: 'low' },
        { key: 'err_qda_train', label: 'Error QDA (train)', format: 'pct', better: 'low' },
        { key: 'err_lda_test', label: 'Error LDA (test)', format: 'pct', better: 'low' },
        { key: 'err_qda_test', label: 'Error QDA (test)', format: 'pct', better: 'low' },
        { key: 'bayes', label: 'Error de Bayes teórico (app)', format: 'pct' },
        { key: 'auc_lda', label: 'AUC LDA', format: 'num', digits: 4, better: 'high' },
        { key: 'auc_qda', label: 'AUC QDA', format: 'num', digits: 4, better: 'high' },
        { key: 'md01', label: 'Mahalanobis² 0→1 (original)', format: 'num', digits: 3 },
        { key: 'md10', label: 'Mahalanobis² 1→0 (original)', format: 'num', digits: 3 },
        { key: 'md01_corr', label: 'Mahalanobis² 0→1 (axis=0)', format: 'num', digits: 3 },
        { key: 'md10_corr', label: 'Mahalanobis² 1→0 (axis=0)', format: 'num', digits: 3 },
        { key: 'md_teorica', label: 'Mahalanobis² teórica d + dist²/σ²', format: 'num', digits: 3 },
        { key: 'cm_lda', label: 'Confusión LDA (test)', format: 'matrix' },
        { key: 'cm_qda', label: 'Confusión QDA (test)', format: 'matrix' },
      ],
      observe: [
        'Ejecuta 3, 0, −3 y −10 dB y guarda cada ejecución: tendrás la tabla de Q2 en la comparación.',
        'Compara el error de test con el «error de Bayes teórico»: ¿por qué el empírico puede quedar ligeramente por encima o por debajo?',
        'Pon 20 muestras por clase: ¿quién se aleja más de su error de train, LDA o QDA?',
        'Con priors [0.8, 0.2], mira la matriz de confusión y el AUC: ¿qué cambia y qué no?',
      ],
      source: src([6, 7, 8, 9, 10, 13, 14, 18, 27, 31, 34, 36, 37, 39], { note: 'Las líneas marcadas «(app)» y el parámetro «priors» son añadidos de la app.' }),
      fidelity: [
        { metrics: { err_lda_train: 0.0085, err_qda_train: 0.0085, err_lda_test: 0.0095, err_qda_test: 0.01, md01: 26.462714729624665, md10: 26.825706588222634, sigma2: 0.0417656 }, tol: 1e-6, source: 'Salidas guardadas de las celdas 11, 27, 36 y 39 (SNR = 3 dB)' },
        { params: { SNR: '-10' }, metrics: { err_lda_test: 0.289, err_qda_test: 0.2925, md01: 4.162378, md10: 4.341241 }, tol: 1e-5, source: 'Report resuelto de Lab1 (Q2 y Q5, ejecución original con media escalar)' },
        { params: { SNR: '-3' }, metrics: { err_lda_test: 0.111, err_qda_test: 0.111, auc_lda: 0.957424, auc_qda: 0.957379 }, tol: 1e-5, source: 'Report resuelto de Lab1 (Q2 y Q4)' },
      ],
    },
  ],
  activities: [
    // ---------------------------------------------------------- nivel 1: leer y explicar
    {
      kind: 'mcq',
      id: 'lab1_1.read.energy',
      skills: ['codigo'],
      blocks: ['b_energy'],
      source: src([8]),
      prompt: '¿Qué calcula este fragmento y para qué se usa en el notebook?',
      code: 'energy = 0\nfor i in range(0,n_classes):\n  energy = energy + np.dot(M_Means[i],M_Means[i])\nenergy = energy / n_classes',
      options: [
        { text: 'La energía media de los vectores de medias, numerador de la SNR', correct: true },
        { text: 'La varianza del ruido de cada clase', feedback: 'La varianza se obtiene después, dividiendo esta energía por la SNR.' },
        { text: 'La distancia entre las dos medias', feedback: 'La distancia sería ‖μ0 − μ1‖; aquí se suman ‖μi‖².' },
        { text: 'La covarianza entre features', feedback: 'No interviene ninguna covarianza.' },
      ],
      explanation: '`np.dot(v, v)` = ‖v‖². Se promedia sobre clases: $E=\\frac1c\\sum\\|\\mu_i\\|^2$, y luego σ² = E / SNR / d.',
    },
    {
      kind: 'mcq',
      id: 'lab1_1.read.attrs',
      skills: ['api'],
      blocks: ['b_fit', 'b_sklearn'],
      source: src([27]),
      prompt: 'En `lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)` y `ldamodel = lda.fit(X_train, y_train)`, ¿cuál de estos es un **atributo aprendido**?',
      options: [
        { text: '`ldamodel.covariance_`', correct: true, feedback: 'Termina en `_` y solo existe tras `fit`.' },
        { text: '`solver`', feedback: 'Es un argumento (hiperparámetro) del constructor.' },
        { text: '`y_tpred_lda`', feedback: 'Es el valor devuelto por `predict`, guardado en una variable.' },
        { text: '`store_covariance`', feedback: 'Argumento del constructor que decide si se guarda `covariance_`.' },
      ],
      explanation: 'Convención de scikit-learn: lo que se aprende de los datos termina en guion bajo (`priors_`, `means_`, `covariance_`, `coef_`).',
    },
    {
      kind: 'text',
      id: 'lab1_1.read.mahal',
      skills: ['codigo', 'concepto'],
      blocks: ['b_mahal_train'],
      source: src([31]),
      prompt: 'Explica con tus palabras qué devuelve `mahalanobis(x=xdata0, data=xdata1)` y qué representa `np.mean(md01)`.',
      code: 'md01 = mahalanobis(x = xdata0, data = xdata1)\nmdist01 = np.mean(md01)',
      rubric: [
        'Devuelve un vector con una distancia por cada muestra de la clase 0.',
        'Cada distancia se mide respecto a la media y la covarianza estimadas de la clase 1.',
        'Es la distancia al cuadrado (no se toma raíz).',
        'mdist01 es el promedio: una medida de separación entre clases en unidades de dispersión.',
      ],
      model: 'Devuelve un array (1000,) con $D_M^2(x_j)=(x_j-\\hat\\mu_1)^T\\hat C_1^{-1}(x_j-\\hat\\mu_1)$ para cada muestra $x_j$ de la clase 0, usando la media y la covarianza estimadas de la clase 1. Es una distancia al cuadrado. Su media resume lo lejos que están, en promedio, las muestras de la clase 0 de la distribución de la clase 1: cuanto mayor, más separadas están las clases y menor debería ser el error.',
    },
    // ---------------------------------------------------------- nivel 2: predecir
    {
      kind: 'predict',
      id: 'lab1_1.pred.shape_xdata',
      skills: ['codigo'],
      blocks: ['b_generate'],
      source: src([14]),
      prompt: 'Tras ejecutar la celda 14, ¿qué forma tiene `xdata`? Escríbela como tupla.',
      setup: SETUP_PARAMS + 'from numpy.random import default_rng\nrng = default_rng(5)\nM_covar = np.array([0.04 * np.eye(3)] * 2)\n',
      code: 'xdata = np.empty((0,n_feat), float)\nlabels = np.empty((0), int)\nfor i in range(0,n_classes):\n  ydata = rng.multivariate_normal(mean=M_Means[i,:], cov= M_covar[i,:,:], size= n_samples)\n  ylabels = i * np.ones((n_samples),dtype=int)\n  xdata = np.append(xdata, ydata, axis=0)\n  labels = np.append(labels, ylabels, axis=0)',
      expr: 'xdata.shape',
      answerKind: 'shape',
      explanation: '2 clases × 1000 muestras apiladas por filas (`axis=0`) y 3 features en columnas: (2000, 3).',
    },
    {
      kind: 'predict',
      id: 'lab1_1.pred.sigma0',
      skills: ['concepto', 'codigo'],
      blocks: ['b_sigma'],
      source: src([9]),
      prompt: 'Con `energy = 0.25`, `n_feat = 3` y **SNR = 0 dB**, ¿cuánto vale `sig` al final de la celda 9? (4 decimales)',
      setup: 'energy = 0.25\nn_feat = 3\nSNR = 0',
      code: 'SNR = 10 ** (SNR/10)\nsig = energy / SNR\nsig = sig / n_feat',
      expr: 'sig',
      answerKind: 'numeric',
      tolerance: 0.0005,
      explanation: '0 dB → SNR lineal = 1 → σ²_total = 0.25 → por feature 0.25/3 ≈ 0.0833.',
    },
    {
      kind: 'predict',
      id: 'lab1_1.pred.cm_sum',
      skills: ['interpretacion'],
      blocks: ['b_test_eval'],
      source: src([36]),
      prompt: 'En la matriz de confusión de test, ¿cuánto suma cada **fila**?',
      setup: lab1_1Setup(3),
      code: 'lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)\ncm = confusion_matrix(y_test, lda.predict(X_test))',
      expr: 'int(cm.sum(axis=1)[0])',
      answerKind: 'numeric',
      tolerance: 0,
      explanation: 'Las filas son clases **reales**: cada una suma el número de muestras de esa clase en test (1000). Las columnas (predichas) no tienen por qué sumar 1000.',
    },
    {
      kind: 'predict',
      id: 'lab1_1.pred.qda_cov',
      skills: ['api'],
      blocks: ['b_fit'],
      source: src([27]),
      prompt: '¿Qué devuelve `len(qdamodel.covariance_)`?',
      setup: lab1_1Setup(3),
      code: 'qdamodel = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)',
      expr: 'len(qdamodel.covariance_)',
      answerKind: 'numeric',
      tolerance: 0,
      explanation: 'QDA guarda una covarianza **por clase**: una lista de 2 matrices (3, 3).',
    },
    // ---------------------------------------------------------- nivel 3: completar
    {
      kind: 'code',
      level: 3,
      id: 'lab1_1.fill.generate',
      title: 'Completa la generación de una clase',
      skills: ['codigo', 'api'],
      blocks: ['b_generate'],
      source: src([14]),
      practice: 'rng.multivariate_normal',
      prompt: 'Rellena los huecos `___` para generar `n_samples` muestras de la clase 1 y sus etiquetas.',
      available: '`np`, `rng` (Generator con semilla 5), `M_Means` (2, 3), `M_covar` (2, 3, 3), `n_samples = 1000`.',
      produce: '`ydata` con forma (1000, 3) y `ylabels` (1000,) lleno de unos (enteros).',
      setup: SETUP_PARAMS + 'from numpy.random import default_rng\nrng = default_rng(seed=5)\nM_covar = np.array([0.0417656 * np.eye(3)] * 2)\n',
      starter: 'i = 1\nydata = rng.___(mean=___, cov=___, size=___)\nylabels = i * np.ones((n_samples), dtype=___)',
      solution: 'i = 1\nydata = rng.multivariate_normal(mean=M_Means[i,:], cov=M_covar[i,:,:], size=n_samples)\nylabels = i * np.ones((n_samples), dtype=int)',
      alternatives: ['i = 1\nydata = rng.multivariate_normal(M_Means[i], M_covar[i], n_samples)\nylabels = np.full(n_samples, 1)'],
      wrong: [{ code: 'i = 1\nydata = rng.multivariate_normal(mean=M_Means[:,i], cov=M_covar[i,:,:], size=n_samples)\nylabels = i * np.ones((n_samples), dtype=int)', expect: 'dimensiones' }],
      checks: "c.shape('ydata', (1000, 3))\nc.same('ydata', rtol=1e-9)\nc.same('ylabels')\nc.true(np.issubdtype(np.asarray(ns['ylabels']).dtype, np.integer), 'Etiquetas enteras', 'Las etiquetas deberían ser enteras (dtype=int).', 'api')",
      hints: ['El método del Generator para gaussianas multivariantes se llama `multivariate_normal`.', 'La media de la clase i es la **fila** i de `M_Means`: `M_Means[i,:]`.', 'La covarianza de la clase i es `M_covar[i,:,:]`; el tamaño es `n_samples`.'],
    },
    {
      kind: 'code',
      level: 3,
      id: 'lab1_1.fill.fit',
      title: 'Completa el entrenamiento y la evaluación de LDA',
      skills: ['api', 'codigo'],
      blocks: ['b_fit', 'b_test_eval'],
      source: src([27, 36]),
      prompt: 'Rellena los huecos para entrenar LDA con los datos de entrenamiento y calcular su **error** y su matriz de confusión en **test**.',
      available: '`X_train` (2000, 3), `y_train` (2000,), `X_test`, `y_test` (SNR = 3 dB, semilla 5); `LinearDiscriminantAnalysis`, `accuracy_score`, `confusion_matrix` importados.',
      produce: '`ldamodel` entrenado, `linear_error` (float) y `cm` (2, 2).',
      setup: lab1_1Setup(3),
      starter: 'lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)\nldamodel = lda.___(___, ___)\ny_test_lda = ldamodel.___(___)\nlinear_error = ___ - accuracy_score(___, ___)\ncm = confusion_matrix(___, ___)',
      solution: 'lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)\nldamodel = lda.fit(X_train, y_train)\ny_test_lda = ldamodel.predict(X_test)\nlinear_error = 1. - accuracy_score(y_test, y_test_lda)\ncm = confusion_matrix(y_test, y_test_lda)',
      alternatives: ['ldamodel = LinearDiscriminantAnalysis().fit(X_train, y_train)\npred = ldamodel.predict(X_test)\nlinear_error = np.mean(pred != y_test)\ncm = confusion_matrix(y_test, pred)'],
      wrong: [
        { code: 'ldamodel = LinearDiscriminantAnalysis().fit(X_train, y_train)\npred = ldamodel.predict(X_test)\nlinear_error = accuracy_score(y_test, pred)\ncm = confusion_matrix(y_test, pred)', expect: 'razonamiento' },
        { code: 'ldamodel = LinearDiscriminantAnalysis().fit(X_train, y_train)\npred = ldamodel.predict(X_test)\nlinear_error = 1 - accuracy_score(y_test, pred)\ncm = confusion_matrix(pred, y_test)', expect: 'razonamiento' },
      ],
      checks: "c.fitted('ldamodel', 'LinearDiscriminantAnalysis', ['means_'])\nc.same('linear_error', rtol=1e-9, desc='El error de test')\nc.same('cm', desc='La matriz de confusión (filas = reales)')",
      hints: ['Primero `fit` con los datos de **entrenamiento**; después `predict` sobre **test**.', 'El error es 1 menos la tasa de acierto.', '`confusion_matrix(y_true, y_pred)`: el orden importa (filas = valores reales).'],
      explanation: 'Valores esperados con SNR = 3 dB: error 0.0095, matriz [[992, 8], [11, 989]] (coinciden con la salida guardada de la celda 36).',
    },
    // ---------------------------------------------------------- nivel 4: corregir
    {
      kind: 'code',
      level: 4,
      id: 'lab1_1.fix.mahal',
      title: 'Corrige la función de Mahalanobis',
      skills: ['codigo', 'concepto'],
      blocks: ['b_mahal_train'],
      source: src([31], { note: 'Errata del original' }),
      prompt: 'La función del notebook tiene dos problemas: resta una media **escalar** y la condición `if not cov` falla si se le pasa una matriz. Corrígela para que funcione en general. Se probará con datos cuyas features tienen medias **distintas** y también pasando `cov` explícitamente.',
      available: '`np`, `sp` (scipy con `scipy.linalg`), `A` (300, 3) y `B` (400, 3) con medias distintas por feature.',
      produce: 'Una función `mahalanobis(x=None, data=None, cov=None)` que devuelva $D_M^2$ de cada fila de `x`; además `dAB = np.mean(mahalanobis(x=A, data=B))` y `dAB_cov = np.mean(mahalanobis(x=A, data=B, cov=np.cov(B.T)))`.',
      setup: 'import numpy as np\nimport scipy as sp\nimport scipy.linalg\nrng = np.random.default_rng(1)\nA = rng.normal([0, 2, -1], [1, 0.5, 2], size=(300, 3))\nB = rng.normal([1, -1, 3], [0.7, 1.5, 1], size=(400, 3))',
      starter: 'def mahalanobis(x=None, data=None, cov=None):\n  x_minus_mu = x - np.mean(data)\n  if not cov:\n    cov = np.cov(data.T)\n  inv_covmat = sp.linalg.inv(cov)\n  left_term = np.dot(x_minus_mu, inv_covmat)\n  mahal = np.dot(left_term, x_minus_mu.T)\n  return mahal.diagonal()\n\ndAB = np.mean(mahalanobis(x=A, data=B))\ndAB_cov = np.mean(mahalanobis(x=A, data=B, cov=np.cov(B.T)))',
      solution: 'def mahalanobis(x=None, data=None, cov=None):\n  x_minus_mu = x - np.mean(data, axis=0)\n  if cov is None:\n    cov = np.cov(data.T)\n  inv_covmat = sp.linalg.inv(cov)\n  left_term = np.dot(x_minus_mu, inv_covmat)\n  mahal = np.dot(left_term, x_minus_mu.T)\n  return mahal.diagonal()\n\ndAB = np.mean(mahalanobis(x=A, data=B))\ndAB_cov = np.mean(mahalanobis(x=A, data=B, cov=np.cov(B.T)))',
      alternatives: ['def mahalanobis(x=None, data=None, cov=None):\n    mu = data.mean(axis=0)\n    C = np.cov(data, rowvar=False) if cov is None else cov\n    d = x - mu\n    return np.einsum("ij,jk,ik->i", d, np.linalg.inv(C), d)\n\ndAB = np.mean(mahalanobis(x=A, data=B))\ndAB_cov = np.mean(mahalanobis(x=A, data=B, cov=np.cov(B.T)))'],
      wrong: [{ code: 'def mahalanobis(x=None, data=None, cov=None):\n  x_minus_mu = x - np.mean(data)\n  if cov is None:\n    cov = np.cov(data.T)\n  inv_covmat = sp.linalg.inv(cov)\n  return np.dot(np.dot(x_minus_mu, inv_covmat), x_minus_mu.T).diagonal()\n\ndAB = np.mean(mahalanobis(x=A, data=B))\ndAB_cov = np.mean(mahalanobis(x=A, data=B, cov=np.cov(B.T)))', expect: 'razonamiento' }],
      checks: "c.same('dAB', rtol=1e-8, desc='La distancia media A→B')\nc.same('dAB_cov', rtol=1e-8, desc='La distancia con cov explícita')\nc.true(callable(ns.get('mahalanobis')), 'La función existe', 'Define la función mahalanobis', 'ejecucion')",
      hints: ['La media que hay que restar es el **vector** de medias de las columnas de `data`.', '`np.mean(data, axis=0)` devuelve una media por feature.', 'Para comprobar si no se ha pasado covarianza usa `cov is None`; `not array` es ambiguo.'],
      explanation: 'En el notebook el error pasa desapercibido porque las tres features de cada clase tienen la misma media; con medias distintas el resultado original sería falso.',
    },
    {
      kind: 'code',
      level: 4,
      id: 'lab1_1.fix.snr',
      title: 'Encuentra el error en el cálculo de σ²',
      skills: ['concepto', 'codigo'],
      blocks: ['b_sigma'],
      source: src([9]),
      prompt: 'Este código pretende calcular la varianza por feature `sig` para SNR = −3 dB, pero contiene **dos** errores de concepto. Corrígelo.',
      available: '`energy = 0.25`, `n_feat = 3`, `SNR = -3` (dB).',
      produce: '`sig` = σ² por feature.',
      setup: 'energy = 0.25\nn_feat = 3\nSNR = -3',
      starter: 'SNR = 10 ** (SNR * 10)\nsig = energy * SNR\nsig = sig / n_feat',
      solution: 'SNR = 10 ** (SNR/10)\nsig = energy / SNR\nsig = sig / n_feat',
      alternatives: ['sig = energy / (10 ** (SNR / 10)) / n_feat'],
      checks: "c.same('sig', rtol=1e-9, desc='σ² por feature')",
      hints: ['La conversión de dB a lineal es $10^{x/10}$.', 'A más SNR, **menos** ruido: σ² debe ser inversamente proporcional a la SNR.'],
      explanation: '−3 dB → 0.501 lineal → σ² = 0.25/0.501/3 ≈ 0.1663.',
    },
    {
      kind: 'code',
      level: 4,
      id: 'lab1_1.fix.priors',
      title: 'Corrige la impresión de las medias',
      skills: ['api'],
      blocks: ['b_fit'],
      source: src([27], { note: 'Errata del original' }),
      prompt: 'El notebook imprime `ldamodel.priors_` con la etiqueta «LDA model means». Guarda en `medias` las medias que ha estimado el modelo y en `medias_manual` las mismas medias calculadas a mano con NumPy a partir de `X_train` e `y_train`.',
      available: '`ldamodel` entrenado sobre `X_train`, `y_train` (SNR = 3 dB).',
      produce: '`medias` y `medias_manual`, ambas (2, 3).',
      setup: lab1_1Setup(3) + 'ldamodel = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)\n',
      starter: "medias = ldamodel.priors_\nmedias_manual = ...\nprint('LDA model means:', medias)",
      solution: "medias = ldamodel.means_\nmedias_manual = np.array([X_train[y_train == k].mean(axis=0) for k in [0, 1]])\nprint('LDA model means:', medias)",
      alternatives: ["medias = ldamodel.means_\nmedias_manual = np.vstack([X_train[y_train==0].mean(0), X_train[y_train==1].mean(0)])"],
      wrong: [{ code: 'medias = ldamodel.means_\nmedias_manual = np.array([X_train[y_train == k].mean() for k in [0, 1]])', expect: 'dimensiones' }],
      checks: "c.same('medias', rtol=1e-9)\nc.same('medias_manual', rtol=1e-9)",
      hints: ['Los atributos aprendidos de LDA incluyen `priors_`, `means_` y `covariance_`.', 'Para la versión manual filtra por clase con una máscara booleana y promedia por columnas (`axis=0`).'],
    },
    // ---------------------------------------------------------- nivel 5: de memoria
    {
      kind: 'code',
      level: 5,
      id: 'lab1_1.mem.dataset',
      title: 'Genera el dataset del Case 1 desde cero',
      skills: ['escritura', 'codigo'],
      blocks: ['b_design', 'b_energy', 'b_sigma', 'b_covar', 'b_generate'],
      source: src([7, 8, 9, 10, 13, 14]),
      prompt: 'Sin mirar el notebook: para **SNR = 0 dB**, construye las medias, σ², las covarianzas y genera el conjunto de entrenamiento como en las celdas 7–14 (2 clases, 1000 muestras por clase, 3 features, `dist = 1`, semilla 5).',
      available: 'Nada: importa lo que necesites.',
      produce: '`M_Means` (2, 3), `sig` (float), `M_covar` (2, 3, 3), `xdata` (2000, 3) y `labels` (2000,) en orden de clase, generados con `default_rng(seed=5)` y `multivariate_normal` clase a clase.',
      setup: '',
      starter: '',
      solution: lab1_1Setup(0).split('np.random.seed(0)')[0],
      alternatives: ['import numpy as np\nrng = np.random.default_rng(5)\nn_feat, n_samples = 3, 1000\nM_Means = 0.5 * np.array([[1., 1, 1], [-1, -1, -1]]) / np.sqrt(3)\nenergy = np.mean(np.sum(M_Means**2, axis=1))\nsig = energy / 10**(0/10) / n_feat\nM_covar = np.stack([sig * np.eye(3), sig * np.eye(3)])\nxdata = np.vstack([rng.multivariate_normal(M_Means[k], M_covar[k], n_samples) for k in range(2)])\nlabels = np.repeat([0, 1], n_samples)'],
      checks: "c.same('M_Means', rtol=1e-9)\nc.same('sig', rtol=1e-9, desc='σ² por feature')\nc.same('M_covar', rtol=1e-9)\nc.shape('xdata', (2000, 3))\nc.same('xdata', rtol=1e-8, desc='Las muestras (misma semilla y orden)')\nc.same('labels')",
      hints: ['Medias: $\\pm 0.5\\cdot$dist$\\cdot(1,1,1)/\\sqrt{3}$.', 'σ² = energía / SNR lineal / n_feat, con SNR lineal = $10^{0/10}$.', 'Para reproducir las muestras, genera primero la clase 0 y después la clase 1 con el mismo `rng`.'],
    },
    {
      kind: 'code',
      level: 5,
      id: 'lab1_1.mem.evaluate',
      title: 'Entrena y evalúa LDA y QDA sin ayuda',
      skills: ['escritura', 'api'],
      blocks: ['b_fit', 'b_test_eval'],
      source: src([27, 36]),
      prompt: 'Entrena un clasificador lineal y uno cuadrático con los datos de entrenamiento y calcula, en **test**, el error y la matriz de confusión de cada uno.',
      available: '`X_train`, `y_train`, `X_test`, `y_test` (SNR = −3 dB). No hay nada importado de scikit-learn.',
      produce: '`err_lda`, `err_qda` (float) y `cm_lda`, `cm_qda` (2, 2).',
      setup: lab1_1Setup(-3).replace('from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nfrom sklearn.metrics import accuracy_score, confusion_matrix\n', ''),
      starter: '',
      solution: 'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nfrom sklearn.metrics import accuracy_score, confusion_matrix\nlda = LinearDiscriminantAnalysis().fit(X_train, y_train)\nqda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)\nerr_lda = 1 - accuracy_score(y_test, lda.predict(X_test))\nerr_qda = 1 - accuracy_score(y_test, qda.predict(X_test))\ncm_lda = confusion_matrix(y_test, lda.predict(X_test))\ncm_qda = confusion_matrix(y_test, qda.predict(X_test))',
      checks: "c.same('err_lda', rtol=1e-9)\nc.same('err_qda', rtol=1e-9)\nc.same('cm_lda')\nc.same('cm_qda')",
      hints: ['Las clases están en `sklearn.discriminant_analysis`; las métricas, en `sklearn.metrics`.', 'Entrena con train; evalúa con test.'],
      explanation: 'A −3 dB ambos cometen 222 errores de 2000 (11.1 %).',
    },
    // ---------------------------------------------------------- nivel 6: adaptar
    {
      kind: 'code',
      level: 6,
      id: 'lab1_1.adapt.auc',
      title: 'Adapta: AUC sin dibujar la ROC',
      skills: ['api', 'metodo'],
      blocks: ['b_roc_test'],
      source: src([37]),
      practice: 'sklearn.metrics.roc_auc_score / roc_curve',
      prompt: 'El report solo pide el valor del AUC en test para −10 dB. Calcula el AUC de LDA y de QDA **sin** usar `RocCurveDisplay`, a partir de las probabilidades a posteriori de la clase positiva (1).',
      available: '`ldamodel`, `qdamodel` entrenados; `X_test`, `y_test` (SNR = −10 dB).',
      produce: '`auc_lda`, `auc_qda` (float).',
      setup: lab1_1Setup(-10) + 'ldamodel = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)\nqdamodel = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)\n',
      starter: '',
      solution: 'from sklearn.metrics import roc_auc_score\nauc_lda = roc_auc_score(y_test, ldamodel.predict_proba(X_test)[:, 1])\nauc_qda = roc_auc_score(y_test, qdamodel.predict_proba(X_test)[:, 1])',
      alternatives: ['from sklearn.metrics import roc_curve, auc\nf, t, _ = roc_curve(y_test, ldamodel.predict_proba(X_test)[:, 1])\nauc_lda = auc(f, t)\nf, t, _ = roc_curve(y_test, qdamodel.predict_proba(X_test)[:, 1])\nauc_qda = auc(f, t)', 'from sklearn.metrics import roc_auc_score\nauc_lda = roc_auc_score(y_test, ldamodel.decision_function(X_test))\nauc_qda = roc_auc_score(y_test, qdamodel.decision_function(X_test))'],
      wrong: [{ code: 'from sklearn.metrics import roc_auc_score\nauc_lda = roc_auc_score(y_test, ldamodel.predict(X_test))\nauc_qda = roc_auc_score(y_test, qdamodel.predict(X_test))', expect: 'razonamiento' }],
      checks: "c.same('auc_lda', rtol=1e-6, desc='AUC de LDA')\nc.same('auc_qda', rtol=1e-6, desc='AUC de QDA')",
      hints: ['`predict_proba(X)` devuelve una columna por clase; la positiva es la segunda: `[:, 1]`.', 'Con `predict` (etiquetas) solo tendrías un umbral: el AUC saldría menor.'],
      explanation: 'A −10 dB el AUC es ≈ 0.79 para los dos modelos.',
    },
    {
      kind: 'code',
      level: 6,
      id: 'lab1_1.adapt.three',
      title: 'Adapta: tres clases en lugar de dos',
      skills: ['escritura', 'metodo'],
      blocks: ['b_design', 'b_generate', 'b_fit'],
      source: src([7, 14, 27]),
      transfer: true,
      prompt: 'Modifica el diseño para **3 clases** con medias $(1,1,1)$, $(-1,-1,-1)$ y $(1,-1,1)$ (multiplicadas por $0.5/\\sqrt3$), misma $C=\\sigma^2 I$ con σ² = 0.2 y 500 muestras por clase. Genera train con `default_rng(seed=5)` (clase 0, 1 y 2 en orden), entrena LDA y calcula su error de entrenamiento.',
      available: 'Nada importado.',
      produce: '`xdata` (1500, 3), `labels` (1500,), `err_train` (float).',
      setup: '',
      starter: '',
      solution: 'import numpy as np\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.metrics import accuracy_score\nrng = np.random.default_rng(seed=5)\nM_Means = 0.5 * np.array([[1, 1, 1], [-1, -1, -1], [1, -1, 1]]) / np.sqrt(3)\nC = 0.2 * np.eye(3)\nxdata = np.vstack([rng.multivariate_normal(M_Means[k], C, 500) for k in range(3)])\nlabels = np.repeat([0, 1, 2], 500)\nlda = LinearDiscriminantAnalysis().fit(xdata, labels)\nerr_train = 1 - accuracy_score(labels, lda.predict(xdata))',
      checks: "c.shape('xdata', (1500, 3))\nc.same('xdata', rtol=1e-8)\nc.same('labels')\nc.same('err_train', rtol=1e-9)",
      hints: ['`M_Means` pasa a tener 3 filas.', 'Cada clase usa la misma covarianza `0.2 * np.eye(3)`.', 'LDA es multiclase de forma nativa: el mismo `fit` sirve.'],
    },
    // ---------------------------------------------------------- interpretación
    {
      kind: 'mcq',
      id: 'lab1_1.int.ldaqda',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_test_eval'],
      source: src([36]),
      prompt: 'A 3 dB, LDA comete 19 errores en test y QDA 20. ¿Qué conclusión está **justificada**?',
      options: [
        { text: 'Ambos rinden prácticamente igual; con el Case 1 era lo esperado', correct: true },
        { text: 'LDA es mejor que QDA en general', feedback: 'Una diferencia de 1 muestra en una sola semilla no permite generalizar.' },
        { text: 'QDA sobreajusta gravemente', feedback: 'Su error de test es casi igual al de train; no hay sobreajuste grave.' },
        { text: 'Los datos no son gaussianos', feedback: 'Se han generado gaussianos por construcción.' },
      ],
      explanation: 'Con covarianzas iguales la frontera óptima es lineal; QDA puede representarla, pero estima más parámetros. La diferencia es ruido muestral.',
    },
    {
      kind: 'mcq',
      id: 'lab1_1.int.trainvstest',
      skills: ['interpretacion'],
      blocks: ['b_test_eval', 'b_fit'],
      source: src([27, 36]),
      prompt: 'Error LDA: 0.85 % en train y 0.95 % en test. ¿Cómo se interpreta?',
      options: [
        { text: 'El train es ligeramente optimista; el modelo generaliza bien', correct: true },
        { text: 'Hay sobreajuste importante', feedback: 'Una diferencia de 0.1 puntos con 2000 muestras es compatible con el azar.' },
        { text: 'El test está mal generado', feedback: 'Se genera con el mismo modelo y el mismo rng.' },
        { text: 'Se debería usar el error de train para el report', feedback: 'El report pide expresamente el test.' },
      ],
      explanation: 'Con pocos parámetros (LDA: 2 medias + 1 covarianza) y muchas muestras, el error de entrenamiento casi coincide con el de test.',
    },
    {
      kind: 'mcq',
      id: 'lab1_1.int.roc',
      skills: ['interpretacion'],
      blocks: ['b_roc_test'],
      figure: { notebook: 'lab1_1', cell: 37, output: 0 },
      source: src([37]),
      prompt: 'La figura (salida guardada, SNR = 3 dB) muestra las ROC en test. ¿Qué afirmación es **incorrecta**?',
      options: [
        { text: 'El AUC cambia si se modifican los priors del clasificador', correct: true, feedback: 'Correcto: es la afirmación falsa. Los priors desplazan el umbral, es decir, eligen otro punto de la misma curva.' },
        { text: 'Cada punto de la curva corresponde a un umbral sobre P(y=1|x)' },
        { text: 'Un AUC cercano a 1 indica clases casi separables' },
        { text: 'La diagonal corresponde a un clasificador aleatorio' },
      ],
      explanation: 'La ROC barre todos los umbrales; cambiar priors equivale a cambiar de umbral, no la curva. El AUC solo cambia si cambian las puntuaciones relativas (otro modelo u otros datos).',
    },
    {
      kind: 'text',
      id: 'lab1_1.int.q2',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_test_eval'],
      source: { notebook: NB, cells: [36], doc: 'lab1_guide', section: 'Q2', origin: 'original' },
      prompt: 'Q2 · Con la tabla de errores de test (LDA: 0.95 %, 4.25 %, 11.10 %, 28.90 %; QDA: 1.00 %, 4.25 %, 11.10 %, 29.25 % para 3, 0, −3, −10 dB), redacta la discusión.',
      rubric: [
        'El error crece al bajar la SNR porque aumenta el solapamiento entre clases.',
        'LDA y QDA obtienen errores casi iguales.',
        'Se justifica por el Case 1: covarianzas iguales → frontera óptima lineal; LDA está bien especificado.',
        'QDA estima más parámetros sin beneficio; diferencias de pocas muestras no son significativas (una sola semilla).',
        'Opcional: relación con el error de Bayes teórico.',
      ],
      model: 'El error aumenta de forma monótona al disminuir la SNR: con más varianza de ruido las dos gaussianas se solapan más y cualquier clasificador comete más errores (de ≈1 % a ≈29 %). LDA y QDA son prácticamente equivalentes porque los datos siguen el Case 1, con la misma covarianza isotrópica en ambas clases: la frontera MAP óptima es un plano y LDA es el modelo correcto. QDA puede representar esa frontera, pero estima dos covarianzas: no gana nada y, con muestras finitas, puede perder ligeramente (−10 dB: 29.25 % frente a 28.90 %). Las diferencias son de pocas muestras en una sola realización, compatibles con la variación muestral.',
    },
    {
      kind: 'text',
      id: 'lab1_1.int.q5',
      skills: ['interpretacion', 'concepto'],
      blocks: ['b_mahal_test'],
      source: { notebook: NB, cells: [39], doc: 'lab1_guide', section: 'Q5', origin: 'original' },
      prompt: 'Q5 · ¿Por qué `mahalanobis(x=xdata0, data=xdata1)` y `mahalanobis(x=xdata1, data=xdata0)` dan resultados distintos?',
      rubric: [
        'Cada orden mide las muestras de una clase respecto a la media y la covarianza de la otra.',
        'Media y covarianza se estiman con muestras distintas, así que no coinciden exactamente.',
        'Por tanto la medida no es simétrica (aunque los parámetros poblacionales sean iguales).',
        'Con una covarianza común y midiendo entre centroides sería simétrica.',
      ],
      model: 'En el primer caso se promedian las distancias de las muestras de la clase 0 a la distribución estimada de la clase 1 ($\\hat\\mu_1$, $\\hat C_1$); en el segundo, las de la clase 1 a la de la clase 0 ($\\hat\\mu_0$, $\\hat C_0$). Aunque poblacionalmente $C_0=C_1$, las estimaciones muestrales difieren y además cada promedio recorre muestras distintas, así que los valores no coinciden. Es una medida asimétrica; si se usara una covarianza común y se midiera entre centroides, sería simétrica.',
    },
    {
      kind: 'mcq',
      id: 'lab1_1.int.misconception',
      skills: ['interpretacion', 'concepto'],
      blocks: ['b_test_eval', 'b_theory'],
      source: src([24, 25, 36]),
      prompt: '«Como QDA es más flexible que LDA, siempre obtendrá un error de test menor o igual». ¿Qué error conceptual contiene?',
      options: [
        { text: 'Confunde capacidad del modelo con rendimiento: más parámetros implican más varianza de estimación', correct: true },
        { text: 'Ninguno: un modelo más general nunca puede ser peor', feedback: 'Puede serlo en test: estima más parámetros con los mismos datos.' },
        { text: 'QDA no es más flexible que LDA', feedback: 'Sí lo es: LDA es un caso particular de QDA.' },
      ],
      explanation: 'LDA es un caso particular de QDA (covarianza común). Si el supuesto se cumple, LDA estima menos parámetros y puede generalizar mejor, sobre todo con pocas muestras (pruébalo con 20 muestras por clase).',
    },
    {
      kind: 'numeric',
      id: 'lab1_1.int.bayes',
      skills: ['concepto'],
      blocks: ['b_sigma', 'b_test_eval'],
      source: { origin: 'app', note: 'Cálculo teórico añadido por la app a partir del modelo del notebook' },
      prompt: 'Con dos gaussianas de covarianza $\\sigma^2 I$, priors iguales y medias separadas una distancia 1, el error de Bayes es $\\Phi\\left(-\\frac{1}{2\\sigma}\\right)$. Para σ² = 0.0833 (0 dB), ¿cuánto vale en %? (1 decimal)',
      answer: 4.2,
      tolerance: 0.15,
      unit: '%',
      explanation: 'σ = 0.2887; 1/(2σ) = 1.732; Φ(−1.732) ≈ 0.0416 → 4.2 %. El error empírico de LDA a 0 dB fue 4.25 %.',
    },
    {
      kind: 'dynamic',
      id: 'lab1_1.dyn.sigma',
      skills: ['concepto', 'codigo'],
      blocks: ['b_sigma'],
      source: src([9], { note: 'Variante generada' }),
      answerKind: 'numeric',
      generator: `import numpy as np
rng = np.random.default_rng(SEED)
snr = int(rng.choice([-12, -9, -6, -5, -2, 1, 2, 4, 5, 6, 8, 12]))
d = int(rng.choice([2, 3, 4]))
energy = 0.25
val = energy / 10**(snr/10) / d
variante = {
  'prompt': f'Con energy = 0.25, n_feat = {d} y SNR = {snr} dB, ¿cuánto vale sig (σ² por feature) al final de la celda 9? (4 decimales)',
  'answer': round(val, 6), 'tolerance': 5e-4,
  'explanation': f'SNR lineal = 10^({snr}/10) = {10**(snr/10):.4f}; σ² total = 0.25 / {10**(snr/10):.4f} = {energy/10**(snr/10):.4f}; por feature: / {d} = {val:.4f}.'
}`,
    },
  ],
  recall: ['lab1_1.fill.generate', 'lab1_1.fill.fit', 'lab1_1.fix.snr', 'lab1_1.fix.mahal', 'lab1_1.mem.dataset', 'lab1_1.mem.evaluate', 'lab1_1.adapt.auc'],
  practice: [
    {
      id: 'lab1_1.practice.report',
      title: 'Mini-report a −3 dB (Q2–Q5)',
      scenario: 'Te piden completar el report para **SNR = −3 dB** sin el notebook delante: genera los datos, evalúa en test, calcula el AUC y las distancias de Mahalanobis (corregidas) y justifica los resultados.',
      steps: ['lab1_1.fix.snr', 'lab1_1.mem.evaluate', 'lab1_1.adapt.auc', 'lab1_1.fix.mahal', 'lab1_1.int.q2', 'lab1_1.int.q5'],
      source: { notebook: NB, doc: 'lab1_guide', section: 'Q2–Q5', origin: 'original' },
    },
  ],
  notes: [
    'La app reproduce exactamente las salidas guardadas con SNR = 3 dB (errores, matrices y covarianzas), verificado con `npm run verify`.',
  ],
};
