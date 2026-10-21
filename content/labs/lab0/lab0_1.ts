import type { NotebookContent } from '../../../src/content/types';

const NB = 'lab0_1';
const src = (cells: number[], extra: Partial<{ section: string; note: string; origin: 'original' | 'ampliacion' }> = {}) => ({
  notebook: NB,
  cells,
  origin: 'original' as const,
  ...extra,
});

/** Estado del notebook tras las celdas 5–15: las cuatro muestras con semilla 1. */
const SETUP = `import numpy as np
import scipy.stats as ss
from numpy.random import default_rng
n = 100000
nbins = 50
rng = default_rng(seed=1)
nmu, nsigma = 3, 5
data_normal = rng.normal(nmu, nsigma, n)
data_uniform = rng.uniform(-0.5, 0.5, n)
data_laplace = rng.laplace(0, 1, n)
data_rayleigh = rng.rayleigh(5, n)
`;

const SETUP_PLOT = SETUP + 'import matplotlib.pyplot as plt\nimport seaborn as sns\n';

const EXPERIMENT = `import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from numpy.random import default_rng

# celda 7 · tamaño, bins y semilla
n = {{n}}
nbins = {{nbins}}
rng = default_rng(seed={{seed}})

# celdas 9-15 · las cuatro muestras, en el mismo orden que el notebook
nmu, nsigma = {{nmu}}, {{nsigma}}
data_normal = rng.normal(nmu, nsigma, n)
data_uniform = rng.uniform( -0.5, 0.5, n)
data_laplace = rng.laplace( 0, 1, n)
data_rayleigh = rng.rayleigh(5, n)

x = {{dist}}                       # (app) muestra que se analiza
npoints = {{npoints}}

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
# celda 17 · histograma con KDE (no es un ajuste normal)
sns.histplot(data=x, bins=nbins, kde=True, ax=ax[0])
ax[0].set_title('Histograma + KDE')
# celda 19 · CDF empírica (rojo) frente a la normal con la misma media y desviación (azul)
xx = np.linspace(np.min(x),np.max(x),npoints)
ncdf = ss.norm.cdf(xx,np.mean(x),np.std(x))
ax[1].plot(xx,ncdf)
ax[1].hist(x, npoints, density=True,histtype='step',cumulative=True,color='r')
ax[1].set_title('CDF')
# celda 23 · gráfico Q-Q frente a la normal
ss.probplot(x, plot=ax[2])
ax[2].set_title('Q-Q')
plt.show()

# celda 21 · momentos (curtosis de Fisher: normal = 0)
# (app) referencias teóricas y máxima diferencia entre la CDF empírica y la normal (celda 26)
teorica = {'data_normal': (0, 0), 'data_uniform': (0, -1.2), 'data_laplace': (0, 3), 'data_rayleigh': (0.631, 0.245)}
z = np.linspace(-4, 4, 3001)
empirical = np.searchsorted(np.sort(x), x.mean() + x.std() * z, side='right') / len(x)
resultados = {
  'media': np.mean(x), 'std': np.std(x),
  'skew': ss.skew(x), 'kurt': ss.kurtosis(x),
  'skew_teorica': teorica['{{dist}}'][0], 'kurt_teorica': teorica['{{dist}}'][1],
  'max_dif_cdf': np.max(np.abs(empirical - ss.norm.cdf(z))),
}
`;

export const lab0_1: NotebookContent = {
  id: NB,
  lab: 'lab0',
  title: 'Lab 0 · Parte 1 — Gaussianidad de distribuciones sintéticas',
  short: 'EDA: distribuciones sintéticas',
  origin: 'original',
  context: {
    goal: `Aprender a **reconocer si una muestra es gaussiana** con las herramientas de EDA antes de usar clasificadores que lo suponen (LDA/QDA en Lab1).

El notebook genera cuatro muestras de distribuciones conocidas — **normal, uniforme, Laplace y Rayleigh** — y las compara con cuatro herramientas: histograma (con KDE), **CDF** empírica frente a la normal ajustada, **asimetría** y **curtosis**, y **gráfico Q-Q**. Como conocemos la verdad, sirve para calibrar el ojo: qué aspecto tiene «ser normal» y cómo se ven la cola pesada (Laplace), la cola ligera y acotada (uniforme) y la asimetría (Rayleigh).

Responde a la **Q1** de la guía: describir y comparar las medidas obtenidas para las cuatro distribuciones.`,
    dataset: `Datos **generados** con \`numpy.random.default_rng(seed=1)\`, **n = 100 000** muestras de cada distribución: $\\mathcal N(3, 5^2)$, $\\mathcal U(-0.5, 0.5)$, Laplace(0, 1) y Rayleigh(5). No hay clases: cada muestra es un vector 1D.`,
    variables: [
      { name: 'n, nbins', meaning: 'Tamaño de cada muestra y número de intervalos de los histogramas.', shape: 'escalares' },
      { name: 'rng', meaning: 'Generador aleatorio (Generator PCG64) con semilla 1.', shape: '—' },
      { name: 'data_normal, data_uniform, data_laplace, data_rayleigh', meaning: 'Las cuatro muestras.', shape: '(100000,) cada una' },
      { name: 'all_data', meaning: 'Las cuatro muestras apiladas por filas.', shape: '(4, 100000)' },
      { name: 'dkurt, dskew', meaning: 'Curtosis (exceso, Fisher) y asimetría de cada muestra.', shape: '(4,)' },
      { name: 'x, ncdf', meaning: 'Rejilla de evaluación y CDF de la normal con media y desviación de la muestra.', shape: '(10000,)' },
    ],
    questions: [{ id: 'Q1', text: 'Describe brevemente las conclusiones del análisis de las cuatro distribuciones (normal, Rayleigh, Laplace y uniforme).', blocks: ['b_histplot', 'b_cdf', 'b_moments', 'b_qq'], doc: 'lab0_guide' }],
    notation: [
      { symbol: '$\\mathcal N(\\mu,\\sigma^2)$', meaning: 'normal de media μ y varianza σ²', code: 'rng.normal(nmu, nsigma, n) (nsigma es σ, no σ²)' },
      { symbol: '$F_X(x)=P(X\\le x)$', meaning: 'función de distribución acumulada', code: 'ss.norm.cdf, hist(cumulative=True)' },
      { symbol: '$Sk=\\frac{E[(x-\\mu)^3]}{\\sigma^3}$', meaning: 'asimetría (skewness)', code: 'ss.skew' },
      { symbol: '$K-3=\\frac{E[(x-\\mu)^4]}{\\sigma^4}-3$', meaning: 'exceso de curtosis (definición de Fisher, normal = 0)', code: 'ss.kurtosis' },
      { symbol: 'Q-Q', meaning: 'cuantiles de la muestra frente a cuantiles teóricos normales', code: 'ss.probplot' },
    ],
  },
  flow: [
    { stage: 'contexto', label: 'Contexto', blocks: ['b_intro', 'b_imports'], summary: 'Qué es EDA y qué librerías se usan.' },
    { stage: 'datos', label: 'Datos', blocks: ['b_rng', 'b_normal', 'b_uniform', 'b_laplace', 'b_rayleigh'], summary: 'Cuatro muestras de 100 000 valores con semilla 1.' },
    { stage: 'exploracion', label: 'Exploración', blocks: ['b_histplot', 'b_cdf', 'b_moments', 'b_qq'], summary: 'Histograma+KDE, CDF, momentos y Q-Q.' },
    { stage: 'interpretacion', label: 'Interpretación', blocks: ['b_extra_moments', 'b_extra_ecdf'], summary: 'Evidencia adicional para Q1 (ampliación).' },
  ],
  blocks: [
    {
      id: 'b_intro',
      title: 'Título, Colab, Python y qué es EDA',
      cells: [0, 1, 2, 3],
      stage: 'contexto',
      problem: 'Situar el laboratorio: qué es el análisis exploratorio y qué se va a hacer en esta parte.',
      purpose: 'Define el objetivo: usar EDA para analizar la **gaussianidad** de datos sintéticos (parte 1) antes de explorar un dataset real (parte 2, Iris).',
      inputs: 'Texto.',
      outputs: 'El plan de la sesión.',
      prereqs: ['eda', 'gaussianidad'],
      apis: [],
      lines: [],
      interpretation: 'EDA no demuestra hipótesis: **resume y visualiza** para decidir qué supuestos son razonables. Aquí el supuesto que interesa es la normalidad, porque los clasificadores MAP gaussianos de Lab1 lo necesitan.',
      next: 'Se importan las librerías.',
      whatIf: [],
      check: {
        kind: 'mcq',
        id: 'lab0_1.check.intro',
        skills: ['concepto'],
        source: src([2]),
        prompt: '¿Por qué interesa saber si una feature es gaussiana en esta asignatura?',
        options: [
          { text: 'Porque los clasificadores MAP gaussianos (LDA/QDA) suponen densidades normales por clase', correct: true },
          { text: 'Porque solo se pueden dibujar histogramas de datos normales', feedback: 'Un histograma se puede dibujar con cualquier distribución.' },
          { text: 'Porque scikit-learn rechaza datos no gaussianos', feedback: 'Los acepta; simplemente el modelo puede ser inadecuado.' },
        ],
        explanation: 'Si las densidades por clase son gaussianas, la regla MAP tiene forma cerrada (Lab1). EDA ayuda a valorar ese supuesto (slides Lab0, p. 14).',
      },
    },
    {
      id: 'b_imports',
      title: 'Importación de librerías',
      cells: [4, 5],
      stage: 'contexto',
      problem: 'Disponer de arrays, gráficos, tablas y funciones estadísticas.',
      purpose: 'Fija los alias `np`, `plt`, `pd`, `sns`, `ss` que usa el resto del notebook. La celda 4 explica qué aporta cada librería.',
      inputs: 'Nada.',
      outputs: 'Módulos importados.',
      prereqs: [],
      apis: [
        { name: 'import numpy as np', lib: 'numpy', kind: 'módulo', does: 'Arrays, generación aleatoria, mean/std/min/max, linspace, vstack.' },
        { name: 'import matplotlib.pyplot as plt', lib: 'matplotlib', kind: 'módulo', does: 'hist, plot, subplot, bar, title.' },
        { name: 'import pandas as pd', lib: 'pandas', kind: 'módulo', does: 'No se usa en esta parte (sí en la parte 2).' },
        { name: 'import seaborn as sns', lib: 'seaborn', kind: 'módulo', does: 'histplot con KDE (celda 17).' },
        { name: 'import scipy.stats as ss', lib: 'scipy.stats', kind: 'módulo', does: 'norm.cdf, kurtosis, skew, probplot.' },
      ],
      lines: [{ cell: 5, lines: [5, 5], text: '`scipy.stats` con alias `ss`: todas las funciones estadísticas del notebook cuelgan de aquí (`ss.norm.cdf`, `ss.kurtosis`, `ss.skew`, `ss.probplot`).' }],
      interpretation: 'Sin salida. Los alias son una convención: en el examen se espera reconocer `ss.` como `scipy.stats`.',
      next: 'Parámetros y generador aleatorio.',
      whatIf: [{ change: 'Importar solo `import scipy`', effect: '`ss.kurtosis` daría NameError; habría que escribir `scipy.stats.kurtosis` (y en versiones antiguas importar el submódulo).' }],
    },
    {
      id: 'b_rng',
      title: 'Tamaño de muestra, bins y semilla',
      cells: [6, 7],
      stage: 'datos',
      problem: 'Fijar cuántos valores se generan, cómo se dibujan y que la secuencia aleatoria sea reproducible.',
      purpose: '`n` y la semilla determinan los **datos**; `nbins` solo la **visualización**. Con `default_rng(seed=1)` cada ejecución produce los mismos números, así que las conclusiones de Q1 son repetibles.',
      inputs: 'Constantes.',
      outputs: '`n = 100000`, `nbins = 50`, `rng`.',
      prereqs: ['semilla'],
      apis: [{ name: 'default_rng', lib: 'numpy.random', kind: 'función', does: 'Crea un Generator a partir de una semilla.', args: [{ name: 'seed', value: '1', meaning: 'estado inicial' }], returns: 'Generator' }],
      lines: [
        { cell: 7, lines: [1, 2], text: '`n` es el tamaño de **cada** muestra. Con 100 000 valores los momentos estimados son muy precisos (error típico de la asimetría ≈ $\\sqrt{6/n}\\approx 0.008$).' },
        { cell: 7, lines: [6, 6], text: 'Las cuatro muestras salen del mismo `rng` en orden: normal, uniforme, Laplace, Rayleigh.' },
      ],
      interpretation: 'Con muestras tan grandes, las diferencias entre distribuciones no se deben al azar: lo que ves es la forma real de cada distribución.',
      next: 'Primera muestra: normal.',
      whatIf: [
        { change: 'n = 100', effect: 'Histogramas irregulares y momentos ruidosos: una normal puede mostrar asimetría o curtosis de ±0.5 por azar.' },
        { change: 'Cambiar la semilla', effect: 'Otros números, mismas conclusiones (con n grande).' },
      ],
    },
    {
      id: 'b_normal',
      title: 'Muestra normal N(3, 5²) e histograma',
      cells: [8, 9],
      stage: 'datos',
      problem: 'Generar la distribución de referencia.',
      purpose: 'Es el patrón con el que se comparan las demás: simétrica, campana, colas «normales».',
      inputs: '`rng`, `n`, `nbins`.',
      outputs: '`data_normal` (100000,) y su histograma normalizado.',
      prereqs: ['histograma', 'media', 'desviacion'],
      apis: [
        { name: 'rng.normal', lib: 'numpy.random.Generator', kind: 'método', does: 'Muestras normales.', args: [{ name: 'loc', value: 'nmu = 3', meaning: 'media' }, { name: 'scale', value: 'nsigma = 5', meaning: 'desviación típica (no varianza)' }, { name: 'size', value: 'n', meaning: 'número de muestras' }], returns: 'array (n,)' },
        { name: 'plt.hist', lib: 'matplotlib', kind: 'función', does: 'Histograma.', args: [{ name: 'bins', value: 'nbins', meaning: 'número de intervalos' }, { name: 'density', value: 'True', meaning: 'altura = densidad: el área total vale 1' }] },
      ],
      lines: [
        { cell: 9, lines: [1, 2], text: 'El segundo argumento es la **desviación típica** σ = 5 (varianza 25).' },
        { cell: 9, lines: [3, 3], text: '`density=True` divide los conteos por n·(ancho del bin): el histograma es comparable con una densidad de probabilidad.' },
      ],
      shapes: [{ expr: 'data_normal', shape: '(100000,)', meaning: 'vector 1D' }],
      interpretation: 'Campana simétrica centrada en 3; casi todos los valores en $3\\pm 15$ (±3σ).',
      next: 'Uniforme, Laplace y Rayleigh siguen el mismo patrón.',
      whatIf: [{ change: 'Quitar `density=True`', effect: 'El eje y mostraría conteos; la forma es la misma, pero no se puede comparar con una pdf.' }],
    },
    {
      id: 'b_uniform',
      title: 'Muestra uniforme U(−0.5, 0.5)',
      cells: [10, 11],
      stage: 'datos',
      problem: 'Generar una distribución simétrica pero **no** normal.',
      purpose: 'Contraejemplo clave: asimetría 0 como la normal, pero acotada y con colas inexistentes (curtosis −1.2).',
      inputs: '`rng`.',
      outputs: '`data_uniform` e histograma plano.',
      prereqs: ['histograma'],
      apis: [{ name: 'rng.uniform', lib: 'numpy.random.Generator', kind: 'método', does: 'Muestras uniformes en [low, high).', args: [{ name: 'low, high', value: '-0.5, 0.5', meaning: 'extremos' }] }],
      lines: [{ cell: 11, lines: [1, 1], text: 'Varianza teórica $(b-a)^2/12 = 1/12$, σ ≈ 0.289.' }],
      interpretation: 'Histograma rectangular: no hay valores fuera de [−0.5, 0.5]. Simetría **no** implica normalidad.',
      whatIf: [],
    },
    {
      id: 'b_laplace',
      title: 'Muestra de Laplace(0, 1)',
      cells: [12, 13],
      stage: 'datos',
      problem: 'Generar una distribución simétrica de **colas pesadas**.',
      purpose: 'Pico más agudo que la normal y más valores extremos: curtosis de exceso teórica 3.',
      inputs: '`rng`.',
      outputs: '`data_laplace` e histograma puntiagudo.',
      prereqs: ['curtosis'],
      apis: [{ name: 'rng.laplace', lib: 'numpy.random.Generator', kind: 'método', does: 'Distribución de Laplace (doble exponencial).', args: [{ name: 'loc, scale', value: '0, 1', meaning: 'centro y escala b (varianza 2b²)' }] }],
      lines: [{ cell: 13, lines: [1, 1], text: 'Varianza teórica $2b^2 = 2$, σ ≈ 1.414.' }],
      interpretation: 'Pico estrecho y colas que caen más despacio que la normal.',
      whatIf: [],
    },
    {
      id: 'b_rayleigh',
      title: 'Muestra de Rayleigh(5)',
      cells: [14, 15],
      stage: 'datos',
      problem: 'Generar una distribución **asimétrica** y positiva.',
      purpose: 'Contraejemplo de asimetría: solo valores ≥ 0 y cola larga a la derecha (asimetría teórica ≈ 0.631).',
      inputs: '`rng`.',
      outputs: '`data_rayleigh` e histograma sesgado.',
      prereqs: ['asimetria'],
      apis: [{ name: 'rng.rayleigh', lib: 'numpy.random.Generator', kind: 'método', does: 'Distribución de Rayleigh (módulo de un vector gaussiano 2D).', args: [{ name: 'scale', value: '5', meaning: 'parámetro σ (moda)' }] }],
      lines: [{ cell: 15, lines: [1, 1], text: 'Media teórica $\\sigma\\sqrt{\\pi/2}\\approx 6.27$; la moda está en 5.' }],
      interpretation: 'Moda a la izquierda y cola larga a la derecha: asimetría positiva.',
      next: 'Se comparan las cuatro con histplot+KDE.',
      whatIf: [],
    },
    {
      id: 'b_histplot',
      title: 'Histogramas con seaborn y KDE',
      cells: [16, 17],
      stage: 'exploracion',
      problem: 'Ver las cuatro formas juntas en una figura 2×2.',
      purpose: 'Comparación visual inmediata de forma, colas y simetría.',
      inputs: 'Las cuatro muestras.',
      outputs: 'Figura con 4 histogramas (conteos) y su curva KDE.',
      prereqs: ['histograma', 'kde'],
      apis: [
        { name: 'plt.subplots', lib: 'matplotlib', kind: 'función', does: 'Figura y matriz de ejes.', args: [{ name: 'ncols, nrows', value: '2, 2', meaning: 'rejilla 2×2' }], returns: '(fig, ax) con ax de forma (2, 2)' },
        { name: 'sns.histplot', lib: 'seaborn', kind: 'función', does: 'Histograma de conteos.', args: [{ name: 'data', meaning: 'vector a representar' }, { name: 'kde', value: 'True', meaning: 'superpone una estimación de densidad por kernel' }, { name: 'ax', value: 'ax[0][0]…', meaning: 'eje donde dibujar' }] },
      ],
      lines: [
        { cell: 17, lines: [1, 1], text: '`norm` se importa pero **no se usa**: es un resto del antiguo `distplot(fit=norm)`.' },
        { cell: 17, lines: [4, 7], text: 'Cada `ax=ax[i][j]` coloca el histograma en una posición de la rejilla 2×2.' },
      ],
      shapes: [{ expr: 'ax', shape: '(2, 2)', meaning: 'array de ejes; ax[0][1] = fila 0, columna 1' }],
      interpretation: 'La curva es una **KDE**, no la normal ajustada: sigue la forma de los datos (plana en la uniforme, picuda en Laplace). Para comparar con la normal hay que mirar la CDF o el Q-Q.',
      corrections: [{ cell: 17, kind: 'concepto', original: "fig.suptitle('Histograms - Fitting a normal distribution', fontsize=14)\nsns.histplot(data=data_normal, kde=True, ax=ax[0][0])", fixed: "x = np.linspace(data_normal.min(), data_normal.max(), 200)\nsns.histplot(data=data_normal, stat='density', ax=ax[0][0])\nax[0][0].plot(x, ss.norm.pdf(x, data_normal.mean(), data_normal.std()))", why: 'El título habla de ajustar una normal, pero `kde=True` dibuja una estimación no paramétrica. Para superponer la normal ajustada hay que dibujar `ss.norm.pdf` con la media y la desviación de la muestra (y el histograma en densidad).' }],
      whatIf: [{ change: 'Cambiar `kde=True` por `stat="density"` y superponer `ss.norm.pdf`', effect: 'Verías la normal ajustada: encaja en la normal, sobra en las colas de la uniforme y falta en el pico de Laplace.' }],
    },
    {
      id: 'b_cdf',
      title: 'CDF empírica frente a la normal ajustada',
      cells: [18, 19],
      stage: 'exploracion',
      problem: 'Comparar cuantitativamente la distribución acumulada de cada muestra con la de una normal de igual media y desviación.',
      purpose: 'Si la muestra es normal, la escalera roja (CDF empírica) se superpone a la curva azul. Las desviaciones sistemáticas revelan colas o asimetría.',
      inputs: 'Las cuatro muestras.',
      outputs: 'Figura 2×2: CDF normal (azul) y CDF empírica (rojo) por distribución.',
      prereqs: ['cdf', 'media', 'desviacion'],
      apis: [
        { name: 'np.linspace', lib: 'numpy', kind: 'función', does: 'npoints valores equiespaciados entre el mínimo y el máximo de la muestra.' },
        { name: 'ss.norm.cdf', lib: 'scipy.stats', kind: 'método', does: 'CDF normal.', args: [{ name: 'x', meaning: 'puntos de evaluación' }, { name: 'loc', value: 'np.mean(data)', meaning: 'media estimada' }, { name: 'scale', value: 'np.std(data)', meaning: 'desviación estimada (ddof=0)' }] },
        { name: 'plt.hist(..., cumulative=True, histtype="step")', lib: 'matplotlib', kind: 'función', does: 'Histograma acumulado dibujado como línea: con density=True es la CDF empírica.', args: [{ name: 'bins', value: 'npoints = 10000', meaning: 'muchos bins → escalera fina' }] },
        { name: 'plt.subplot(2,2,k)', lib: 'matplotlib', kind: 'función', does: 'Activa el subplot k (1..4) para las llamadas plt.* siguientes.' },
      ],
      lines: [
        { cell: 19, lines: [6, 7], text: 'La normal de comparación usa la media y la desviación **de la propia muestra**: compara formas, no posiciones.' },
        { cell: 19, lines: [9, 9], text: 'Con `density=True` y `cumulative=True` el último escalón vale 1: es la CDF empírica.' },
        { cell: 19, lines: [7, 7], text: 'El `;` final es innecesario en Python (herencia de MATLAB).' },
      ],
      shapes: [{ expr: 'x, ncdf', shape: '(10000,)', meaning: 'rejilla y CDF normal en ella' }],
      interpretation: 'Normal: curvas indistinguibles. Uniforme: la empírica es una recta y cruza la normal en forma de S. Laplace: la empírica sube más rápido en el centro (pico) y más despacio en los extremos. Rayleigh: desplazada por la asimetría, empieza en 0.',
      next: 'Se cuantifica con asimetría y curtosis.',
      whatIf: [{ change: 'Usar `np.var` como `scale`', effect: 'Se compararía con una normal de desviación σ² en vez de σ: las curvas no encajarían ni siquiera para la normal.' }],
    },
    {
      id: 'b_moments',
      title: 'Curtosis y asimetría de las cuatro muestras',
      cells: [20, 21],
      stage: 'exploracion',
      problem: 'Resumir la forma de cada distribución con dos números.',
      purpose: 'La asimetría mide simetría y la curtosis (de Fisher) el peso de las colas; para una normal ambas ≈ 0. Son necesarias pero **no suficientes** para la normalidad.',
      inputs: 'Las cuatro muestras.',
      outputs: '`dkurt` ≈ [0.020, −1.197, 2.832, 0.237] y `dskew` ≈ [0.014, 0.005, 0.031, 0.624] y dos diagramas de barras.',
      prereqs: ['asimetria', 'curtosis'],
      apis: [
        { name: 'np.vstack', lib: 'numpy', kind: 'función', does: 'Apila vectores como filas.', returns: 'array (4, 100000)' },
        { name: 'ss.kurtosis', lib: 'scipy.stats', kind: 'función', does: 'Curtosis.', args: [{ name: 'axis', value: '1', meaning: 'una curtosis por fila (por distribución)' }, { name: 'fisher', value: 'True (defecto)', meaning: 'resta 3: normal = 0' }, { name: 'bias', value: 'True (defecto)', meaning: 'estimador sesgado (momentos muestrales)' }], returns: 'array (4,)' },
        { name: 'ss.skew', lib: 'scipy.stats', kind: 'función', does: 'Asimetría.', args: [{ name: 'axis', value: '1', meaning: 'por fila' }], returns: 'array (4,)' },
        { name: 'plt.bar / plt.xticks', lib: 'matplotlib', kind: 'función', does: 'Barras y etiquetas del eje x.' },
      ],
      lines: [
        { cell: 21, lines: [1, 1], text: 'Cada fila de `all_data` es una distribución: forma (4, 100000).' },
        { cell: 21, lines: [2, 3], text: '`axis=1` calcula a lo largo de las columnas, es decir, **un valor por fila**. Con el `axis=0` por defecto obtendrías 100 000 valores sin sentido (uno por posición).' },
        { cell: 21, lines: [8, 10], text: 'Se crea una rejilla 1×2 pero luego se usa `plt.subplot(2,1,k)` (2×1): los ejes iniciales se sustituyen. Funciona, pero es inconsistente.' },
      ],
      shapes: [
        { expr: 'all_data', shape: '(4, 100000)', meaning: 'filas = distribuciones' },
        { expr: 'dkurt, dskew', shape: '(4,)', meaning: 'un valor por distribución' },
      ],
      interpretation: 'Normal ≈ (0, 0). Uniforme: asimetría 0 pero curtosis −1.2 (colas ligeras). Laplace: asimetría 0 y curtosis ≈ 3 (colas pesadas). Rayleigh: asimetría ≈ 0.62 (cola derecha) y curtosis ≈ 0.24. Uniforme y Laplace muestran que asimetría 0 no basta.',
      next: 'Q-Q plot: comparación cuantil a cuantil.',
      whatIf: [{ change: '`ss.kurtosis(..., fisher=False)`', effect: 'Curtosis de Pearson: la normal valdría 3 y todas se desplazarían +3.' }],
    },
    {
      id: 'b_qq',
      title: 'Gráficos cuantil-cuantil',
      cells: [22, 23],
      stage: 'exploracion',
      problem: 'Comparar cada cuantil de la muestra con el que tendría una normal.',
      purpose: 'Es la herramienta visual más sensible a colas: si los puntos siguen la recta, la normal es plausible; las curvas en S o en U señalan colas o asimetría.',
      inputs: 'Las cuatro muestras.',
      outputs: 'Figura 2×2 de Q-Q plots con la recta de ajuste y R².',
      prereqs: ['qqplot', 'cuantil'],
      apis: [{ name: 'ss.probplot', lib: 'scipy.stats', kind: 'función', does: 'Calcula los cuantiles teóricos (normal por defecto) frente a los ordenados de la muestra y ajusta una recta.', args: [{ name: 'plot', value: 'plt', meaning: 'objeto con método plot donde dibujar (plt o un eje)' }, { name: 'dist', value: "'norm' (defecto)", meaning: 'distribución teórica' }], returns: '((cuantiles teóricos, ordenados), (pendiente, ordenada, r))' }],
      lines: [{ cell: 23, lines: [5, 5], text: 'Dibuja en el subplot activo; la pendiente de la recta estima σ y la ordenada μ.' }],
      interpretation: 'Normal: puntos sobre la recta. Uniforme: S tumbada (extremos por dentro: colas más cortas). Laplace: S invertida (extremos por fuera: colas más largas). Rayleigh: curva convexa (asimetría derecha).',
      next: 'Ampliación: valores numéricos y diferencia de CDF.',
      whatIf: [{ change: 'Usar n = 50', effect: 'Los extremos se dispersan por azar incluso en la normal: hay que ser prudente con muestras pequeñas (como Iris, 50 por clase).' }],
    },
    {
      id: 'b_extra_moments',
      title: 'Ampliación: momentos con valores anotados',
      cells: [24, 25],
      stage: 'interpretacion',
      origin: 'ampliacion',
      problem: 'Tener los valores numéricos de asimetría y curtosis en la figura para citarlos en Q1.',
      purpose: 'Repite el cálculo de la celda 21 con un formato más legible (barras anotadas). **No es parte del notebook original**: se añadió para preparar el report.',
      inputs: 'Las cuatro muestras.',
      outputs: '`skewness`, `excess` (4,) y figura anotada.',
      prereqs: ['asimetria', 'curtosis'],
      apis: [{ name: 'ax.annotate', lib: 'matplotlib', kind: 'método', does: 'Escribe el valor encima de cada barra.' }, { name: 'ax.axhline', lib: 'matplotlib', kind: 'método', does: 'Línea horizontal de referencia en 0 (valor normal).' }],
      lines: [{ cell: 25, lines: [4, 5], text: 'Mismo cálculo que la celda 21 (`axis=1`): los valores coinciden.' }],
      interpretation: 'Permite leer directamente: Laplace ≈ 2.83 de exceso de curtosis, uniforme ≈ −1.20, Rayleigh ≈ 0.62 de asimetría.',
      whatIf: [],
    },
    {
      id: 'b_extra_ecdf',
      title: 'Ampliación: ECDF − CDF normal en escala estandarizada',
      cells: [26],
      stage: 'interpretacion',
      origin: 'ampliacion',
      problem: 'Ver la discrepancia con la normal de forma más precisa que superponiendo curvas.',
      purpose: 'Dibuja la **diferencia** entre la CDF empírica exacta y la normal en la variable estandarizada $z=(x-\\bar x)/s$. Es descriptivo, no un p-valor. **Añadido**, no forma parte del original.',
      inputs: 'Las cuatro muestras.',
      outputs: 'Figura 2×2 de diferencias en [−0.1, 0.1].',
      prereqs: ['cdf'],
      apis: [{ name: 'np.searchsorted', lib: 'numpy', kind: 'función', does: 'Posición de cada punto de la rejilla en la muestra ordenada: número de valores ≤ punto (con side="right").' }],
      lines: [
        { cell: 26, lines: [7, 8], text: 'ECDF exacta: proporción de valores ≤ cada punto de la rejilla en unidades de la muestra.' },
        { cell: 26, lines: [9, 9], text: 'Diferencia con Φ(z). Su máximo absoluto es el estadístico de Kolmogorov-Smirnov (con parámetros estimados).' },
      ],
      interpretation: 'Normal: diferencia ≈ 0 (máx ≈ 0.002). Uniforme y Laplace: ondas simétricas (máx ≈ 0.058 y 0.062). Rayleigh: onda asimétrica (máx ≈ 0.047).',
      whatIf: [],
    },
  ],
  experiments: [
    {
      id: 'lab0_1.exp.dist',
      title: 'Una distribución, cuatro herramientas',
      goal: 'Elige una de las cuatro muestras del notebook y mira a la vez histograma+KDE, CDF y Q-Q, con sus momentos. Cambia el tamaño de muestra para ver cuándo las herramientas dejan de ser fiables, y los bins para ver qué es solo visual.',
      blocks: ['b_rng', 'b_normal', 'b_histplot', 'b_cdf', 'b_moments', 'b_qq'],
      template: EXPERIMENT,
      seedParam: 'seed',
      params: [
        { id: 'dist', label: 'Distribución', kind: 'datos', control: 'select', default: 'data_normal', options: [{ value: 'data_normal', label: 'Normal' }, { value: 'data_uniform', label: 'Uniforme' }, { value: 'data_laplace', label: 'Laplace' }, { value: 'data_rayleigh', label: 'Rayleigh' }], variable: 'x = …', computation: 'Selecciona qué muestra se analiza (las cuatro se generan siempre, en el orden del notebook).', result: 'Cambian forma, CDF, Q-Q y momentos según la distribución.', why: 'Cada distribución tiene colas y simetría distintas.' },
        { id: 'n', label: 'Tamaño n', kind: 'datos', control: 'select', default: '100000', options: [{ value: '30', label: '30' }, { value: '50', label: '50 (como una clase de Iris)' }, { value: '500', label: '500' }, { value: '5000', label: '5000' }, { value: '100000', label: '100 000 (original)' }], variable: 'n', computation: 'Número de valores generados para cada distribución.', result: 'Con n pequeño, los momentos oscilan mucho y una normal puede parecer asimétrica o picuda.', why: 'La varianza de los estimadores de asimetría y curtosis decrece como 1/n.' },
        { id: 'seed', label: 'Semilla', kind: 'datos', control: 'range', min: 0, max: 50, step: 1, default: '1', variable: 'default_rng(seed=…)', computation: 'Otra secuencia aleatoria.', result: 'Con n grande apenas cambia; con n pequeño, mucho.', why: 'Variabilidad muestral.' },
        { id: 'nmu', label: 'Media de la normal', kind: 'datos', control: 'select', default: '3', options: [{ value: '0', label: '0' }, { value: '3', label: '3 (original)' }, { value: '10', label: '10' }], variable: 'nmu', computation: 'Desplaza data_normal (y solo ella).', result: 'Cambia la media; asimetría, curtosis, forma de la CDF y Q-Q no cambian.', why: 'Los momentos estandarizados no dependen de la posición.' },
        { id: 'nsigma', label: 'Desviación de la normal', kind: 'datos', control: 'select', default: '5', options: [{ value: '1', label: '1' }, { value: '5', label: '5 (original)' }, { value: '20', label: '20' }], variable: 'nsigma', computation: 'Escala data_normal.', result: 'Cambia la desviación típica; las medidas de forma no.', why: 'La asimetría y la curtosis están normalizadas por σ.' },
        { id: 'nbins', label: 'Bins del histograma', kind: 'visualizacion', control: 'select', default: '50', options: [{ value: '5', label: '5' }, { value: '20', label: '20' }, { value: '50', label: '50 (original)' }, { value: '200', label: '200' }], variable: 'nbins', computation: 'Solo el dibujo del histograma.', result: 'Con pocos bins se pierde la forma; con demasiados (y n pequeño) se ve ruido. Las métricas no cambian.', why: 'El número de bins no interviene en ningún cálculo estadístico.' },
        { id: 'npoints', label: 'Puntos de la CDF', kind: 'visualizacion', control: 'select', default: '10000', options: [{ value: '100', label: '100' }, { value: '1000', label: '1000' }, { value: '10000', label: '10 000 (original)' }], variable: 'npoints', computation: 'Resolución de la rejilla y bins de la CDF acumulada.', result: 'Escalera más o menos fina; métricas iguales.', why: 'Solo cambia la discretización del dibujo.' },
      ],
      metrics: [
        { key: 'media', label: 'Media', format: 'num', digits: 4 },
        { key: 'std', label: 'Desviación típica', format: 'num', digits: 4 },
        { key: 'skew', label: 'Asimetría', format: 'num', digits: 4 },
        { key: 'skew_teorica', label: 'Asimetría teórica', format: 'num', digits: 3 },
        { key: 'kurt', label: 'Curtosis (exceso)', format: 'num', digits: 4 },
        { key: 'kurt_teorica', label: 'Curtosis teórica', format: 'num', digits: 3 },
        { key: 'max_dif_cdf', label: 'Máx |ECDF − Φ| (app)', format: 'num', digits: 4, better: 'low' },
      ],
      observe: [
        'Pasa por las cuatro distribuciones con n = 100 000: ¿qué herramienta distingue mejor la uniforme de la normal?',
        'Pon la normal con n = 50 y pulsa varias veces «Nueva muestra»: ¿cuánto oscila la curtosis? ¿Qué implica para Iris (50 flores por clase)?',
        'Cambia los bins: ¿cambia alguna métrica? ¿Por qué el número de bins es un parámetro de visualización?',
        'Cambia la media y la desviación de la normal: ¿qué métricas se mueven y cuáles no?',
      ],
      source: src([7, 9, 11, 13, 15, 17, 19, 21, 23, 26], { note: 'La selección de distribución, las referencias teóricas y la diferencia máxima de CDF son añadidos de la app.' }),
      fidelity: [
        { metrics: { kurt: 0.01950599007746323, skew: 0.014189860673160708, media: 2.977047139785621, std: 4.982665940924694, max_dif_cdf: 0.0016634386118963285 }, tol: 1e-6, source: 'Celda 21 recalculada con tu entorno local (seed 1, n = 100 000)' },
        { params: { dist: 'data_uniform' }, metrics: { kurt: -1.197480999277135, skew: 0.00527575785004402 }, tol: 1e-6, source: 'Celda 21 (uniforme)' },
        { params: { dist: 'data_laplace' }, metrics: { kurt: 2.832184327661956, skew: 0.03060398427754898, max_dif_cdf: 0.06191111304323438 }, tol: 1e-6, source: 'Celdas 21 y 26 (Laplace)' },
        { params: { dist: 'data_rayleigh' }, metrics: { kurt: 0.2370316573665998, skew: 0.6238096598566125 }, tol: 1e-6, source: 'Celda 21 (Rayleigh)' },
      ],
    },
  ],
  activities: [
    // ---------------------------------------------------------- nivel 1
    {
      kind: 'mcq',
      id: 'lab0_1.read.vstack',
      skills: ['codigo', 'api'],
      blocks: ['b_moments'],
      source: src([21]),
      prompt: '¿Qué devuelve `ss.kurtosis(all_data, axis=1)` en este notebook?',
      code: 'all_data = np.vstack((data_normal,data_uniform, data_laplace, data_rayleigh))\ndkurt = ss.kurtosis(all_data,axis=1)',
      options: [
        { text: 'Un vector de 4 valores: el exceso de curtosis de cada distribución', correct: true },
        { text: 'Un único número: la curtosis de todos los datos juntos', feedback: 'Eso sería `axis=None`.' },
        { text: '100 000 valores, uno por posición', feedback: 'Eso ocurriría con `axis=0` (el valor por defecto).' },
        { text: 'La curtosis de Pearson (normal = 3)', feedback: 'Por defecto `fisher=True`: se resta 3.' },
      ],
      explanation: '`vstack` pone cada muestra en una fila (4, 100000); `axis=1` recorre las columnas y deja una curtosis por fila.',
    },
    {
      kind: 'mcq',
      id: 'lab0_1.read.density',
      skills: ['api'],
      blocks: ['b_normal', 'b_cdf'],
      source: src([9, 19]),
      prompt: 'En `plt.hist(data_normal, npoints, density=True, histtype="step", cumulative=True, color="r")`, ¿qué consigue la combinación `density=True` y `cumulative=True`?',
      options: [
        { text: 'Dibujar la CDF empírica (sube de 0 a 1)', correct: true },
        { text: 'Dibujar la densidad normal ajustada', feedback: 'La normal se dibuja aparte con `ss.norm.cdf`.' },
        { text: 'Contar cuántos valores hay en cada bin', feedback: 'Eso es el histograma normal sin `cumulative`.' },
      ],
      explanation: 'Normalizar (área 1) y acumular da la proporción de datos ≤ x, es decir, la CDF empírica; `histtype="step"` la dibuja como línea.',
    },
    {
      kind: 'text',
      id: 'lab0_1.read.cdf',
      skills: ['codigo', 'concepto'],
      blocks: ['b_cdf'],
      source: src([19]),
      prompt: 'Explica qué compara cada subplot de la celda 19 y por qué la normal se construye con `np.mean(data)` y `np.std(data)`.',
      code: 'x = np.linspace(np.min(data_laplace),np.max(data_laplace),npoints)\nncdf = ss.norm.cdf(x,np.mean(data_laplace),np.std(data_laplace))\nplt.plot(x,ncdf)\nplt.hist(data_laplace, npoints, density=True,histtype=\'step\',cumulative=True,color=\'r\')',
      rubric: [
        'Compara la CDF empírica de la muestra (rojo) con la CDF de una normal (azul).',
        'La normal tiene la misma media y desviación que la muestra, para comparar la forma y no la posición o la escala.',
        'Si la muestra es normal, las curvas se superponen; las discrepancias indican colas o asimetría.',
        'x es una rejilla entre el mínimo y el máximo de la muestra.',
      ],
      model: 'Cada subplot superpone la CDF empírica de una muestra (histograma acumulado normalizado, en rojo) y la CDF de una normal $\\mathcal N(\\bar x, s^2)$ evaluada en una rejilla entre el mínimo y el máximo (azul). Se usa la media y la desviación de la propia muestra para que ambas distribuciones tengan el mismo centro y la misma escala: así cualquier diferencia se debe a la **forma** (colas, asimetría). Si los datos son normales, las curvas coinciden.',
    },
    // ---------------------------------------------------------- nivel 2
    {
      kind: 'predict',
      id: 'lab0_1.pred.alldata',
      skills: ['codigo'],
      blocks: ['b_moments'],
      source: src([21]),
      prompt: '¿Qué forma tiene `all_data`?',
      setup: SETUP,
      code: 'all_data = np.vstack((data_normal,data_uniform, data_laplace, data_rayleigh))',
      expr: 'all_data.shape',
      answerKind: 'shape',
      explanation: '`vstack` apila los 4 vectores de 100 000 valores como filas: (4, 100000).',
    },
    {
      kind: 'predict',
      id: 'lab0_1.pred.axis0',
      skills: ['codigo', 'api'],
      blocks: ['b_moments'],
      source: src([21]),
      prompt: 'Si se olvida `axis=1`, ¿qué forma tiene el resultado de `ss.kurtosis(all_data)`?',
      setup: SETUP + 'all_data = np.vstack((data_normal, data_uniform, data_laplace, data_rayleigh))\n',
      code: 'dkurt_mal = ss.kurtosis(all_data)',
      expr: 'dkurt_mal.shape',
      answerKind: 'shape',
      explanation: 'El valor por defecto es `axis=0`: calcula la curtosis de cada **columna** (4 valores de distribuciones distintas mezcladas), 100 000 resultados sin sentido.',
    },
    {
      kind: 'predict',
      id: 'lab0_1.pred.kurt_uniform',
      skills: ['concepto'],
      blocks: ['b_moments', 'b_uniform'],
      source: src([21]),
      prompt: 'Sin ejecutar: ¿qué exceso de curtosis esperas para `data_uniform` (1 decimal)?',
      setup: SETUP,
      code: 'k = ss.kurtosis(data_uniform)',
      expr: 'float(k)',
      answerKind: 'numeric',
      tolerance: 0.06,
      explanation: 'La uniforme tiene curtosis de exceso teórica −1.2: colas más ligeras que la normal (de hecho, no tiene colas).',
    },
    // ---------------------------------------------------------- nivel 3
    {
      kind: 'code',
      level: 3,
      id: 'lab0_1.fill.moments',
      title: 'Completa el cálculo de momentos',
      skills: ['codigo', 'api'],
      blocks: ['b_moments'],
      source: src([21]),
      practice: 'scipy.stats.kurtosis / skew',
      prompt: 'Rellena los huecos para obtener la curtosis y la asimetría de cada una de las cuatro distribuciones.',
      available: '`np`, `ss` (scipy.stats) y las cuatro muestras `data_normal`, `data_uniform`, `data_laplace`, `data_rayleigh` (100000,) con semilla 1.',
      produce: '`all_data` (4, 100000), `dkurt` y `dskew` (4,).',
      setup: SETUP,
      starter: 'all_data = np.___((data_normal, data_uniform, data_laplace, data_rayleigh))\ndkurt = ss.___(all_data, axis=___)\ndskew = ss.___(all_data, axis=___)',
      solution: 'all_data = np.vstack((data_normal, data_uniform, data_laplace, data_rayleigh))\ndkurt = ss.kurtosis(all_data, axis=1)\ndskew = ss.skew(all_data, axis=1)',
      alternatives: ['samples = [data_normal, data_uniform, data_laplace, data_rayleigh]\nall_data = np.array(samples)\ndkurt = np.array([ss.kurtosis(s) for s in samples])\ndskew = np.array([ss.skew(s) for s in samples])'],
      wrong: [{ code: 'all_data = np.vstack((data_normal, data_uniform, data_laplace, data_rayleigh))\ndkurt = ss.kurtosis(all_data, axis=0)\ndskew = ss.skew(all_data, axis=0)', expect: 'dimensiones' }],
      checks: "c.shape('all_data', (4, 100000))\nc.same('dkurt', rtol=1e-9, desc='La curtosis por distribución')\nc.same('dskew', rtol=1e-9, desc='La asimetría por distribución')",
      hints: ['La función que apila vectores como filas es `np.vstack`.', 'Las funciones están en scipy.stats: `kurtosis` y `skew`.', 'Quieres un valor por fila: `axis=1`.'],
    },
    {
      kind: 'code',
      level: 3,
      id: 'lab0_1.fill.cdf',
      title: 'Completa la comparación de CDF para Rayleigh',
      skills: ['api', 'codigo'],
      blocks: ['b_cdf'],
      source: src([19]),
      practice: 'scipy.stats.norm.cdf',
      prompt: 'Completa el código para evaluar la CDF normal ajustada a `data_rayleigh` en una rejilla de 10 000 puntos y dibujarla junto a la CDF empírica.',
      available: '`np`, `ss`, `plt` y las cuatro muestras.',
      produce: '`x` (10000,), `ncdf` (10000,) y una figura.',
      setup: SETUP_PLOT,
      starter: 'npoints = 10000\nx = np.linspace(np.___(data_rayleigh), np.___(data_rayleigh), npoints)\nncdf = ss.norm.___(x, ___, ___)\nplt.plot(x, ncdf)\nplt.hist(data_rayleigh, npoints, density=True, histtype="step", cumulative=___, color="r")\nplt.show()',
      solution: 'npoints = 10000\nx = np.linspace(np.min(data_rayleigh), np.max(data_rayleigh), npoints)\nncdf = ss.norm.cdf(x, np.mean(data_rayleigh), np.std(data_rayleigh))\nplt.plot(x, ncdf)\nplt.hist(data_rayleigh, npoints, density=True, histtype="step", cumulative=True, color="r")\nplt.show()',
      alternatives: ['npoints = 10000\nx = np.linspace(data_rayleigh.min(), data_rayleigh.max(), npoints)\nncdf = ss.norm(loc=data_rayleigh.mean(), scale=data_rayleigh.std()).cdf(x)\nplt.plot(x, ncdf)\nplt.hist(data_rayleigh, npoints, density=True, histtype="step", cumulative=True)\nplt.show()'],
      wrong: [{ code: 'npoints = 10000\nx = np.linspace(np.min(data_rayleigh), np.max(data_rayleigh), npoints)\nncdf = ss.norm.cdf(x, np.mean(data_rayleigh), np.var(data_rayleigh))\nplt.plot(x, ncdf)\nplt.show()', expect: 'razonamiento' }],
      checks: "c.same('x', rtol=1e-9)\nc.same('ncdf', rtol=1e-7, desc='La CDF normal ajustada')\nc.figures(1)",
      hints: ['La rejilla va del mínimo al máximo de la muestra.', 'El método de la distribución normal para la acumulada es `cdf`.', '`loc` es la media y `scale` la **desviación típica** (no la varianza).'],
    },
    // ---------------------------------------------------------- nivel 4
    {
      kind: 'code',
      level: 4,
      id: 'lab0_1.fix.pearson',
      title: 'Corrige: curtosis de Pearson frente a exceso',
      skills: ['concepto', 'api'],
      blocks: ['b_moments'],
      source: src([20, 21]),
      prompt: 'Este código pretende comprobar si `data_laplace` tiene colas más pesadas que la normal comparando su curtosis con **0**, pero obtiene un valor cercano a 6 y concluye mal. Corrígelo para que `kurt` sea el **exceso** de curtosis y `pesada` sea `True` si las colas son más pesadas que las de la normal.',
      available: '`ss`, `data_laplace`.',
      produce: '`kurt` (float) y `pesada` (bool).',
      setup: SETUP,
      starter: 'kurt = ss.kurtosis(data_laplace, fisher=False)\npesada = kurt > 6',
      solution: 'kurt = ss.kurtosis(data_laplace)\npesada = kurt > 0',
      alternatives: ['kurt = ss.kurtosis(data_laplace, fisher=False) - 3\npesada = bool(kurt > 0)'],
      wrong: [{ code: 'kurt = ss.kurtosis(data_laplace, fisher=False)\npesada = kurt > 0', expect: 'razonamiento' }],
      checks: "c.same('kurt', rtol=1e-9, desc='El exceso de curtosis')\nc.same('pesada')",
      hints: ['Con `fisher=False` la normal vale 3 (Pearson).', 'El exceso de curtosis es la curtosis de Pearson menos 3; scipy lo da por defecto.'],
    },
    {
      kind: 'code',
      level: 4,
      id: 'lab0_1.fix.axis',
      title: 'Corrige el eje del cálculo',
      skills: ['codigo'],
      blocks: ['b_moments'],
      source: src([21]),
      prompt: 'El diagrama de barras falla porque `dskew` no tiene 4 valores. Corrige el código sin cambiar `all_data`.',
      available: '`np`, `ss`, `plt`, las cuatro muestras.',
      produce: '`dskew` (4,) y la figura de barras.',
      setup: SETUP_PLOT,
      starter: "all_data = np.vstack((data_normal, data_uniform, data_laplace, data_rayleigh))\ndskew = ss.skew(all_data)\npos = np.arange(4)\nplt.bar(pos, dskew)\nplt.xticks(pos, ('Normal', 'Uniform', 'Laplace', 'Rayleigh'))\nplt.show()",
      solution: "all_data = np.vstack((data_normal, data_uniform, data_laplace, data_rayleigh))\ndskew = ss.skew(all_data, axis=1)\npos = np.arange(4)\nplt.bar(pos, dskew)\nplt.xticks(pos, ('Normal', 'Uniform', 'Laplace', 'Rayleigh'))\nplt.show()",
      checks: "c.same('dskew', rtol=1e-9)\nc.figures(1)",
      hints: ['¿Qué forma tiene `ss.skew(all_data)`? Ejecútalo y mira `.shape`.', 'Cada distribución está en una fila: necesitas un valor por fila.'],
    },
    // ---------------------------------------------------------- nivel 5
    {
      kind: 'code',
      level: 5,
      id: 'lab0_1.mem.samples',
      title: 'Genera las cuatro muestras y sus momentos',
      skills: ['escritura', 'api'],
      blocks: ['b_rng', 'b_normal', 'b_uniform', 'b_laplace', 'b_rayleigh', 'b_moments'],
      source: src([7, 9, 11, 13, 15, 21]),
      prompt: 'Sin mirar: reproduce las celdas 7–21. Genera, con `default_rng(seed=1)` y n = 100 000, en este orden: normal (media 3, desviación 5), uniforme en [−0.5, 0.5], Laplace(0, 1) y Rayleigh(5); calcula el exceso de curtosis y la asimetría de cada una.',
      available: 'Nada importado.',
      produce: '`data_normal`, `data_uniform`, `data_laplace`, `data_rayleigh` (100000,), `dkurt`, `dskew` (4,) en ese orden.',
      setup: '',
      starter: '',
      solution: SETUP + 'all_data = np.vstack((data_normal, data_uniform, data_laplace, data_rayleigh))\ndkurt = ss.kurtosis(all_data, axis=1)\ndskew = ss.skew(all_data, axis=1)',
      alternatives: ['import numpy as np\nfrom scipy import stats\nrng = np.random.default_rng(1)\nn = 100_000\ndata_normal = rng.normal(loc=3, scale=5, size=n)\ndata_uniform = rng.uniform(low=-0.5, high=0.5, size=n)\ndata_laplace = rng.laplace(0, 1, size=n)\ndata_rayleigh = rng.rayleigh(scale=5, size=n)\nS = [data_normal, data_uniform, data_laplace, data_rayleigh]\ndkurt = np.array([stats.kurtosis(s) for s in S])\ndskew = np.array([stats.skew(s) for s in S])'],
      checks: "for v in ['data_normal', 'data_uniform', 'data_laplace', 'data_rayleigh']:\n    c.same(v, rtol=1e-9)\nc.same('dkurt', rtol=1e-9)\nc.same('dskew', rtol=1e-9)",
      hints: ['`from numpy.random import default_rng`; los métodos se llaman como las distribuciones.', 'El orden de generación importa: todas salen del mismo rng.', 'Momentos: `scipy.stats.kurtosis` y `skew` con `axis=1` sobre la pila.'],
    },
    {
      kind: 'code',
      level: 5,
      id: 'lab0_1.mem.ks',
      title: 'Distancia máxima entre la ECDF y la normal',
      skills: ['escritura', 'concepto'],
      blocks: ['b_extra_ecdf', 'b_cdf'],
      source: src([26], { origin: 'ampliacion' }),
      prompt: 'Escribe una función `max_dif(x)` que estandarice la muestra con su media y su desviación (ddof=0), evalúe la ECDF exacta en la rejilla `z = np.linspace(-4, 4, 3001)` y devuelva el máximo de |ECDF − Φ(z)|. Aplícala a `data_laplace`.',
      available: '`np`, `ss`, las cuatro muestras.',
      produce: '`max_dif` (función) y `d_laplace` (float).',
      setup: SETUP,
      starter: '',
      solution: "def max_dif(x):\n    z = np.linspace(-4, 4, 3001)\n    grid = x.mean() + x.std() * z\n    empirical = np.searchsorted(np.sort(x), grid, side='right') / len(x)\n    return np.max(np.abs(empirical - ss.norm.cdf(z)))\nd_laplace = max_dif(data_laplace)",
      alternatives: ["def max_dif(x):\n    z = np.linspace(-4, 4, 3001)\n    xs = (x - x.mean()) / x.std()\n    ecdf = np.array([(xs <= v).mean() for v in z])\n    return np.abs(ecdf - ss.norm.cdf(z)).max()\nd_laplace = max_dif(data_laplace)"],
      checks: "c.same('d_laplace', rtol=1e-6)\nc.true(callable(ns.get('max_dif')), 'La función existe', 'Define la función max_dif', 'ejecucion')",
      hints: ['ECDF en v = proporción de valores ≤ v.', 'Puedes estandarizar los datos o la rejilla: $x = \\bar x + s z$.', '`ss.norm.cdf(z)` es Φ(z) de la normal estándar.'],
    },
    // ---------------------------------------------------------- nivel 6
    {
      kind: 'code',
      level: 6,
      id: 'lab0_1.adapt.expo',
      title: 'Adapta: ¿es gaussiana una exponencial?',
      skills: ['metodo', 'escritura'],
      blocks: ['b_moments', 'b_qq'],
      source: src([21, 23]),
      transfer: true,
      prompt: 'Genera n = 100 000 valores de una exponencial de escala 2 con `default_rng(seed=7)`, calcula su asimetría y su exceso de curtosis, dibuja su Q-Q plot frente a la normal y decide si se puede considerar gaussiana (`gaussiana = False/True`).',
      available: 'Nada importado.',
      produce: '`data_exp` (100000,), `sk`, `ku` (float), `gaussiana` (bool) y una figura Q-Q.',
      setup: '',
      starter: '',
      solution: 'import numpy as np\nimport scipy.stats as ss\nimport matplotlib.pyplot as plt\nrng = np.random.default_rng(seed=7)\ndata_exp = rng.exponential(2, 100000)\nsk = ss.skew(data_exp)\nku = ss.kurtosis(data_exp)\nss.probplot(data_exp, plot=plt)\nplt.show()\ngaussiana = False',
      checks: "c.same('data_exp', rtol=1e-9)\nc.same('sk', rtol=1e-9)\nc.same('ku', rtol=1e-9)\nc.same('gaussiana')\nc.figures(1)",
      hints: ['`rng.exponential(scale, size)`.', 'La exponencial tiene asimetría teórica 2 y curtosis de exceso 6.'],
      explanation: 'Asimetría ≈ 2 y curtosis ≈ 6: muy lejos de (0, 0), y el Q-Q muestra una curva convexa marcada. No es gaussiana.',
    },
    {
      kind: 'code',
      level: 6,
      id: 'lab0_1.adapt.wine',
      title: 'Adapta: momentos de un dataset real (wine)',
      skills: ['metodo', 'api'],
      blocks: ['b_moments'],
      source: { origin: 'app', note: 'Transferencia con sklearn.datasets.load_wine (incluido en scikit-learn)' },
      transfer: true,
      prompt: 'Carga el dataset *wine* de scikit-learn como DataFrame, calcula para **todas** las features la asimetría y el exceso de curtosis con scipy (estimador sesgado por defecto) y guarda en `mas_asimetrica` el nombre de la feature con mayor |asimetría|.',
      available: 'Nada importado. `from sklearn.datasets import load_wine` funciona sin conexión.',
      produce: '`asim` y `curt` (arrays de 13 valores en el orden de las columnas) y `mas_asimetrica` (str).',
      setup: '',
      starter: '',
      solution: 'import numpy as np\nimport scipy.stats as ss\nfrom sklearn.datasets import load_wine\nwine = load_wine(as_frame=True)\nX = wine.data\nasim = ss.skew(X, axis=0)\ncurt = ss.kurtosis(X, axis=0)\nmas_asimetrica = X.columns[np.argmax(np.abs(asim))]',
      alternatives: ['import scipy.stats as ss\nfrom sklearn.datasets import load_wine\nX = load_wine(as_frame=True).data\nasim = X.apply(lambda c: ss.skew(c)).to_numpy()\ncurt = X.apply(lambda c: ss.kurtosis(c)).to_numpy()\nmas_asimetrica = X.columns[abs(asim).argmax()]'],
      wrong: [{ code: 'import scipy.stats as ss\nfrom sklearn.datasets import load_wine\nX = load_wine(as_frame=True).data\nasim = X.skew().to_numpy()\ncurt = X.kurt().to_numpy()\nmas_asimetrica = X.columns[abs(asim).argmax()]', expect: 'razonamiento' }],
      checks: "c.same('asim', rtol=1e-8, desc='Las asimetrías (scipy, sesgado)')\nc.same('curt', rtol=1e-8, desc='Las curtosis (scipy, sesgado)')\nc.same('mas_asimetrica')",
      hints: ['`load_wine(as_frame=True).data` es un DataFrame de 178×13.', 'Aquí las features están en columnas: `axis=0`.', 'pandas (`.skew()`, `.kurt()`) usa el estimador **insesgado**: da valores ligeramente distintos.'],
    },
    // ---------------------------------------------------------- interpretación
    {
      kind: 'mcq',
      id: 'lab0_1.int.qq',
      skills: ['interpretacion'],
      blocks: ['b_qq'],
      figure: { notebook: NB, cell: 23, output: 0 },
      source: src([23]),
      prompt: 'En los Q-Q plots guardados, ¿qué patrón corresponde a la distribución **uniforme**?',
      options: [
        { text: 'Una S: los extremos se doblan hacia dentro (colas más cortas que la normal)', correct: true },
        { text: 'Puntos sobre la recta en todo el rango', feedback: 'Ese es el caso de la normal.' },
        { text: 'Extremos que se alejan hacia fuera de la recta', feedback: 'Es el patrón de colas pesadas (Laplace).' },
        { text: 'Una curva convexa en un solo sentido', feedback: 'Es la firma de la asimetría (Rayleigh).' },
      ],
      explanation: 'La uniforme no tiene valores lejanos: sus cuantiles extremos son menores (en valor absoluto) que los de la normal, de ahí la S.',
    },
    {
      kind: 'mcq',
      id: 'lab0_1.int.bars',
      skills: ['interpretacion'],
      blocks: ['b_moments'],
      figure: { notebook: NB, cell: 21, output: 0 },
      source: src([21]),
      prompt: 'Según las barras de curtosis y asimetría, ¿qué distribución tiene las colas más pesadas y cuál es asimétrica?',
      options: [
        { text: 'Colas pesadas: Laplace; asimétrica: Rayleigh', correct: true },
        { text: 'Colas pesadas: uniforme; asimétrica: normal', feedback: 'La uniforme tiene curtosis negativa (colas ligeras).' },
        { text: 'Colas pesadas: Rayleigh; asimétrica: Laplace', feedback: 'Laplace es simétrica.' },
      ],
      explanation: 'Curtosis de Laplace ≈ 2.8 (teórica 3); asimetría de Rayleigh ≈ 0.62 (teórica 0.63).',
    },
    {
      kind: 'mcq',
      id: 'lab0_1.int.misconception',
      skills: ['interpretacion', 'concepto'],
      blocks: ['b_moments', 'b_uniform'],
      source: src([21]),
      prompt: '«La asimetría de `data_uniform` es 0.005, prácticamente 0, así que es una muestra gaussiana». ¿Qué error conceptual contiene?',
      options: [
        { text: 'Asimetría nula es necesaria pero no suficiente: también hay que mirar curtosis, CDF y Q-Q', correct: true },
        { text: 'Ninguno: asimetría 0 implica normalidad', feedback: 'La uniforme y la Laplace son contraejemplos.' },
        { text: 'El error es que 0.005 no es suficientemente pequeño', feedback: 'El valor sí es pequeño; el problema es la conclusión.' },
      ],
      explanation: 'Toda distribución simétrica tiene asimetría 0. La uniforme tiene curtosis −1.2 y un Q-Q en S: no es normal (slides Lab0, p. 17).',
    },
    {
      kind: 'mcq',
      id: 'lab0_1.int.kde',
      skills: ['interpretacion', 'api'],
      blocks: ['b_histplot'],
      figure: { notebook: NB, cell: 17, output: 0 },
      source: src([17]),
      prompt: 'El título de la figura dice «Fitting a normal distribution». ¿Qué representa realmente la curva sobre cada histograma?',
      options: [
        { text: 'Una estimación de densidad por kernel (KDE), no una normal', correct: true },
        { text: 'La normal con la media y desviación de la muestra', feedback: 'Si lo fuera, sería una campana también sobre la uniforme.' },
        { text: 'La CDF de la muestra', feedback: 'La CDF es creciente; esta curva tiene forma de densidad.' },
      ],
      explanation: '`kde=True` en `sns.histplot` dibuja una KDE: sigue la forma de los datos (plana en la uniforme). Es una errata conceptual del título heredada de `distplot(fit=norm)`.',
    },
    {
      kind: 'text',
      id: 'lab0_1.int.q1',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_histplot', 'b_cdf', 'b_moments', 'b_qq'],
      source: { notebook: NB, cells: [17, 19, 21, 23], doc: 'lab0_guide', section: 'Q1', origin: 'original' },
      prompt: 'Q1 · Resume las conclusiones del análisis de las cuatro distribuciones, citando al menos una herramienta para cada una.',
      rubric: [
        'Normal: histograma en campana, CDF superpuesta, Q-Q recto y momentos ≈ 0.',
        'Uniforme: simétrica pero acotada; curtosis ≈ −1.2 y Q-Q en S (colas ligeras).',
        'Laplace: simétrica con pico agudo y colas pesadas; curtosis ≈ 3.',
        'Rayleigh: asimetría positiva ≈ 0.6, solo valores positivos, Q-Q curvado.',
        'Conclusión: la simetría no basta; hay que combinar varias herramientas.',
      ],
      model: 'La muestra **normal** es simétrica, su histograma tiene forma de campana, su CDF empírica coincide con la normal ajustada, el Q-Q sigue la recta y asimetría y curtosis son ≈ 0. La **uniforme** también es simétrica (asimetría ≈ 0), pero está acotada: su curtosis es ≈ −1.2 y el Q-Q forma una S porque no tiene colas. La **Laplace** es simétrica con un pico más agudo y colas pesadas: curtosis ≈ 2.8 y Q-Q con extremos por fuera de la recta. La **Rayleigh** es asimétrica a la derecha (≈ 0.62), solo toma valores positivos y su Q-Q es convexo. Conclusión: ninguna medida aislada demuestra normalidad; la simetría es necesaria pero no suficiente, y el Q-Q es la herramienta visual más sensible a las colas.',
    },
    {
      kind: 'numeric',
      id: 'lab0_1.int.sem',
      skills: ['concepto'],
      blocks: ['b_rng'],
      source: { origin: 'app', note: 'Resultado estadístico estándar: error típico de la asimetría ≈ √(6/n)' },
      prompt: 'Para una muestra normal, el error típico de la asimetría es aproximadamente $\\sqrt{6/n}$. ¿Cuánto vale con n = 50 (tamaño de una clase de Iris)? (2 decimales)',
      answer: 0.35,
      tolerance: 0.01,
      explanation: '$\\sqrt{6/50}\\approx 0.346$. Con 50 datos, asimetrías de ±0.3 son compatibles con una normal: con Iris hay que ser prudente.',
    },
    {
      kind: 'dynamic',
      id: 'lab0_1.dyn.which',
      skills: ['interpretacion', 'concepto'],
      blocks: ['b_moments'],
      source: src([21], { note: 'Variante generada' }),
      answerKind: 'mcq',
      generator: `import numpy as np
import scipy.stats as ss
rng = np.random.default_rng(SEED)
opciones = ['Normal', 'Uniforme', 'Laplace', 'Rayleigh']
k = int(rng.integers(4))
n = int(rng.choice([2000, 5000, 20000]))
x = [rng.normal(0, 1, n), rng.uniform(-1, 1, n), rng.laplace(0, 1, n), rng.rayleigh(2, n)][k]
variante = {
  'prompt': f'Una muestra de n = {n} valores tiene asimetría {ss.skew(x):.3f} y exceso de curtosis {ss.kurtosis(x):.3f}. ¿De cuál de las cuatro distribuciones del notebook procede con más probabilidad?',
  'options': opciones, 'answer': opciones[k],
  'explanation': 'Referencias teóricas (asimetría, curtosis): normal (0, 0), uniforme (0, −1.2), Laplace (0, 3), Rayleigh (0.63, 0.25).'
}`,
    },
    // ---------------------------------------------------------- cobertura: imports y ampliación de momentos
    {
      kind: 'mcq',
      id: 'lab0_1.cov.imports',
      skills: ['api'],
      blocks: ['b_imports'],
      source: src([5]),
      prompt: 'En la celda 5 se importan cinco librerías con alias. ¿Qué alias necesitas para llamar a `probplot`, `kurtosis` y `norm.cdf` tal como aparecen en el notebook?',
      options: [
        { text: '`ss`, porque son funciones de `scipy.stats`', correct: true },
        { text: '`np`, porque son funciones numéricas', feedback: 'NumPy no tiene `probplot` ni `kurtosis`; están en `scipy.stats`.' },
        { text: '`sns`, porque se usan para dibujar', feedback: '`probplot` dibuja con matplotlib (`plot=plt`), pero pertenece a `scipy.stats`.' },
        { text: '`pd`, porque los datos están en un DataFrame', feedback: 'En lab0_1 los datos son arrays de NumPy, no DataFrames.' },
      ],
      explanation: '`import scipy.stats as ss` da acceso a distribuciones (`ss.norm`), momentos (`ss.skew`, `ss.kurtosis`) y gráficos de probabilidad (`ss.probplot`). Sin esa línea, las celdas 19, 21 y 23 fallarían con `NameError: name \'ss\' is not defined`.',
    },
    {
      kind: 'predict',
      id: 'lab0_1.cov.vstack',
      skills: ['codigo', 'api'],
      blocks: ['b_extra_moments'],
      source: src([25], { origin: 'ampliacion' }),
      prompt: 'En la celda 25 (ampliación) se calculan los momentos con `ss.skew(np.vstack(samples), axis=1)`. ¿Qué forma tiene `np.vstack(samples)`? Escríbela como tupla.',
      setup: SETUP,
      code: 'samples = [data_normal, data_uniform, data_laplace, data_rayleigh]\nM = np.vstack(samples)\nskewness = ss.skew(M, axis=1)',
      expr: 'M.shape',
      answerKind: 'shape',
      explanation: '`vstack` apila las 4 muestras de 100 000 valores como **filas**: (4, 100000). Por eso se usa `axis=1`: un momento por fila (por distribución). Con `axis=0` saldrían 100 000 valores sin sentido, uno por posición.',
    },
  ],
  recall: ['lab0_1.fill.moments', 'lab0_1.fill.cdf', 'lab0_1.fix.axis', 'lab0_1.fix.pearson', 'lab0_1.mem.samples', 'lab0_1.mem.ks', 'lab0_1.adapt.expo'],
  practice: [
    {
      id: 'lab0_1.practice.q1',
      title: 'Q1 completa sin el notebook',
      scenario: 'Prepara la respuesta a **Q1**: genera las muestras, calcula sus momentos, interpreta las figuras y redacta la conclusión. Después, aplica lo mismo a una distribución nueva (exponencial).',
      steps: ['lab0_1.mem.samples', 'lab0_1.int.bars', 'lab0_1.int.qq', 'lab0_1.int.misconception', 'lab0_1.int.q1', 'lab0_1.adapt.expo'],
      source: { notebook: NB, doc: 'lab0_guide', section: 'Q1', origin: 'original' },
    },
  ],
  notes: ['Las celdas 24–26 («Additional evidence for Q1») son una ampliación añadida al notebook, no material de la asignatura.'],
};
