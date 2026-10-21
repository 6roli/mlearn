import type { NotebookContent } from '../../../src/content/types';

const NB = 'lab0_3';
const sol = (cells: number[], extra: Partial<{ section: string; note: string }> = {}) => ({
  notebook: NB,
  cells,
  origin: 'solucion' as const,
  ...extra,
});
const guide = (section: string) => ({ doc: 'lab0_guide', section, origin: 'original' as const });

const IRIS = `import numpy as np
import pandas as pd
from scipy import stats
iris = pd.read_csv('iris.csv')
features = list(iris.columns[:-1])
`;

const EXPERIMENT = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

iris = pd.read_csv('iris.csv')
feature = {{feature}}
especie = {{especie}}
confianza = {{confianza}}
ddof = {{ddof}}
alfa = {{alfa}}

# Q6 (a) · intervalo «manual»: media, desviación muestral y t crítico con n - 1 grados de libertad
x = iris[iris['species'] == especie][feature].to_numpy()
n = len(x)
media = np.mean(x)
s = np.std(x, ddof=ddof)
t_crit = stats.t.ppf(1 - (1 - confianza) / 2, df=n - 1)
ic_manual = (media - t_crit * s / np.sqrt(n), media + t_crit * s / np.sqrt(n))

# Q6 (b) · con stats.t.interval (celda 3); stats.sem usa siempre ddof=1
ic_funcion = stats.t.interval(confianza, n - 1, loc=media, scale=stats.sem(x))
print('IC manual :', ic_manual)
print('IC función:', ic_funcion)

# Q7 · contrastes de normalidad (celda 5)
sw = stats.shapiro(x)
ad = stats.anderson(x, dist='norm')
indice = np.flatnonzero(np.isclose(ad.significance_level, 100*alfa))[0]
critico = ad.critical_values[indice]
print('Shapiro-Wilk: W = %.6f, p = %.6f' % (sw.statistic, sw.pvalue))
print('Anderson-Darling: A2 = %.4f, valor crítico (alfa=%s) = %.3f' % (ad.statistic, alfa, critico))

# figura: IC de las tres especies (izquierda) y Q-Q plot de la selección (derecha)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (esp, grupo) in enumerate(iris.groupby('species', sort=False)):
  xs = grupo[feature]
  a, b = stats.t.interval(confianza, len(xs)-1, loc=xs.mean(), scale=stats.sem(xs))
  ax[0].errorbar(k, xs.mean(), yerr=[[xs.mean()-a], [b-xs.mean()]], fmt='o', capsize=8, color='C%d' % k)
ax[0].set_xticks(range(3))
ax[0].set_xticklabels(['setosa', 'versicolor', 'virginica'])
ax[0].set_ylabel(feature + ' (cm)')
ax[0].set_title('Media e IC al %g %%' % (100*confianza))
stats.probplot(x, plot=ax[1])
ax[1].set_title('Q-Q normal · ' + especie + ' · ' + feature)
plt.show()

resultados = {
  'n': n, 'media': media, 's': s, 't_crit': t_crit,
  'ic_low': ic_manual[0], 'ic_high': ic_manual[1],
  'ic_low_fun': ic_funcion[0], 'ic_high_fun': ic_funcion[1],
  'semiancho': t_crit * s / np.sqrt(n),
  'W': sw.statistic, 'p': sw.pvalue, 'A2': ad.statistic, 'critico': critico,
  'SW': 'Rechazar' if sw.pvalue < alfa else 'No rechazar',
  'AD': 'Rechazar' if ad.statistic > critico else 'No rechazar',
}
`;

const CLASS_OPTS = [
  { value: "'Iris-setosa'", label: 'setosa (clase 1)' },
  { value: "'Iris-versicolor'", label: 'versicolor (clase 2)' },
  { value: "'Iris-virginica'", label: 'virginica (clase 3)' },
];
const FEAT_OPTS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].map((f) => ({ value: `'${f}'`, label: f }));

export const lab0_3: NotebookContent = {
  id: NB,
  lab: 'lab0',
  title: 'Lab 0 · Parte 3 — Intervalos de confianza y contrastes de normalidad (Q6–Q7)',
  short: 'IC y contrastes (Q6–Q7)',
  origin: 'solucion',
  context: {
    goal: `La guía de Lab0 pide **crear** un notebook nuevo, \`Mlearn_lab0_3.ipynb\`, para pasar de describir los datos a hacer **inferencia**:

- **Q6**: elegir una feature y calcular, para cada especie, la media y los intervalos de confianza al 95 %, 99 % y 99.9 %, (a) «a mano» con el valor crítico \`stats.t.ppf\`, la media y la desviación típica muestral, y (b) con \`stats.t.interval\`.
- **Q7**: contrastar, para las 4 features y las 3 clases, $H_0$: «la feature K de la clase J es gaussiana», con **Shapiro-Wilk** (\`shapiro\`) y **Anderson-Darling** (\`anderson\`), a $\\alpha = 0.05$ y $0.01$, e interpretar p-valores y valores críticos.

El archivo de la carpeta es **tu solución** (no material de la asignatura). Aquí se usa como referencia para validar tus respuestas: sus bloques de código permanecen **ocultos** hasta que intentas el ejercicio correspondiente.`,
    dataset: 'Iris (`iris.csv`): 150 flores, 50 por especie, 4 features en cm. La guía numera las clases: clase 1 = setosa, clase 2 = versicolor, clase 3 = virginica. Tu solución usa `sepal_length` para Q6 (49 grados de libertad).',
    variables: [
      { name: 'iris', meaning: 'DataFrame con las 150 flores.', shape: '(150, 5)' },
      { name: 'features', meaning: 'Nombres de las 4 columnas numéricas.', shape: 'lista de 4' },
      { name: 'grupo, x', meaning: 'Filas de una especie y la columna elegida.', shape: '(50, 5), (50,)' },
      { name: 'resultados / filas', meaning: 'Lista de diccionarios, una fila por especie (Q6) o por especie × feature (Q7).', shape: '3 / 12 filas' },
      { name: 'sw, ad', meaning: 'Resultados de `shapiro` (statistic, pvalue) y `anderson` (statistic, critical_values, significance_level).', shape: '—' },
      { name: 'resumen', meaning: 'DataFrame con todas las decisiones de Q7.', shape: '(12, 11)' },
    ],
    classes: [
      { id: 'setosa', label: 'Iris-setosa (clase 1)' },
      { id: 'versicolor', label: 'Iris-versicolor (clase 2)' },
      { id: 'virginica', label: 'Iris-virginica (clase 3)' },
    ],
    questions: [
      { id: 'Q6', text: 'Media e IC al 95 %, 99 % y 99.9 % de una feature para las tres clases: manual (t.ppf, mean, std) y con `stats.t.interval`.', blocks: ['b_q6', 'b_q6_table'], doc: 'lab0_guide' },
      { id: 'Q7', text: 'Shapiro-Wilk y Anderson-Darling para todas las features y clases a α = 0.05 y 0.01; significado del p-valor y del valor crítico.', blocks: ['b_q7', 'b_q7_tables'], doc: 'lab0_guide' },
    ],
    notation: [
      { symbol: '$\\bar x$', meaning: 'media muestral', code: 'x.mean() / np.mean(x)' },
      { symbol: '$s$', meaning: 'desviación típica muestral (con n − 1)', code: 'np.std(x, ddof=1) / x.std()' },
      { symbol: '$s/\\sqrt n$', meaning: 'error estándar de la media', code: 'stats.sem(x)' },
      { symbol: '$t^*_{\\alpha/2,\\,n-1}$', meaning: 'valor crítico de la t de Student', code: 'stats.t.ppf(1 - alpha/2, n - 1)' },
      { symbol: '$\\bar x \\pm t^* s/\\sqrt n$', meaning: 'IC bilateral al nivel 1 − α', code: 'stats.t.interval(conf, n-1, loc, scale)' },
      { symbol: '$H_0$', meaning: 'la feature es gaussiana en esa clase' },
      { symbol: '$W$, $p$', meaning: 'estadístico y p-valor de Shapiro-Wilk; se rechaza si p < α', code: 'stats.shapiro(x)' },
      { symbol: '$A^2$', meaning: 'estadístico de Anderson-Darling; se rechaza si A² > valor crítico', code: 'stats.anderson(x, dist="norm")' },
    ],
  },
  flow: [
    { stage: 'contexto', label: 'Contexto', blocks: ['b_intro'], summary: 'Qué piden Q6 y Q7.' },
    { stage: 'datos', label: 'Datos', blocks: ['b_load'], summary: 'Leer iris.csv y comprobar 50 por clase.' },
    { stage: 'metodo', label: 'Método', blocks: ['b_q6', 'b_q7'], summary: 'IC de Student y dos contrastes por clase y feature.' },
    { stage: 'resultados', label: 'Resultados', blocks: ['b_q6_table', 'b_q7_tables'], summary: 'Tablas del report.' },
    { stage: 'interpretacion', label: 'Interpretación', blocks: [], summary: 'p-valor, valor crítico y qué se puede concluir (pestaña Interpretar).' },
  ],
  blocks: [
    {
      id: 'b_intro',
      title: 'Cabecera: alcance y versión de SciPy',
      cells: [0],
      stage: 'contexto',
      origin: 'solucion',
      problem: 'Situar el notebook: es el que pide crear la guía para Q6 y Q7.',
      purpose: 'Indica cómo ejecutarlo (junto a iris.csv), la correspondencia clase 1–3 ↔ especie y la versión de SciPy con la que se validó.',
      inputs: 'Texto.',
      outputs: 'Advertencia sobre la API de Anderson-Darling.',
      prereqs: ['intervalo-confianza', 'contraste', 'anderson'],
      apis: [],
      lines: [],
      interpretation: '«La tabla de valores críticos de Anderson-Darling usa la API de SciPy anterior a 1.19»: hasta SciPy 1.18, `stats.anderson` devuelve `critical_values` y `significance_level` (en %: 15, 10, 5, 2.5, 1) y no un p-valor. Las versiones nuevas cambian ese resultado (añaden otra forma de obtener la significación), así que el acceso `ad.critical_values[indice]` de la celda 5 puede dejar de funcionar o avisar. La app usa SciPy 1.18.0 y tu entorno 1.18.1: ambos con la API antigua.',
      next: 'Carga de datos.',
      whatIf: [{ change: 'Ejecutar con SciPy ≥ 1.19', effect: 'Revisa la documentación de `scipy.stats.anderson`: la decisión debe seguir siendo «rechazar si A² supera el valor crítico al nivel α».' }],
    },
    {
      id: 'b_load',
      title: 'Imports y carga de Iris',
      cells: [1],
      stage: 'datos',
      origin: 'solucion',
      hiddenUntil: 'lab0_3.fill.load',
      data: ['iris.csv'],
      problem: 'Leer el CSV desde Colab o desde la carpeta local y comprobar el tamaño de cada clase.',
      purpose: 'Q6 necesita n por clase (para los grados de libertad) y Q7 la lista de features.',
      inputs: '`iris.csv`.',
      outputs: '`iris` (150, 5), `features` (4 nombres); imprime 50 flores por especie.',
      prereqs: ['dataframe', 'groupby'],
      apis: [
        { name: 'from scipy import stats', lib: 'scipy', kind: 'módulo', does: 'Distribuciones (t) y contrastes (shapiro, anderson).' },
        { name: 'Path.exists', lib: 'pathlib', kind: 'método', does: 'Comprueba si el archivo existe para elegir la ruta.' },
        { name: 'pd.read_csv', lib: 'pandas', kind: 'función', does: 'Lee el CSV en un DataFrame.' },
        { name: 'groupby(...).size()', lib: 'pandas', kind: 'método', does: 'Número de filas de cada grupo.' },
      ],
      lines: [
        { cell: 1, lines: [5, 7], text: 'Primero busca `iris.csv` junto al notebook (Colab); si no, una ruta relativa de tu carpeta. La app ejecuta con el CSV en el directorio actual.' },
        { cell: 1, lines: [9, 9], text: 'Todas las columnas menos la última (`species`).' },
        { cell: 1, lines: [10, 10], text: 'Salida: 50, 50, 50 → df = 49 en Q6.' },
      ],
      shapes: [{ expr: 'iris', shape: '(150, 5)', meaning: '4 features + especie' }],
      interpretation: 'Con 50 observaciones por clase el valor crítico t con 49 g.l. (2.0096 al 95 %) es algo mayor que el normal (1.96): el IC es un poco más ancho por estimar σ.',
      next: 'Q6.',
      whatIf: [{ change: 'Usar `iris.columns[:4]`', effect: 'Equivalente aquí; `[:-1]` no depende de cuántas features haya.' }],
    },
    {
      id: 'b_q6',
      title: 'Q6 · Intervalos de confianza de Student',
      cells: [2, 3],
      stage: 'metodo',
      origin: 'solucion',
      hiddenUntil: 'lab0_3.mem.ci',
      data: ['iris.csv'],
      problem: 'Estimar la media poblacional de `sepal_length` de cada especie con un rango que refleje la incertidumbre muestral.',
      purpose: 'Responde Q6 (b) con `stats.t.interval`. La parte (a) «manual» se practica en los ejercicios y debe dar exactamente lo mismo.',
      inputs: '`iris`.',
      outputs: 'Tabla con media e IC al 95 %, 99 % y 99.9 % por especie. Setosa: media 5.006, IC 95 % [4.9058, 5.1062].',
      prereqs: ['intervalo-confianza', 't-student', 'grados-libertad', 'error-estandar'],
      apis: [
        { name: 'groupby("species", sort=False)', lib: 'pandas', kind: 'método', does: 'Itera por especie en el orden del archivo (setosa, versicolor, virginica).' },
        { name: 'stats.t.interval', lib: 'scipy.stats', kind: 'función', does: 'Intervalo central que contiene la probabilidad `confidence` de una t.', args: [{ name: 'confidence', value: '0.95 / 0.99 / 0.999', meaning: 'nivel 1 − α (en versiones antiguas se llamaba alpha)' }, { name: 'df', value: 'len(x)-1', meaning: 'grados de libertad' }, { name: 'loc', value: 'x.mean()', meaning: 'centro: media muestral' }, { name: 'scale', value: 'stats.sem(x)', meaning: 'error estándar s/√n' }], returns: 'tupla (inferior, superior)' },
        { name: 'stats.sem', lib: 'scipy.stats', kind: 'función', does: 'Error estándar de la media, con ddof=1 por defecto.' },
        { name: 'stats.t.ppf', lib: 'scipy.stats', kind: 'función', does: 'Inversa de la CDF (versión manual): $t^* = $ `t.ppf(1 - α/2, n - 1)`.' },
      ],
      lines: [
        { cell: 3, lines: [2, 3], text: 'Un grupo por especie; `x` es una Series de 50 valores.' },
        { cell: 3, lines: [6, 6], text: 'Equivale a $\\bar x \\pm t^*_{\\alpha/2,49}\\,s/\\sqrt{50}$. La versión manual: `t = stats.t.ppf(1-(1-c)/2, n-1)`; `m ± t*np.std(x, ddof=1)/np.sqrt(n)`.' },
        { cell: 3, lines: [8, 8], text: 'Imprime la tabla; las tuplas salen con todos los decimales.' },
      ],
      shapes: [{ expr: 'x', shape: '(50,)', meaning: 'una especie, una feature' }],
      interpretation: 'Al subir el nivel de confianza el intervalo se **ensancha** (más seguridad = menos precisión). Los IC de las tres especies **no se solapan** ni al 99.9 %: hay evidencia de que sus medias poblacionales de sepal_length son distintas. Interpretación correcta: el procedimiento captura la media real en el 95 % de las muestras posibles; **no** que μ tenga un 95 % de probabilidad de estar en este intervalo concreto.',
      next: 'Q7: contrastes de normalidad.',
      whatIf: [
        { change: 'Usar `np.std(x)` sin ddof en la versión manual', effect: 'NumPy usa ddof=0 por defecto: s sale un factor $\\sqrt{49/50}\\approx 0.99$ menor y el intervalo, algo más estrecho que el de `t.interval`.' },
        { change: 'Usar 1.96 (normal) en lugar de t*', effect: 'IC ligeramente más estrecho: ignora la incertidumbre de estimar σ.' },
      ],
    },
    {
      id: 'b_q7',
      title: 'Q7 · Shapiro-Wilk y Anderson-Darling para 12 combinaciones',
      cells: [4, 5],
      stage: 'metodo',
      origin: 'solucion',
      hiddenUntil: 'lab0_3.mem.sw',
      data: ['iris.csv'],
      problem: 'Decidir, para cada especie y feature, si hay evidencia contra la normalidad.',
      purpose: 'Responde Q7: estadístico, p-valor (SW), valor crítico (AD) y decisión a α = 0.05 y 0.01, sin corrección por comparaciones múltiples.',
      inputs: '`iris`, `features`.',
      outputs: '`resultados`: 12 filas con W, p, A², valores críticos y decisiones.',
      prereqs: ['shapiro', 'anderson', 'p-valor', 'valor-critico', 'h0', 'significacion'],
      apis: [
        { name: 'stats.shapiro', lib: 'scipy.stats', kind: 'función', does: 'Contraste de Shapiro-Wilk.', returns: 'resultado con statistic (W) y pvalue' },
        { name: 'stats.anderson', lib: 'scipy.stats', kind: 'función', does: 'Contraste de Anderson-Darling.', args: [{ name: 'dist', value: "'norm'", meaning: 'distribución de referencia' }], returns: 'statistic (A²), critical_values, significance_level (en %)' },
        { name: 'np.isclose / np.flatnonzero', lib: 'numpy', kind: 'función', does: 'Encuentra la posición del nivel 5 % o 1 % en `significance_level` sin comparar floats con ==.' },
      ],
      lines: [
        { cell: 5, lines: [5, 6], text: 'Dos contrastes sobre los mismos 50 valores.' },
        { cell: 5, lines: [9, 10], text: '`significance_level` = [15, 10, 5, 2.5, 1] (en %). Se busca 100·α y se toma el valor crítico de esa posición: 0.740 al 5 % y 1.019 al 1 % (n = 50).' },
        { cell: 5, lines: [11, 11], text: 'Shapiro-Wilk: rechazar $H_0$ si p < α.' },
        { cell: 5, lines: [13, 13], text: 'Anderson-Darling: rechazar si A² > valor crítico (el sentido es el contrario que con el p-valor).' },
      ],
      shapes: [{ expr: 'pd.DataFrame(resultados)', shape: '(12, 11)', meaning: '3 especies × 4 features' }],
      interpretation: 'Setosa·petal_width se rechaza con ambos tests a los dos niveles (W = 0.814, p ≈ 2·10⁻⁶; A² = 4.31): sus medidas son muy discretas (0.1–0.6) y asimétricas. Versicolor·petal_width se rechaza al 5 % y no al 1 %. Setosa·petal_length muestra un **desacuerdo**: SW no rechaza al 5 % (p = 0.0547) pero AD sí (A² = 1.011 > 0.740). **No rechazar ≠ demostrar normalidad**: solo indica que con 50 datos no hay evidencia suficiente. Con 12 contrastes a α = 0.05 se esperan ≈ 0.6 rechazos falsos aunque todo fuera gaussiano.',
      next: 'Tablas resumen para el report.',
      whatIf: [
        { change: 'Comparar `ad.statistic < critico` para rechazar', effect: 'Invertiría todas las decisiones de AD: error de razonamiento típico.' },
        { change: 'Usar α = 0.01', effect: 'Menos rechazos: solo setosa·petal_width.' },
      ],
    },
    {
      id: 'b_q6_table',
      title: 'Tabla de Q6 para el report',
      cells: [6],
      stage: 'resultados',
      origin: 'solucion',
      hiddenUntil: 'lab0_3.mem.ci',
      data: ['iris.csv'],
      problem: 'Presentar los IC con 4 decimales en el formato de la tabla de la guía.',
      purpose: 'Rellena «Mean / CI at 95 % / 99 % / 99.9 %» por clase.',
      inputs: '`iris`.',
      outputs: 'Tabla: setosa 5.0060 [4.9058, 5.1062] [4.8724, 5.1396] [4.8315, 5.1805]; versicolor 5.9360 …; virginica 6.5880 ….',
      prereqs: ['intervalo-confianza'],
      apis: [{ name: "f'{a:.4f}'", lib: 'python', kind: 'operador', does: 'Formato con 4 decimales.' }],
      lines: [{ cell: 6, lines: [7, 8], text: 'Mismo cálculo que la celda 3, desempaquetado en (a, b) y formateado.' }],
      interpretation: 'La anchura crece con el nivel: setosa 0.200 (95 %), 0.267 (99 %), 0.349 (99.9 %). Virginica tiene intervalos más anchos porque su desviación típica es mayor (0.636 frente a 0.352).',
      whatIf: [{ change: 'Elegir petal_length como feature', effect: 'Mismo procedimiento; las medias (1.46, 4.26, 5.55) están aún más separadas.' }],
    },
    {
      id: 'b_q7_tables',
      title: 'Tablas de Q7 para el report',
      cells: [7, 8],
      stage: 'resultados',
      origin: 'solucion',
      hiddenUntil: 'lab0_3.fix.ad',
      data: ['iris.csv'],
      problem: 'Separar las columnas de cada test para las dos tablas de la guía.',
      purpose: 'Tabla SW (decisión y p-valor) y tabla AD (decisión, valor crítico y estadístico).',
      inputs: '`resultados` de la celda 5.',
      outputs: 'Dos tablas de 12 filas.',
      prereqs: ['dataframe'],
      apis: [{ name: 'DataFrame.to_string', lib: 'pandas', kind: 'método', does: 'Imprime sin índice y con formato numérico.', args: [{ name: 'float_format', meaning: 'función que formatea cada float' }] }],
      lines: [
        { cell: 7, lines: [2, 2], text: '`resumen` se crea aquí a partir de la lista `resultados` de la celda 5: si se ejecuta antes la celda 6, `resultados` sigue siendo el de Q7 porque la celda 6 usa otra lista (`filas`).' },
        { cell: 8, lines: [2, 4], text: 'Mismo DataFrame, otras columnas: los valores críticos son iguales en las 12 filas porque solo dependen de n.' },
      ],
      interpretation: 'Valor crítico: umbral del estadístico por encima del cual los datos serían demasiado improbables bajo H₀ al nivel α. p-valor: probabilidad, bajo H₀, de un estadístico al menos tan extremo como el observado. Ambas reglas dicen lo mismo con lenguajes distintos.',
      whatIf: [{ change: 'Ejecutar la celda 7 sin haber ejecutado la 5', effect: '`resultados` sería la lista de Q6 (si se ejecutó la 3) y faltarían columnas: KeyError.' }],
    },
  ],
  experiments: [
    {
      id: 'lab0_3.exp.ic',
      title: 'Nivel de confianza, feature, clase, ddof y α',
      goal: 'Ver cómo cambia el IC con el nivel de confianza y con la elección de ddof, y cómo cambian las decisiones de los contrastes con α. Los datos son siempre los 150 Iris: no hay semilla.',
      blocks: ['b_q6', 'b_q7'],
      data: ['iris.csv'],
      template: EXPERIMENT,
      params: [
        { id: 'feature', label: 'Feature', kind: 'datos', control: 'select', default: "'sepal_length'", options: FEAT_OPTS, variable: 'feature', computation: 'Columna sobre la que se calculan IC y contrastes.', result: 'Medias, anchuras y decisiones distintas por feature.', why: 'Cada feature tiene su propia dispersión y forma.' },
        { id: 'especie', label: 'Especie', kind: 'datos', control: 'select', default: "'Iris-setosa'", options: CLASS_OPTS, variable: 'especie', computation: 'Filtra las 50 flores de una clase.', result: 'IC y contrastes de esa clase (la figura izquierda muestra las tres).', why: 'Q6 y Q7 se calculan por clase.' },
        { id: 'confianza', label: 'Nivel de confianza', kind: 'evaluacion', control: 'select', default: '0.95', options: [{ value: '0.9', label: '90 %' }, { value: '0.95', label: '95 %' }, { value: '0.99', label: '99 %' }, { value: '0.999', label: '99.9 %' }], variable: 'confianza', computation: 'Cambia el cuantil `t.ppf(1 - (1 - c)/2, n - 1)`.', result: 'Más confianza → t* mayor → intervalo más ancho.', why: 'Para capturar μ con más frecuencia hay que aceptar un rango mayor.' },
        { id: 'ddof', label: 'ddof de np.std', kind: 'preprocesado', control: 'select', default: '1', options: [{ value: '1', label: '1 (muestral, correcto)' }, { value: '0', label: '0 (por defecto de NumPy)' }], variable: 'np.std(x, ddof=…)', computation: 'Divide por n − ddof al calcular s.', result: 'Con ddof=0 el IC manual queda un 1 % más estrecho y deja de coincidir con `t.interval`.', why: '`stats.sem` usa ddof=1; el IC de Student requiere la desviación muestral con n − 1.' },
        { id: 'alfa', label: 'α de los contrastes', kind: 'evaluacion', control: 'select', default: '0.05', options: [{ value: '0.1', label: '0.10' }, { value: '0.05', label: '0.05' }, { value: '0.01', label: '0.01' }], variable: 'alfa', computation: 'Umbral del p-valor (SW) y posición del valor crítico (AD).', result: 'α menor → menos rechazos.', why: 'α es la probabilidad aceptada de rechazar H₀ siendo cierta (error de tipo I).' },
      ],
      metrics: [
        { key: 'media', label: 'Media muestral', format: 'num', digits: 4 },
        { key: 's', label: 'Desviación s', format: 'num', digits: 4 },
        { key: 't_crit', label: 't crítico', format: 'num', digits: 4 },
        { key: 'ic_low', label: 'IC manual · inferior', format: 'num', digits: 4 },
        { key: 'ic_high', label: 'IC manual · superior', format: 'num', digits: 4 },
        { key: 'ic_low_fun', label: 't.interval · inferior', format: 'num', digits: 4 },
        { key: 'ic_high_fun', label: 't.interval · superior', format: 'num', digits: 4 },
        { key: 'semiancho', label: 'Semiancho del IC', format: 'num', digits: 4 },
        { key: 'W', label: 'Shapiro-Wilk W', format: 'num', digits: 6 },
        { key: 'p', label: 'p-valor SW', format: 'num', digits: 6 },
        { key: 'SW', label: 'Decisión SW', format: 'text' },
        { key: 'A2', label: 'Anderson-Darling A²', format: 'num', digits: 4 },
        { key: 'critico', label: 'Valor crítico AD', format: 'num', digits: 3 },
        { key: 'AD', label: 'Decisión AD', format: 'text' },
      ],
      observe: [
        'Pasa de 95 % a 99.9 %: ¿cuánto crece el semiancho? ¿Cambia la media?',
        'Pon ddof = 0: compara el IC manual con el de `t.interval`. ¿Cuál está bien y por qué?',
        'Elige setosa · petal_length: ¿coinciden SW y AD al 5 %? ¿y al 1 %?',
        'Elige setosa · petal_width y mira el Q-Q plot: ¿qué forma tiene la desviación de la recta?',
      ],
      source: sol([3, 5], { note: 'La versión manual del IC, la figura y la opción ddof son de la app (guía Q6a).' }),
      fidelity: [
        { metrics: { media: 5.006, ic_low: 4.905823539299264, ic_high: 5.106176460700737, ic_low_fun: 4.905823539299264, W: 0.977699, p: 0.459513, A2: 0.407986, critico: 0.74 }, tol: 1e-5, source: 'Salidas guardadas de las celdas 3 y 5 (setosa · sepal_length)' },
        { params: { confianza: '0.999', especie: "'Iris-virginica'" }, metrics: { ic_low: 6.273215831432418, ic_high: 6.902784168567583 }, tol: 1e-9, source: 'Salida guardada de la celda 3 (virginica, 99.9 %)' },
        { params: { feature: "'petal_width'" }, metrics: { W: 0.813817, A2: 4.307008 }, tol: 1e-5, source: 'Salida guardada de la celda 5 (setosa · petal_width)' },
      ],
    },
  ],
  activities: [
    // ---------------------------------------------------------- nivel 1
    {
      kind: 'mcq',
      id: 'lab0_3.read.sem',
      skills: ['api', 'concepto'],
      blocks: ['b_q6'],
      source: sol([3]),
      prompt: 'En `stats.t.interval(0.95, len(x)-1, loc=x.mean(), scale=stats.sem(x))`, ¿qué es `scale`?',
      options: [
        { text: 'El error estándar de la media, $s/\\sqrt n$ con ddof = 1', correct: true },
        { text: 'La desviación típica de los datos, $s$', feedback: 'Eso daría un intervalo para una observación individual, no para la media.' },
        { text: 'La varianza muestral', feedback: 'La escala de la t es una desviación, no una varianza.' },
        { text: 'El nivel de significación', feedback: 'El nivel es el primer argumento (0.95).' },
      ],
      explanation: '`stats.sem(x)` = `np.std(x, ddof=1) / np.sqrt(len(x))`: la desviación típica de la media muestral.',
    },
    {
      kind: 'text',
      id: 'lab0_3.read.ci',
      skills: ['concepto', 'interpretacion'],
      blocks: ['b_q6'],
      source: { ...guide('Q6'), notebook: NB, cells: [3] },
      prompt: 'El IC al 95 % de la media de `sepal_length` en setosa es [4.906, 5.106]. Explica qué significa (y qué **no** significa).',
      rubric: [
        'Es un rango para la media poblacional μ, no para flores individuales.',
        'El 95 % se refiere al procedimiento: en muchas muestras repetidas, el 95 % de los intervalos contendría μ.',
        'No significa que μ tenga un 95 % de probabilidad de estar en este intervalo concreto (μ es fija).',
        'Se basa en la t de Student con 49 g.l. porque σ se estima con s.',
      ],
      model: 'Con el procedimiento usado, si se repitiera el muestreo de 50 setosas muchas veces, el 95 % de los intervalos construidos así contendrían la media poblacional de sepal_length. Este intervalo concreto la contiene o no; no se puede asignar una probabilidad a μ, que es fija. Es un intervalo para la media, no para el 95 % de las flores, y usa la t de Student con 49 grados de libertad porque la desviación poblacional se sustituye por la muestral.',
    },
    {
      kind: 'mcq',
      id: 'lab0_3.read.adrule',
      skills: ['codigo', 'concepto'],
      blocks: ['b_q7'],
      source: sol([5]),
      prompt: 'En la celda 5, ¿por qué la regla de Shapiro-Wilk usa `pvalue < alfa` y la de Anderson-Darling `statistic > critico`?',
      options: [
        { text: 'Son la misma regla en dos escalas: p pequeño ⇔ estadístico más extremo que el crítico', correct: true },
        { text: 'Porque Anderson-Darling contrasta la hipótesis contraria', feedback: 'Ambos contrastan H₀: «los datos son gaussianos».' },
        { text: 'Porque la de AD es más exigente', feedback: 'No es una cuestión de exigencia: es cómo `anderson` devuelve su resultado (sin p-valor).' },
      ],
      explanation: '`shapiro` devuelve un p-valor; `anderson` (API ≤ 1.18) devuelve el estadístico y sus valores críticos para niveles fijos. Rechazar cuando A² supera el crítico al nivel α equivale a rechazar cuando p < α.',
    },
    // ---------------------------------------------------------- nivel 2
    {
      kind: 'predict',
      id: 'lab0_3.pred.tcrit',
      skills: ['api', 'concepto'],
      blocks: ['b_q6'],
      source: { ...guide('Q6'), notebook: NB, cells: [3] },
      prompt: '¿Cuánto vale el valor crítico `stats.t.ppf(0.975, 49)`? (4 decimales)',
      setup: 'from scipy import stats',
      code: 't_crit = stats.t.ppf(0.975, 49)',
      expr: 't_crit',
      answerKind: 'numeric',
      tolerance: 0.0006,
      explanation: '2.0096: algo mayor que 1.96 de la normal porque hay que estimar σ con 50 datos.',
    },
    {
      kind: 'predict',
      id: 'lab0_3.pred.rows',
      skills: ['codigo'],
      blocks: ['b_q7'],
      source: sol([5]),
      prompt: '¿Cuántas filas tiene `pd.DataFrame(resultados)` al final de la celda 5?',
      setup: IRIS,
      code: "resultados = []\nfor especie, grupo in iris.groupby('species', sort=False):\n    for variable in features:\n        resultados.append({'species': especie, 'feature': variable})\ndf = pd.DataFrame(resultados)",
      expr: 'len(df)',
      answerKind: 'numeric',
      tolerance: 0,
      data: ['iris.csv'],
      explanation: '3 especies × 4 features = 12 contrastes (con dos niveles α en columnas).',
    },
    {
      kind: 'predict',
      id: 'lab0_3.pred.crit',
      skills: ['api'],
      blocks: ['b_q7'],
      source: sol([5]),
      prompt: '¿Qué devuelve `stats.anderson(x, dist="norm").significance_level` (la lista de niveles en %)? Elige.',
      setup: IRIS,
      code: "x = iris[iris['species'] == 'Iris-setosa']['sepal_length']\nad = stats.anderson(x, dist='norm')\nniveles = str([float(v) for v in ad.significance_level])",
      expr: 'niveles',
      answerKind: 'mcq',
      options: ['[15.0, 10.0, 5.0, 2.5, 1.0]', '[0.15, 0.1, 0.05, 0.025, 0.01]', '[0.05, 0.01]', '[1.0, 2.5, 5.0, 10.0, 15.0]'],
      data: ['iris.csv'],
      explanation: 'En porcentaje y de mayor a menor: por eso la celda 5 busca `100*alfa` con `np.isclose`.',
    },
    // ---------------------------------------------------------- nivel 3
    {
      kind: 'code',
      level: 3,
      id: 'lab0_3.fill.load',
      title: 'Completa la carga y el recuento por especie',
      skills: ['codigo', 'api'],
      blocks: ['b_load'],
      source: sol([1]),
      prompt: 'Rellena los huecos para leer `iris.csv`, obtener la lista de features (todas las columnas menos la última) y contar cuántas flores hay de cada especie.',
      available: '`pd` importado; `iris.csv` en el directorio de trabajo.',
      produce: '`iris` (150, 5), `features` (lista de 4 nombres) y `conteo` (Series con 50, 50, 50).',
      setup: 'import pandas as pd',
      data: ['iris.csv'],
      starter: "iris = pd.___('iris.csv')\nfeatures = list(iris.columns[___])\nconteo = iris.groupby('___').___()",
      solution: "iris = pd.read_csv('iris.csv')\nfeatures = list(iris.columns[:-1])\nconteo = iris.groupby('species').size()",
      alternatives: ["iris = pd.read_csv('iris.csv')\nfeatures = [c for c in iris.columns if c != 'species']\nconteo = iris['species'].value_counts().sort_index()"],
      checks: "c.shape('iris', (150, 5))\nc.true(ns.get('features') == ref['features'], 'Lista de features correcta', 'La lista debe tener las 4 columnas numéricas en orden.', 'razonamiento')\nc.same('conteo')",
      hints: ['La función de pandas para CSV es `read_csv`.', 'Todas menos la última: `[:-1]`.', 'Agrupa por la columna `species` y usa `.size()`.'],
    },
    {
      kind: 'code',
      level: 3,
      id: 'lab0_3.fill.interval',
      title: 'Completa la llamada a t.interval',
      skills: ['api'],
      blocks: ['b_q6'],
      source: sol([3]),
      practice: 'scipy.stats.t.interval',
      prompt: 'Completa la llamada para obtener el IC al 99 % de la media de `x` (petal_length de versicolor).',
      available: '`stats` (scipy.stats), `x`: Series de 50 valores.',
      produce: '`ic99`: tupla (inferior, superior).',
      setup: IRIS + "x = iris[iris['species'] == 'Iris-versicolor']['petal_length']\n",
      data: ['iris.csv'],
      starter: 'ic99 = stats.t.interval(___, ___, loc=___, scale=___)',
      solution: 'ic99 = stats.t.interval(0.99, len(x) - 1, loc=x.mean(), scale=stats.sem(x))',
      alternatives: ['ic99 = stats.t.interval(confidence=0.99, df=49, loc=np.mean(x), scale=np.std(x, ddof=1) / np.sqrt(50))'],
      wrong: [{ code: 'ic99 = stats.t.interval(0.99, len(x), loc=x.mean(), scale=stats.sem(x))', expect: 'razonamiento' }],
      checks: "c.same('ic99', rtol=1e-10, desc='El IC al 99 %')",
      hints: ['El primer argumento es el nivel de confianza (0.99).', 'Grados de libertad: n − 1.', 'Centro: la media; escala: el error estándar `stats.sem(x)`.'],
    },
    // ---------------------------------------------------------- nivel 4
    {
      kind: 'code',
      level: 4,
      id: 'lab0_3.fix.manual',
      title: 'Corrige el IC «manual» (Q6 a)',
      skills: ['codigo', 'concepto'],
      blocks: ['b_q6'],
      source: { ...guide('Q6 (a)'), notebook: NB, cells: [3] },
      prompt: 'Este IC manual al 95 % no coincide con `stats.t.interval`. Tiene **tres** errores de concepto. Corrígelo.',
      available: '`np`, `stats`, `x`: sepal_length de setosa (50 valores).',
      produce: '`ic`: tupla (inferior, superior) igual a la de `t.interval`.',
      setup: IRIS + "x = iris[iris['species'] == 'Iris-setosa']['sepal_length'].to_numpy()\n",
      data: ['iris.csv'],
      starter: 'n = len(x)\nm = np.mean(x)\ns = np.std(x)\nt = stats.t.ppf(0.95, n)\nic = (m - t * s / n, m + t * s / n)',
      solution: 'n = len(x)\nm = np.mean(x)\ns = np.std(x, ddof=1)\nt = stats.t.ppf(0.975, n - 1)\nic = (m - t * s / np.sqrt(n), m + t * s / np.sqrt(n))',
      alternatives: ['n = len(x)\nm = x.mean()\nse = stats.sem(x)\nt = stats.t.ppf(1 - 0.05 / 2, df=n - 1)\nic = (m - t * se, m + t * se)'],
      wrong: [{ code: 'n = len(x)\nm = np.mean(x)\ns = np.std(x)\nt = stats.t.ppf(0.975, n - 1)\nic = (m - t * s / np.sqrt(n), m + t * s / np.sqrt(n))', expect: 'razonamiento' }],
      checks: "c.same('ic', rtol=1e-10, desc='El intervalo')",
      hints: ['IC bilateral al 95 %: cada cola tiene 2.5 % → cuantil 0.975.', 'Grados de libertad n − 1; desviación muestral con `ddof=1` (NumPy usa 0 por defecto).', 'El error estándar es s/√n, no s/n.'],
      explanation: 'Resultado: (4.9058, 5.1062), igual que la celda 3.',
    },
    {
      kind: 'code',
      level: 4,
      id: 'lab0_3.fix.ad',
      title: 'Corrige la decisión de Anderson-Darling',
      skills: ['api', 'concepto'],
      blocks: ['b_q7', 'b_q7_tables'],
      source: sol([5]),
      prompt: 'El código debe decidir, para setosa · petal_length, si se rechaza la normalidad con Anderson-Darling a α = 0.05 y a α = 0.01. Toma el valor crítico de una posición equivocada y compara al revés. Corrígelo.',
      available: '`np`, `stats`, `x`: petal_length de setosa.',
      produce: '`crit05`, `crit01` (valores críticos) y `dec05`, `dec01`: "Rechazar" o "No rechazar".',
      setup: IRIS + "x = iris[iris['species'] == 'Iris-setosa']['petal_length']\n",
      data: ['iris.csv'],
      starter: "ad = stats.anderson(x, dist='norm')\ncrit05 = ad.critical_values[0]\ncrit01 = ad.critical_values[1]\ndec05 = 'Rechazar' if ad.statistic < crit05 else 'No rechazar'\ndec01 = 'Rechazar' if ad.statistic < crit01 else 'No rechazar'",
      solution: "ad = stats.anderson(x, dist='norm')\ncrit05 = ad.critical_values[np.flatnonzero(np.isclose(ad.significance_level, 5))[0]]\ncrit01 = ad.critical_values[np.flatnonzero(np.isclose(ad.significance_level, 1))[0]]\ndec05 = 'Rechazar' if ad.statistic > crit05 else 'No rechazar'\ndec01 = 'Rechazar' if ad.statistic > crit01 else 'No rechazar'",
      alternatives: ["ad = stats.anderson(x, dist='norm')\nniveles = list(ad.significance_level)\ncrit05 = ad.critical_values[niveles.index(5.0)]\ncrit01 = ad.critical_values[niveles.index(1.0)]\ndec05 = 'Rechazar' if ad.statistic > crit05 else 'No rechazar'\ndec01 = 'Rechazar' if ad.statistic > crit01 else 'No rechazar'"],
      checks: "c.same('crit05', rtol=1e-9)\nc.same('crit01', rtol=1e-9)\nc.same('dec05')\nc.same('dec01')",
      hints: ['Mira `ad.significance_level`: [15, 10, 5, 2.5, 1].', 'Se rechaza si el estadístico **supera** el valor crítico.'],
      explanation: 'A² = 1.011: supera 0.740 (5 %) → Rechazar; no supera 1.019 (1 %) → No rechazar. Shapiro-Wilk no rechaza al 5 % (p = 0.0547): los dos tests no siempre coinciden cerca del umbral.',
    },
    // ---------------------------------------------------------- nivel 5
    {
      kind: 'code',
      level: 5,
      id: 'lab0_3.mem.ci',
      title: 'Q6 de memoria: tabla de IC de las tres especies',
      skills: ['escritura', 'api'],
      blocks: ['b_q6', 'b_q6_table'],
      source: { ...guide('Q6'), notebook: NB, cells: [3, 6] },
      prompt: 'Sin mirar: para `sepal_length`, calcula la media de cada especie y sus IC al 95 %, 99 % y 99.9 % (versión manual o con `t.interval`, como prefieras).',
      available: '`iris` (DataFrame), `np`, `stats`. Orden de especies: setosa, versicolor, virginica.',
      produce: '`medias` (3,) y `ic` array (3, 3, 2): especie × nivel (0.95, 0.99, 0.999) × (inferior, superior).',
      setup: IRIS,
      data: ['iris.csv'],
      starter: '',
      solution: "niveles = [0.95, 0.99, 0.999]\nespecies = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']\nmedias = np.array([iris[iris.species == e].sepal_length.mean() for e in especies])\nic = np.array([[stats.t.interval(c, 49, loc=iris[iris.species == e].sepal_length.mean(), scale=stats.sem(iris[iris.species == e].sepal_length)) for c in niveles] for e in especies])",
      alternatives: [
        "medias, ic = [], []\nfor especie, grupo in iris.groupby('species', sort=False):\n    x = grupo['sepal_length'].to_numpy()\n    n, m, s = len(x), x.mean(), x.std(ddof=1)\n    medias.append(m)\n    fila = []\n    for c in [0.95, 0.99, 0.999]:\n        t = stats.t.ppf(1 - (1 - c) / 2, n - 1)\n        fila.append([m - t * s / np.sqrt(n), m + t * s / np.sqrt(n)])\n    ic.append(fila)\nmedias, ic = np.array(medias), np.array(ic)",
      ],
      wrong: [{ code: "medias, ic = [], []\nfor especie, grupo in iris.groupby('species', sort=False):\n    x = grupo['sepal_length'].to_numpy()\n    n, m, s = len(x), x.mean(), x.std()\n    medias.append(m)\n    ic.append([[m - stats.t.ppf(1 - (1 - c) / 2, n - 1) * s / np.sqrt(n), m + stats.t.ppf(1 - (1 - c) / 2, n - 1) * s / np.sqrt(n)] for c in [0.95, 0.99, 0.999]])\nmedias, ic = np.array(medias), np.array(ic)", expect: 'razonamiento' }],
      checks: "c.same('medias', rtol=1e-10)\nc.shape('ic', (3, 3, 2))\nc.same('ic', rtol=1e-9, desc='Los intervalos')",
      hints: ['Para cada especie: n = 50, media, s con ddof = 1.', 't* = `stats.t.ppf(1 - (1 - c)/2, n - 1)` y semiancho t*·s/√n; o directamente `stats.t.interval`.', 'Apila los resultados con `np.array` en el orden especie × nivel × (inf, sup).'],
      explanation: 'Si usaste `x.std()` de NumPy sin ddof, tu IC sería un 1 % más estrecho: es el fallo más frecuente (la guía dice «use Numpy mean and std»).',
    },
    {
      kind: 'code',
      level: 5,
      id: 'lab0_3.mem.sw',
      title: 'Q7 de memoria: Shapiro-Wilk para las 12 combinaciones',
      skills: ['escritura', 'api'],
      blocks: ['b_q7'],
      source: { ...guide('Q7'), notebook: NB, cells: [5] },
      prompt: 'Sin mirar: aplica Shapiro-Wilk a cada especie y feature y decide si se rechaza la normalidad a α = 0.05.',
      available: '`iris`, `features` (4 nombres), `np`, `stats`, `pd`.',
      produce: '`pvalores` array (3, 4) (filas: setosa, versicolor, virginica; columnas: `features`) y `rechazo05` array booleano (3, 4).',
      setup: IRIS,
      data: ['iris.csv'],
      starter: '',
      solution: "especies = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']\npvalores = np.array([[stats.shapiro(iris[iris.species == e][f]).pvalue for f in features] for e in especies])\nrechazo05 = pvalores < 0.05",
      alternatives: ["pvalores = np.zeros((3, 4))\nfor i, (esp, g) in enumerate(iris.groupby('species', sort=False)):\n    for j, f in enumerate(features):\n        W, p = stats.shapiro(g[f])\n        pvalores[i, j] = p\nrechazo05 = pvalores < 0.05"],
      wrong: [{ code: "especies = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']\npvalores = np.array([[stats.shapiro(iris[iris.species == e][f]).pvalue for f in features] for e in especies])\nrechazo05 = pvalores > 0.05", expect: 'razonamiento' }],
      checks: "c.same('pvalores', rtol=1e-6)\nc.same('rechazo05', desc='Las decisiones')",
      hints: ['`stats.shapiro(x)` devuelve (statistic, pvalue).', 'Dos bucles: especies (filas) y features (columnas).', 'Se rechaza H₀ cuando p < α.'],
      explanation: 'Se rechaza en setosa·petal_width (p ≈ 2·10⁻⁶) y versicolor·petal_width (p = 0.027).',
    },
    // ---------------------------------------------------------- nivel 6
    {
      kind: 'code',
      level: 6,
      id: 'lab0_3.adapt.wine',
      title: 'Adapta Q6 a otro dataset (wine)',
      skills: ['metodo', 'escritura'],
      blocks: ['b_q6'],
      source: { origin: 'app', note: 'Transferencia: dataset wine de scikit-learn (incluido, sin descarga)' },
      transfer: true,
      prompt: 'Con el dataset **wine** de scikit-learn (3 cultivares, 13 features), calcula para la feature `alcohol` la media y el IC al 99 % de cada clase. Ojo: las clases no tienen el mismo tamaño.',
      available: '`wine = load_wine(as_frame=True).frame` con columnas de features y `target` (0, 1, 2); `np`, `stats`.',
      produce: '`n_clase` (3,), `ic99` (3, 2) en orden de clase 0, 1, 2.',
      setup: 'import numpy as np\nfrom scipy import stats\nfrom sklearn.datasets import load_wine\nwine = load_wine(as_frame=True).frame',
      starter: '',
      solution: "n_clase = np.array([(wine.target == k).sum() for k in range(3)])\nic99 = np.array([stats.t.interval(0.99, n - 1, loc=wine.alcohol[wine.target == k].mean(), scale=stats.sem(wine.alcohol[wine.target == k])) for k, n in zip(range(3), n_clase)])",
      alternatives: ["g = wine.groupby('target')['alcohol']\nn_clase = g.size().to_numpy()\nic99 = np.array([stats.t.interval(0.99, len(x) - 1, loc=x.mean(), scale=stats.sem(x)) for _, x in g])"],
      wrong: [{ code: "n_clase = np.array([(wine.target == k).sum() for k in range(3)])\nic99 = np.array([stats.t.interval(0.99, 49, loc=wine.alcohol[wine.target == k].mean(), scale=stats.sem(wine.alcohol[wine.target == k])) for k in range(3)])", expect: 'razonamiento' }],
      checks: "c.same('n_clase')\nc.same('ic99', rtol=1e-9)",
      hints: ['Los grados de libertad dependen del tamaño de cada clase (59, 71 y 48).', 'Reutiliza `stats.t.interval` con `len(x) - 1`.'],
      explanation: 'Copiar «49 grados de libertad» de Iris es el error típico al transferir: df = n − 1 de cada grupo.',
    },
    // ---------------------------------------------------------- interpretación
    {
      kind: 'mcq',
      id: 'lab0_3.int.notreject',
      skills: ['interpretacion', 'concepto'],
      blocks: ['b_q7'],
      source: { ...guide('Q7'), notebook: NB, cells: [4, 5] },
      prompt: 'Shapiro-Wilk para setosa · sepal_length da p = 0.46. ¿Qué interpretación contiene un **error conceptual**?',
      options: [
        { text: '«Queda demostrado que sepal_length de setosa es gaussiana»', correct: true, feedback: 'Correcto, es la errónea: no rechazar H₀ no la demuestra; solo no hay evidencia en contra con 50 datos.' },
        { text: '«Con estos datos no hay evidencia para rechazar la normalidad a α = 0.05»' },
        { text: '«El supuesto gaussiano de LDA/QDA es razonable para esta feature y clase»' },
        { text: '«Si H₀ fuera cierta, un W tan bajo o más saldría el 46 % de las veces»' },
      ],
      explanation: 'Un contraste solo puede rechazar o no rechazar. Con muestras pequeñas la potencia es baja; con muestras muy grandes se rechazan desviaciones irrelevantes.',
    },
    {
      kind: 'mcq',
      id: 'lab0_3.int.disagree',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_q7', 'b_q7_tables'],
      source: sol([5, 8]),
      prompt: 'Setosa · petal_length: SW p = 0.0547 (no rechaza al 5 %) y AD A² = 1.011 > 0.740 (rechaza al 5 %). ¿Cómo lo explicas?',
      options: [
        { text: 'Están cerca del umbral y miden la desviación de forma distinta (AD pesa más las colas): conviene mirar el Q-Q plot', correct: true },
        { text: 'Uno de los dos está mal calculado', feedback: 'Ambos coinciden con la salida guardada.' },
        { text: 'Anderson-Darling siempre rechaza más', feedback: 'No en general: depende de dónde está la desviación.' },
        { text: 'Hay que elegir el que dé el resultado deseado', feedback: 'Eso sería un mal uso de los contrastes.' },
      ],
      explanation: 'Con p ≈ α la decisión es frágil. Petal_length de setosa toma pocos valores distintos (datos redondeados a 0.1 cm), lo que afecta a los dos tests de forma diferente.',
    },
    {
      kind: 'numeric',
      id: 'lab0_3.int.multiple',
      skills: ['concepto', 'metodo'],
      blocks: ['b_q7'],
      source: sol([4], { note: '«sin corrección múltiple»' }),
      prompt: 'Si las 12 combinaciones fueran realmente gaussianas e independientes, ¿cuántos rechazos falsos esperarías en promedio con α = 0.05?',
      answer: 0.6,
      tolerance: 0.01,
      explanation: '12 × 0.05 = 0.6. Por eso el notebook advierte que las decisiones no tienen corrección por comparaciones múltiples (p. ej. Bonferroni: α/12).',
    },
    {
      kind: 'text',
      id: 'lab0_3.int.q7',
      skills: ['interpretacion', 'concepto'],
      blocks: ['b_q7', 'b_q7_tables'],
      source: { ...guide('Q7'), notebook: NB, cells: [7, 8] },
      prompt: 'Q7 · Explica el significado del p-valor y del valor crítico e interpreta los resultados de las dos tablas.',
      rubric: [
        'p-valor: probabilidad bajo H₀ de un estadístico al menos tan extremo como el observado; se rechaza si p < α.',
        'Valor crítico: umbral del estadístico para el nivel α; en AD se rechaza si A² > crítico.',
        'Setosa·petal_width se rechaza con ambos tests a los dos niveles.',
        'Versicolor·petal_width se rechaza al 5 % y no al 1 %; setosa·petal_length divide a los dos tests al 5 %.',
        'No rechazar no prueba la normalidad; 12 contrastes sin corrección múltiple.',
      ],
      model: 'El p-valor es la probabilidad, suponiendo que la feature es gaussiana, de observar un estadístico tan extremo o más que el obtenido; si es menor que α se rechaza H₀. El valor crítico es el umbral del estadístico que deja una probabilidad α en la cola de rechazo; en Anderson-Darling se rechaza si A² lo supera (0.740 al 5 % y 1.019 al 1 % con n = 50). Las tablas muestran que la mayoría de combinaciones no se rechazan; setosa·petal_width se rechaza con claridad (W = 0.814, A² = 4.31), versicolor·petal_width solo al 5 % y setosa·petal_length solo con AD al 5 %. No rechazar no demuestra normalidad, y con 12 contrastes cabe esperar algún rechazo por azar.',
    },
    {
      kind: 'dynamic',
      id: 'lab0_3.dyn.ci',
      skills: ['api', 'interpretacion'],
      blocks: ['b_q6'],
      data: ['iris.csv'],
      source: sol([3], { note: 'Variante generada' }),
      answerKind: 'numeric',
      generator: `import numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(SEED)
iris = pd.read_csv('iris.csv')
f = str(rng.choice(list(iris.columns[:-1])))
e = str(rng.choice(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
c = float(rng.choice([0.9, 0.95, 0.99]))
lado = str(rng.choice(['inferior', 'superior']))
x = iris[iris.species == e][f]
a, b = stats.t.interval(c, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
val = a if lado == 'inferior' else b
variante = {
  'prompt': f'Para {f} de {e}, ¿cuál es el extremo **{lado}** del IC al {int(c*100)} % de la media? (3 decimales). Calcúlalo como harías en el examen.',
  'answer': round(float(val), 6), 'tolerance': 0.0015,
  'explanation': f'media = {x.mean():.4f}, s = {x.std(ddof=1):.4f}, n = 50, t* = {stats.t.ppf(1-(1-c)/2, 49):.4f} → IC = [{a:.4f}, {b:.4f}].'
}`,
    },
  ],
  recall: ['lab0_3.fill.load', 'lab0_3.fill.interval', 'lab0_3.fix.manual', 'lab0_3.fix.ad', 'lab0_3.mem.ci', 'lab0_3.mem.sw', 'lab0_3.adapt.wine'],
  practice: [
    {
      id: 'lab0_3.practice.q6q7',
      title: 'Reconstruye Q6 y Q7 en una hoja en blanco',
      scenario: 'Te sientas delante de un notebook vacío con `iris.csv`. Carga los datos, calcula la tabla de Q6 (manual y con la función deben coincidir), realiza los 12 contrastes de Q7 y redacta la interpretación de los p-valores y valores críticos.',
      steps: ['lab0_3.fill.load', 'lab0_3.mem.ci', 'lab0_3.read.ci', 'lab0_3.mem.sw', 'lab0_3.fix.ad', 'lab0_3.int.notreject', 'lab0_3.int.q7'],
      source: { ...guide('Q6–Q7') },
    },
  ],
  notes: [
    'Este notebook es tu solución: la app la usa para validar los ejercicios (las salidas guardadas coinciden con las de la app) y mantiene su código oculto hasta que intentas cada ejercicio.',
  ],
};
