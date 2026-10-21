import type { NotebookContent } from '../../../src/content/types';

const NB = 'lab2_3';
const DATA = ['BD_phoneme.csv'];
const src = (cells: number[], extra: Partial<{ section: string; note: string; doc: string }> = {}) => ({
  notebook: NB,
  cells,
  origin: 'original' as const,
  ...extra,
});

/** Lectura y centrado (celdas 4 y 7), sin recortar features. */
const SETUP_RAW = `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv("BD_phoneme.csv", header=None)
X = np.array(df.iloc[:, 0:256])
y = np.array(df.iloc[:, 256])
X = X - X.mean(axis=0)
`;

/** Estado tras la celda 13: 64 primeras features y partición estratificada. */
const SETUP_SPLIT = `${SETUP_RAW}X = X[:, np.array(range(0, 64))]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=5, shuffle=True, stratify=y)
`;

const EXP_MANUAL = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# celdas 4 y 7 · lectura y centrado (broadcasting)
df = pd.read_csv("BD_phoneme.csv", header=None)
X = np.array(df.iloc[:,0:256])
y = np.array(df.iloc[:,256])
Xm = X.mean(axis=0)
X = X - Xm
nfft = 256
classes = ['aa','ao','dc','iy','sh']
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']   # (app) paleta fija por clase

# celda 17 · selección MANUAL de dos features (columnas originales)
V_coor = {{V_coor}}
Xall = X                                   # (app) conservamos las 256 para el espectro
X = X[:,V_coor]

# celdas 13 y 17 · partición estratificada
trainsize= {{trainsize}}
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= trainsize, random_state={{seed}}, shuffle=True, stratify = y)

lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True)
ldamodel = lda.fit(X_train, y_train)
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qdamodel = qda.fit(X_train, y_train)
lda_train_error = 1. - accuracy_score(y_train, ldamodel.predict(X_train))
lda_test_error = 1. - accuracy_score(y_test, ldamodel.predict(X_test))
qda_train_error = 1. - accuracy_score(y_train, qdamodel.predict(X_train))
qda_test_error = 1. - accuracy_score(y_test, qdamodel.predict(X_test))
print('LDA train error: %f' %lda_train_error)
print('LDA test error: %f' %lda_test_error)
print('QDA train error: %f' %qda_train_error)
print('QDA test error: %f' %qda_test_error)

# (app) espectro medio por clase (celda 9) marcando las features elegidas
fmax = 8*nfft/256
freqax = np.array(range(0,nfft))*fmax/nfft
plt.figure(figsize=(9,3.5))
for idc, nc in enumerate(classes):
  plt.plot(np.array(range(0,nfft)), Xall[y==idc+1].mean(axis=0), color=colors[idc], label=nc)
for v in V_coor:
  plt.axvline(v, color='k', linestyle='--', linewidth=1)
plt.xlim(0, {{xmax}})
plt.xlabel('feature (frecuencia = feature · 8/256 kHz)')
plt.ylabel('media (centrada)')
plt.legend()
plt.title('Espectro medio por clase y features seleccionadas')
plt.show()

# celda 21 · regiones de decisión con las dos features
h = {{h}} # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
cmap = ListedColormap(colors)              # (app) mismas clases, mismos colores en puntos y regiones
fig, ax = plt.subplots(figsize=(14,7),nrows=1, ncols=2)
for k, (model, name) in enumerate([(lda, 'LDA boundaries'), (qda, 'QDA boundaries')]):
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
  ax[k].contourf(xx, yy, Z, levels=np.arange(0.5, 6), cmap=cmap, alpha=0.2)
  for idc, nc in enumerate(classes):
    idx = y== (idc+1)
    ax[k].scatter(X[idx,0], X[idx,1], color = colors[idc], label=nc, alpha=0.5, s=8)
  ax[k].set_xlim(xx.min(), xx.max())
  ax[k].set_ylim(yy.min(), yy.max())
  ax[k].set_xlabel('feature %d' % V_coor[0])
  ax[k].set_ylabel('feature %d' % V_coor[1])
  ax[k].set_title(name)
ax[0].legend()
plt.show()

cm_lda = confusion_matrix(y_test, ldamodel.predict(X_test))
resultados = {
  'lda_train': lda_train_error, 'lda_test': lda_test_error,
  'qda_train': qda_train_error, 'qda_test': qda_test_error,
  'cm_lda': cm_lda, 'cm_qda': confusion_matrix(y_test, qdamodel.predict(X_test)),
  'aa_predichas_lda': int(cm_lda[:, 0].sum()),
  'f1_khz': V_coor[0] * 8 / 256, 'f2_khz': V_coor[1] * 8 / 256,
}
`;

const EXP_NFIRST = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# celdas 4 y 7 · lectura y centrado
df = pd.read_csv("BD_phoneme.csv", header=None)
X = np.array(df.iloc[:,0:256])
y = np.array(df.iloc[:,256])
Xm = X.mean(axis=0)
X = X - Xm

# celda 11 · selección de las primeras features (un subconjunto: frecuencias bajas)
V_coor = np.array(range(0,{{nfirst}}))
nfeat  = len(V_coor)
X = X[:,V_coor]
print(X.shape)

# celda 13 · partición
trainsize= {{trainsize}}
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= trainsize, random_state={{seed}}, shuffle=True, stratify = y)

# celda 15 · LDA y QDA
lda = LinearDiscriminantAnalysis(solver="svd",store_covariance=True)
ldamodel = lda.fit(X_train, y_train)
qda = QuadraticDiscriminantAnalysis(store_covariance=True, reg_param={{reg}})
qdamodel = qda.fit(X_train, y_train)
lda_train_error = 1. - accuracy_score(y_train, ldamodel.predict(X_train))
lda_test_error = 1. - accuracy_score(y_test, ldamodel.predict(X_test))
qda_train_error = 1. - accuracy_score(y_train, qdamodel.predict(X_train))
qda_test_error = 1. - accuracy_score(y_test, qdamodel.predict(X_test))
print('LDA train error: %f' %lda_train_error)
print('LDA test error: %f' %lda_test_error)
print('QDA train error: %f' %qda_train_error)
print('QDA test error: %f' %qda_test_error)

# (app) resumen visual train vs test
fig, ax = plt.subplots(figsize=(6,3.5))
ax.bar([0, 1, 3, 4], [lda_train_error, lda_test_error, qda_train_error, qda_test_error],
       color=['tab:blue', 'tab:cyan', 'tab:orange', 'gold'])
ax.set_xticks([0, 1, 3, 4])
ax.set_xticklabels(['LDA train', 'LDA test', 'QDA train', 'QDA test'])
ax.set_ylabel('error')
ax.set_title('%d features · train %d / test %d muestras' % (nfeat, len(y_train), len(y_test)))
plt.show()

resultados = {
  'nfeat': nfeat, 'n_train': len(y_train),
  'lda_train': lda_train_error, 'lda_test': lda_test_error,
  'qda_train': qda_train_error, 'qda_test': qda_test_error,
  'gap_qda': qda_test_error - qda_train_error,
  'param_lda': 5 * nfeat + nfeat * (nfeat + 1) // 2,
  'param_qda': 5 * nfeat + 5 * nfeat * (nfeat + 1) // 2,
  'min_train_clase': int(np.bincount(y_train)[1:].min()),
  'cm_lda': confusion_matrix(y_test, ldamodel.predict(X_test)),
  'cm_qda': confusion_matrix(y_test, qdamodel.predict(X_test)),
}
`;

export const lab2_3: NotebookContent = {
  id: NB,
  lab: 'lab2',
  title: 'Lab 2 · Parte 3 — Phoneme: todas las features frente a una selección manual',
  short: 'Phoneme: LDA/QDA y selección manual',
  origin: 'original',
  context: {
    goal: `Pasar de datos sintéticos a un **problema real**: clasificar fotogramas de voz en 5 fonemas a partir de su log-periodograma. El notebook centra los datos, dibuja los espectros por clase, se queda con las **64 primeras** features (0–2 kHz), parte en train/test estratificado (70/30), y compara LDA y QDA. Después repite el análisis con **solo dos features elegidas a mano** (\`V_coor\`) y dibuja las fronteras.

Es el lab donde aparece la **selección de características en sentido estricto**: \`V_coor\` elige columnas originales (un subconjunto), a diferencia de PCA/MDA (parte 2 y notebook 4), que construyen features nuevas. La guía lo plantea así: «suponga que por coste computacional solo podemos usar dos features por observación».

Preguntas de la guía/report (Lab2): Q5 espectros, Q6 errores con todas las features, Q7 matrices de confusión en test, Q8 qué dos features elegir y comparar, Q9 scatter y fronteras.`,
    dataset: `**Phoneme** (Hastie, Buja y Tibshirani, 1995; TIMIT): 4509 fotogramas de 32 ms de 50 hablantes, cada uno descrito por un log-periodograma de **256** valores (0–8 kHz, 31.25 Hz por feature). Archivo \`BD_phoneme.csv\` **sin cabecera**: columnas 0–255 = features, columna 256 = clase 1–5. Clases (guía, p. 3): 1 = **aa** (695, vocal de *dark*), 2 = **ao** (1022, primera vocal de *water*), 3 = **dcl** (757, *dark*; el notebook la escribe «dc»), 4 = **iy** (1163, vocal de *she*), 5 = **sh** (872, *she*). Las clases están desequilibradas.`,
    variables: [
      { name: 'df', meaning: 'DataFrame leído con header=None: columnas numeradas 0–256.', shape: '(4509, 257)' },
      { name: 'X', meaning: 'Matriz de features (se centra y luego se recorta a 64 y a 2 columnas).', shape: '(4509, 256) → (4509, 64) → (4509, 2)' },
      { name: 'y', meaning: 'Etiquetas 1–5 (no empiezan en 0).', shape: '(4509,)' },
      { name: 'Xm', meaning: 'Media de cada feature (vector).', shape: '(256,)' },
      { name: 'xdm', meaning: 'Espectro medio de cada clase.', shape: '(5, 256)' },
      { name: 'V_coor', meaning: 'Índices de las features seleccionadas.', shape: '(64,) o lista de 2' },
      { name: 'X_train, X_test', meaning: 'Partición estratificada 70/30 (random_state=5).', shape: '(3156, 64), (1353, 64)' },
      { name: 'ldamodel, qdamodel', meaning: 'Clasificadores entrenados.', shape: '—' },
    ],
    classes: [
      { id: '1', label: 'aa (1)' },
      { id: '2', label: 'ao (2)' },
      { id: '3', label: 'dcl / «dc» (3)' },
      { id: '4', label: 'iy (4)' },
      { id: '5', label: 'sh (5)' },
    ],
    questions: [
      { id: 'Q5', text: 'Incluir los espectros de los fonemas.', blocks: ['b_spectrum'], doc: 'lab2_guide' },
      { id: 'Q6', text: 'Errores de train y test de LC y QC con todas las features (las 64 primeras).', blocks: ['b_classify64'], doc: 'lab2_guide' },
      { id: 'Q7', text: 'Matrices de confusión en test de LC y QC con todas las features.', blocks: ['b_classify64'], doc: 'lab2_guide' },
      { id: 'Q8', text: '¿Qué dos features elegirías? Errores de train y test y comparación con el caso anterior.', blocks: ['b_manual'], doc: 'lab2_guide' },
      { id: 'Q9', text: 'Scatter plot y fronteras de decisión con dos features.', blocks: ['b_lines', 'b_regions'], doc: 'lab2_guide' },
    ],
    notation: [
      { symbol: '$d$', meaning: 'número de features (256, 64 o 2)', code: 'nfeat' },
      { symbol: '$c = 5$', meaning: 'número de clases', code: 'nclass' },
      { symbol: '$f_k = k \\cdot 8/256$ kHz', meaning: 'frecuencia de la feature k', code: 'freqax' },
      { symbol: '$\\log|TF\\{x[n]\\}|^2$', meaning: 'log-periodograma de un fotograma (slides Lab2, p. 12)' },
      { symbol: '$x - \\bar x$', meaning: 'centrado por feature (broadcasting)', code: 'X - Xm' },
      { symbol: '$\\{x_{10}, x_{30}\\}$', meaning: 'subconjunto elegido a mano', code: 'V_coor = [10, 30]' },
    ],
  },
  flow: [
    { stage: 'contexto', label: 'Contexto', blocks: ['b_intro', 'b_imports'], summary: 'Problema real de 5 clases; librerías y train_test_split.' },
    { stage: 'datos', label: 'Datos', blocks: ['b_read'], summary: 'CSV sin cabecera: 4509 × 257.' },
    { stage: 'exploracion', label: 'Exploración', blocks: ['b_spectrum'], summary: 'Espectros por clase y espectro medio: ¿qué frecuencias distinguen?' },
    { stage: 'preparacion', label: 'Preparación', blocks: ['b_prepare', 'b_first64', 'b_split'], summary: 'Centrado, 64 primeras features, partición estratificada.' },
    { stage: 'metodo', label: 'Método', blocks: ['b_classify64', 'b_manual'], summary: 'LDA/QDA con 64 features y con 2 elegidas a mano.' },
    { stage: 'resultados', label: 'Resultados', blocks: ['b_lines', 'b_regions'], summary: 'Fronteras en el plano de las dos features.' },
    { stage: 'interpretacion', label: 'Interpretación', blocks: [], summary: 'Q6–Q9: sobreajuste de QDA, confusión aa/ao, coste de quedarse con 2 features.' },
  ],
  blocks: [
    {
      id: 'b_intro',
      title: 'Título: Phoneme, todas las features o un subconjunto',
      cells: [0, 1],
      stage: 'contexto',
      problem: 'Situar el experimento: clasificar con todas las features disponibles o con un subconjunto elegido manualmente.',
      purpose: 'Anuncia las dos mitades del notebook (Q6–Q7 con todas las features; Q8–Q9 con dos). El título «Feature selection using PCA and MDA» es el del lab completo; aquí todavía no hay PCA ni MDA.',
      inputs: 'Texto.',
      outputs: 'El plan del notebook.',
      prereqs: ['seleccion-features', 'extraccion-features'],
      apis: [],
      lines: [],
      interpretation: '«A manually selected subset of features» es **selección** en sentido estricto: se conservan columnas originales con su significado físico (frecuencias). PCA/MDA (notebook 2 y notebook 4) son **extracción**: crean combinaciones lineales nuevas.',
      next: 'Imports.',
      whatIf: [],
      check: {
        kind: 'mcq',
        id: 'lab2_3.check.intro',
        skills: ['concepto', 'metodo'],
        source: src([1]),
        prompt: '`V_coor = [10, 30]` y después `X = X[:, V_coor]`. ¿Qué tipo de reducción es?',
        options: [
          { text: 'Selección de características: se quedan dos columnas originales', correct: true },
          { text: 'Extracción: se construyen dos combinaciones lineales de las 64 features', feedback: 'Eso sería PCA o MDA (`transform`), no indexar columnas.' },
          { text: 'Normalización de las features', feedback: 'Indexar no cambia los valores, solo cuántas columnas hay.' },
        ],
        explanation: 'Indexar columnas elige un subconjunto: cada feature conserva su significado (la frecuencia 10·31.25 Hz ≈ 0.31 kHz).',
      },
    },
    {
      id: 'b_imports',
      title: 'Importación de librerías',
      cells: [2],
      stage: 'contexto',
      problem: 'Disponer de pandas, NumPy, matplotlib y las herramientas de scikit-learn.',
      purpose: 'Novedad respecto a Lab1: `train_test_split`, porque ahora los datos son **fijos** y no se puede generar un test nuevo: hay que reservar una parte.',
      inputs: 'Nada.',
      outputs: 'Módulos y clases importados.',
      prereqs: ['train-test'],
      apis: [
        { name: 'train_test_split', lib: 'sklearn.model_selection', kind: 'función', does: 'Parte arrays en train y test de forma aleatoria (opcionalmente estratificada).' },
        { name: 'LinearDiscriminantAnalysis / QuadraticDiscriminantAnalysis', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Los mismos clasificadores MAP de Lab1.' },
        { name: 'accuracy_score, confusion_matrix', lib: 'sklearn.metrics', kind: 'función', does: 'Tasa de acierto y matriz de confusión.' },
      ],
      lines: [
        { cell: 2, lines: [3, 4], text: '`seaborn` y `scipy.stats` se importan pero no se usan en este notebook.' },
        { cell: 2, lines: [6, 6], text: '`train_test_split` está en `model_selection`, no en `metrics` ni en `datasets`.' },
      ],
      interpretation: 'Sin generador de datos, la única forma honesta de estimar el error en datos nuevos es separar un test **antes** de entrenar.',
      next: 'Lectura del CSV.',
      whatIf: [{ change: 'Olvidar importar `train_test_split`', effect: '`NameError` en la celda 13 (error de API/import).' }],
    },
    {
      id: 'b_read',
      title: 'Lectura del dataset (CSV sin cabecera)',
      cells: [3, 4, 5],
      stage: 'datos',
      data: DATA,
      problem: 'Cargar la tabla de log-periodogramas.',
      purpose: 'El archivo no tiene fila de nombres: con `header=None` pandas numera las columnas 0–256 y no se pierde la primera muestra.',
      inputs: '`BD_phoneme.csv` (9.8 MB).',
      outputs: '`df` (4509, 257): 256 columnas float y la etiqueta entera en la columna 256.',
      prereqs: ['dataframe'],
      apis: [
        { name: 'pd.read_csv', lib: 'pandas', kind: 'función', does: 'Lee un CSV a DataFrame.', args: [{ name: 'header', value: 'None', meaning: 'no hay fila de cabecera; columnas 0..256' }] },
        { name: 'DataFrame.info', lib: 'pandas', kind: 'método', does: 'Resumen: filas, columnas, tipos y memoria. Imprime y devuelve None (por eso aparece «None»).' },
        { name: 'DataFrame.head', lib: 'pandas', kind: 'método', does: 'Primeras 5 filas.' },
      ],
      lines: [
        { cell: 4, lines: [1, 1], text: 'Sin `header=None`, la primera fila de números se tomaría como nombres de columna: quedarían 4508 muestras y columnas llamadas «9.8577»…' },
        { cell: 4, lines: [2, 2], text: '`print(df.info())`: `info()` ya imprime; su valor devuelto es `None`, que también se imprime.' },
      ],
      shapes: [{ expr: 'df', shape: '(4509, 257)', meaning: '256 features + 1 etiqueta' }],
      interpretation: '`dtypes: float64(256), int64(1)`: la última columna es la clase (entero). No hay valores nulos. La columna «speaker» del dataset original no está en este CSV.',
      versionNotes: [{ cell: 4, text: 'La salida guardada es de una pandas antigua (`<class \'pandas.core.frame.DataFrame\'>`); con pandas 3 la clase se imprime como `pandas.DataFrame`. El contenido (4509 filas, 257 columnas, tipos) es el mismo.' }],
      next: 'Se convierte a arrays NumPy y se centra.',
      whatIf: [{ change: 'Leer sin `header=None`', effect: 'Se pierde una muestra y las columnas tienen nombres numéricos raros; `df.iloc[:,256]` seguiría funcionando por posición.' }],
    },
    {
      id: 'b_prepare',
      title: 'Preparar X e y y centrar en media (broadcasting)',
      cells: [6, 7],
      stage: 'preparacion',
      problem: 'Separar features y etiqueta y restar la media de cada feature.',
      purpose: 'Pasa a la representación de scikit-learn (matriz X y vector y) y **centra** cada feature. Centrar no cambia los errores de LDA/QDA (son invariantes a traslaciones), pero hace los espectros comparables y es lo que haría PCA internamente.',
      inputs: '`df`.',
      outputs: '`X` (4509, 256) centrada, `y` (4509,), `Xm` (256,), `nclass = 5`, `nfft = 256`.',
      prereqs: ['centrado', 'broadcasting', 'matriz-features'],
      apis: [
        { name: 'df.iloc[:, 0:256]', lib: 'pandas', kind: 'operador', does: 'Selección por posición: todas las filas, columnas 0–255.' },
        { name: 'np.array', lib: 'numpy', kind: 'función', does: 'Convierte el DataFrame/Series a ndarray.' },
        { name: 'X.mean(axis=0)', lib: 'numpy', kind: 'método', does: 'Media de cada **columna** (feature).', returns: '(256,)' },
      ],
      lines: [
        { cell: 7, lines: [1, 4], text: 'X = columnas 0–255; y = columna 256. Imprime (4509, 256) y (4509,).' },
        { cell: 7, lines: [7, 7], text: '`axis=0` recorre las filas: una media por columna → (256,). Con `axis=1` sería una media por muestra (4509,).' },
        { cell: 7, lines: [8, 8], text: '**Broadcasting**: (4509, 256) − (256,) resta el vector a cada fila. Ahora cada columna tiene media 0.' },
      ],
      shapes: [
        { expr: 'X', shape: '(4509, 256)', meaning: 'muestras × frecuencias' },
        { expr: 'Xm', shape: '(256,)', meaning: 'media por feature' },
        { expr: 'y', shape: '(4509,)', meaning: 'clases 1..5' },
      ],
      interpretation: 'Se centra con la media de **todo** el dataset (train + test). Para una traslación no tiene efecto en LDA/QDA; con transformaciones que aprenden de los datos (PCA, escalado) habría que calcularlas solo con train.',
      next: 'Espectros por clase.',
      whatIf: [
        { change: 'Usar `X.mean(axis=1)`', effect: '`ValueError` de broadcasting: (4509, 256) − (4509,) no encaja (dimensiones).' },
        { change: 'No centrar', effect: 'Mismos errores de LDA/QDA; los espectros se dibujan en valores absolutos (positivos) en vez de alrededor de 0.' },
      ],
    },
    {
      id: 'b_spectrum',
      title: 'Log-periodogramas por clase y espectro medio (Q5)',
      cells: [8, 9],
      stage: 'exploracion',
      problem: 'Ver qué frecuencias distinguen a cada fonema.',
      purpose: 'Es la exploración que justifica la selección manual de Q8: buscar frecuencias donde los espectros **medios** de las clases están separados.',
      inputs: '`X`, `y`, `nfft`.',
      outputs: 'Figura 3×2: un panel por clase con todos sus espectros y un sexto panel con las 5 medias (`xdm`).',
      prereqs: ['log-periodograma', 'media'],
      apis: [
        { name: 'plt.plot', lib: 'matplotlib', kind: 'función', does: 'Una curva por fotograma (miles de curvas: tarda ~9 s en la app).' },
        { name: 'xd.mean(axis=0)', lib: 'numpy', kind: 'método', does: 'Espectro medio de la clase.' },
      ],
      lines: [
        { cell: 9, lines: [1, 2], text: 'Nombres y colores de las clases. «dc» es el fonema **dcl** de la guía.' },
        { cell: 9, lines: [4, 5], text: 'Eje de frecuencias: `fmax = 8` kHz; la feature k está a k·8/256 kHz (31.25 Hz por feature).' },
        { cell: 9, lines: [11, 13], text: 'Las clases son 1..5, por eso `y == idc+1`. `xdm[idc]` guarda la media de la clase.' },
        { cell: 9, lines: [15, 16], text: 'Dibuja **cada** fotograma de la clase (cientos de curvas superpuestas).' },
        { cell: 9, lines: [20, 22], text: 'Panel 6: las 5 medias. El eje x está en **número de feature** (0–255), no en kHz.' },
      ],
      shapes: [{ expr: 'xdm', shape: '(5, 256)', meaning: 'espectro medio por clase' }],
      interpretation: 'Las vocales **aa** y **ao** tienen espectros medios muy parecidos (máximos de energía en frecuencias bajas, < 1 kHz): será la confusión principal. **sh** concentra energía en frecuencias altas (sibilante) y **dcl** tiene poca energía: son fáciles. Las mayores diferencias entre clases están en las primeras ~64 features (0–2 kHz), lo que justifica el recorte de la celda 11 (slides, p. 12).',
      next: 'Se seleccionan las 64 primeras features.',
      whatIf: [{ change: 'Dibujar solo las medias', effect: 'Mucho más rápido; se pierde la variabilidad dentro de cada clase, que es la que decide el solapamiento.' }],
    },
    {
      id: 'b_first64',
      title: 'Selección de las 64 primeras features',
      cells: [10, 11],
      stage: 'preparacion',
      problem: 'Reducir de 256 a 64 features.',
      purpose: 'Primera **selección de características**: quedarse con las frecuencias de 0 a 2 kHz (las slides dicen que en Lab2 se trabaja con estas 64). Reduce el número de parámetros que tiene que estimar QDA.',
      inputs: '`X` (4509, 256).',
      outputs: '`V_coor` (64,), `nfeat = 64`, `X` (4509, 64).',
      prereqs: ['seleccion-features', 'maldicion-dimension'],
      apis: [{ name: 'X[:, V_coor]', lib: 'numpy', kind: 'operador', does: 'Indexación avanzada: todas las filas, columnas indicadas en V_coor.', returns: 'copia (4509, len(V_coor))' }],
      lines: [
        { cell: 11, lines: [2, 2], text: '`np.array(range(0,64))` = índices 0..63 (equivale a `np.arange(64)`).' },
        { cell: 11, lines: [7, 7], text: '**Sobrescribe X**: a partir de aquí X solo tiene 64 columnas.' },
      ],
      shapes: [{ expr: 'X', shape: '(4509, 64)', meaning: 'frecuencias 0–2 kHz' }],
      interpretation: 'La guía llama a estas 64 «todas las features» en Q6–Q7. Es una selección por conocimiento del dominio (frecuencias bajas), no un método automático.',
      next: 'Partición train/test.',
      whatIf: [{ change: 'Usar las 256 features', effect: 'LDA mejora poco; QDA estima 5 covarianzas 256×256 con ~490 muestras en la clase más pequeña y sobreajusta mucho más (ver notebook 4).' }],
    },
    {
      id: 'b_split',
      title: 'Partición estratificada en train y test',
      cells: [12, 13],
      stage: 'preparacion',
      problem: 'Reservar datos que el modelo no verá para estimar su error.',
      purpose: 'Separa un 70 % para entrenar y un 30 % para evaluar, barajando y manteniendo la **proporción de clases** en ambos (clases desequilibradas).',
      inputs: '`X` (4509, 64), `y`.',
      outputs: '`X_train` (3156, 64), `X_test` (1353, 64), `y_train`, `y_test`.',
      prereqs: ['train-test', 'estratificado'],
      apis: [{ name: 'train_test_split', lib: 'sklearn.model_selection', kind: 'función', does: 'Partición aleatoria.', args: [{ name: 'train_size', value: '0.7', meaning: 'proporción de train (el test es el resto)' }, { name: 'random_state', value: '5', meaning: 'semilla: misma partición en cada ejecución' }, { name: 'shuffle', value: 'True', meaning: 'barajar antes de partir' }, { name: 'stratify', value: 'y', meaning: 'mantener las proporciones de clase' }], returns: 'X_train, X_test, y_train, y_test (en ese orden)' }],
      lines: [{ cell: 13, lines: [4, 4], text: 'El orden de los valores devueltos es **X_train, X_test, y_train, y_test** (no X_train, y_train, …): error típico.' }],
      shapes: [
        { expr: 'X_train', shape: '(3156, 64)', meaning: 'floor(0.7 · 4509) muestras' },
        { expr: 'X_test', shape: '(1353, 64)', meaning: 'el resto' },
      ],
      interpretation: 'Con `stratify=y`, aa sigue siendo ≈15 % en train y test. Cambiar `random_state` da otra partición y errores algo distintos: es variabilidad de la estimación.',
      next: 'Clasificación con 64 features.',
      whatIf: [
        { change: 'Quitar `stratify=y`', effect: 'Las proporciones de clase en test pueden variar algo; con 4509 muestras el efecto es pequeño, pero con clases raras podría quedar casi vacía una clase.' },
        { change: 'Cambiar el orden de las variables al recibir el resultado', effect: 'Entrenarías con X_test o con etiquetas como features: errores absurdos o `ValueError` de dimensiones.' },
      ],
    },
    {
      id: 'b_classify64',
      title: 'LDA y QDA con 64 features: errores y matrices de confusión (Q6, Q7)',
      cells: [14, 15],
      stage: 'metodo',
      problem: 'Medir el rendimiento de los dos clasificadores en train y en test.',
      purpose: 'Responde Q6 y Q7. Compara un modelo con covarianza común (LDA, 2080 parámetros de covarianza) con otro con una covarianza por clase (QDA, 10400).',
      inputs: '`X_train`, `y_train`, `X_test`, `y_test`.',
      outputs: 'LDA: train 7.67 %, test 7.91 %. QDA: train 5.26 %, test 11.23 %. Matrices 5×5.',
      prereqs: ['lda', 'qda', 'matriz-confusion', 'sobreajuste'],
      apis: [
        { name: 'LinearDiscriminantAnalysis(solver="svd", store_covariance=True)', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Clasificador lineal gaussiano.', learned: ['priors_', 'means_', 'covariance_'] },
        { name: 'QuadraticDiscriminantAnalysis(store_covariance=True)', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Clasificador cuadrático gaussiano.', learned: ['covariance_'] },
        { name: 'confusion_matrix(y_true, y_pred)', lib: 'sklearn.metrics', kind: 'función', does: 'Filas = clase real (1..5 en orden), columnas = predicha.' },
      ],
      lines: [
        { cell: 15, lines: [3, 5], text: 'Un único `fit` con train; `predict` sobre train y sobre test.' },
        { cell: 15, lines: [8, 16], text: 'Error y matriz de confusión de train y de test para LDA.' },
        { cell: 15, lines: [19, 32], text: 'Lo mismo para QDA.' },
      ],
      shapes: [
        { expr: 'confusion_matrix(...)', shape: '(5, 5)', meaning: 'orden de filas: aa, ao, dcl, iy, sh' },
        { expr: 'ldamodel.covariance_', shape: '(64, 64)', meaning: 'covarianza común' },
        { expr: 'qdamodel.covariance_', shape: 'lista de 5 arrays (64, 64)', meaning: 'una por clase' },
      ],
      interpretation: '**LDA** generaliza bien (train ≈ test). **QDA** tiene el menor error de train pero el mayor de test: **sobreajuste**, porque estima 5 covarianzas de 64×64 (2080 parámetros cada una) con entre 487 y 814 muestras por clase en train. Casi todos los errores son **aa ↔ ao** (bloque 2×2 superior izquierdo: 51 + 47 en test para LDA), las dos vocales con espectros parecidos; dcl, iy y sh se clasifican casi perfectamente.',
      next: 'Repetir con solo dos features (Q8).',
      whatIf: [
        { change: 'Regularizar QDA (`reg_param=0.1`)', effect: 'Reduce la varianza de las covarianzas estimadas: el error de test de QDA baja (ver experimento «primeras features»).' },
        { change: 'Usar menos features (p. ej. 16)', effect: 'QDA sobreajusta menos; LDA empeora un poco.' },
      ],
    },
    {
      id: 'b_manual',
      title: 'Selección manual de dos features: V_coor = [10, 30] (Q8)',
      cells: [16, 17],
      stage: 'metodo',
      problem: 'Clasificar con solo dos features elegidas a la vista de los espectros.',
      purpose: 'Simula una restricción computacional (guía 5.2). Repite selección, partición y clasificación en un solo bloque.',
      inputs: '`X` (4509, 64) del bloque anterior, `y`.',
      outputs: '`X` (4509, 2). LDA: train 25.5 %, test 27.9 %; QDA: train 25.2 %, test 28.3 %.',
      prereqs: ['seleccion-features', 'frontera'],
      apis: [{ name: 'X[:, V_coor]', lib: 'numpy', kind: 'operador', does: 'Selecciona las columnas 10 y 30.' }],
      lines: [
        { cell: 17, lines: [3, 3], text: 'Features 10 y 30 → 0.31 kHz y 0.94 kHz. Los índices son sobre las 64 columnas, que coinciden con las originales porque son las primeras.' },
        { cell: 17, lines: [7, 7], text: '**Reasigna X sobre la X ya recortada**: si ejecutas esta celda dos veces, la segunda falla (`IndexError`: X ya solo tiene 2 columnas).' },
        { cell: 17, lines: [13, 13], text: 'Misma partición (`random_state=5`, `stratify=y`): los resultados son comparables con los de 64 features.' },
      ],
      shapes: [{ expr: 'X', shape: '(4509, 2)', meaning: 'features 10 y 30' }],
      interpretation: 'Con 2 features el error se triplica (≈8 % → ≈28 %): se pierde información. LDA y QDA quedan casi iguales porque en 2D QDA solo estima 3 parámetros de covarianza por clase: ya no sobreajusta. La matriz de confusión de LDA muestra que **nunca predice aa** (columna 1 a cero): con estas dos frecuencias aa queda dentro de la región de ao. Otra pareja puede ir mejor: es lo que explora el experimento.',
      corrections: [{ cell: 17, kind: 'concepto', original: 'X = X[:,V_coor]', fixed: 'X2 = X[:, V_coor]   # no sobrescribir X', why: 'Reasignar X hace que la celda no sea re-ejecutable y que los índices dependan de la selección anterior. Guardar el subconjunto en otra variable evita ambos problemas.' }],
      next: 'Dibujar los datos y las fronteras en el plano de las dos features.',
      whatIf: [
        { change: 'Elegir features donde aa y ao difieran más', effect: 'Menos confusión aa/ao; el error total puede bajar unos puntos.' },
        { change: 'Ejecutar la celda dos veces', effect: '`IndexError: index 10 is out of bounds for axis 1 with size 2` (dimensiones).' },
      ],
    },
    {
      id: 'b_lines',
      title: 'Scatter con las rectas de LDA (uno contra todos)',
      cells: [18, 19],
      stage: 'resultados',
      problem: 'Visualizar las dos features por clase y las rectas que define LDA.',
      purpose: 'Aprovecha que en 2D cada fila de `coef_` con su `intercept_` define una recta $w_1x_1+w_2x_2+b=0$.',
      inputs: '`X` (4509, 2), `y`, `lda` entrenado en 2D.',
      outputs: 'Scatter de las 5 clases con 5 rectas.',
      prereqs: ['frontera', 'scatter'],
      apis: [
        { name: 'lda.coef_', lib: 'sklearn', kind: 'atributo', does: 'Pesos del discriminante lineal de cada clase.', returns: '(5, 2)' },
        { name: 'lda.intercept_', lib: 'sklearn', kind: 'atributo', does: 'Término independiente de cada clase.', returns: '(5,)' },
      ],
      lines: [
        { cell: 19, lines: [13, 16], text: 'Etiqueta `class %d` con l = 1..5 (aquí sí coinciden con y).' },
        { cell: 19, lines: [21, 24], text: 'Despeja $x_2 = -(b + w_1 x_1)/w_2$ para cada clase.' },
      ],
      interpretation: 'Igual que en Lab1 parte 2: cada recta es $g_i(x)=0$ para la clase i, **no** la frontera entre dos clases (la frontera entre i y j es $g_i = g_j$). Por eso las regiones del bloque siguiente son más fiables. Se dibujan **todos** los datos (train + test).',
      corrections: [{ cell: 19, kind: 'concepto', original: '# Plot the hyperplanes: one vs all', fixed: '# rectas g_i(x) = 0; la frontera i–j es g_i(x) = g_j(x)', why: 'Con 5 clases, `predict` elige la clase de discriminante máximo; las rectas por clase no delimitan las regiones.' }],
      next: 'Regiones de decisión por malla.',
      whatIf: [],
    },
    {
      id: 'b_regions',
      title: 'Regiones de decisión de LDA y QDA (Q9)',
      cells: [20, 21],
      stage: 'resultados',
      problem: 'Ver qué clase asigna cada modelo a cada punto del plano.',
      purpose: 'Clasifica una malla de puntos y pinta las regiones; superpone los datos. Responde Q9.',
      inputs: '`X` (2 columnas), `lda`, `qda`.',
      outputs: 'Dos paneles: regiones lineales (LDA) y cuadráticas (QDA).',
      prereqs: ['meshgrid', 'frontera'],
      apis: [
        { name: 'np.meshgrid', lib: 'numpy', kind: 'función', does: 'Rejilla de puntos con paso `h`.' },
        { name: 'np.c_[xx.ravel(), yy.ravel()]', lib: 'numpy', kind: 'operador', does: 'Lista de puntos (n, 2) para `predict`.' },
        { name: 'plt.contourf', lib: 'matplotlib', kind: 'función', does: 'Rellena las regiones según la clase predicha.' },
      ],
      lines: [
        { cell: 21, lines: [3, 3], text: '`h = .1`: resolución del dibujo (no afecta al modelo).' },
        { cell: 21, lines: [16, 18], text: 'Predice cada punto de la malla y lo reorganiza con la forma de la rejilla.' },
        { cell: 21, lines: [21, 23], text: '**Errata de etiquetas**: la leyenda usa `idc` = 0..4 mientras las clases son 1..5 (y la celda 19 usa 1..5).' },
        { cell: 21, lines: [25, 25], text: '`cmap=plt.cm.tab10` sobre valores 1..5: los colores de las regiones **no** coinciden con los de los puntos (`r,g,b,c,m`). El experimento usa la misma paleta para ambos.' },
      ],
      interpretation: 'Las regiones de LDA son polígonos convexos separados por rectas; las de QDA, curvas. Las nubes de aa y ao se solapan casi por completo en este plano, por eso ningún modelo las separa; dcl, iy y sh ocupan zonas distintas. En 2D los dos modelos tienen errores casi iguales.',
      next: 'Siguiente notebook (propuesto): reducir con PCA y MDA en lugar de elegir a mano.',
      whatIf: [{ change: 'Aumentar `h` a 0.5', effect: 'Dibujo más rápido y pixelado; mismas métricas (parámetro solo de visualización).' }],
    },
  ],
  experiments: [
    {
      id: 'lab2_3.exp.manual',
      title: 'Elegir dos features a mano (Q8–Q9)',
      goal: 'Probar distintas parejas `V_coor` con la misma partición y ver cómo cambian los errores, la confusión aa/ao y las regiones. El espectro medio marca las frecuencias elegidas: busca zonas donde las curvas medias de las clases se separen.',
      blocks: ['b_spectrum', 'b_manual', 'b_regions'],
      data: DATA,
      template: EXP_MANUAL,
      seedParam: 'seed',
      params: [
        { id: 'V_coor', label: 'Features (V_coor)', kind: 'preprocesado', control: 'select', default: '[10, 30]', options: [{ value: '[10, 30]', label: '[10, 30] (notebook)' }, { value: '[5, 20]', label: '[5, 20]' }, { value: '[3, 12]', label: '[3, 12]' }, { value: '[20, 40]', label: '[20, 40]' }, { value: '[15, 55]', label: '[15, 55]' }, { value: '[0, 63]', label: '[0, 63]' }, { value: '[100, 200]', label: '[100, 200] (fuera de las 64)' }], variable: 'V_coor', computation: 'Cambia qué dos columnas de X se conservan.', result: 'Cambian los errores, qué clases se confunden y la forma de las regiones.', why: 'Cada frecuencia separa mejor unos fonemas que otros; con dos features solo se aprovecha una pequeña parte de la información.' },
        { id: 'trainsize', label: 'Proporción de train', kind: 'datos', control: 'select', default: '0.7', options: [{ value: '0.3', label: '0.3' }, { value: '0.5', label: '0.5' }, { value: '0.7', label: '0.7 (original)' }, { value: '0.9', label: '0.9' }], variable: 'trainsize', computation: 'Tamaño de train y de test.', result: 'En 2D apenas cambia el error (pocos parámetros que estimar).', why: 'Con 2 features los modelos estiman pocos parámetros: no necesitan muchos datos.' },
        { id: 'seed', label: 'random_state de la partición', kind: 'datos', control: 'range', min: 0, max: 30, step: 1, default: '5', variable: 'random_state', computation: 'Otra partición aleatoria (mismos datos).', result: 'Variaciones de ±1 punto en el error de test.', why: 'Es la variabilidad de estimar el error con un único test de 1353 muestras.' },
        { id: 'h', label: 'Paso de la malla h', kind: 'visualizacion', control: 'select', default: '.1', options: [{ value: '.1', label: '0.1 (original)' }, { value: '.25', label: '0.25' }, { value: '.5', label: '0.5' }], variable: 'h', computation: 'Solo la resolución del dibujo de regiones.', result: 'Las métricas no cambian.', why: 'La malla se usa únicamente para pintar.' },
        { id: 'xmax', label: 'Zoom del espectro', kind: 'visualizacion', control: 'select', default: '64', options: [{ value: '64', label: 'features 0–63' }, { value: '255', label: 'features 0–255' }], variable: 'plt.xlim', computation: 'Ninguno.', result: 'Solo el rango del eje x.', why: 'Parámetro de visualización.' },
      ],
      metrics: [
        { key: 'lda_train', label: 'Error LDA (train)', format: 'pct', better: 'low' },
        { key: 'lda_test', label: 'Error LDA (test)', format: 'pct', better: 'low' },
        { key: 'qda_train', label: 'Error QDA (train)', format: 'pct', better: 'low' },
        { key: 'qda_test', label: 'Error QDA (test)', format: 'pct', better: 'low' },
        { key: 'aa_predichas_lda', label: 'Muestras de test predichas como aa (LDA)', format: 'int' },
        { key: 'f1_khz', label: 'Frecuencia feature 1 (kHz)', format: 'num', digits: 3 },
        { key: 'f2_khz', label: 'Frecuencia feature 2 (kHz)', format: 'num', digits: 3 },
        { key: 'cm_lda', label: 'Confusión LDA (test)', format: 'matrix' },
        { key: 'cm_qda', label: 'Confusión QDA (test)', format: 'matrix' },
      ],
      observe: [
        'Con [10, 30], ¿cuántas muestras predice LDA como aa? ¿Qué te dice la columna 1 de la matriz?',
        'Busca una pareja con menos error que [10, 30] mirando el espectro medio. ¿Elegiste frecuencias donde aa y ao se separan?',
        'Compara LDA y QDA en 2D con lo que pasaba en 64D: ¿por qué ya no sobreajusta QDA?',
        'Cambia solo `random_state`: ¿cuánto varía el error? Úsalo para no sobreinterpretar diferencias de 1 punto.',
      ],
      source: src([9, 17, 21], { note: 'La paleta común, el espectro marcado y el registro de métricas son añadidos de la app.' }),
      fidelity: [
        { metrics: { lda_train: 0.25507, lda_test: 0.279379, qda_train: 0.251901, qda_test: 0.283075 }, tol: 1e-5, source: 'Salida guardada de la celda 17 (V_coor = [10, 30])' },
      ],
    },
    {
      id: 'lab2_3.exp.nfirst',
      title: 'Cuántas de las primeras features usar (Q6) y sobreajuste de QDA',
      goal: 'Cambiar el número de features iniciales (selección de las k primeras frecuencias) y el tamaño de train para ver cuándo QDA sobreajusta y si la regularización lo corrige.',
      blocks: ['b_first64', 'b_split', 'b_classify64'],
      data: DATA,
      template: EXP_NFIRST,
      seedParam: 'seed',
      params: [
        { id: 'nfirst', label: 'Número de primeras features', kind: 'preprocesado', control: 'select', default: '64', options: [{ value: '2', label: '2' }, { value: '8', label: '8' }, { value: '16', label: '16' }, { value: '32', label: '32' }, { value: '64', label: '64 (original)' }, { value: '128', label: '128' }, { value: '256', label: '256 (todas)' }], variable: 'V_coor = np.array(range(0, k))', computation: 'Cambia la dimensión d: LDA estima d(d+1)/2 parámetros de covarianza; QDA, 5 veces más.', result: 'Al crecer d, el error de train de QDA baja hacia 0 y el de test sube: crece la brecha train–test.', why: 'Más parámetros con las mismas muestras → mayor varianza de estimación (maldición de la dimensionalidad).' },
        { id: 'trainsize', label: 'Proporción de train', kind: 'datos', control: 'select', default: '0.7', options: [{ value: '0.3', label: '0.3' }, { value: '0.5', label: '0.5' }, { value: '0.7', label: '0.7 (original)' }, { value: '0.9', label: '0.9' }], variable: 'trainsize', computation: 'Cuántas muestras ve cada modelo.', result: 'Con menos train, QDA en 64D empeora mucho más que LDA.', why: 'QDA necesita más datos por clase para estimar sus covarianzas.' },
        { id: 'seed', label: 'random_state', kind: 'datos', control: 'range', min: 0, max: 30, step: 1, default: '5', variable: 'random_state', computation: 'Otra partición.', result: 'Pequeñas variaciones.', why: 'Variabilidad de la partición.' },
        { id: 'reg', label: 'reg_param de QDA', kind: 'modelo', control: 'select', default: '0.0', options: [{ value: '0.0', label: '0 (por defecto)' }, { value: '0.01', label: '0.01' }, { value: '0.1', label: '0.1' }, { value: '0.5', label: '0.5' }], variable: 'reg_param', computation: 'Encoge cada covarianza de QDA hacia la identidad: $(1-r)\\Sigma_k + rI$.', result: 'Menos sobreajuste: el error de train sube un poco y el de test baja.', why: 'Regularizar reduce la varianza de unas covarianzas mal estimadas.', origin: 'ampliacion' },
      ],
      metrics: [
        { key: 'nfeat', label: 'Features usadas', format: 'int' },
        { key: 'n_train', label: 'Muestras de train', format: 'int' },
        { key: 'min_train_clase', label: 'Muestras de train de la clase menor', format: 'int' },
        { key: 'param_lda', label: 'Parámetros LDA (medias + covarianza)', format: 'int' },
        { key: 'param_qda', label: 'Parámetros QDA', format: 'int' },
        { key: 'lda_train', label: 'Error LDA (train)', format: 'pct', better: 'low' },
        { key: 'lda_test', label: 'Error LDA (test)', format: 'pct', better: 'low' },
        { key: 'qda_train', label: 'Error QDA (train)', format: 'pct', better: 'low' },
        { key: 'qda_test', label: 'Error QDA (test)', format: 'pct', better: 'low' },
        { key: 'gap_qda', label: 'Brecha QDA (test − train)', format: 'pct', better: 'low' },
        { key: 'cm_lda', label: 'Confusión LDA (test)', format: 'matrix' },
        { key: 'cm_qda', label: 'Confusión QDA (test)', format: 'matrix' },
      ],
      observe: [
        'Con 64 features: ¿qué modelo tiene menor error de train? ¿Y de test? Relaciónalo con el número de parámetros.',
        'Sube a 256 features: ¿qué le pasa a la brecha de QDA?',
        'Con 64 features prueba reg_param = 0.1: ¿se reduce el sobreajuste?',
        'Con 8 features, ¿sigue sobreajustando QDA?',
      ],
      source: src([11, 13, 15], { note: 'El gráfico de barras, el recuento de parámetros y reg_param son añadidos de la app.' }),
      fidelity: [
        { metrics: { lda_train: 0.076679, lda_test: 0.079084, qda_train: 0.052598, qda_test: 0.112343 }, tol: 1e-5, source: 'Salida guardada de la celda 15 (64 features)' },
      ],
    },
  ],
  activities: [
    // ------------------------------------------------ nivel 1
    {
      kind: 'mcq',
      id: 'lab2_3.read.broadcast',
      skills: ['codigo'],
      blocks: ['b_prepare'],
      source: src([7]),
      prompt: '¿Qué hacen estas dos líneas y qué forma tiene `Xm`?',
      code: 'Xm = X.mean(axis=0)\nX = X - Xm',
      options: [
        { text: 'Calculan la media de cada feature (256,) y se la restan a cada fila: cada columna queda con media 0', correct: true },
        { text: 'Calculan la media de cada muestra (4509,) y la restan a cada fila', feedback: 'Eso sería `axis=1`, y además la resta (4509,256) − (4509,) fallaría.' },
        { text: 'Normalizan cada feature a varianza 1', feedback: 'Solo restan la media; no dividen por la desviación típica.' },
        { text: 'Seleccionan las features con media positiva', feedback: 'No seleccionan nada: X conserva sus 256 columnas.' },
      ],
      explanation: '`axis=0` colapsa las filas → un valor por columna. NumPy «difunde» el vector (256,) a cada una de las 4509 filas.',
    },
    {
      kind: 'mcq',
      id: 'lab2_3.read.split',
      skills: ['api'],
      blocks: ['b_split'],
      source: src([13]),
      prompt: '¿En qué orden devuelve sus resultados `train_test_split(X, y, ...)`?',
      options: [
        { text: 'X_train, X_test, y_train, y_test', correct: true },
        { text: 'X_train, y_train, X_test, y_test', feedback: 'Orden muy habitual al escribir de memoria, pero incorrecto: primero las dos partes de X, luego las de y.' },
        { text: 'X_test, X_train, y_test, y_train' },
      ],
      explanation: 'Para cada array de entrada devuelve (train, test), en el orden de entrada: X → X_train, X_test; y → y_train, y_test.',
    },
    {
      kind: 'text',
      id: 'lab2_3.read.stratify',
      skills: ['concepto', 'api'],
      blocks: ['b_split'],
      source: src([13]),
      prompt: 'Explica para qué sirven `random_state=5` y `stratify=y` en la celda 13 y qué ocurriría sin cada uno.',
      rubric: [
        'random_state fija la semilla: la partición es la misma en cada ejecución (reproducible y comparable entre celdas 13 y 17).',
        'Sin random_state, cada ejecución daría otra partición y errores algo distintos.',
        'stratify=y mantiene la proporción de cada clase en train y test.',
        'Es importante porque las clases están desequilibradas (695 aa frente a 1163 iy).',
      ],
      model: '`random_state=5` fija la aleatoriedad de la partición: cada vez que se ejecuta salen los mismos conjuntos, lo que permite comparar los resultados con 64 features (celda 13) y con 2 (celda 17) sobre la misma partición. Sin él, los errores cambiarían ligeramente en cada ejecución. `stratify=y` hace que train y test tengan las mismas proporciones de clases que el dataset completo; como las clases están desequilibradas (aa 15 %, iy 26 %), evita que por azar una clase quede infrarrepresentada en el test.',
    },
    // ------------------------------------------------ nivel 2
    {
      kind: 'predict',
      id: 'lab2_3.pred.shape_train',
      skills: ['codigo'],
      blocks: ['b_split'],
      data: DATA,
      source: src([13]),
      prompt: 'Tras la celda 13 (64 features, `train_size=0.7`), ¿qué forma tiene `X_train`?',
      setup: SETUP_RAW + 'X = X[:, np.array(range(0, 64))]\n',
      code: 'X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=5, shuffle=True, stratify=y)',
      expr: 'X_train.shape',
      answerKind: 'shape',
      explanation: 'floor(0.7 · 4509) = 3156 filas y 64 columnas; el test tiene 1353.',
    },
    {
      kind: 'predict',
      id: 'lab2_3.pred.freq',
      skills: ['concepto', 'codigo'],
      blocks: ['b_spectrum', 'b_manual'],
      source: src([9, 17]),
      prompt: 'Con `fmax = 8` y `nfft = 256`, ¿a qué frecuencia (kHz) corresponde la feature 30? (3 decimales)',
      setup: 'import numpy as np\nnfft = 256',
      code: 'fmax = 8*nfft/256\nfreqax = np.array(range(0,nfft))*fmax/nfft',
      expr: 'float(freqax[30])',
      answerKind: 'numeric',
      tolerance: 0.001,
      explanation: '30 · 8/256 = 0.9375 kHz. Cada feature avanza 31.25 Hz; las 64 primeras cubren 0–2 kHz.',
    },
    {
      kind: 'predict',
      id: 'lab2_3.pred.aa',
      skills: ['interpretacion'],
      blocks: ['b_manual'],
      data: DATA,
      source: src([17]),
      prompt: 'Con `V_coor = [10, 30]`, ¿cuántas muestras de **test** predice LDA como clase 1 (aa)?',
      setup: SETUP_SPLIT,
      code: 'V = [10, 30]\nlda = LinearDiscriminantAnalysis().fit(X_train[:, V], y_train)\npred = lda.predict(X_test[:, V])',
      expr: 'int((pred == 1).sum())',
      answerKind: 'numeric',
      tolerance: 0,
      explanation: 'Ninguna: en el plano (f10, f30) la nube de aa queda dentro de la de ao, más numerosa, y la regla MAP asigna toda esa zona a ao. Por eso la primera columna de la matriz de confusión es 0 y todas las aa son errores.',
    },
    // ------------------------------------------------ nivel 3
    {
      kind: 'code',
      level: 3,
      id: 'lab2_3.fill.center',
      title: 'Completa la preparación de X, y y el centrado',
      skills: ['codigo', 'api'],
      blocks: ['b_read', 'b_prepare'],
      data: DATA,
      source: src([4, 7]),
      prompt: 'Rellena los huecos para leer el CSV sin cabecera, separar features y etiqueta y centrar cada feature.',
      available: '`np`, `pd`; el archivo `BD_phoneme.csv` en la carpeta de trabajo.',
      produce: '`X` (4509, 256) centrada, `y` (4509,) con valores 1–5, `Xm` (256,).',
      setup: 'import numpy as np\nimport pandas as pd',
      starter: 'df = pd.read_csv("BD_phoneme.csv", header=___)\nX = np.array(df.iloc[:, ___])\ny = np.array(df.iloc[:, ___])\nXm = X.mean(axis=___)\nX = X - Xm',
      solution: 'df = pd.read_csv("BD_phoneme.csv", header=None)\nX = np.array(df.iloc[:, 0:256])\ny = np.array(df.iloc[:, 256])\nXm = X.mean(axis=0)\nX = X - Xm',
      alternatives: ['data = pd.read_csv("BD_phoneme.csv", header=None).to_numpy()\nX = data[:, :256]\ny = data[:, 256].astype(int)\nXm = X.mean(0)\nX = X - Xm'],
      wrong: [
        { code: 'df = pd.read_csv("BD_phoneme.csv", header=None)\nX = np.array(df.iloc[:, 0:256])\ny = np.array(df.iloc[:, 256])\nXm = X.mean(axis=1)\nX = X - Xm', expect: 'dimensiones' },
        { code: 'df = pd.read_csv("BD_phoneme.csv")\nX = np.array(df.iloc[:, 0:256])\ny = np.array(df.iloc[:, 256])\nXm = X.mean(axis=0)\nX = X - Xm', expect: 'dimensiones' },
      ],
      checks: "c.shape('X', (4509, 256))\nc.same('X', rtol=1e-9, desc='X centrada')\nc.same('y')\nc.same('Xm', rtol=1e-9)",
      hints: ['El archivo no tiene fila de nombres de columna.', 'Las features son las columnas 0 a 255; la etiqueta, la 256.', 'Una media por **columna**: el eje que se recorre es el de las filas.'],
    },
    {
      kind: 'code',
      level: 3,
      id: 'lab2_3.fill.split',
      title: 'Completa la partición estratificada',
      skills: ['api'],
      blocks: ['b_split'],
      data: DATA,
      source: src([13]),
      practice: 'train_test_split',
      prompt: 'Completa la llamada para obtener un 70 % de train, barajado, estratificado por clase y reproducible con semilla 5.',
      available: '`X` (4509, 64) centrada, `y` (4509,), `train_test_split` importado.',
      produce: '`X_train`, `X_test`, `y_train`, `y_test` iguales a los del notebook.',
      setup: SETUP_RAW + 'X = X[:, np.array(range(0, 64))]\n',
      starter: 'X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=___, random_state=___, shuffle=True, stratify=___)',
      solution: 'X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=5, shuffle=True, stratify=y)',
      alternatives: ['X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, stratify=y)'],
      wrong: [{ code: 'X_train, y_train, X_test, y_test = train_test_split(X, y, train_size=0.7, random_state=5, shuffle=True, stratify=y)', expect: 'dimensiones' }],
      checks: "c.same('X_train', rtol=1e-12)\nc.same('X_test', rtol=1e-12)\nc.same('y_train')\nc.same('y_test')",
      hints: ['`train_size` es una proporción entre 0 y 1.', '`stratify` recibe las etiquetas cuyas proporciones se quieren mantener.'],
      explanation: '`test_size=0.3` es equivalente a `train_size=0.7`, y `shuffle=True` es el valor por defecto.',
    },
    // ------------------------------------------------ nivel 4
    {
      kind: 'code',
      level: 4,
      id: 'lab2_3.fix.rerun',
      title: 'Corrige la selección que no se puede re-ejecutar',
      skills: ['codigo', 'concepto'],
      blocks: ['b_manual'],
      data: DATA,
      source: src([17], { note: 'Problema de la celda original al ejecutarla dos veces' }),
      prompt: 'Este código reproduce lo que ocurre si ejecutas la celda 17 dos veces: falla. Corrígelo para que la selección de las features 10 y 30 **no destruya** la matriz de 64 features y pueda repetirse.',
      available: '`X` (4509, 64) centrada, `y`.',
      produce: '`X2` (4509, 2) con las columnas 10 y 30 de X, y `X` intacta (4509, 64).',
      setup: SETUP_RAW + 'X = X[:, np.array(range(0, 64))]\n',
      starter: 'V_coor = [10, 30]\nX = X[:, V_coor]\nX = X[:, V_coor]   # segunda ejecución de la celda\nX2 = X',
      solution: 'V_coor = [10, 30]\nX2 = X[:, V_coor]\nX2 = X[:, V_coor]   # ahora se puede repetir',
      alternatives: ['X2 = X[:, [10, 30]].copy()'],
      wrong: [{ code: 'V_coor = [10, 30]\nX = X[:, V_coor]\nX = X[:, V_coor]\nX2 = X', expect: 'dimensiones' }],
      checks: "c.shape('X2', (4509, 2))\nc.same('X2', rtol=1e-12)\nc.shape('X', (4509, 64), desc='X intacta')",
      hints: ['El error dice que el índice 30 está fuera de rango para un eje de tamaño 2: ¿qué contiene X tras la primera selección?', 'Guarda el subconjunto en otra variable en lugar de sobrescribir X.'],
    },
    {
      kind: 'code',
      level: 4,
      id: 'lab2_3.fix.confusion',
      title: 'Corrige la matriz de confusión de test',
      skills: ['api', 'codigo'],
      blocks: ['b_classify64'],
      data: DATA,
      source: src([15]),
      prompt: 'El código mezcla variables de train y de test al calcular la matriz de confusión de test de LDA. Corrígelo.',
      available: '`X_train`, `X_test`, `y_train`, `y_test` (64 features), LDA importado.',
      produce: '`cm_test` (5, 5) con la matriz de confusión de **test** de LDA y `lda_test_error`.',
      setup: SETUP_SPLIT,
      starter: 'ldamodel = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)\ny_tpred_lda = ldamodel.predict(X_train)\ny_testpred_lda = ldamodel.predict(X_test)\nlda_test_error = 1. - accuracy_score(y_test, y_testpred_lda)\ncm_test = confusion_matrix(y_test, y_tpred_lda)',
      solution: 'ldamodel = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)\ny_tpred_lda = ldamodel.predict(X_train)\ny_testpred_lda = ldamodel.predict(X_test)\nlda_test_error = 1. - accuracy_score(y_test, y_testpred_lda)\ncm_test = confusion_matrix(y_test, y_testpred_lda)',
      wrong: [{ code: 'ldamodel = LinearDiscriminantAnalysis().fit(X_train, y_train)\ny_tpred_lda = ldamodel.predict(X_train)\nlda_test_error = 1. - accuracy_score(y_test, ldamodel.predict(X_test))\ncm_test = confusion_matrix(y_test, y_tpred_lda)', expect: 'dimensiones' }],
      checks: "c.same('cm_test', desc='Matriz de confusión de test')\nc.same('lda_test_error', rtol=1e-9)",
      hints: ['Lee el error: «inconsistent numbers of samples» indica que los dos argumentos tienen longitudes distintas (1353 frente a 3156).', 'Las predicciones de test son `y_testpred_lda`.'],
      explanation: 'Matriz esperada: [[157, 51, 0, 0, 0], [47, 260, 0, 0, 0], [0, 0, 221, 4, 2], [0, 0, 1, 348, 0], [1, 0, 0, 1, 260]] (salida guardada de la celda 15).',
    },
    // ------------------------------------------------ nivel 5
    {
      kind: 'code',
      level: 5,
      id: 'lab2_3.mem.pipeline',
      title: 'Escribe de memoria el pipeline con 64 features',
      skills: ['escritura', 'api'],
      blocks: ['b_first64', 'b_split', 'b_classify64'],
      data: DATA,
      source: src([11, 13, 15]),
      prompt: 'Partiendo de `X` (256 features centradas) e `y`: quédate con las 64 primeras features, parte 70/30 estratificado con semilla 5, entrena LDA y QDA y calcula sus errores de train y test.',
      available: '`X` (4509, 256), `y`. **Nada de scikit-learn importado.**',
      produce: '`lda_train_error`, `lda_test_error`, `qda_train_error`, `qda_test_error` (float).',
      setup: 'import numpy as np\nimport pandas as pd\ndf = pd.read_csv("BD_phoneme.csv", header=None)\nX = np.array(df.iloc[:, 0:256])\ny = np.array(df.iloc[:, 256])\nX = X - X.mean(axis=0)',
      starter: '',
      solution: 'from sklearn.model_selection import train_test_split\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nfrom sklearn.metrics import accuracy_score\nX = X[:, :64]\nX_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=5, shuffle=True, stratify=y)\nlda = LinearDiscriminantAnalysis().fit(X_train, y_train)\nqda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)\nlda_train_error = 1 - accuracy_score(y_train, lda.predict(X_train))\nlda_test_error = 1 - accuracy_score(y_test, lda.predict(X_test))\nqda_train_error = 1 - accuracy_score(y_train, qda.predict(X_train))\nqda_test_error = 1 - accuracy_score(y_test, qda.predict(X_test))',
      alternatives: ['from sklearn.model_selection import train_test_split\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA\nXtr, Xte, ytr, yte = train_test_split(X[:, np.arange(64)], y, test_size=0.3, random_state=5, stratify=y)\nm1, m2 = LDA().fit(Xtr, ytr), QDA().fit(Xtr, ytr)\nlda_train_error = 1 - m1.score(Xtr, ytr)\nlda_test_error = 1 - m1.score(Xte, yte)\nqda_train_error = 1 - m2.score(Xtr, ytr)\nqda_test_error = 1 - m2.score(Xte, yte)'],
      wrong: [{ code: 'from sklearn.model_selection import train_test_split\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nX_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=5, stratify=y)\nlda = LinearDiscriminantAnalysis().fit(X_train, y_train)\nqda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)\nlda_train_error = 1 - lda.score(X_train, y_train)\nlda_test_error = 1 - lda.score(X_test, y_test)\nqda_train_error = 1 - qda.score(X_train, y_train)\nqda_test_error = 1 - qda.score(X_test, y_test)', expect: 'razonamiento' }],
      checks: "c.same('lda_train_error', rtol=1e-9)\nc.same('lda_test_error', rtol=1e-9)\nc.same('qda_train_error', rtol=1e-9)\nc.same('qda_test_error', rtol=1e-9)",
      hints: ['Selecciona columnas con `X[:, :64]` (o `V_coor`).', '`score(X, y)` de un clasificador devuelve la accuracy: error = 1 − score.'],
      explanation: 'Valores esperados: LDA 7.67 % / 7.91 %, QDA 5.26 % / 11.23 %.',
    },
    {
      kind: 'code',
      level: 5,
      id: 'lab2_3.mem.meanspec',
      title: 'Calcula y dibuja el espectro medio por clase',
      skills: ['escritura', 'codigo'],
      blocks: ['b_spectrum'],
      data: DATA,
      source: src([9]),
      prompt: 'Sin mirar la celda 9: calcula el espectro medio de cada clase en una matriz `xdm` (fila 0 = clase 1, …, fila 4 = clase 5) y dibuja las 5 curvas en una figura.',
      available: '`X` (4509, 256) centrada, `y` (valores 1–5), `np`, `plt`.',
      produce: '`xdm` (5, 256) y una figura con las 5 medias.',
      setup: SETUP_RAW + 'import matplotlib.pyplot as plt\n',
      starter: '',
      solution: 'xdm = np.zeros((5, 256))\nfor idc in range(5):\n    xdm[idc, :] = X[y == idc + 1].mean(axis=0)\nfor idc in range(5):\n    plt.plot(xdm[idc], label=str(idc + 1))\nplt.legend()\nplt.show()',
      alternatives: ['xdm = np.vstack([X[y == k].mean(0) for k in [1, 2, 3, 4, 5]])\nplt.plot(xdm.T)\nplt.show()'],
      wrong: [{ code: 'xdm = np.vstack([X[y == k].mean(0) for k in range(5)])\nplt.plot(xdm.T)\nplt.show()', expect: 'razonamiento' }],
      checks: "c.shape('xdm', (5, 256))\nc.same('xdm', rtol=1e-9, desc='Espectros medios')\nc.figures(1)",
      hints: ['Las etiquetas empiezan en 1, no en 0.', 'Filtra las filas de una clase con una máscara booleana y promedia por columnas.'],
      explanation: 'Un error típico es usar `y == 0`, que no selecciona ninguna fila: la media de un array vacío es NaN.',
    },
    // ------------------------------------------------ nivel 6
    {
      kind: 'code',
      level: 6,
      id: 'lab2_3.adapt.fscore',
      title: 'Adapta: elige las dos features automáticamente (selección por F de ANOVA)',
      skills: ['metodo', 'escritura'],
      blocks: ['b_manual'],
      data: DATA,
      transfer: true,
      source: { notebook: NB, cells: [17], origin: 'ampliacion', note: 'Selección automática con sklearn.feature_selection (slide 3 de Lab2); no la hace el notebook' },
      practice: 'sklearn.feature_selection.f_classif',
      prompt: 'En lugar de mirar los espectros, elige las 2 features (de las 64) con mayor estadístico **F de ANOVA** calculado **solo con train** (`f_classif`). Después entrena LDA con esas dos features y calcula su error de test.',
      available: '`X_train`, `X_test`, `y_train`, `y_test` (64 features, partición del notebook).',
      produce: '`idx`: lista o array con los 2 índices elegidos, en orden creciente; `err_test`: error de test de LDA con ellas.',
      setup: SETUP_SPLIT,
      starter: '',
      solution: 'from sklearn.feature_selection import f_classif\nF, p = f_classif(X_train, y_train)\nidx = np.sort(np.argsort(F)[-2:])\nlda = LinearDiscriminantAnalysis().fit(X_train[:, idx], y_train)\nerr_test = 1 - accuracy_score(y_test, lda.predict(X_test[:, idx]))',
      alternatives: ['from sklearn.feature_selection import SelectKBest, f_classif\nsel = SelectKBest(f_classif, k=2).fit(X_train, y_train)\nidx = np.flatnonzero(sel.get_support())\nlda = LinearDiscriminantAnalysis().fit(sel.transform(X_train), y_train)\nerr_test = 1 - lda.score(sel.transform(X_test), y_test)'],
      wrong: [{ code: 'from sklearn.feature_selection import f_classif\nF, p = f_classif(X_test, y_test)\nidx = np.sort(np.argsort(F)[-2:])\nlda = LinearDiscriminantAnalysis().fit(X_train[:, idx], y_train)\nerr_test = 1 - accuracy_score(y_test, lda.predict(X_test[:, idx]))', expect: 'razonamiento' }],
      checks: "c.same('idx', desc='Índices elegidos')\nc.same('err_test', rtol=1e-9)",
      hints: ['`f_classif(X, y)` devuelve el estadístico F y el p-valor de cada columna.', 'Los dos mayores: `np.argsort(F)[-2:]`.', 'La selección también es parte del entrenamiento: usa solo train.'],
      explanation: 'Esta es selección de características **automática y supervisada** (filtro univariante). Compara su error con el de [10, 30]. Elegir las features con el test es una fuga de información.',
    },
    {
      kind: 'code',
      level: 6,
      id: 'lab2_3.adapt.recall',
      title: 'Adapta: tasa de acierto por clase a partir de la matriz de confusión',
      skills: ['interpretacion', 'codigo'],
      blocks: ['b_classify64'],
      data: DATA,
      source: src([15]),
      prompt: 'Con QDA y 64 features, calcula la tasa de acierto de **cada clase** en test (recall) a partir de la matriz de confusión e identifica la clase peor clasificada (su etiqueta 1–5).',
      available: '`X_train`, `X_test`, `y_train`, `y_test` (64 features); QDA y `confusion_matrix` importados.',
      produce: '`recall` (5,) en el orden de las clases 1..5 y `peor` (int, etiqueta 1–5).',
      setup: SETUP_SPLIT,
      starter: '',
      solution: 'qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)\ncm = confusion_matrix(y_test, qda.predict(X_test))\nrecall = np.diag(cm) / cm.sum(axis=1)\npeor = int(np.argmin(recall) + 1)',
      alternatives: ['from sklearn.metrics import recall_score\nqda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)\nrecall = recall_score(y_test, qda.predict(X_test), average=None)\npeor = int(qda.classes_[np.argmin(recall)])'],
      wrong: [{ code: 'qda = QuadraticDiscriminantAnalysis().fit(X_train, y_train)\ncm = confusion_matrix(y_test, qda.predict(X_test))\nrecall = np.diag(cm) / cm.sum(axis=0)\npeor = int(np.argmin(recall) + 1)', expect: 'razonamiento' }],
      checks: "c.same('recall', rtol=1e-9, desc='Recall por clase')\nc.same('peor')",
      hints: ['Las filas de la matriz son las clases reales.', 'Aciertos de cada clase = diagonal; total de cada clase = suma de su fila.'],
      explanation: 'aa es la peor clase (≈68 % de acierto en test): se confunde con ao.',
    },
    // ------------------------------------------------ interpretación
    {
      kind: 'mcq',
      id: 'lab2_3.int.spectrum',
      skills: ['interpretacion'],
      blocks: ['b_spectrum'],
      figure: { notebook: NB, cell: 9, output: 0 },
      source: src([9], { section: 'Q5' }),
      prompt: 'Observa los log-periodogramas (salida guardada; el último panel son las medias). ¿Qué pareja de fonemas prevés que será más difícil de separar?',
      options: [
        { text: 'aa y ao', correct: true, feedback: 'Sus espectros medios casi se superponen: dos vocales posteriores.' },
        { text: 'iy y sh', feedback: 'sh tiene energía en altas frecuencias y iy no: se distinguen bien.' },
        { text: 'dcl y sh', feedback: 'dcl tiene muy poca energía; sh mucha en altas frecuencias.' },
      ],
      explanation: 'La matriz de confusión lo confirma: casi todos los errores están en el bloque aa/ao.',
    },
    {
      kind: 'mcq',
      id: 'lab2_3.int.qdaoverfit',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_classify64'],
      source: src([15], { section: 'Q6' }),
      prompt: 'Con 64 features: LDA 7.7 % (train) / 7.9 % (test); QDA 5.3 % / 11.2 %. ¿Qué explicación es la correcta?',
      options: [
        { text: 'QDA sobreajusta: estima una covarianza 64×64 por clase con pocas muestras por clase', correct: true },
        { text: 'QDA es peor porque los datos no son gaussianos', feedback: 'Si fuera solo eso, también fallaría en train; aquí falla en test, no en train.' },
        { text: 'El test está desequilibrado', feedback: 'La partición está estratificada.' },
        { text: 'LDA sobreajusta porque tiene un error de train mayor', feedback: 'Sobreajustar es tener train mucho mejor que test; LDA tiene ambos parecidos.' },
      ],
      explanation: 'QDA estima 5 × 2080 = 10 400 parámetros de covarianza frente a 2080 de LDA. Con 487–814 muestras de train por clase, sus covarianzas se ajustan al ruido del train.',
    },
    {
      kind: 'numeric',
      id: 'lab2_3.int.params',
      skills: ['concepto'],
      blocks: ['b_classify64'],
      source: src([15], { note: 'Cálculo de la app' }),
      prompt: '¿Cuántos parámetros **de covarianza** (valores distintos de matrices simétricas) estima QDA con 64 features y 5 clases?',
      answer: 10400,
      tolerance: 0,
      explanation: 'Cada covarianza simétrica 64×64 tiene 64·65/2 = 2080 valores distintos; QDA estima 5: 10 400. LDA estima una sola: 2080.',
    },
    {
      kind: 'mcq',
      id: 'lab2_3.int.regions',
      skills: ['interpretacion'],
      blocks: ['b_regions'],
      figure: { notebook: NB, cell: 21, output: 0 },
      source: src([21], { section: 'Q9' }),
      prompt: 'Regiones de decisión con las features 10 y 30 (salida guardada). ¿Qué afirmación es **incorrecta**?',
      options: [
        { text: 'Los colores de las regiones coinciden con los colores de los puntos de cada clase', correct: true, feedback: 'Correcto, es falsa: las regiones usan `tab10` y los puntos `r,g,b,c,m`; los colores no se corresponden.' },
        { text: 'Las fronteras de LDA son rectas y las de QDA, curvas' },
        { text: 'aa y ao se solapan en este plano, por eso se confunden' },
        { text: 'La resolución h solo afecta al dibujo, no al error' },
      ],
      explanation: 'Detalle del notebook: para comparar regiones y puntos hay que usar la misma paleta (el experimento lo corrige).',
    },
    {
      kind: 'text',
      id: 'lab2_3.int.q8',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b_manual'],
      source: { notebook: NB, cells: [17], doc: 'lab2_guide', section: 'Q8', origin: 'original' },
      prompt: 'Q8 · ¿Qué dos features elegirías y por qué? Compara los errores con 2 features (≈26 % train, ≈28 % test para ambos modelos) con los de 64 features.',
      rubric: [
        'Justifica la elección con los espectros medios: frecuencias donde las clases se separan más (especialmente aa y ao).',
        'Indica la frecuencia en kHz de las features (k · 8/256).',
        'El error sube mucho al pasar de 64 a 2 features: se pierde información discriminante.',
        'En 2D LDA y QDA rinden igual: QDA ya no sobreajusta porque estima pocos parámetros.',
        'Señala la confusión aa/ao (LDA no predice aa con [10, 30]).',
      ],
      model: 'Elegiría frecuencias donde los espectros medios de las clases difieren más, sobre todo entre aa y ao, que son las más parecidas; con [10, 30] (0.31 y 0.94 kHz) se separan bien dcl, iy y sh pero aa queda dentro de ao, y LDA no predice ninguna aa. Con dos features el error pasa de ≈8 % a ≈28 % porque se descarta casi toda la información espectral. LDA y QDA quedan casi iguales: en 2D QDA solo estima 3 parámetros de covarianza por clase y deja de sobreajustar, al contrario que con 64 features. La elección manual es una selección de características: conserva el significado de las frecuencias, pero no garantiza ser la mejor pareja.',
    },
    {
      kind: 'mcq',
      id: 'lab2_3.int.misconception',
      skills: ['concepto', 'metodo'],
      blocks: ['b_first64', 'b_manual'],
      source: src([11, 17]),
      prompt: '«Quedarse con las 64 primeras features es lo mismo que aplicar PCA con 64 componentes». ¿Qué error contiene?',
      options: [
        { text: 'Confunde selección con extracción: PCA crea 64 combinaciones lineales de las 256 features, elegidas por varianza', correct: true },
        { text: 'Ninguno: ambos dejan 64 columnas', feedback: 'Tener el mismo número de columnas no implica las mismas columnas.' },
        { text: 'Solo difieren en que PCA centra los datos', feedback: 'La diferencia principal es que PCA mezcla todas las features.' },
      ],
      explanation: 'Seleccionar conserva frecuencias concretas (0–2 kHz); PCA proyecta sobre direcciones de máxima varianza, que combinan todas las frecuencias. Lo verás en el notebook 4.',
    },
    {
      kind: 'dynamic',
      id: 'lab2_3.dyn.freq',
      skills: ['concepto'],
      blocks: ['b_spectrum'],
      source: src([9], { note: 'Variante generada' }),
      answerKind: 'numeric',
      generator: `import numpy as np
rng = np.random.default_rng(SEED)
k = int(rng.integers(1, 256))
val = k * 8 / 256
variante = {
  'prompt': f'En Phoneme (256 features entre 0 y 8 kHz), ¿a qué frecuencia en kHz corresponde la feature {k}? (4 decimales)',
  'answer': round(val, 6), 'tolerance': 5e-4,
  'explanation': f'Cada feature avanza 8/256 = 0.03125 kHz: {k} · 0.03125 = {val:.4f} kHz.'
}`,
    },
  ],
  recall: ['lab2_3.fill.center', 'lab2_3.fill.split', 'lab2_3.fix.rerun', 'lab2_3.fix.confusion', 'lab2_3.mem.pipeline', 'lab2_3.mem.meanspec', 'lab2_3.adapt.recall'],
  practice: [
    {
      id: 'lab2_3.practice.report',
      title: 'Mini-report Q6–Q9 sin el notebook',
      scenario: 'Reconstruye los resultados de Q6–Q9: prepara los datos, clasifica con 64 features, analiza qué clase falla, elige dos features (a mano o con un criterio) y justifica el cambio de error.',
      steps: ['lab2_3.fill.center', 'lab2_3.mem.pipeline', 'lab2_3.adapt.recall', 'lab2_3.int.qdaoverfit', 'lab2_3.pred.aa', 'lab2_3.int.q8'],
      source: { notebook: NB, doc: 'lab2_guide', section: 'Q6–Q9', origin: 'original' },
    },
  ],
  notes: [
    'La app reproduce exactamente los errores guardados de las celdas 15 y 17 (verificado con `npm run verify -- lab2_3`).',
    'La celda 9 dibuja miles de curvas: en la app tarda unos 9 s.',
  ],
};
