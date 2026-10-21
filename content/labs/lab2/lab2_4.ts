import type { NotebookContent } from '../../../src/content/types';

// Notebook PROPUESTO por la guía de Lab2 (apartado 6): lo escribe el alumno y no está en los materiales.
// Todo el código de este archivo es la solución de referencia de la app, no de la asignatura.
const NB = 'lab2_4';
const DATA = ['BD_phoneme.csv'];
const guide = (section: string, note?: string) => ({ doc: 'lab2_guide', section, origin: 'app' as const, note: note ?? 'Enunciado de la guía; solución de referencia de la app' });

export const LOAD = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# dataset completo: 256 features (guía, apartado 6), mismo preprocesado y partición que lab2_3
df = pd.read_csv("BD_phoneme.csv", header=None)
X = np.array(df.iloc[:, 0:256])
y = np.array(df.iloc[:, 256])
X = X - X.mean(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=5, shuffle=True, stratify=y)
`;

export const PCA_FIT = `# PCA se ajusta SOLO con el conjunto de entrenamiento
pca = PCA()
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
`;

export const PCA_LOOP = `dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 64, 80, 100, 128, 160, 200, 256]
lda_tr, lda_te, qda_tr, qda_te = [], [], [], []
for nfeat in dims:
    Ztr, Zte = X_train_pca[:, :nfeat], X_test_pca[:, :nfeat]   # primeras nfeat componentes
    lda = LinearDiscriminantAnalysis().fit(Ztr, y_train)
    qda = QuadraticDiscriminantAnalysis().fit(Ztr, y_train)
    lda_tr.append(1 - accuracy_score(y_train, lda.predict(Ztr)))
    lda_te.append(1 - accuracy_score(y_test, lda.predict(Zte)))
    qda_tr.append(1 - accuracy_score(y_train, qda.predict(Ztr)))
    qda_te.append(1 - accuracy_score(y_test, qda.predict(Zte)))
`;

export const PCA_PLOT = `plt.figure(figsize=(9, 5))
plt.plot(dims, lda_tr, 'b--', label='LC train')
plt.plot(dims, lda_te, 'b-', label='LC test')
plt.plot(dims, qda_tr, 'r--', label='QC train')
plt.plot(dims, qda_te, 'r-', label='QC test')
plt.xscale('log')
plt.xlabel('número de componentes PCA (nfeat)')
plt.ylabel('probabilidad de error')
plt.legend()
plt.title('PCA: error frente a la dimensión')
plt.show()
`;

export const MDA_LOOP = `n_classes = len(np.unique(y_train))
dmax = min(X_train.shape[1], n_classes - 1)
print('dmax =', dmax)
mda_dims = list(range(1, dmax + 1))
mlda_tr, mlda_te, mqda_tr, mqda_te = [], [], [], []
for d in mda_dims:
    mda = LinearDiscriminantAnalysis(n_components=d).fit(X_train, y_train)   # supervisado: usa y_train
    Ztr, Zte = mda.transform(X_train), mda.transform(X_test)
    lda = LinearDiscriminantAnalysis().fit(Ztr, y_train)
    qda = QuadraticDiscriminantAnalysis().fit(Ztr, y_train)
    mlda_tr.append(1 - accuracy_score(y_train, lda.predict(Ztr)))
    mlda_te.append(1 - accuracy_score(y_test, lda.predict(Zte)))
    mqda_tr.append(1 - accuracy_score(y_train, qda.predict(Ztr)))
    mqda_te.append(1 - accuracy_score(y_test, qda.predict(Zte)))
`;

export const MDA_PLOT = `plt.figure(figsize=(7, 4.5))
plt.plot(mda_dims, mlda_tr, 'b--o', label='LC train')
plt.plot(mda_dims, mlda_te, 'b-o', label='LC test')
plt.plot(mda_dims, mqda_tr, 'r--o', label='QC train')
plt.plot(mda_dims, mqda_te, 'r-o', label='QC test')
plt.xticks(mda_dims)
plt.xlabel("dimensión d' (MDA)")
plt.ylabel('probabilidad de error')
plt.legend()
plt.title('MDA: error frente a la dimensión (d\\' ≤ c − 1 = 4)')
plt.show()
`;

const MDA_BLOCKS = `# Ampliación (slides, p. 8 y 18): más de c − 1 features calculando MDA por grupos de features
Ztr_list, Zte_list = [], []
for g in range(4):                          # 4 grupos de 64 frecuencias consecutivas
    cols = slice(64 * g, 64 * (g + 1))
    mda_g = LinearDiscriminantAnalysis(n_components=4).fit(X_train[:, cols], y_train)
    Ztr_list.append(mda_g.transform(X_train[:, cols]))
    Zte_list.append(mda_g.transform(X_test[:, cols]))
Ztr_b, Zte_b = np.hstack(Ztr_list), np.hstack(Zte_list)   # (n, 16)
qda_b = QuadraticDiscriminantAnalysis().fit(Ztr_b, y_train)
print('QDA con 16 features MDA por bloques: test', 1 - accuracy_score(y_test, qda_b.predict(Zte_b)))
`;

const EXP = `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# lectura y centrado (como lab2_3), 256 features
df = pd.read_csv("BD_phoneme.csv", header=None)
X = np.array(df.iloc[:, 0:256])
y = np.array(df.iloc[:, 256])
X = X - X.mean(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size={{trainsize}}, random_state={{seed}}, shuffle=True, stratify=y)

metodo = {{metodo}}
if metodo == 'PCA':
    dims = {{dims}}
    pca = PCA().fit(X_train)                  # solo train
    P_train, P_test = pca.transform(X_train), pca.transform(X_test)
else:
    dmax = min(X_train.shape[1], len(np.unique(y_train)) - 1)
    dims = list(range(1, dmax + 1))

lda_tr, lda_te, qda_tr, qda_te = [], [], [], []
for d in dims:
    if metodo == 'PCA':
        Ztr, Zte = P_train[:, :d], P_test[:, :d]
    else:
        mda = LinearDiscriminantAnalysis(n_components=d).fit(X_train, y_train)
        Ztr, Zte = mda.transform(X_train), mda.transform(X_test)
    lda = LinearDiscriminantAnalysis().fit(Ztr, y_train)
    qda = QuadraticDiscriminantAnalysis(reg_param={{reg}}).fit(Ztr, y_train)
    lda_tr.append(1 - accuracy_score(y_train, lda.predict(Ztr)))
    lda_te.append(1 - accuracy_score(y_test, lda.predict(Zte)))
    qda_tr.append(1 - accuracy_score(y_train, qda.predict(Ztr)))
    qda_te.append(1 - accuracy_score(y_test, qda.predict(Zte)))

plt.figure(figsize=(9, 5))
plt.plot(dims, lda_tr, 'b--', marker='.', label='LC train')
plt.plot(dims, lda_te, 'b-', marker='.', label='LC test')
plt.plot(dims, qda_tr, 'r--', marker='.', label='QC train')
plt.plot(dims, qda_te, 'r-', marker='.', label='QC test')
if {{logx}}:
    plt.xscale('log')
plt.xlabel('dimensión (' + metodo + ')')
plt.ylabel('probabilidad de error')
plt.legend()
plt.title(metodo + ': error de train y test frente a la dimensión')
plt.show()

i_l, i_q = int(np.argmin(lda_te)), int(np.argmin(qda_te))
resultados = {
  'n_dims': len(dims), 'dim_max': dims[-1],
  'lda_best_dim': dims[i_l], 'lda_best_test': lda_te[i_l],
  'qda_best_dim': dims[i_q], 'qda_best_test': qda_te[i_q],
  'lda_test_last': lda_te[-1], 'qda_train_last': qda_tr[-1], 'qda_test_last': qda_te[-1],
  'gap_qda_last': qda_te[-1] - qda_tr[-1],
}
`;


export const lab2_4: NotebookContent = {
  id: NB,
  lab: 'lab2',
  title: 'Lab 2 · Parte 4 (propuesta) — Phoneme: reducción de dimensión con PCA y MDA',
  short: 'Phoneme: PCA vs MDA (propuesto)',
  origin: 'app',
  proposed: true,
  context: {
    goal: `La guía de Lab2 (apartado 6) pide **escribir un notebook propio** (\`Mlearn_lab2_4_Phoneme_PCA_MDA_<apellidos>.ipynb\`) que sustituya la selección manual de lab2_3 por una **reducción de dimensión aprendida**:

1. Leer el dataset **completo** (vectores de 256 features).
2. Reducir con **PCA** y clasificar con LC (LDA) y QC (QDA) para un número de features \`nfeat\` de 1 a 256, mostrando en **una figura** las cuatro curvas de error (train/test × LC/QC).
3. Elegir la dimensión más adecuada para cada clasificador **sin sobreajustar** (el error de test no debe ser mucho mayor que el de train).
4. Repetir con **MDA** para $d'$ de 1 a $d_{max}$ y comparar ambos métodos (Q10–Q13).

Regla clave de la guía: *la matriz de proyección y el clasificador se calculan con el conjunto de entrenamiento; el test solo se usa para evaluar*.

> Este notebook **no está en tus materiales**: el código y los resultados son la **solución de referencia de la app**, verificada ejecutándola, no la de la asignatura.`,
    dataset: 'Phoneme completo: 4509 × 256 log-periodogramas centrados, 5 clases (aa, ao, dcl, iy, sh). Misma partición estratificada que lab2_3 (`train_size=0.7`, `random_state=5`): 3156 muestras de train y 1353 de test.',
    variables: [
      { name: 'X_train, X_test', meaning: 'Partición con las 256 features.', shape: '(3156, 256), (1353, 256)' },
      { name: 'pca', meaning: 'PCA ajustado con X_train (todas las componentes).', shape: 'components_ (256, 256)' },
      { name: 'X_train_pca, X_test_pca', meaning: 'Proyecciones sobre las 256 componentes; se usan las nfeat primeras.', shape: '(3156, 256), (1353, 256)' },
      { name: 'dims', meaning: 'Dimensiones evaluadas.', shape: 'lista' },
      { name: 'lda_tr, lda_te, qda_tr, qda_te', meaning: 'Errores de train y test por dimensión.', shape: 'len(dims)' },
      { name: 'dmax', meaning: 'Máximo de componentes MDA: min(d, c − 1) = 4.', shape: 'escalar' },
      { name: 'mda', meaning: 'LinearDiscriminantAnalysis usado como proyección (transform).', shape: 'scalings_ (256, 4)' },
    ],
    classes: [
      { id: '1', label: 'aa' },
      { id: '2', label: 'ao' },
      { id: '3', label: 'dcl' },
      { id: '4', label: 'iy' },
      { id: '5', label: 'sh' },
    ],
    questions: [
      { id: 'Q10', text: 'Curvas de error de LC y QC en train y test frente al número de features (PCA).', blocks: ['b4_loop_pca', 'b4_plot_pca'], doc: 'lab2_guide' },
      { id: 'Q11', text: '¿Qué dimensión es la más adecuada para LC y cuál para QC? Evitar el sobreajuste.', blocks: ['b4_choose_pca'], doc: 'lab2_guide' },
      { id: 'Q12', text: '¿Cuál es el máximo número de features dmax con MDA? Curvas de error.', blocks: ['b4_mda', 'b4_plot_mda'], doc: 'lab2_guide' },
      { id: 'Q13', text: 'Comparar PCA y MDA en Phoneme entre 1 y dmax features.', blocks: ['b4_compare'], doc: 'lab2_guide' },
    ],
    notation: [
      { symbol: '$y = W^T x$', meaning: 'reducción lineal de d a d\' features (slides, p. 4)', code: 'pca.transform(X) / mda.transform(X)' },
      { symbol: '$S_T = S_B + S_C$', meaning: 'dispersión total = entre clases + intra-clase (slides, p. 5)' },
      { symbol: 'PCA: $S_T w_i = \\lambda_i w_i$', meaning: 'autovectores de mayor autovalor de la dispersión total', code: 'pca.components_' },
      { symbol: "MDA: $S_C^{-1}S_B w_j = \\lambda_j w_j$", meaning: 'autovectores generalizados; máximo $\\min(d, c-1)$', code: 'LinearDiscriminantAnalysis(n_components=d)' },
      { symbol: "$d' \\le \\min(d, c-1)$", meaning: 'límite de MDA: con c = 5 clases, 4', code: 'dmax' },
      { symbol: 'nfeat', meaning: 'número de componentes PCA usadas', code: 'X_train_pca[:, :nfeat]' },
    ],
  },
  flow: [
    { stage: 'contexto', label: 'Enunciado', blocks: ['b4_task'], summary: 'Qué pide la guía (apartado 6).' },
    { stage: 'datos', label: 'Datos', blocks: ['b4_load'], summary: '256 features, misma partición que lab2_3.' },
    { stage: 'metodo', label: 'PCA', blocks: ['b4_pca', 'b4_loop_pca'], summary: 'Ajustar con train y recorrer nfeat.' },
    { stage: 'resultados', label: 'Curvas PCA', blocks: ['b4_plot_pca', 'b4_choose_pca'], summary: 'Q10–Q11.' },
    { stage: 'metodo', label: 'MDA', blocks: ['b4_mda', 'b4_plot_mda', 'b4_blocks'], summary: "d' = 1..4; ampliación por bloques." },
    { stage: 'interpretacion', label: 'Comparación', blocks: ['b4_compare'], summary: 'Q12–Q13: PCA frente a MDA.' },
  ],
  blocks: [
    {
      id: 'b4_task',
      title: 'Enunciado: qué debe hacer el notebook 4',
      cells: [],
      stage: 'contexto',
      origin: 'app',
      noRun: true,
      problem: 'Sustituir la selección manual de features por una reducción automática y elegir la dimensión.',
      purpose: 'Fija las reglas: dataset completo, PCA y MDA, cuatro curvas en una figura, y **ajustar proyección y clasificador solo con train**.',
      inputs: 'Guía de Lab2, apartado 6; report Q10–Q13.',
      outputs: 'El plan del notebook.',
      prereqs: ['pca', 'mda', 'extraccion-features', 'sobreajuste'],
      apis: [],
      lines: [],
      interpretation: 'Aquí la reducción es **extracción**: las nuevas features son combinaciones lineales de las 256 frecuencias, a diferencia de `V_coor` (lab2_3), que seleccionaba columnas.',
      next: 'Carga de datos.',
      whatIf: [],
      check: {
        kind: 'mcq',
        id: 'lab2_4.check.task',
        skills: ['concepto'],
        source: guide('apartado 6'),
        prompt: 'Según la guía, ¿con qué datos se calcula la matriz de proyección de PCA?',
        options: [
          { text: 'Solo con el conjunto de entrenamiento', correct: true },
          { text: 'Con todo el dataset, porque PCA no usa etiquetas', feedback: 'Aunque no use etiquetas, aprender W con el test filtra información del test al modelo.' },
          { text: 'Con el test, para que la proyección se adapte a los datos evaluados', feedback: 'Así el test deja de ser independiente.' },
        ],
        explanation: '«Use the training dataset to find the projection matrix and to train the classifier. The test dataset should only be used for evaluating the error probabilities.»',
      },
    },
    {
      id: 'b4_load',
      title: 'Carga del dataset completo y partición',
      cells: [],
      stage: 'datos',
      origin: 'app',
      data: DATA,
      runCode: LOAD,
      problem: 'Disponer de las 256 features y de una partición train/test.',
      purpose: 'Reutiliza el preprocesado de lab2_3 (lectura, centrado, partición estratificada con semilla 5) pero **sin** recortar a 64 features, como pide la guía.',
      inputs: '`BD_phoneme.csv`.',
      outputs: '`X_train` (3156, 256), `X_test` (1353, 256), `y_train`, `y_test`.',
      prereqs: ['train-test', 'estratificado', 'centrado'],
      apis: [
        { name: 'train_test_split', lib: 'sklearn.model_selection', kind: 'función', does: 'Misma partición que lab2_3 (mismos argumentos).' },
      ],
      lines: [],
      shapes: [
        { expr: 'X_train', shape: '(3156, 256)', meaning: 'todas las frecuencias' },
        { expr: 'X_test', shape: '(1353, 256)', meaning: '' },
      ],
      interpretation: 'Usar la **misma** partición que lab2_3 permite comparar directamente con los errores de 64 features y de la selección manual. El centrado es irrelevante para PCA (lo hace internamente) y para LDA/QDA.',
      next: 'Ajuste de PCA.',
      whatIf: [{ change: 'Recortar a 64 features antes de PCA', effect: 'Se estudiaría otro problema (0–2 kHz); la guía pide el vector completo.' }],
    },
    {
      id: 'b4_pca',
      title: 'Ajustar PCA con train y proyectar train y test',
      cells: [],
      stage: 'metodo',
      origin: 'app',
      runCode: PCA_FIT,
      problem: 'Obtener las direcciones de máxima varianza a partir de los datos de entrenamiento.',
      purpose: 'Se ajusta **una vez** un PCA con todas las componentes y después se usan las `nfeat` primeras columnas de la proyección: las componentes de PCA están **anidadas** (las d primeras de PCA completo coinciden, salvo el signo, con `PCA(n_components=d)`), así que es equivalente y mucho más rápido que reajustar en cada iteración.',
      inputs: '`X_train`, `X_test`.',
      outputs: '`pca` (components_ 256×256), `X_train_pca` (3156, 256), `X_test_pca` (1353, 256).',
      prereqs: ['pca', 'proyeccion', 'varianza-explicada'],
      apis: [
        { name: 'PCA()', lib: 'sklearn.decomposition', kind: 'clase', does: 'Sin `n_components` conserva todas las componentes (min(n_samples, n_features)).', learned: ['components_', 'explained_variance_', 'explained_variance_ratio_', 'mean_'] },
        { name: 'pca.fit(X_train)', lib: 'sklearn', kind: 'método', does: 'Aprende media y componentes **solo con train**.' },
        { name: 'pca.transform(X)', lib: 'sklearn', kind: 'método', does: 'Centra con la media de train y proyecta: (X − mean_) · components_ᵀ.', returns: '(n, n_components)' },
      ],
      lines: [],
      shapes: [{ expr: 'pca.components_', shape: '(256, 256)', meaning: 'fila k = dirección de la componente k' }],
      interpretation: 'PCA ordena las direcciones por varianza explicada, **no** por capacidad de separar clases: las primeras componentes recogen la energía global del espectro, que no siempre es lo que distingue aa de ao.',
      next: 'Bucle sobre nfeat.',
      whatIf: [
        { change: '`pca.fit(X)` con todo el dataset', effect: 'Fuga de información del test: la estimación del error deja de ser independiente (en este caso el efecto numérico es pequeño, pero el procedimiento es incorrecto).' },
        { change: '`pca.fit_transform(X_test)`', effect: 'Se aprendería otra proyección distinta para el test: train y test estarían en ejes diferentes y el clasificador fallaría.' },
      ],
    },
    {
      id: 'b4_loop_pca',
      title: 'Bucle: LDA y QDA para cada número de componentes',
      cells: [],
      stage: 'metodo',
      origin: 'app',
      runCode: PCA_LOOP,
      problem: 'Medir cómo cambian los errores al aumentar la dimensión.',
      purpose: 'Para cada `nfeat` entrena LDA y QDA sobre las primeras componentes y guarda los 4 errores. La rejilla de la app (23 valores entre 1 y 256) tarda ≈10 s; la guía pide 1..256 completo, que es el mismo código con `dims = range(1, 257)`.',
      inputs: '`X_train_pca`, `X_test_pca`, `y_train`, `y_test`.',
      outputs: 'Listas `lda_tr`, `lda_te`, `qda_tr`, `qda_te`.',
      prereqs: ['lda', 'qda', 'sobreajuste'],
      apis: [
        { name: 'X_train_pca[:, :nfeat]', lib: 'numpy', kind: 'operador', does: 'Primeras nfeat componentes.' },
        { name: 'LinearDiscriminantAnalysis / QuadraticDiscriminantAnalysis', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Clasificadores reentrenados en cada dimensión.' },
      ],
      lines: [],
      interpretation: 'Cada punto de las curvas es un modelo distinto entrenado con train y evaluado con test. Los clasificadores **se reentrenan** en cada dimensión; la proyección no (por estar anidada).',
      next: 'Dibujar las cuatro curvas.',
      whatIf: [{ change: 'Evaluar todas las dimensiones 1..256', effect: 'Curvas más suaves; mismas conclusiones; unas 10 veces más lento en la app.' }],
    },
    {
      id: 'b4_plot_pca',
      title: 'Figura con las cuatro curvas de error (Q10)',
      cells: [],
      stage: 'resultados',
      origin: 'app',
      runCode: PCA_PLOT,
      problem: 'Mostrar en una figura train/test × LC/QC, como pide la guía.',
      purpose: 'Visualizar sobreajuste: brecha entre línea discontinua (train) y continua (test).',
      inputs: 'Listas de errores y `dims`.',
      outputs: 'Figura con 4 curvas (eje x logarítmico para ver bien las dimensiones bajas).',
      prereqs: ['sobreajuste'],
      apis: [{ name: 'plt.plot / plt.xscale("log")', lib: 'matplotlib', kind: 'función', does: 'Curvas y escala logarítmica en x.' }],
      lines: [],
      interpretation: 'Con 1–3 componentes ambos clasificadores fallan mucho (40 % → 12 %). LC baja hasta ≈7 % hacia 10–20 componentes y luego se mantiene casi plano. QC baja a ≈8 % con 7–10 componentes y, a partir de ~30, su error de **train** sigue bajando hasta ≈0 mientras el de **test** sube hasta ≈17 % con 256: sobreajuste clásico.',
      next: 'Elegir la dimensión (Q11).',
      whatIf: [{ change: 'Eje x lineal', effect: 'Las dimensiones bajas, donde ocurre lo interesante, quedan comprimidas a la izquierda.' }],
    },
    {
      id: 'b4_choose_pca',
      title: 'Elegir la dimensión para LC y para QC (Q11)',
      cells: [],
      stage: 'interpretacion',
      origin: 'app',
      noRun: true,
      problem: 'Decidir cuántas componentes usar sin sobreajustar.',
      purpose: 'Responde Q11 con el criterio de la guía: buen error de test y test no mucho mayor que train, preferiblemente con pocas dimensiones.',
      inputs: 'Curvas del bloque anterior.',
      outputs: 'Una dimensión razonable para cada clasificador y su justificación.',
      prereqs: ['sobreajuste', 'maldicion-dimension'],
      apis: [],
      lines: [],
      interpretation: '**LC**: hacia 10–20 componentes el error de test (≈7.1–7.3 %) ya es casi el mínimo (≈6.8 % con 100) y train ≈ test; añadir más apenas mejora. **QC**: el mejor test aparece con ≈7–8 componentes (≈8 %), con train y test parecidos; por encima de ~30 la brecha crece. La elección con el **mínimo error de train** sería un error conceptual: en QC llevaría a 256 componentes y al peor test. En rigor, elegir la dimensión mirando el test convierte el test en validación: lo correcto sería reservar un conjunto de validación o usar validación cruzada (Lab5).',
      next: 'MDA.',
      whatIf: [{ change: 'Regularizar QDA (`reg_param`)', effect: 'Suaviza el sobreajuste de QC en dimensiones altas (ver experimento).' }],
    },
    {
      id: 'b4_mda',
      title: "MDA: d' de 1 a dmax = min(d, c − 1) = 4 (Q12)",
      cells: [],
      stage: 'metodo',
      origin: 'app',
      runCode: MDA_LOOP,
      problem: 'Reducir con una proyección **supervisada** que maximiza la separación entre clases.',
      purpose: 'En scikit-learn, MDA (Fisher) es `LinearDiscriminantAnalysis(n_components=d).fit(X, y).transform(X)`. Como $S_B$ tiene rango $c-1$, como mucho hay **4** direcciones discriminantes con 5 clases.',
      inputs: '`X_train`, `y_train`, `X_test`.',
      outputs: '`dmax = 4` y listas de errores `mlda_tr`, `mlda_te`, `mqda_tr`, `mqda_te`.',
      prereqs: ['mda', 'scatter-matrix', 'autovalor'],
      apis: [
        { name: 'LinearDiscriminantAnalysis(n_components=d)', lib: 'sklearn.discriminant_analysis', kind: 'clase', does: 'Usado como transformador: proyecta en las d direcciones de Fisher.', args: [{ name: 'n_components', value: 'd ≤ 4', meaning: 'si d > min(n_features, n_classes − 1) lanza ValueError' }], learned: ['scalings_', 'explained_variance_ratio_'] },
        { name: 'mda.transform(X)', lib: 'sklearn', kind: 'método', does: 'Proyección sobre las direcciones discriminantes.', returns: '(n, d)' },
      ],
      lines: [],
      shapes: [{ expr: 'mda.transform(X_test)', shape: "(1353, d')", meaning: 'd\' ≤ 4' }],
      interpretation: 'A diferencia de PCA, MDA **usa las etiquetas** de train para elegir las direcciones. Con d\' = 4: LC 6.2 % train / 7.3 % test, QC 5.9 % / 7.4 %: QC ya no sobreajusta porque trabaja en 4 dimensiones.',
      next: 'Curvas de MDA.',
      whatIf: [{ change: '`n_components=10`', effect: '`ValueError: n_components cannot be larger than min(n_features, n_classes - 1)`.' }],
    },
    {
      id: 'b4_plot_mda',
      title: 'Curvas de error de MDA',
      cells: [],
      stage: 'resultados',
      origin: 'app',
      runCode: MDA_PLOT,
      problem: 'Visualizar las cuatro curvas para d\' = 1..4.',
      purpose: 'Igual que con PCA, en una figura (Q12).',
      inputs: 'Listas de errores de MDA.',
      outputs: 'Figura con 4 curvas sobre 4 dimensiones.',
      prereqs: ['mda'],
      apis: [{ name: 'plt.plot', lib: 'matplotlib', kind: 'función', does: 'Curvas con marcadores.' }],
      lines: [],
      interpretation: 'El error cae de forma pronunciada con cada dimensión (≈47 % → 28 % → 14 % → 7 %): las 4 direcciones aportan información discriminante, y la última es necesaria para separar aa de ao.',
      next: 'Ampliación por bloques y comparación.',
      whatIf: [],
    },
    {
      id: 'b4_blocks',
      title: 'Ampliación: MDA por grupos para obtener más de c − 1 features',
      cells: [],
      stage: 'metodo',
      origin: 'ampliacion',
      runCode: MDA_BLOCKS,
      problem: 'Superar el límite de 4 features de MDA.',
      purpose: 'Las slides (p. 8: «if we need more than min(d, c−1) features we can define groups of features in x and compute MDA on each group»; p. 18: «in blocks of 4 features») sugieren calcular MDA por separado en grupos de features y concatenar. Aquí: 4 grupos de 64 frecuencias × 4 = 16 features.',
      inputs: '`X_train`, `X_test`, `y_train`.',
      outputs: '`Ztr_b` (3156, 16), `Zte_b` (1353, 16) y el error de QDA.',
      prereqs: ['mda'],
      apis: [{ name: 'np.hstack', lib: 'numpy', kind: 'función', does: 'Concatena las proyecciones por columnas.' }],
      lines: [],
      interpretation: 'Es una interpretación de la app de lo que indican las slides; la guía no lo pide explícitamente. Permite explorar dimensiones intermedias con un criterio supervisado.',
      whatIf: [{ change: 'Grupos de otro tamaño', effect: 'Cambia el número total de features (4 por grupo).' }],
    },
    {
      id: 'b4_compare',
      title: 'PCA frente a MDA en Phoneme (Q13)',
      cells: [],
      stage: 'interpretacion',
      origin: 'app',
      noRun: true,
      problem: 'Comparar los dos métodos con el mismo número de features (1 a dmax).',
      purpose: 'Responde Q13.',
      inputs: 'Curvas de PCA y MDA.',
      outputs: 'Conclusiones.',
      prereqs: ['pca', 'mda'],
      apis: [],
      lines: [],
      interpretation: 'Con **4 features**, MDA (LC 7.3 %, QC 7.4 % en test) es mucho mejor que PCA (≈10.8 % y ≈10.1 %): MDA elige direcciones que separan las clases; PCA, direcciones de máxima varianza, que no tienen por qué ser discriminantes (slides, p. 7). Con 1–2 features PCA es incluso mejor que MDA en este caso, porque la primera componente de PCA ya separa algo y MDA con 1 dimensión solo puede ordenar las clases en una recta. Además, **LC sobre las 4 componentes MDA da el mismo error que LDA con las 256 features** (7.32 %): el clasificador LDA solo usa las $c-1$ direcciones discriminantes. PCA necesita ~10–20 componentes para igualarlo.',
      whatIf: [],
    },
  ],
  experiments: [
    {
      id: 'lab2_4.exp.curves',
      title: 'Curvas de error frente a la dimensión: PCA o MDA',
      goal: 'Generar las curvas de Q10 y Q12 cambiando el método, la rejilla de dimensiones, la partición y la regularización de QDA. Todo se ajusta con train; el test solo evalúa.',
      blocks: ['b4_pca', 'b4_loop_pca', 'b4_plot_pca', 'b4_mda', 'b4_plot_mda'],
      data: DATA,
      template: EXP,
      seedParam: 'seed',
      params: [
        { id: 'metodo', label: 'Método de reducción', kind: 'preprocesado', control: 'select', default: "'PCA'", options: [{ value: "'PCA'", label: 'PCA (no supervisado)' }, { value: "'MDA'", label: 'MDA (supervisado)' }], variable: 'metodo', computation: 'PCA: proyección por varianza aprendida con X_train. MDA: proyección de Fisher aprendida con X_train e y_train, d\' ≤ 4.', result: 'MDA alcanza con 4 features lo que PCA necesita con 10–20.', why: 'MDA maximiza la separación entre clases; PCA ignora las etiquetas.' },
        { id: 'dims', label: 'Dimensiones (solo PCA)', kind: 'evaluacion', control: 'select', default: '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 64, 80, 100, 128, 160, 200, 256]', options: [{ value: '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 64, 80, 100, 128, 160, 200, 256]', label: 'Rejilla 1…256 (rápida, ≈10 s)' }, { value: 'list(range(1, 41))', label: '1…40 una a una' }, { value: 'list(range(1, 257))', label: '1…256 completa (lenta, ~1–2 min)' }], variable: 'dims', computation: 'Qué dimensiones se evalúan.', result: 'Solo cambia la resolución de las curvas.', why: 'Cada dimensión es un entrenamiento más; los resultados en los puntos comunes son idénticos.' },
        { id: 'reg', label: 'reg_param de QDA', kind: 'modelo', control: 'select', default: '0.0', options: [{ value: '0.0', label: '0 (por defecto)' }, { value: '0.05', label: '0.05' }, { value: '0.2', label: '0.2' }], variable: 'reg_param', computation: 'Regulariza las covarianzas de QDA.', result: 'Reduce la subida del error de test de QC en dimensiones altas.', why: 'Menos varianza en las covarianzas estimadas.', origin: 'ampliacion' },
        { id: 'trainsize', label: 'Proporción de train', kind: 'datos', control: 'select', default: '0.7', options: [{ value: '0.4', label: '0.4' }, { value: '0.7', label: '0.7 (guía)' }, { value: '0.9', label: '0.9' }], variable: 'train_size', computation: 'Muestras disponibles para estimar proyección y clasificador.', result: 'Con menos train, QC sobreajusta antes (en dimensiones más bajas).', why: 'QDA necesita más muestras por clase cuantas más dimensiones hay.' },
        { id: 'seed', label: 'random_state', kind: 'datos', control: 'range', min: 0, max: 30, step: 1, default: '5', variable: 'random_state', computation: 'Otra partición.', result: 'La dimensión óptima puede moverse algo.', why: 'Las curvas tienen variabilidad muestral: no sobreinterpretar el mínimo exacto.' },
        { id: 'logx', label: 'Eje x logarítmico', kind: 'visualizacion', control: 'toggle', default: 'True', variable: 'plt.xscale', computation: 'Ninguno.', result: 'Solo la escala del eje.', why: 'Parámetro de visualización.' },
      ],
      metrics: [
        { key: 'n_dims', label: 'Dimensiones evaluadas', format: 'int' },
        { key: 'dim_max', label: 'Dimensión máxima', format: 'int' },
        { key: 'lda_best_dim', label: 'Dimensión con menor error de test (LC)', format: 'int' },
        { key: 'lda_best_test', label: 'Menor error de test (LC)', format: 'pct', better: 'low' },
        { key: 'qda_best_dim', label: 'Dimensión con menor error de test (QC)', format: 'int' },
        { key: 'qda_best_test', label: 'Menor error de test (QC)', format: 'pct', better: 'low' },
        { key: 'lda_test_last', label: 'Error de test LC en la dimensión máxima', format: 'pct', better: 'low' },
        { key: 'qda_train_last', label: 'Error de train QC en la dimensión máxima', format: 'pct' },
        { key: 'qda_test_last', label: 'Error de test QC en la dimensión máxima', format: 'pct', better: 'low' },
        { key: 'gap_qda_last', label: 'Brecha QC (test − train) en la dimensión máxima', format: 'pct', better: 'low' },
      ],
      observe: [
        'Con PCA, ¿a partir de qué dimensión el error de train de QC sigue bajando pero el de test sube?',
        'Cambia a MDA: compara el error con 4 dimensiones con el de PCA con 4.',
        'Con MDA y 4 dimensiones, compara el error de test de LC con el de PCA con 256: ¿por qué coinciden?',
        'Prueba reg_param = 0.2 con PCA: ¿qué le pasa a la brecha de QC en 256?',
      ],
      source: guide('apartado 6, Q10–Q13'),
      fidelity: [
        { metrics: { lda_test_last: 0.07317073170731703, qda_train_last: 0.000316856780735, qda_test_last: 0.16999260901699928 }, tol: 1e-4, source: 'Ejecución de referencia de la app (PCA, 256 componentes)' },
        { params: { metodo: "'MDA'" }, metrics: { dim_max: 4, lda_test_last: 0.07317073170731703, qda_test_last: 0.0739 }, tol: 2e-3, source: 'Ejecución de referencia de la app (MDA, d\' = 4)' },
      ],
    },
  ],
  activities: [
    // ------------------------------------------------ nivel 1
    {
      kind: 'mcq',
      id: 'lab2_4.read.leak',
      skills: ['concepto', 'metodo'],
      blocks: ['b4_pca'],
      source: guide('apartado 6'),
      prompt: '¿Por qué `pca.fit(X_train)` y no `pca.fit(X)` antes de evaluar el error de test?',
      options: [
        { text: 'Porque la proyección forma parte del modelo: aprenderla con el test introduce información del test en el entrenamiento', correct: true },
        { text: 'Porque PCA no admite más de 3156 muestras', feedback: 'No hay tal límite.' },
        { text: 'Porque con X el resultado sería exactamente igual, así que da igual', feedback: 'Puede parecerse, pero el procedimiento deja de garantizar una evaluación independiente.' },
        { text: 'Porque PCA necesita las etiquetas de train', feedback: 'PCA no usa etiquetas; MDA sí.' },
      ],
      explanation: 'Todo lo que se «aprende» de los datos (medias, componentes, escalas, selección de features) debe salir solo de train (slides Lab2, p. 4).',
    },
    {
      kind: 'numeric',
      id: 'lab2_4.read.dmax',
      skills: ['concepto'],
      blocks: ['b4_mda'],
      source: guide('Q12'),
      prompt: 'Phoneme tiene 256 features y 5 clases. ¿Cuál es el número máximo de componentes de MDA, dmax?',
      answer: 4,
      tolerance: 0,
      explanation: "$d' \\le \\min(d, c-1) = \\min(256, 4) = 4$: la matriz de dispersión entre clases $S_B$ (suma de c términos con media común) tiene rango como mucho $c-1$.",
    },
    {
      kind: 'mcq',
      id: 'lab2_4.read.ncomp',
      skills: ['api'],
      blocks: ['b4_mda'],
      source: guide('Q12'),
      prompt: '¿Qué ocurre al ejecutar `LinearDiscriminantAnalysis(n_components=6).fit(X_train, y_train)` con 5 clases?',
      options: [
        { text: 'Lanza un ValueError porque n_components > min(n_features, n_classes − 1)', correct: true },
        { text: 'Devuelve 6 componentes, las dos últimas nulas', feedback: 'scikit-learn lo comprueba y lanza un error.' },
        { text: 'Devuelve 4 componentes y avisa', feedback: 'No recorta silenciosamente: falla.' },
      ],
      explanation: 'Es un error de **ejecución** que delata un error de **concepto**: MDA no puede dar más de c − 1 direcciones.',
    },
    // ------------------------------------------------ nivel 2
    {
      kind: 'predict',
      id: 'lab2_4.pred.components',
      skills: ['api', 'codigo'],
      blocks: ['b4_pca'],
      data: DATA,
      source: guide('Q10'),
      prompt: 'Tras `pca = PCA().fit(X_train)` con X_train de forma (3156, 256), ¿qué forma tiene `pca.components_`?',
      setup: LOAD,
      code: 'pca = PCA().fit(X_train)',
      expr: 'pca.components_.shape',
      answerKind: 'shape',
      explanation: '(n_components, n_features) = (256, 256): sin `n_components` se conservan min(3156, 256) = 256 componentes. Cada fila es una dirección en el espacio de las 256 frecuencias.',
    },
    {
      kind: 'predict',
      id: 'lab2_4.pred.mdashape',
      skills: ['codigo'],
      blocks: ['b4_mda'],
      data: DATA,
      source: guide('Q12'),
      prompt: '¿Qué forma tiene `mda.transform(X_test)` si `mda = LinearDiscriminantAnalysis(n_components=3).fit(X_train, y_train)`?',
      setup: LOAD,
      code: 'mda = LinearDiscriminantAnalysis(n_components=3).fit(X_train, y_train)\nZ = mda.transform(X_test)',
      expr: 'Z.shape',
      answerKind: 'shape',
      explanation: '1353 muestras de test × 3 direcciones discriminantes.',
    },
    // ------------------------------------------------ nivel 3
    {
      kind: 'code',
      level: 3,
      id: 'lab2_4.fill.pca',
      title: 'Completa la reducción con PCA a 10 componentes',
      skills: ['api', 'codigo'],
      blocks: ['b4_pca'],
      data: DATA,
      source: guide('Q10'),
      practice: 'sklearn.decomposition.PCA',
      prompt: 'Rellena los huecos para ajustar PCA con 10 componentes **solo con train** y proyectar train y test.',
      available: '`X_train` (3156, 256), `X_test` (1353, 256), `PCA` importado.',
      produce: '`X_train_pca` (3156, 10) y `X_test_pca` (1353, 10).',
      setup: LOAD,
      starter: 'pca = PCA(n_components=___)\npca.fit(___)\nX_train_pca = pca.___(X_train)\nX_test_pca = pca.___(X_test)',
      solution: 'pca = PCA(n_components=10)\npca.fit(X_train)\nX_train_pca = pca.transform(X_train)\nX_test_pca = pca.transform(X_test)',
      alternatives: ['pca = PCA(10)\nX_train_pca = pca.fit_transform(X_train)\nX_test_pca = pca.transform(X_test)'],
      wrong: [{ code: 'pca = PCA(n_components=10)\nX_train_pca = pca.fit_transform(X_train)\nX_test_pca = pca.fit_transform(X_test)', expect: 'razonamiento' }],
      checks: "c.shape('X_train_pca', (3156, 10))\nc.same('X_train_pca', rtol=1e-6, atol=1e-6)\nc.same('X_test_pca', rtol=1e-6, atol=1e-6, desc='Proyección del test con la PCA de train')",
      hints: ['`fit` solo con los datos de entrenamiento.', 'Para aplicar la proyección aprendida se usa `transform`, también con el test.'],
      explanation: '`fit_transform(X_test)` aprendería otra PCA distinta para el test: sus ejes no serían los mismos que los de train.',
    },
    // ------------------------------------------------ nivel 4
    {
      kind: 'code',
      level: 4,
      id: 'lab2_4.fix.leak',
      title: 'Corrige la fuga de información del test',
      skills: ['metodo', 'codigo'],
      blocks: ['b4_pca'],
      data: DATA,
      source: guide('apartado 6, nota'),
      prompt: 'Este código calcula la PCA con **todo** el dataset antes de evaluar. Corrígelo para que cumpla la regla de la guía.',
      available: '`X`, `y` (completos), `X_train`, `X_test`, `y_train`, `y_test`; PCA, LDA y `accuracy_score` importados.',
      produce: '`X_test_pca` (1353, 10) y `err_test`: error de test de LDA con 10 componentes.',
      setup: LOAD,
      starter: 'pca = PCA(n_components=10).fit(X)\nX_train_pca = pca.transform(X_train)\nX_test_pca = pca.transform(X_test)\nlda = LinearDiscriminantAnalysis().fit(X_train_pca, y_train)\nerr_test = 1 - accuracy_score(y_test, lda.predict(X_test_pca))',
      solution: 'pca = PCA(n_components=10).fit(X_train)\nX_train_pca = pca.transform(X_train)\nX_test_pca = pca.transform(X_test)\nlda = LinearDiscriminantAnalysis().fit(X_train_pca, y_train)\nerr_test = 1 - accuracy_score(y_test, lda.predict(X_test_pca))',
      wrong: [{ code: 'pca = PCA(n_components=10).fit(X)\nX_train_pca = pca.transform(X_train)\nX_test_pca = pca.transform(X_test)\nlda = LinearDiscriminantAnalysis().fit(X_train_pca, y_train)\nerr_test = 1 - accuracy_score(y_test, lda.predict(X_test_pca))', expect: 'razonamiento' }],
      checks: "c.same('X_test_pca', rtol=1e-6, atol=1e-6, desc='Proyección del test (PCA ajustada con train)')\nc.same('err_test', rtol=1e-9)",
      hints: ['¿Con qué datos se ha llamado a `fit`?', 'El test solo puede aparecer en `transform` y en la evaluación.'],
    },
    {
      kind: 'code',
      level: 4,
      id: 'lab2_4.fix.mdacomp',
      title: 'Corrige el número de componentes de MDA',
      skills: ['api', 'concepto'],
      blocks: ['b4_mda'],
      data: DATA,
      source: guide('Q12'),
      prompt: 'Se quiere la proyección MDA con **el máximo** número de componentes posible, pero el código falla. Calcula `dmax` a partir de los datos (no lo escribas a mano) y úsalo.',
      available: '`X_train`, `X_test`, `y_train`; `LinearDiscriminantAnalysis` importado.',
      produce: '`dmax` (int) y `X_test_mda` con forma (1353, dmax).',
      setup: LOAD,
      starter: 'mda = LinearDiscriminantAnalysis(n_components=10).fit(X_train, y_train)\nX_test_mda = mda.transform(X_test)',
      solution: 'dmax = min(X_train.shape[1], len(np.unique(y_train)) - 1)\nmda = LinearDiscriminantAnalysis(n_components=dmax).fit(X_train, y_train)\nX_test_mda = mda.transform(X_test)',
      alternatives: ['dmax = min(X_train.shape[1], np.unique(y_train).size - 1)\nX_test_mda = LinearDiscriminantAnalysis().fit(X_train, y_train).transform(X_test)'],
      wrong: [{ code: 'mda = LinearDiscriminantAnalysis(n_components=10).fit(X_train, y_train)\nX_test_mda = mda.transform(X_test)', expect: 'ejecucion' }],
      checks: "c.same('dmax')\nc.shape('X_test_mda', (1353, 4))\nc.same('X_test_mda', rtol=1e-6, atol=1e-8)",
      hints: ['El límite es min(d, c − 1).', '`np.unique(y_train)` da las clases distintas.', 'Sin `n_components`, LDA.transform ya usa min(d, c − 1).'],
    },
    // ------------------------------------------------ nivel 5
    {
      kind: 'code',
      level: 5,
      id: 'lab2_4.mem.pcaloop',
      title: 'Escribe el bucle de PCA y las cuatro curvas',
      skills: ['escritura', 'metodo'],
      blocks: ['b4_loop_pca', 'b4_plot_pca'],
      data: DATA,
      source: guide('Q10'),
      prompt: 'Para `dims = [1, 2, 5, 10, 20, 50]`: reduce con PCA (ajustada con train), entrena LDA y QDA en cada dimensión y guarda los errores de train y test. Dibuja las cuatro curvas en **una** figura.',
      available: '`X_train`, `X_test`, `y_train`, `y_test` (256 features). Nada de scikit-learn importado; `np` y `plt` sí.',
      produce: 'Listas o arrays `lda_tr`, `lda_te`, `qda_tr`, `qda_te` (longitud 6, en el orden de `dims`) y una figura.',
      setup: LOAD.replace('from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nfrom sklearn.metrics import accuracy_score\n', ''),
      starter: 'dims = [1, 2, 5, 10, 20, 50]\n',
      solution: 'from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nfrom sklearn.metrics import accuracy_score\ndims = [1, 2, 5, 10, 20, 50]\npca = PCA().fit(X_train)\nPtr, Pte = pca.transform(X_train), pca.transform(X_test)\nlda_tr, lda_te, qda_tr, qda_te = [], [], [], []\nfor k in dims:\n    l = LinearDiscriminantAnalysis().fit(Ptr[:, :k], y_train)\n    q = QuadraticDiscriminantAnalysis().fit(Ptr[:, :k], y_train)\n    lda_tr.append(1 - accuracy_score(y_train, l.predict(Ptr[:, :k])))\n    lda_te.append(1 - accuracy_score(y_test, l.predict(Pte[:, :k])))\n    qda_tr.append(1 - accuracy_score(y_train, q.predict(Ptr[:, :k])))\n    qda_te.append(1 - accuracy_score(y_test, q.predict(Pte[:, :k])))\nplt.plot(dims, lda_tr, label="LC train")\nplt.plot(dims, lda_te, label="LC test")\nplt.plot(dims, qda_tr, label="QC train")\nplt.plot(dims, qda_te, label="QC test")\nplt.legend()\nplt.show()',
      alternatives: ['from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA\ndims = [1, 2, 5, 10, 20, 50]\nres = {"lda_tr": [], "lda_te": [], "qda_tr": [], "qda_te": []}\nfor k in dims:\n    p = PCA(n_components=k).fit(X_train)\n    Ztr, Zte = p.transform(X_train), p.transform(X_test)\n    for name, M in (("lda", LDA), ("qda", QDA)):\n        m = M().fit(Ztr, y_train)\n        res[name + "_tr"].append(1 - m.score(Ztr, y_train))\n        res[name + "_te"].append(1 - m.score(Zte, y_test))\nlda_tr, lda_te, qda_tr, qda_te = res["lda_tr"], res["lda_te"], res["qda_tr"], res["qda_te"]\nfig, ax = plt.subplots()\nfor key in res:\n    ax.plot(dims, res[key], label=key)\nax.legend()\nplt.show()'],
      wrong: [{ code: 'from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA\ndims = [1, 2, 5, 10, 20, 50]\nlda_tr, lda_te, qda_tr, qda_te = [], [], [], []\nfor k in dims:\n    p = PCA(n_components=k).fit(X_train)\n    Ztr = p.transform(X_train)\n    Zte = PCA(n_components=k).fit_transform(X_test)\n    l, q = LDA().fit(Ztr, y_train), QDA().fit(Ztr, y_train)\n    lda_tr.append(1 - l.score(Ztr, y_train)); lda_te.append(1 - l.score(Zte, y_test))\n    qda_tr.append(1 - q.score(Ztr, y_train)); qda_te.append(1 - q.score(Zte, y_test))\nplt.plot(dims, lda_te)\nplt.show()', expect: 'razonamiento' }],
      checks: "c.same('lda_tr', rtol=1e-9)\nc.same('lda_te', rtol=1e-9)\nc.same('qda_tr', rtol=1e-9)\nc.same('qda_te', rtol=1e-9)\nc.figures(1)",
      hints: ['Ajusta la PCA una vez con train y usa las k primeras columnas de la proyección (o reajusta con n_components=k: es equivalente).', 'Entrena los clasificadores con train proyectado y evalúa con test proyectado con **la misma** PCA.'],
      explanation: 'La solución alternativa reajusta PCA para cada k y da exactamente los mismos errores: las componentes están anidadas.',
    },
    {
      kind: 'code',
      level: 5,
      id: 'lab2_4.mem.mdaloop',
      title: "Escribe el bucle de MDA para d' = 1..dmax",
      skills: ['escritura', 'metodo'],
      blocks: ['b4_mda', 'b4_plot_mda'],
      data: DATA,
      source: guide('Q12'),
      prompt: "Calcula dmax a partir de los datos y, para d' = 1..dmax, proyecta con MDA (ajustado con train), entrena LDA y QDA y guarda los errores. Dibuja las curvas.",
      available: '`X_train`, `X_test`, `y_train`, `y_test`; LDA, QDA y `accuracy_score` importados; `np`, `plt`.',
      produce: '`dmax` y listas `mlda_tr`, `mlda_te`, `mqda_tr`, `mqda_te` de longitud dmax; una figura.',
      setup: LOAD,
      starter: '',
      solution: MDA_LOOP + MDA_PLOT,
      alternatives: ['dmax = min(X_train.shape[1], len(set(y_train)) - 1)\nout = {k: [] for k in ["mlda_tr", "mlda_te", "mqda_tr", "mqda_te"]}\nfor d in range(1, dmax + 1):\n    m = LinearDiscriminantAnalysis(n_components=d).fit(X_train, y_train)\n    A, B = m.transform(X_train), m.transform(X_test)\n    for pre, M in (("mlda", LinearDiscriminantAnalysis), ("mqda", QuadraticDiscriminantAnalysis)):\n        c_ = M().fit(A, y_train)\n        out[pre + "_tr"].append(1 - c_.score(A, y_train))\n        out[pre + "_te"].append(1 - c_.score(B, y_test))\nmlda_tr, mlda_te, mqda_tr, mqda_te = out["mlda_tr"], out["mlda_te"], out["mqda_tr"], out["mqda_te"]\nplt.plot(range(1, dmax + 1), mlda_te)\nplt.show()'],
      checks: "c.same('dmax')\nc.same('mlda_tr', rtol=1e-9)\nc.same('mlda_te', rtol=1e-9)\nc.same('mqda_tr', rtol=1e-9)\nc.same('mqda_te', rtol=1e-9)\nc.figures(1)",
      hints: ['dmax = min(n_features, n_classes − 1).', 'MDA en scikit-learn: `LinearDiscriminantAnalysis(n_components=d).fit(X_train, y_train).transform(...)`.'],
    },
    // ------------------------------------------------ nivel 6
    {
      kind: 'code',
      level: 6,
      id: 'lab2_4.adapt.digits',
      title: 'Transferencia: PCA frente a MDA en el dataset digits',
      skills: ['metodo', 'escritura'],
      blocks: ['b4_mda', 'b4_compare'],
      transfer: true,
      source: { origin: 'app', note: 'Variante de transferencia con load_digits (lab2_1) y el procedimiento de lab2_4' },
      prompt: 'Aplica el mismo análisis a **digits** (`sklearn.datasets.load_digits`: 1797 imágenes 8×8, 64 features, 10 clases). Parte 70/30 estratificado con `random_state=0`. Calcula el dmax de MDA y compara el error de **test** de LDA usando dmax componentes de MDA frente a dmax componentes de PCA (ambas ajustadas con train).',
      available: 'Nada importado.',
      produce: '`dmax` (int), `err_mda` y `err_pca` (float).',
      setup: '',
      starter: '',
      solution: 'import numpy as np\nfrom sklearn.datasets import load_digits\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nd = load_digits()\nX_train, X_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.3, random_state=0, stratify=d.target)\ndmax = min(X_train.shape[1], len(np.unique(y_train)) - 1)\nmda = LinearDiscriminantAnalysis(n_components=dmax).fit(X_train, y_train)\nerr_mda = 1 - LinearDiscriminantAnalysis().fit(mda.transform(X_train), y_train).score(mda.transform(X_test), y_test)\npca = PCA(n_components=dmax).fit(X_train)\nerr_pca = 1 - LinearDiscriminantAnalysis().fit(pca.transform(X_train), y_train).score(pca.transform(X_test), y_test)',
      checks: "c.same('dmax')\nc.same('err_mda', rtol=1e-9)\nc.same('err_pca', rtol=1e-9)",
      hints: ['Con 10 clases, dmax = min(64, 9).', 'Mismo patrón que en Phoneme: fit con train, transform de ambos, clasificador sobre la proyección.'],
      explanation: 'MDA vuelve a ganar con el mismo número de componentes, porque usa las etiquetas para elegir las direcciones.',
    },
    {
      kind: 'code',
      level: 6,
      id: 'lab2_4.adapt.blocks',
      title: 'Adapta: MDA por grupos de features (más de c − 1 componentes)',
      skills: ['metodo', 'escritura'],
      blocks: ['b4_blocks'],
      data: DATA,
      source: { doc: 'lab2_slides', section: 'p. 8 y 18', origin: 'ampliacion', note: 'Interpretación de la app de «compute MDA on each group»' },
      prompt: 'Divide las 256 features en **8 grupos de 32 frecuencias consecutivas**, calcula en cada grupo una MDA de 4 componentes (ajustada con train), concatena las proyecciones (32 features) y calcula el error de test de QDA.',
      available: '`X_train`, `X_test`, `y_train`, `y_test`; LDA, QDA, `accuracy_score`, `np`.',
      produce: '`Ztr` (3156, 32), `Zte` (1353, 32) y `err_qda` (float).',
      setup: LOAD,
      starter: '',
      solution: 'Ztr_l, Zte_l = [], []\nfor g in range(8):\n    cols = slice(32 * g, 32 * (g + 1))\n    m = LinearDiscriminantAnalysis(n_components=4).fit(X_train[:, cols], y_train)\n    Ztr_l.append(m.transform(X_train[:, cols]))\n    Zte_l.append(m.transform(X_test[:, cols]))\nZtr, Zte = np.hstack(Ztr_l), np.hstack(Zte_l)\nqda = QuadraticDiscriminantAnalysis().fit(Ztr, y_train)\nerr_qda = 1 - accuracy_score(y_test, qda.predict(Zte))',
      checks: "c.shape('Ztr', (3156, 32))\nc.same('Ztr', rtol=1e-6, atol=1e-8)\nc.same('err_qda', rtol=1e-9)",
      hints: ['`X_train[:, 32*g:32*(g+1)]` selecciona el grupo g.', 'Ajusta cada MDA con su grupo de train y transforma con ella train y test.', '`np.hstack` concatena por columnas.'],
    },
    // ------------------------------------------------ interpretación
    {
      kind: 'mcq',
      id: 'lab2_4.int.curves',
      skills: ['interpretacion'],
      blocks: ['b4_plot_pca', 'b4_choose_pca'],
      figure: { experiment: 'lab2_4.exp.curves', figureIndex: 0 },
      source: guide('Q10–Q11'),
      prompt: 'Curvas de PCA (figura calculada por la app). ¿Qué describe mejor el comportamiento de QC por encima de ~30 componentes?',
      options: [
        { text: 'Su error de train sigue bajando hacia 0 mientras el de test sube: sobreajuste', correct: true },
        { text: 'Ambos errores bajan: conviene usar las 256 componentes', feedback: 'El de test sube hasta ≈17 %.' },
        { text: 'Ambos errores suben: las componentes altas son ruido para los dos conjuntos', feedback: 'El de train baja: QC memoriza el train.' },
      ],
      explanation: 'Con d componentes QDA estima 5·d(d+1)/2 parámetros de covarianza: con 256 son 164 480 para ~3000 muestras.',
    },
    {
      kind: 'text',
      id: 'lab2_4.int.q11',
      skills: ['interpretacion', 'metodo'],
      blocks: ['b4_choose_pca'],
      source: guide('Q11'),
      prompt: 'Q11 · ¿Qué dimensión elegirías para el clasificador lineal y cuál para el cuadrático? Justifica con las curvas y con el criterio de no sobreajustar.',
      rubric: [
        'Para LC, una dimensión de ~10–20 componentes: el error de test ya es casi mínimo y train ≈ test.',
        'Para QC, una dimensión menor (~7–10): antes de que el error de test empiece a subir.',
        'Justifica con la brecha train–test: por encima de ~30 componentes QC sobreajusta.',
        'No se elige por el mínimo error de train.',
        'Menciona que elegir con el test lo usa como validación (lo correcto sería validación o validación cruzada).',
      ],
      model: 'Para el clasificador lineal elegiría unas 15–20 componentes: el error de test (≈7.1 %) está casi en su mínimo y coincide con el de train, así que no hay sobreajuste; usar más apenas mejora y cuesta más. Para el cuadrático elegiría unas 7–8 componentes (≈8 % en test, train parecido): a partir de ~30 su error de train sigue bajando hacia 0 pero el de test sube hasta ≈17 % con 256, señal de sobreajuste por el número de parámetros de las covarianzas. No se debe elegir el mínimo error de train. Además, como la dimensión se elige mirando el test, este pasa a ser un conjunto de validación: para una estimación final honesta habría que usar validación cruzada o un tercer conjunto.',
    },
    {
      kind: 'mcq',
      id: 'lab2_4.int.q13',
      skills: ['metodo', 'interpretacion'],
      blocks: ['b4_compare'],
      source: guide('Q13'),
      prompt: 'Con 4 features, MDA da ≈7.3 % de error de test con LC y PCA ≈10.8 %. ¿Cuál es la explicación?',
      options: [
        { text: 'MDA elige direcciones que maximizan la separación entre clases usando las etiquetas; PCA elige las de máxima varianza sin mirarlas', correct: true },
        { text: 'MDA es no lineal y PCA lineal', feedback: 'Ambas son proyecciones lineales $y = W^Tx$.' },
        { text: 'PCA pierde varianza y MDA no', feedback: 'Con 4 dimensiones ambas descartan mucha varianza; MDA conserva la que discrimina.' },
        { text: 'MDA usa más datos de entrenamiento', feedback: 'Ambas usan el mismo X_train.' },
      ],
      explanation: 'La varianza máxima no implica separabilidad (slides Lab2, p. 7). En Phoneme las primeras componentes de PCA recogen la energía global del espectro, común a varias clases.',
    },
    {
      kind: 'mcq',
      id: 'lab2_4.int.lda4',
      skills: ['concepto', 'metodo'],
      blocks: ['b4_compare'],
      source: { doc: 'lab2_slides', section: 'p. 8 y 11', origin: 'app', note: 'Observación de la ejecución de referencia' },
      prompt: 'LC entrenado sobre las 4 componentes MDA obtiene **exactamente** el mismo error (train y test) que LC con las 256 features. ¿Por qué?',
      options: [
        { text: 'Porque la regla de decisión de LDA solo depende de las c − 1 direcciones discriminantes, que son justo las que conserva MDA', correct: true },
        { text: 'Es una casualidad de esta partición', feedback: 'Ocurre siempre (salvo detalles numéricos): es una propiedad de LDA.' },
        { text: 'Porque MDA y PCA coinciden con 4 componentes', feedback: 'No coinciden: PCA con 4 da un error mucho mayor.' },
      ],
      explanation: 'Con covarianza común, los discriminantes de LDA difieren solo en el subespacio generado por las medias (en métrica de Mahalanobis): ese subespacio es el de MDA, de dimensión ≤ c − 1.',
    },
    {
      kind: 'mcq',
      id: 'lab2_4.int.misconception',
      skills: ['concepto', 'interpretacion'],
      blocks: ['b4_choose_pca'],
      source: guide('Q11'),
      prompt: '«El número óptimo de componentes de PCA para QC es 256, porque es donde su error de entrenamiento es mínimo (≈0 %).» ¿Qué error contiene?',
      options: [
        { text: 'Usa el error de entrenamiento para elegir: premia el sobreajuste; hay que mirar el error de test (o de validación)', correct: true },
        { text: 'Ninguno: más componentes siempre conservan más información', feedback: 'Más información no implica mejor generalización cuando hay que estimar más parámetros.' },
        { text: 'Que PCA no admite 256 componentes', feedback: 'Sí las admite (min(3156, 256)).' },
      ],
      explanation: 'Con 256 componentes QC tiene ≈0 % en train y ≈17 % en test: es la peor elección.',
    },
    {
      kind: 'dynamic',
      id: 'lab2_4.dyn.dmax',
      skills: ['concepto'],
      blocks: ['b4_mda'],
      source: guide('Q12', 'Variante generada'),
      answerKind: 'numeric',
      generator: `import numpy as np
rng = np.random.default_rng(SEED)
d = int(rng.choice([2, 3, 8, 13, 30, 64, 256]))
c = int(rng.choice([2, 3, 5, 10, 26]))
variante = {
  'prompt': f'Un dataset tiene {d} features y {c} clases. ¿Cuántas componentes MDA (LinearDiscriminantAnalysis.transform) se pueden obtener como máximo?',
  'answer': min(d, c - 1), 'tolerance': 0,
  'explanation': f"d' ≤ min(d, c − 1) = min({d}, {c - 1}) = {min(d, c - 1)}."
}`,
    },
  ],
  recall: ['lab2_4.fill.pca', 'lab2_4.fix.leak', 'lab2_4.fix.mdacomp', 'lab2_4.mem.pcaloop', 'lab2_4.mem.mdaloop', 'lab2_4.adapt.digits'],
  practice: [
    {
      id: 'lab2_4.practice.q10q13',
      title: 'Notebook 4 completo (Q10–Q13)',
      scenario: 'Escribe el notebook que pide la guía: curvas de PCA, elección de dimensión, curvas de MDA y comparación. Recuerda: proyección y clasificador se aprenden **solo con train**.',
      steps: ['lab2_4.read.leak', 'lab2_4.mem.pcaloop', 'lab2_4.int.q11', 'lab2_4.read.dmax', 'lab2_4.mem.mdaloop', 'lab2_4.int.q13'],
      source: { doc: 'lab2_guide', section: 'apartado 6, Q10–Q13', origin: 'app' },
    },
  ],
  notes: [
    'Notebook propuesto: no existe en tus materiales. Código, figuras y valores son la solución de referencia de la app, verificada con `npm run verify -- lab2_4`.',
    'La rejilla rápida de dimensiones (23 valores) tarda ≈10 s en la app; la rejilla completa 1…256 de la guía, del orden de un minuto o más.',
  ],
};
