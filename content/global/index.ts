import type { GlobalChallenge } from '../../src/content/types';

// Retos de «Ponte a prueba» que combinan varios labs. Cada reto declara los labs que necesita: si alguno
// no está registrado, el reto no aparece. Las preguntas son de elaboración propia (no son de examen).

const IRIS = `import numpy as np
import pandas as pd
iris = pd.read_csv('iris.csv')
features = list(iris.columns[:-1])
`;

const WINE = `import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
`;

const CANCER = `import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
bc = load_breast_cancer()
X, y = bc.data[:, :10], bc.target          # las 10 features «mean»; y = 1 benigno, 0 maligno
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
`;

export const GLOBAL_CHALLENGES: GlobalChallenge[] = [
  // ------------------------------------------------------------------ contexto original
  {
    id: 'g.iris.pipeline',
    title: 'Iris: de la exploración a la elección de clasificador',
    requires: ['lab0', 'lab1'],
    mode: 'contexto',
    scenario: `Con el dataset Iris de Lab0 vas a recorrer todas las etapas: **explorar** la gaussianidad por clase, **valorar** si las covarianzas de las clases se parecen, **elegir** entre un clasificador lineal y uno cuadrático y **justificar** el resultado en test.

Datos: \`iris.csv\` (150 × 5). Nadie te dirá qué función usar en cada paso.`,
    steps: [
      {
        kind: 'code', level: 5, id: 'g.iris.shapiro', title: 'Gaussianidad por clase de una feature',
        skills: ['escritura', 'api'], source: { notebook: 'lab0_3', origin: 'app', note: 'Reto global basado en Q7 de Lab0' },
        prompt: 'Calcula, para cada especie, el p-valor del contraste de Shapiro-Wilk de `petal_width`. Guarda los tres p-valores en una lista o array `pvals` en el orden setosa, versicolor, virginica, y en `rechazo_005` una lista de booleanos que diga si se rechaza H0 al 5 %.',
        available: '`iris` (DataFrame), `features`, `np`, `pd`.', produce: '`pvals` (3 valores) y `rechazo_005` (3 booleanos).',
        setup: IRIS, data: ['iris.csv'], starter: '',
        solution: "from scipy import stats\npvals = [stats.shapiro(g['petal_width']).pvalue for _, g in iris.groupby('species', sort=False)]\nrechazo_005 = [p < 0.05 for p in pvals]",
        alternatives: ["import scipy.stats as ss\npvals = np.array([ss.shapiro(iris.loc[iris.species == s, 'petal_width'])[1] for s in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']])\nrechazo_005 = list(pvals < 0.05)"],
        wrong: [{ code: "from scipy import stats\npvals = [stats.shapiro(g['petal_width']).statistic for _, g in iris.groupby('species', sort=False)]\nrechazo_005 = [p < 0.05 for p in pvals]", expect: 'razonamiento' }],
        checks: "c.same('pvals', rtol=1e-6, desc='Los p-valores')\nc.same('rechazo_005', desc='Las decisiones al 5 %')",
        hints: ['Es un contraste de normalidad de `scipy.stats` que devuelve estadístico W y p-valor.', 'Agrupa por `species` o filtra con una máscara por especie.'],
        explanation: 'Se rechaza la normalidad de petal_width en setosa (p ≈ 2·10⁻⁶) y versicolor (p ≈ 0.027), no en virginica. «No rechazar» no demuestra normalidad.',
      },
      {
        kind: 'mcq', id: 'g.iris.cov', skills: ['metodo', 'interpretacion'], source: { notebook: 'lab0_2', origin: 'app' },
        prompt: 'En los boxplots de Lab0, la dispersión de `petal_length` y `petal_width` de setosa es mucho menor que la de virginica. ¿Qué supuesto de los clasificadores de Lab1 queda en duda?',
        options: [
          { text: 'Que todas las clases compartan la misma matriz de covarianza (supuesto de LDA)', correct: true },
          { text: 'Que las clases tengan priors iguales', feedback: 'Las tres especies tienen 50 muestras: los priors son iguales.' },
          { text: 'Que las features sean discretas', feedback: 'Son medidas continuas en cm.' },
          { text: 'Ninguno: el tamaño de las cajas no tiene que ver con la covarianza', feedback: 'La anchura de la caja refleja la dispersión (varianza) de cada clase.' },
        ],
        explanation: 'Dispersión distinta por clase ⇒ covarianzas distintas (Case 3): el modelo de QDA describe mejor los datos, aunque con pocas muestras LDA puede generalizar igual o mejor.',
      },
      {
        kind: 'code', level: 6, id: 'g.iris.ldaqda', title: 'Compara LDA y QDA en Iris',
        skills: ['escritura', 'metodo'], source: { notebook: 'lab1_1', origin: 'app' },
        prompt: 'Divide Iris en train/test (30 % test, `random_state=0`, estratificado por especie) y calcula el error de test de un clasificador gaussiano lineal y de uno cuadrático.',
        available: '`iris`, `features`, `np`, `pd`.', produce: '`err_lda`, `err_qda` (floats en [0, 1]).',
        setup: IRIS, data: ['iris.csv'], starter: '',
        solution: "from sklearn.model_selection import train_test_split\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nX = iris[features].to_numpy()\ny = iris['species'].to_numpy()\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)\nerr_lda = 1 - LinearDiscriminantAnalysis().fit(X_train, y_train).score(X_test, y_test)\nerr_qda = 1 - QuadraticDiscriminantAnalysis().fit(X_train, y_train).score(X_test, y_test)",
        alternatives: ["from sklearn.model_selection import train_test_split\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA\nfrom sklearn.metrics import accuracy_score\nX_train, X_test, y_train, y_test = train_test_split(iris.iloc[:, :4], iris.species, test_size=0.3, random_state=0, stratify=iris.species)\nerr_lda = 1 - accuracy_score(y_test, LDA().fit(X_train, y_train).predict(X_test))\nerr_qda = 1 - accuracy_score(y_test, QDA().fit(X_train, y_train).predict(X_test))"],
        wrong: [{ code: "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis\nX = iris[features].to_numpy(); y = iris['species'].to_numpy()\nerr_lda = 1 - LinearDiscriminantAnalysis().fit(X, y).score(X, y)\nerr_qda = 1 - QuadraticDiscriminantAnalysis().fit(X, y).score(X, y)", expect: 'razonamiento' }],
        checks: "c.same('err_lda', rtol=1e-9, desc='Error de test del lineal')\nc.same('err_qda', rtol=1e-9, desc='Error de test del cuadrático')",
        hints: ['Lab2_3 usa `train_test_split` con `stratify`.', 'Los clasificadores son los de `sklearn.discriminant_analysis`; `score` devuelve la accuracy.'],
      },
      {
        kind: 'text', id: 'g.iris.justify', skills: ['interpretacion', 'metodo'], source: { origin: 'app' },
        prompt: 'Con 45 muestras de test, ambos clasificadores cometen muy pocos errores. ¿Qué clasificador recomendarías y qué **no** puedes concluir de estos resultados?',
        rubric: ['Con tan pocas muestras de test, una diferencia de 1 error (≈2 %) no es concluyente.', 'Los boxplots sugieren covarianzas distintas (argumento a favor de QDA) pero QDA estima más parámetros.', 'Habría que repetir con otras particiones o validación cruzada para comparar.', 'La separabilidad de setosa hace el problema casi trivial para esa clase; los errores están entre versicolor y virginica.'],
        model: 'Ambos son razonables: el error de test es muy bajo para los dos y la diferencia es de una o dos muestras de 45, lo que no permite afirmar que uno sea mejor. Los boxplots y las varianzas por clase indican covarianzas distintas, así que el modelo de QDA es más realista, pero estima tres covarianzas 4×4 con 35 muestras por clase. Para decidir haría falta repetir con varias particiones (o validación cruzada). Los errores se concentran entre versicolor y virginica; setosa se separa perfectamente por las features del pétalo.',
      },
    ],
    review: {
      'g.iris.shapiro': { notebook: 'lab0_3', label: 'Lab0 parte 3 · Q7 (Shapiro-Wilk)' },
      'g.iris.cov': { notebook: 'lab0_2', label: 'Lab0 parte 2 · boxplots' },
      'g.iris.ldaqda': { notebook: 'lab1_1', block: 'b_fit', label: 'Lab1 parte 1 · entrenar LDA y QDA' },
      'g.iris.justify': { notebook: 'lab1_1', block: 'b_test_eval', label: 'Lab1 parte 1 · evaluación en test' },
    },
  },
  {
    id: 'g.bugs.pipeline',
    title: 'Detecta y corrige: un análisis con tres errores',
    requires: ['lab0', 'lab1', 'lab2'],
    mode: 'contexto',
    scenario: 'Un compañero ha escrito este análisis sobre un dataset gaussiano sintético como los de Lab1/Lab2. Se ejecuta sin errores, pero contiene **tres errores de concepto** (uno de exploración, uno de reducción de dimensión y uno de evaluación). Encuéntralos y corrígelos sin cambiar nada más.',
    steps: [
      {
        kind: 'code', level: 4, id: 'g.bugs.fix', title: 'Corrige el análisis',
        skills: ['codigo', 'concepto', 'metodo'], source: { origin: 'app', note: 'Combina Lab0 (curtosis), Lab2 (PCA en train) y Lab1 (matriz de confusión)' },
        prompt: 'Corrige los tres errores. Pistas de contexto: la curtosis debe ser 0 para una normal; la proyección debe aprenderse solo con entrenamiento; las filas de la matriz de confusión son las clases reales.',
        available: '`X_train`, `X_test` (300 × 3 cada uno), `y_train`, `y_test` (3 clases), `np`, `ss` (scipy.stats).',
        produce: '`kurt` (curtosis de exceso de cada feature de X_train), `err_test` (error de test de LDA tras PCA a 2D) y `cm` (3 × 3).',
        setup: "import numpy as np\nimport scipy.stats as ss\nrng = np.random.default_rng(3)\nM = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1]])\nX_train = np.vstack([rng.normal(M[k], 0.8, size=(100, 3)) for k in range(3)])\ny_train = np.repeat([0, 1, 2], 100)\nX_test = np.vstack([rng.normal(M[k], 0.8, size=(100, 3)) for k in range(3)])\ny_test = np.repeat([0, 1, 2], 100)",
        starter: "from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.metrics import confusion_matrix\n\nkurt = ss.kurtosis(X_train, axis=0, fisher=False)\n\npca = PCA(n_components=2).fit(np.vstack([X_train, X_test]))\nZ_train, Z_test = pca.transform(X_train), pca.transform(X_test)\n\nlda = LinearDiscriminantAnalysis().fit(Z_train, y_train)\npred = lda.predict(Z_test)\nerr_test = np.mean(pred != y_test)\ncm = confusion_matrix(pred, y_test)",
        solution: "from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.metrics import confusion_matrix\n\nkurt = ss.kurtosis(X_train, axis=0, fisher=True)\n\npca = PCA(n_components=2).fit(X_train)\nZ_train, Z_test = pca.transform(X_train), pca.transform(X_test)\n\nlda = LinearDiscriminantAnalysis().fit(Z_train, y_train)\npred = lda.predict(Z_test)\nerr_test = np.mean(pred != y_test)\ncm = confusion_matrix(y_test, pred)",
        alternatives: ["from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.metrics import confusion_matrix\nkurt = ss.kurtosis(X_train, axis=0)\npca = PCA(2)\nZ_train = pca.fit_transform(X_train)\nZ_test = pca.transform(X_test)\npred = LinearDiscriminantAnalysis().fit(Z_train, y_train).predict(Z_test)\nerr_test = 1 - np.mean(pred == y_test)\ncm = confusion_matrix(y_test, pred)"],
        wrong: [{ code: "from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.metrics import confusion_matrix\nkurt = ss.kurtosis(X_train, axis=0)\npca = PCA(n_components=2).fit(np.vstack([X_train, X_test]))\nZ_train, Z_test = pca.transform(X_train), pca.transform(X_test)\npred = LinearDiscriminantAnalysis().fit(Z_train, y_train).predict(Z_test)\nerr_test = np.mean(pred != y_test)\ncm = confusion_matrix(y_test, pred)", expect: 'razonamiento' }],
        checks: "c.same('kurt', rtol=1e-9, desc='La curtosis de exceso (Fisher)')\nc.same('err_test', rtol=1e-9, desc='El error de test con PCA ajustado en train')\nc.same('cm', desc='La matriz de confusión (filas = reales)')",
        hints: ['`fisher=False` devuelve la curtosis de Pearson (3 para una normal).', 'Ajustar PCA con train + test filtra información del test en la proyección (slide 4 de Lab2).', 'Mira el orden de argumentos de `confusion_matrix(y_true, y_pred)`.'],
      },
    ],
    review: { 'g.bugs.fix': { notebook: 'lab2_2', label: 'Lab0 (curtosis) · Lab2 parte 2 (PCA en train) · Lab1 (confusión)' } },
  },
  // ------------------------------------------------------------------ transferencia
  {
    id: 'g.wine.transfer',
    title: 'Vinos: elige tú el tratamiento de las features',
    requires: ['lab0', 'lab1', 'lab2'],
    mode: 'transferencia',
    scenario: `Dataset **Wine** (\`sklearn.datasets.load_wine\`): 178 vinos de 3 cultivares, 13 features químicas con escalas muy distintas. Ya está dividido en train/test (30 %, estratificado, \`random_state=0\`).

Tu tarea: explorar las distribuciones, decidir **cómo reducir a 2 dimensiones** (una técnica de los labs, la que tú elijas), clasificar y justificar. No se indica qué técnica conviene: elegirla es parte del reto.`,
    steps: [
      {
        kind: 'code', level: 5, id: 'g.wine.explore', title: 'Explora las features',
        skills: ['escritura', 'api'], transfer: true, source: { origin: 'app', note: 'Transferencia de Lab0' },
        prompt: 'Calcula sobre `X_train` la desviación típica, la asimetría y la curtosis de exceso de cada feature. Guarda también en `mas_dispersa` el **nombre** de la feature con mayor desviación típica.',
        available: '`wine` (Bunch con `feature_names`), `X_train` (124 × 13), `y_train`, `X_test`, `y_test`, `np`, `pd`.',
        produce: '`std`, `skew`, `kurt` (13 valores cada uno) y `mas_dispersa` (str).',
        setup: WINE, starter: '',
        solution: "import scipy.stats as ss\nstd = X_train.std(axis=0)\nskew = ss.skew(X_train, axis=0)\nkurt = ss.kurtosis(X_train, axis=0)\nmas_dispersa = wine.feature_names[int(np.argmax(std))]",
        alternatives: ["df = pd.DataFrame(X_train, columns=wine.feature_names)\nstd = df.std(ddof=0).to_numpy()\nimport scipy.stats as ss\nskew = ss.skew(df)\nkurt = ss.kurtosis(df)\nmas_dispersa = df.std().idxmax()"],
        checks: "c.same('std', rtol=1e-6, desc='Desviaciones típicas (ddof=0)')\nc.same('skew', rtol=1e-6)\nc.same('kurt', rtol=1e-6, desc='Curtosis de exceso')\nc.same('mas_dispersa')",
        hints: ['Las funciones de momentos de Lab0 están en `scipy.stats`.', '`np.argmax` da el índice; `wine.feature_names` el nombre.'],
        explanation: '`proline` tiene una desviación típica ≈ 300 veces mayor que la mayoría de features: cualquier método basado en varianza total quedará dominado por ella.',
      },
      {
        kind: 'mcq', id: 'g.wine.choose', skills: ['metodo'], transfer: true, source: { origin: 'app', note: 'Transferencia de Lab2' },
        prompt: 'Debes reducir a **2 dimensiones** para clasificar las 3 clases con LDA/QDA, sin escalar las features. ¿Qué tratamiento esperas que funcione mejor?',
        options: [
          { text: 'MDA (LinearDiscriminantAnalysis con n_components=2)', correct: true, feedback: 'Supervisado y, con c = 3 clases, admite exactamente c − 1 = 2 componentes que maximizan la separación entre clases.' },
          { text: 'PCA con 2 componentes', feedback: 'PCA maximiza la varianza total: aquí la dominan proline y magnesium por su escala, no por separar clases.' },
          { text: 'Quedarse con las dos features de mayor varianza', feedback: 'Es selección por varianza: mismo problema de escala que PCA, y la varianza no mide separabilidad.' },
          { text: 'Da igual: todas conservan la información', feedback: 'Reducir de 13 a 2 dimensiones siempre descarta información; cuál se descarta depende del criterio.' },
        ],
        explanation: 'Compruébalo en el siguiente paso: la elección la justifica la diferencia entre un criterio no supervisado (varianza) y uno supervisado (separabilidad).',
      },
      {
        kind: 'code', level: 6, id: 'g.wine.apply', title: 'Aplica tu elección y compárala',
        skills: ['escritura', 'metodo'], transfer: true, source: { origin: 'app', note: 'Transferencia de Lab2_2' },
        prompt: 'Calcula el error de **test** de LDA tras reducir a 2D con PCA (`err_pca`) y con MDA (`err_mda`), aprendiendo la proyección **solo con train**. Guarda en `var_pc1` la proporción de varianza que explica la primera componente de PCA.',
        available: '`X_train`, `X_test`, `y_train`, `y_test`, `np`.',
        produce: '`err_pca`, `err_mda`, `var_pc1` (floats).',
        setup: WINE, starter: '',
        solution: "from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\npca = PCA(n_components=2).fit(X_train)\nvar_pc1 = pca.explained_variance_ratio_[0]\nlda = LinearDiscriminantAnalysis().fit(pca.transform(X_train), y_train)\nerr_pca = 1 - lda.score(pca.transform(X_test), y_test)\nmda = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)\nlda2 = LinearDiscriminantAnalysis().fit(mda.transform(X_train), y_train)\nerr_mda = 1 - lda2.score(mda.transform(X_test), y_test)",
        wrong: [{ code: "from sklearn.decomposition import PCA\nfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis\npca = PCA(n_components=2).fit(X_test)\nvar_pc1 = pca.explained_variance_ratio_[0]\nlda = LinearDiscriminantAnalysis().fit(pca.transform(X_train), y_train)\nerr_pca = 1 - lda.score(pca.transform(X_test), y_test)\nmda = LinearDiscriminantAnalysis(n_components=2).fit(X_test, y_test)\nlda2 = LinearDiscriminantAnalysis().fit(mda.transform(X_train), y_train)\nerr_mda = 1 - lda2.score(mda.transform(X_test), y_test)", expect: 'razonamiento' }],
        checks: "c.same('var_pc1', rtol=1e-6, desc='Varianza explicada por PC1')\nc.same('err_pca', rtol=1e-9, desc='Error con PCA')\nc.same('err_mda', rtol=1e-9, desc='Error con MDA')",
        hints: ['Ajusta cada transformación con `fit(X_train, …)` y aplica `transform` a train y a test.', 'MDA en scikit-learn es `LinearDiscriminantAnalysis(n_components=…)` + `transform`.'],
        explanation: 'PC1 explica ≈ 99.8 % de la varianza (es casi proline) y LDA tras PCA falla ≈ 35 %; tras MDA el error de test es 0. Escalar las features (Lab4) cambiaría el resultado de PCA.',
      },
      {
        kind: 'text', id: 'g.wine.justify', skills: ['interpretacion', 'metodo'], transfer: true, source: { origin: 'app' },
        prompt: 'Justifica el resultado: ¿por qué PCA rinde tan mal aquí y qué cambiarías si tuvieras que usar PCA? ¿Qué conclusión **no** puedes sacar?',
        rubric: ['PCA es no supervisado y maximiza la varianza total, no la separación entre clases.', 'Sin escalar, la varianza la domina proline por su escala (unidades), así que PC1 ≈ proline.', 'MDA usa las etiquetas y busca las c − 1 = 2 direcciones más discriminantes.', 'Solución para PCA: estandarizar las features antes (o elegir otra técnica).', 'Con 54 muestras de test, «0 % de error» no significa que el error real sea 0.'],
        model: 'PCA busca las direcciones de máxima varianza sin mirar las etiquetas. Como las features no están escaladas, proline (desviación ≈ 300) acapara casi toda la varianza y PC1 es prácticamente proline: la proyección conserva escala, no separabilidad, y LDA falla un 35 %. MDA, supervisado, encuentra las 2 direcciones (c − 1) que maximizan la separación entre clases frente a la dispersión intra-clase, y separa perfectamente el test. Para usar PCA habría que estandarizar antes cada feature. Un error de 0 en 54 muestras de test no garantiza error real nulo.',
      },
    ],
    review: {
      'g.wine.explore': { notebook: 'lab0_1', label: 'Lab0 · asimetría y curtosis' },
      'g.wine.choose': { notebook: 'lab2_2', label: 'Lab2 parte 2 · PCA vs MDA' },
      'g.wine.apply': { notebook: 'lab2_2', label: 'Lab2 parte 2 · proyección con train' },
      'g.wine.justify': { notebook: 'lab2_1', label: 'Lab2 parte 1 · qué maximiza PCA' },
    },
  },
  {
    id: 'g.cancer.transfer',
    title: 'Diagnóstico binario: ROC, priors y matriz de confusión',
    requires: ['lab1'],
    mode: 'transferencia',
    scenario: 'Dataset **Breast Cancer Wisconsin** (`load_breast_cancer`, solo las 10 features «mean»): 0 = maligno, 1 = benigno. Train/test 70/30 estratificado. Aplica lo de Lab1 parte 1 a un problema real con clases desequilibradas.',
    steps: [
      {
        kind: 'code', level: 6, id: 'g.cancer.auc', title: 'Error, AUC y matriz de confusión',
        skills: ['escritura', 'api'], transfer: true, source: { notebook: 'lab1_1', origin: 'app' },
        prompt: 'Entrena un LDA y calcula en test: el error (`err`), el AUC con la clase positiva 1 (`auc`) y la matriz de confusión (`cm`). Guarda también los priors que ha usado el modelo en `priors`.',
        available: '`X_train`, `X_test`, `y_train`, `y_test`, `np`.', produce: '`err`, `auc` (float), `cm` (2 × 2), `priors` (2).',
        setup: CANCER, starter: '',
        solution: "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.metrics import roc_auc_score, confusion_matrix\nlda = LinearDiscriminantAnalysis().fit(X_train, y_train)\nerr = 1 - lda.score(X_test, y_test)\nauc = roc_auc_score(y_test, lda.predict_proba(X_test)[:, 1])\ncm = confusion_matrix(y_test, lda.predict(X_test))\npriors = lda.priors_",
        alternatives: ["from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.metrics import roc_curve, auc as area, confusion_matrix, accuracy_score\nm = LinearDiscriminantAnalysis().fit(X_train, y_train)\np = m.predict(X_test)\nerr = 1 - accuracy_score(y_test, p)\nf, t, _ = roc_curve(y_test, m.decision_function(X_test))\nauc = area(f, t)\ncm = confusion_matrix(y_test, p)\npriors = np.bincount(y_train) / len(y_train)"],
        checks: "c.same('err', rtol=1e-9)\nc.same('auc', rtol=1e-6, desc='AUC')\nc.same('cm', desc='Matriz de confusión')\nc.same('priors', rtol=1e-9, desc='Priors (frecuencias de train)')",
        hints: ['Para el AUC usa probabilidades o la función de decisión, no las etiquetas predichas.', 'Los priors aprendidos están en un atributo que termina en `_`.'],
      },
      {
        kind: 'mcq', id: 'g.cancer.cost', skills: ['interpretacion', 'metodo'], transfer: true, source: { notebook: 'lab1_1', origin: 'app' },
        prompt: 'En este problema es mucho más grave no detectar un tumor **maligno** (clase 0) que dar una falsa alarma. Sin cambiar de modelo, ¿qué harías?',
        options: [
          { text: 'Aumentar el prior (o bajar el umbral) de la clase maligna: se mueve el punto de trabajo sobre la misma ROC', correct: true },
          { text: 'Cambiar a QDA, que siempre detecta más malignos', feedback: 'QDA cambia el modelo, no el compromiso entre errores; no hay garantía.' },
          { text: 'Usar el AUC como criterio de decisión', feedback: 'El AUC resume todos los umbrales; no elige ninguno.' },
          { text: 'Nada: el error total ya es bajo', feedback: 'El error total pondera igual los dos tipos de error, que aquí tienen costes distintos.' },
        ],
        explanation: 'Los priors y el umbral desplazan la frontera MAP: menos falsos negativos a cambio de más falsos positivos. El AUC no cambia (Lab1, parámetro «priors» del experimento).',
      },
    ],
    review: {
      'g.cancer.auc': { notebook: 'lab1_1', block: 'b_roc_test', label: 'Lab1 parte 1 · ROC en test' },
      'g.cancer.cost': { notebook: 'lab1_1', block: 'b_theory', label: 'Lab1 parte 1 · regla MAP y priors' },
    },
  },
];
