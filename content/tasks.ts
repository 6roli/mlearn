// Tareas de código que piden las guías de cada laboratorio: qué se pedía, qué se ha hecho, el código y
// (tras `npm run tasks`) sus salidas, ejecutadas con tu entorno local. Se muestran en la pestaña
// «G · Tareas del enunciado» de cada notebook y en los PDF.
import type { LabTask } from '../src/content/types';
import { LOAD, PCA_FIT, PCA_LOOP, PCA_PLOT, MDA_LOOP, MDA_PLOT } from './labs/lab2/lab2_4.ts';
import { lab2_2Setup } from './labs/lab2/lab2_2.ts';

const LAB2_2_TABLE = `from sklearn.decomposition import PCA
tablas = {}
for metodo in ['PCA', 'MDA']:
    filas = {'LC': [], 'QC': []}
    for d in [3, 2, 1]:
        if d == 3:
            A, B = X_train, X_test                       # 3 features: los datos originales
        elif metodo == 'PCA':
            pca = PCA(n_components=d).fit(X_train)      # la proyección se aprende SOLO con train
            A, B = pca.transform(X_train), pca.transform(X_test)
        else:
            mda = LinearDiscriminantAnalysis(n_components=d).fit(X_train, y_train)
            A, B = mda.transform(X_train), mda.transform(X_test)
        _, e_lda, _ = lda_classifier(A, y_train, B, y_test)
        _, e_qda, _ = qda_classifier(A, y_train, B, y_test)
        filas['LC'] += [e_lda[1], e_lda[0]]
        filas['QC'] += [e_qda[1], e_qda[0]]
    print(f'\\n{metodo} · SNR = {SNR_dB} dB        3 feat (test, train) | 2 feat (test, train) | 1 feat (test, train)')
    for k, v in filas.items():
        print(f'  {k}  ' + ' | '.join(f'{v[i]:.4f}  {v[i + 1]:.4f}' for i in range(0, 6, 2)))
`;

const LAB2_2_FIG = `
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
col = ['tab:blue', 'tab:orange', 'tab:green']
for a, (nombre, Z) in zip(ax, [('PCA 2D', PCA(2).fit(X_train).transform(X_train)),
                               ('MDA 2D', LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train).transform(X_train))]):
    for k in range(3):
        a.scatter(Z[y_train == k, 0], Z[y_train == k, 1], s=5, alpha=0.5, color=col[k], label=f'class {k}')
    a.set_title(f'{nombre} (train, SNR = {SNR_dB} dB)'); a.legend(markerscale=3)
plt.tight_layout()
plt.show()
`;

const IRIS = `import numpy as np
import pandas as pd
from scipy import stats
iris = pd.read_csv('iris.csv')
features = list(iris.columns[:-1])
`;

// ------------------------------------------------------------------ Lab1: helpers comunes
const LAB1_1 = `import numpy as np
import scipy as sp
import scipy.linalg
import matplotlib.pyplot as plt
from numpy.random import default_rng
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay, roc_auc_score

def mahalanobis(x=None, data=None, cov=None):          # función original del notebook (celda 31)
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()

def mahalanobis_corregida(x, data):                     # media POR FEATURE (axis=0)
    d = x - data.mean(axis=0)
    return np.einsum('ij,jk,ik->i', d, np.linalg.inv(np.cov(data.T)), d)

def experimento(SNR_dB):
    """Celdas 6-39 del notebook con SNR_dB (mismo orden de generación y semilla 5)."""
    dist, n_classes, n_samples, n_feat = 1., 2, 1000, 3
    M_Means = 0.5 * dist * np.array([[1, 1, 1], [-1, -1, -1]]) / np.sqrt(n_feat)
    energy = np.mean([np.dot(m, m) for m in M_Means])
    sig = energy / 10 ** (SNR_dB / 10) / n_feat
    M_covar = np.array([sig * np.eye(n_feat)] * n_classes)
    rng = default_rng(seed=5)
    def genera():
        X = np.vstack([rng.multivariate_normal(M_Means[i], M_covar[i], n_samples) for i in range(n_classes)])
        y = np.repeat(np.arange(n_classes), n_samples)
        rp = np.random.permutation(len(X))
        return X[rp], y[rp]
    X_train, y_train = genera()
    X_test, y_test = genera()
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True).fit(X_train, y_train)
    qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)
    return dict(sig=sig, lda=lda, qda=qda, X_test=X_test, y_test=y_test)
`;

const LAB1_2 = `import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.linalg import eigvals
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

def qpsk(SNR_dB, ro, factor_sigma0=1):
    """Celdas 3-11 de lab1_2: QPSK, 4 clases, 2 features, semilla 5."""
    rng = default_rng(seed=5)
    n_classes, n_samples, n_feat, dist = 4, 1000, 2, 1.
    M_Means = 0.5 * dist * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    energy = np.mean([np.dot(m, m) for m in M_Means])
    sig = energy / 10 ** (SNR_dB / 10)
    sigma = sig * np.array([1., 1, 1, 1])
    sigma[0] *= factor_sigma0                       # Q12: sigma[0] *= 30 ANTES de construir M_covar
    M_covar = np.array([sigma[i] / n_feat * np.array([[1, ro[i]], [ro[i], 1]]) for i in range(n_classes)])
    X = np.vstack([rng.multivariate_normal(M_Means[i], M_covar[i], n_samples) for i in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_samples)
    rp = np.random.permutation(len(X))
    return X[rp], y[rp], M_covar, sigma

def evalua(X, y, titulo):
    """Celdas 15, 19 y 20: LDA y QDA evaluados sobre TRAIN (como el notebook) y regiones de decisión."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (nombre, modelo) in zip(axes, [('LDA', LinearDiscriminantAnalysis()), ('QDA', QuadraticDiscriminantAnalysis())]):
        modelo.fit(X, y)
        pred = modelo.predict(X)
        print(f'{titulo} · {nombre}: error train = {1 - accuracy_score(y, pred):.5f}')
        print(confusion_matrix(y, pred))
        xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, .02),
                             np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, .02))
        Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=np.arange(-0.5, 4), colors=col, alpha=0.15)
        for k in range(4):
            ax.scatter(X[y == k, 0], X[y == k, 1], s=4, color=col[k], alpha=0.5, label=f'class {k}')
        ax.set_title(f'{titulo} · {nombre}')
        ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    plt.show()
`;

export const TASKS: LabTask[] = [
  // ================================================================== Lab0
  {
    id: 'lab0.q4', lab: 'lab0', notebook: 'lab0_2', question: 'Q4',
    title: 'Boxplots de todas las features por especie',
    asks: '«Edit the notebook to create the boxplots for all the features. Analyze boxplots by feature. Are there significant differences between the classes?» El notebook solo trae el de `petal_length` (celdas 38 y 40) y pide, en la celda 41, crear los de *petal width, sepal length y sepal width*.',
    did: 'Se dibujan los **cuatro** boxplots por especie con `sns.boxplot` en una misma figura (tu solución de la celda 42 hacía los tres que faltaban; aquí se añade también `petal_length` para tenerlos juntos). Lectura: en el pétalo, las cajas de las tres especies **no se solapan** (setosa muy separada y con poca dispersión); en el sépalo se solapan bastante, sobre todo versicolor y virginica en `sepal_width`. «Diferencias significativas» aquí es una lectura visual: el contraste formal sería otro.',
    data: ['iris.csv'], author: 'solucion',
    source: { notebook: 'lab0_2', cells: [41, 42], doc: 'lab0_guide', section: 'Q4', origin: 'solucion' },
    code: `import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
iris = pd.read_csv('iris.csv')

features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
for ax, feature in zip(axes, features):
    sns.boxplot(data=iris, x='species', y=feature, hue='species', ax=ax)
    ax.set_title(feature)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelrotation=20)
plt.show()

# resumen numérico de lo que se ve: mediana e IQR por especie
resumen = iris.groupby('species')[features].quantile([0.25, 0.5, 0.75]).unstack()
print(resumen.round(2).T)`,
  },
  {
    id: 'lab0.q6', lab: 'lab0', notebook: 'lab0_3', question: 'Q6',
    title: 'Intervalos de confianza de la media (manual y con stats.t.interval)',
    asks: 'Elegir una feature y calcular la media y los intervalos de confianza al 95 %, 99 % y 99.9 % para las tres clases: **(a) «manualmente»**, con los valores críticos t (`stats.t.ppf`), la media y la desviación típica muestral (NumPy), con df = n − 1; y **(b)** con `stats.t.interval`. Rellenar la tabla Media / IC 95 / IC 99 / IC 99.9 por clase.',
    did: 'Feature: `sepal_length` (la misma que en tu `lab0_3`). Tu notebook solo hacía la parte (b); aquí se añade la **(a)**: $\\bar x \\pm t^*_{1-\\alpha/2,\\,n-1}\\, s/\\sqrt n$ con `np.std(x, ddof=1)` (con el `ddof=0` por defecto de NumPy los intervalos saldrían algo más estrechos). Se comprueba que (a) y (b) coinciden y se imprime la tabla del report. Cuanto más confianza, más ancho el intervalo; los de las tres especies no se solapan, así que sus medias de `sepal_length` son claramente distintas.',
    data: ['iris.csv'], author: 'app',
    source: { notebook: 'lab0_3', cells: [3, 6], doc: 'lab0_guide', section: 'Q6 (a) y (b)', origin: 'app', note: 'La parte (b) procede de tu lab0_3' },
    code: IRIS + `feature = 'sepal_length'
filas = []
for especie, grupo in iris.groupby('species', sort=False):
    x = grupo[feature].to_numpy()
    n = len(x)
    media = np.mean(x)
    s = np.std(x, ddof=1)                      # desviación típica MUESTRAL
    fila = {'Especie': especie, 'Media': round(media, 4)}
    for conf in [0.95, 0.99, 0.999]:
        alpha = 1 - conf
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)          # (a) manual
        a_man = (media - t_crit * s / np.sqrt(n), media + t_crit * s / np.sqrt(n))
        a_fun = stats.t.interval(conf, n - 1, loc=media, scale=stats.sem(x))   # (b) función
        assert np.allclose(a_man, a_fun)       # las dos formas dan lo mismo
        fila[f't* {conf}'] = round(t_crit, 4)
        fila[f'IC {conf}'] = f'[{a_man[0]:.4f}, {a_man[1]:.4f}]'
    filas.append(fila)
print(pd.DataFrame(filas).to_string(index=False))
print()
print('(a) y (b) coinciden en las tres especies y los tres niveles.')`,
  },
  {
    id: 'lab0.q7', lab: 'lab0', notebook: 'lab0_3', question: 'Q7',
    title: 'Contrastes de normalidad: Shapiro-Wilk y Anderson-Darling',
    asks: 'Programar, para **todas las features K y clases J**, los contrastes de Shapiro-Wilk (`shapiro`) y Anderson-Darling (`anderson`) de H0: «la feature K de la clase J es gaussiana», a α = 0.05 y 0.01. Tablas: decisión y p-valor (SW); decisión, valor crítico y estadístico (AD). Explicar el significado del p-valor y del valor crítico.',
    did: 'Es tu código de `lab0_3`, ordenado en dos tablas. **SW:** se rechaza H0 si p < α. **AD:** se rechaza si A² > valor crítico del nivel α (en SciPy ≥ 1.19 la API de valores críticos cambia; con tu 1.18 funciona como aquí). Resultado: se rechaza la normalidad de `petal_width` en setosa (ambos tests, ambos α) y en versicolor al 5 %; AD además rechaza `petal_length` de setosa al 5 % (SW no, p = 0.055). **No rechazar no demuestra** normalidad, y con 12 contrastes a α = 0.05 es esperable algún rechazo por azar.',
    data: ['iris.csv'], author: 'solucion',
    source: { notebook: 'lab0_3', cells: [5, 7, 8], doc: 'lab0_guide', section: 'Q7', origin: 'solucion' },
    code: IRIS + `filas = []
for especie, grupo in iris.groupby('species', sort=False):
    for variable in features:
        x = grupo[variable]
        sw = stats.shapiro(x)
        ad = stats.anderson(x, dist='norm')
        fila = {'species': especie, 'feature': variable, 'W': sw.statistic, 'p': sw.pvalue, 'A2': ad.statistic}
        for alfa in [0.05, 0.01]:
            critico = ad.critical_values[np.flatnonzero(np.isclose(ad.significance_level, 100 * alfa))[0]]
            fila[f'SW {alfa}'] = 'Rechazar' if sw.pvalue < alfa else 'No rechazar'
            fila[f'AD crítico {alfa}'] = critico
            fila[f'AD {alfa}'] = 'Rechazar' if ad.statistic > critico else 'No rechazar'
        filas.append(fila)
r = pd.DataFrame(filas)
print('Shapiro-Wilk')
print(r[['species', 'feature', 'W', 'p', 'SW 0.05', 'SW 0.01']].to_string(index=False, float_format=lambda v: f'{v:.6f}'))
print()
print('Anderson-Darling')
print(r[['species', 'feature', 'A2', 'AD crítico 0.05', 'AD crítico 0.01', 'AD 0.05', 'AD 0.01']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))`,
  },
  // ================================================================== Lab1
  {
    id: 'lab1.q2q5', lab: 'lab1', notebook: 'lab1_1', question: 'Q2–Q5',
    title: 'Ejecutar con SNR = 3, 0, −3 y −10 dB: errores, confusión, ROC y Mahalanobis',
    asks: 'Ejecutar el notebook cuatro veces (SNR = 3, 0, −3, −10 dB) y, **sobre el test**: Q2 tabla de errores de LC y QC; Q3 matrices de confusión a −10 y −3 dB; Q4 curvas ROC a −10 y −3 dB; Q5 distancia de Mahalanobis entre clases en ambos órdenes y por qué difieren.',
    did: 'El código del notebook se encapsula en una función `experimento(SNR_dB)` (mismo orden de generación, semilla 5) y se recorre la lista de SNR. Se imprimen la tabla de Q2, las matrices de Q3, se dibujan las ROC de Q4 con su AUC, y para Q5 se dan la distancia **original** del notebook (media escalar) y la **corregida** (media por feature), junto a su valor teórico $d + \\mathrm{dist}^2/\\sigma^2$. Al bajar la SNR aumenta el error, la ROC se acerca a la diagonal y la distancia baja. Los dos órdenes difieren porque cada uno usa la media y covarianza estimadas de una clase distinta.',
    author: 'app',
    source: { notebook: 'lab1_1', cells: [6, 36, 37, 39], doc: 'lab1_guide', section: 'Q2–Q5', origin: 'app' },
    code: LAB1_1 + `
SNRs = [3, 0, -3, -10]
res = {s: experimento(s) for s in SNRs}

print('Q2 · Error en TEST')
print('       ' + '  '.join(f'{s:>7} dB' for s in SNRs))
for nombre in ['lda', 'qda']:
    errs = [1 - accuracy_score(r['y_test'], r[nombre].predict(r['X_test'])) for r in res.values()]
    print(f"{'LC' if nombre == 'lda' else 'QC'}     " + '  '.join(f'{e:10.4f}' for e in errs))

print('\\nQ3 · Matrices de confusión en TEST (filas = clase real)')
for s in [-3, -10]:
    r = res[s]
    for nombre in ['lda', 'qda']:
        print(f"SNR {s} dB · {'LC' if nombre == 'lda' else 'QC'}:")
        print(confusion_matrix(r['y_test'], r[nombre].predict(r['X_test'])))

print('\\nQ4 · Curvas ROC en TEST')
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for k, s in enumerate([-3, -10]):
    r = res[s]
    RocCurveDisplay.from_estimator(r['lda'], r['X_test'], r['y_test'], ax=ax[k], name='LC (LDA)')
    RocCurveDisplay.from_estimator(r['qda'], r['X_test'], r['y_test'], ax=ax[k], name='QC (QDA)')
    ax[k].plot([0, 1], [0, 1], 'k:', lw=1)
    ax[k].set_title(f'SNR = {s} dB')
plt.tight_layout()
plt.show()

print('Q5 · Distancia de Mahalanobis² media entre clases en TEST')
print(' SNR | original 0→1 | original 1→0 | corregida 0→1 | corregida 1→0 | teórica')
for s, r in res.items():
    X0, X1 = r['X_test'][r['y_test'] == 0], r['X_test'][r['y_test'] == 1]
    print(f"{s:4} | {np.mean(mahalanobis(x=X0, data=X1)):12.4f} | {np.mean(mahalanobis(x=X1, data=X0)):12.4f} | "
          f"{np.mean(mahalanobis_corregida(X0, X1)):13.4f} | {np.mean(mahalanobis_corregida(X1, X0)):13.4f} | {3 + 1 / r['sig']:7.4f}")`,
  },
  {
    id: 'lab1.q6q8', lab: 'lab1', notebook: 'lab1_2', question: 'Q6–Q8',
    title: 'QPSK Case 2: ρ = 0 y ρ = 0.5 a SNR = 10 dB',
    asks: 'Q6: con SNR = 10 dB y ρ = 0 (todas las clases), incluir scatter, fronteras, matrices de confusión y errores de LC y QC; comparar. Q7: lo mismo con ρ = 0.5. Q8: comparar Q6 y Q7.',
    did: 'Se generan los datos con la celda de diseño de `lab1_2` cambiando `ro` (comentando/descomentando las líneas del Case 2) y se evalúa como el notebook, sobre **train**. Con ρ = 0 las nubes son círculos (autovalores iguales, 0.025); con ρ = 0.5 son elipses orientadas según (1, 1) (autovalores 0.0375 y 0.0125). En ambos casos la covarianza es común, así que la frontera óptima es lineal y LDA ≈ QDA; con ρ = 0.5 los errores bajan a 0 porque las elipses se alargan en una dirección que no apunta hacia las clases vecinas.',
    author: 'app',
    source: { notebook: 'lab1_2', cells: [8, 9, 15, 19, 20], doc: 'lab1_guide', section: 'Q6–Q8', origin: 'app' },
    code: LAB1_2 + `
for ro in ([0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]):
    X, y, M_covar, sigma = qpsk(10, ro)
    print(f'ro = {ro}: covarianza de la clase 0 =', M_covar[0].tolist(), '· autovalores =', np.sort(eigvals(M_covar[0]).real))
    evalua(X, y, f'Case 2, ρ = {ro[0]}, SNR 10 dB')`,
  },
  {
    id: 'lab1.q9q11', lab: 'lab1', notebook: 'lab1_2', question: 'Q9–Q11',
    title: 'QPSK Case 3 a 5 y 10 dB; autovalores teóricos y muestrales',
    asks: 'Q9: con ρ = [0.5, 0, −0.5, 0.8] (una covarianza por clase), errores de LC y QC a SNR = +5 y +10 dB. Q10 (report): tabla de autovalores **teóricos** (fórmula de Q1: $\\lambda = \\frac{\\sigma^2}{2}(1\\pm\\rho)$) y **muestrales** de cada clase con `scipy.linalg.eigvals`. Q11: scatter y regiones de LC y QC a 5 y 10 dB, relacionando la forma de los clusters con los autovalores.',
    did: 'Se ejecuta el Case 3 a 5 y 10 dB y se evalúa sobre train como el notebook. La tabla de Q10 compara $\\frac{\\sigma^2}{2}(1\\pm\\rho_k)$ con `eigvals(np.cov(...))` de cada clase a 10 dB. QDA gana a LDA porque puede adaptar una elipse por clase; la clase 4 (ρ = 0.8) es la más alargada (cociente de semiejes $\\sqrt{(1+0.8)/(1-0.8)} = 3$). A 5 dB todas las varianzas son $\\sqrt{10}$ veces mayores y hay más solapamiento.',
    author: 'app',
    source: { notebook: 'lab1_2', cells: [8, 9, 15, 19, 20], doc: 'lab1_report', section: 'Q9–Q11', origin: 'app' },
    code: LAB1_2 + `
ro = [0.5, 0, -0.5, 0.8]
for snr in [5, 10]:
    X, y, M_covar, sigma = qpsk(snr, ro)
    evalua(X, y, f'Case 3, SNR {snr} dB')
    if snr == 10:
        print('Q10 · Autovalores a 10 dB (ordenados de menor a mayor)')
        print('clase |   teóricos σ²/2·(1∓ρ)   |   muestrales (eigvals)')
        for k in range(4):
            teo = sigma[k] / 2 * np.array([1 - abs(ro[k]), 1 + abs(ro[k])])
            mue = np.sort(eigvals(np.cov(X[y == k].T)).real)
            print(f'  {k + 1}   | {teo[0]:.5f}  {teo[1]:.5f} | {mue[0]:.5f}  {mue[1]:.5f}')`,
  },
  {
    id: 'lab1.q12', lab: 'lab1', notebook: 'lab1_2', question: 'Q12 (guía Q11)',
    title: 'Multiplicar la covarianza de la clase 1 por 30',
    asks: 'Con SNR = 10 dB, multiplicar la covarianza de la clase 1 por un número grande (`sigma[0] *= 30`), generar los datos y obtener errores, scatter y fronteras de LC y QC; justificar por qué ahora el cuadrático supera al lineal e indicar el nuevo `sigma[0]` y la covarianza de la clase 1.',
    did: 'Se aplica `sigma[0] *= 30` **antes** de construir `M_covar` (Case 3). `sigma[0]` pasa de 0.05 a 1.5 (es la varianza total, no la desviación típica) y la covarianza de la clase 1 se multiplica por 30. LDA, al promediar una única covarianza común, no puede representar a la vez una clase muy dispersa y tres compactas; QDA sí, con regiones curvas: su error es claramente menor.',
    author: 'app',
    source: { notebook: 'lab1_2', cells: [9, 15, 19, 20], doc: 'lab1_guide', section: 'Q11 de la guía / Q12 del report', origin: 'app' },
    code: LAB1_2 + `
X, y, M_covar, sigma = qpsk(10, [0.5, 0, -0.5, 0.8], factor_sigma0=30)
print('nuevo sigma[0] =', sigma[0])
print('covarianza de la clase 1 =\\n', M_covar[0])
evalua(X, y, 'Case 3, SNR 10 dB, sigma[0] × 30')`,
  },
  // ================================================================== Lab2
  {
    id: 'lab2.q1q4', lab: 'lab2', notebook: 'lab2_2', question: 'Q1–Q4',
    title: 'Errores con 3, 2 y 1 features usando PCA y MDA, a 10 y 0 dB',
    asks: 'Completar las tablas de errores de **train y test** de LC y QC usando **3, 2 y 1 feature**, con **PCA** (Q1 a 10 dB, Q3 a 0 dB) y con **MDA** (Q2 a 10 dB, Q4 a 0 dB), con la misma semilla para generar el mismo dataset; analizar los scatter 2D y 1D.',
    did: 'Se reutiliza el código de `lab2_2` (semilla 5, funciones `lda_classifier`/`qda_classifier`) y se recorre SNR ∈ {10, 0}, método ∈ {PCA, MDA} y dimensión ∈ {3, 2, 1}; la proyección se ajusta **solo con train**. Se imprimen las cuatro tablas del report y una figura con las proyecciones 2D de PCA y MDA. Con MDA a 2D el error de LDA es el mismo que en 3D (MDA conserva lo que LDA usa: con c = 3 clases bastan c − 1 = 2 direcciones); PCA puede perder la dirección que separa las clases porque maximiza varianza, no separabilidad.',
    author: 'app',
    source: { notebook: 'lab2_2', cells: [4, 6, 11, 13, 15, 19, 23, 27], doc: 'lab2_guide', section: 'Q1–Q4', origin: 'app' },
    code: [10, 0].map((snr) => `${lab2_2Setup(snr)}\nSNR_dB = ${snr}\n${LAB2_2_TABLE}`).join('\n') + LAB2_2_FIG,
  },
  {
    id: 'lab2.q8q9', lab: 'lab2', notebook: 'lab2_3', question: 'Q8–Q9',
    title: 'Elegir a mano dos features de Phoneme',
    asks: 'Suponiendo que solo se pueden usar **dos features**, elegirlas a mano a partir de los espectros (Q5) en la variable `V_coor`; dar errores de train y test de LC y QC, compararlos con los de las 64 features (Q8) e incluir scatter y fronteras (Q9).',
    did: 'Se dibuja el espectro medio por clase de las 64 primeras features para justificar la elección y se comparan **varias parejas** (la del notebook, [10, 30], y otras con picos distintos por clase), con la misma partición estratificada (`random_state=5`). Se elige la mejor pareja **por error de entrenamiento** —nunca por el de test— y se dibujan sus regiones. Con 2 features el error sube mucho respecto a 64 (de ≈8 % a ≈25–30 %): aa/ao siguen sin separarse. Es **selección** de features (subconjunto), a diferencia de PCA/MDA.',
    data: ['BD_phoneme.csv'], author: 'app',
    source: { notebook: 'lab2_3', cells: [9, 17, 19, 21], doc: 'lab2_guide', section: 'Q8–Q9', origin: 'app' },
    code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score

df = pd.read_csv("BD_phoneme.csv", header=None)
X = np.array(df.iloc[:, 0:256]); y = np.array(df.iloc[:, 256])
X = X - X.mean(axis=0)
X64 = X[:, :64]
clases = ['aa', 'ao', 'dcl', 'iy', 'sh']
col = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

plt.figure(figsize=(9, 3.5))
for k, c in enumerate(clases):
    plt.plot(X64[y == k + 1].mean(axis=0), color=col[k], label=c)
plt.xlabel('feature (frecuencia, 0–2 kHz)'); plt.ylabel('media (centrada)'); plt.legend(); plt.title('Espectro medio por clase')
plt.show()

def errores(V_coor):
    Xs = X64[:, V_coor]
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, train_size=0.7, random_state=5, shuffle=True, stratify=y)
    out = {}
    for nombre, m in [('LC', LinearDiscriminantAnalysis()), ('QC', QuadraticDiscriminantAnalysis())]:
        m.fit(Xtr, ytr)
        out[nombre] = (1 - accuracy_score(ytr, m.predict(Xtr)), 1 - accuracy_score(yte, m.predict(Xte)), m)
    return out, Xtr, ytr

print('pareja      | LC train  LC test | QC train  QC test')
res = {}
for V in ([10, 30], [5, 20], [3, 40], [15, 45], [8, 50]):
    r, Xtr, ytr = errores(V)
    res[tuple(V)] = (r, Xtr, ytr)
    print(f"{str(V):11} | {r['LC'][0]:.4f}   {r['LC'][1]:.4f} | {r['QC'][0]:.4f}   {r['QC'][1]:.4f}")
r64, _, _ = errores(list(range(64)))
print(f"64 features | {r64['LC'][0]:.4f}   {r64['LC'][1]:.4f} | {r64['QC'][0]:.4f}   {r64['QC'][1]:.4f}")

mejor = min(res, key=lambda v: res[v][0]['LC'][0])      # se elige con el error de TRAIN
print('\\npareja elegida (menor error de train de LC):', list(mejor))
r, Xtr, ytr = res[mejor]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
xx, yy = np.meshgrid(np.arange(Xtr[:, 0].min() - 1, Xtr[:, 0].max() + 1, .1), np.arange(Xtr[:, 1].min() - 1, Xtr[:, 1].max() + 1, .1))
for ax, nombre in zip(axes, ['LC', 'QC']):
    Z = r[nombre][2].predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=np.arange(0.5, 6), colors=col, alpha=0.15)
    for k, c in enumerate(clases):
        ax.scatter(Xtr[ytr == k + 1, 0], Xtr[ytr == k + 1, 1], s=4, color=col[k], label=c, alpha=0.5)
    ax.set_title(f'{nombre} con features {list(mejor)}'); ax.legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.show()`,
  },
  {
    id: 'lab2.q10q11', lab: 'lab2', notebook: 'lab2_4', question: 'Q10–Q11',
    title: 'Notebook 4 · PCA de 1 a 256 dimensiones: curvas de error',
    asks: 'En un notebook nuevo (`Mlearn_lab2_4_Phoneme_PCA_MDA`): leer el dataset completo (256 features), reducir con **PCA ajustado solo con train**, aplicar LC y QC y calcular los errores de train y test para nfeat = 1…256; dibujar las **cuatro curvas** en una figura (Q10) y discutir qué dimensión conviene a cada clasificador sin sobreajustar (Q11).',
    did: 'Se carga y parte el dataset como en `lab2_3` (centrado, 70/30 estratificado, `random_state=5`), se ajusta **un** PCA con todas las componentes sobre train (las d primeras coinciden con `PCA(n_components=d)`) y se evalúa una rejilla de dimensiones de 1 a 256. Para elegir la dimensión se mira dónde el error de test deja de bajar **y** no se separa del de train. **LC**: a partir de ≈15 componentes el test se estabiliza en ≈7 % (el mínimo, 6.8 % con 100, apenas mejora): una dimensión de 15–20 es la elección razonable. **QC**: el mejor test es con ≈7–10 componentes (≈8 %); por encima de ≈30 sobreajusta claramente (con 256, train 0.03 % y test 17 %). La dimensión se elige mirando las curvas; en rigor habría que usar un conjunto de validación y no el test.',
    data: ['BD_phoneme.csv'], author: 'app',
    source: { notebook: 'lab2_4', doc: 'lab2_guide', section: 'apartado 6, Q10–Q11', origin: 'app' },
    code: LOAD + '\n' + PCA_FIT + '\n' + PCA_LOOP + '\n' + PCA_PLOT + `
print('nfeat | LC train  LC test | QC train  QC test')
for d, a, b, c, e in zip(dims, lda_tr, lda_te, qda_tr, qda_te):
    print(f'{d:5} | {a:.4f}   {b:.4f}  | {c:.4f}   {e:.4f}')
print('menor error de test: LC con', dims[int(np.argmin(lda_te))], 'componentes; QC con', dims[int(np.argmin(qda_te))])`,
  },
  {
    id: 'lab2.q12q13', lab: 'lab2', notebook: 'lab2_4', question: 'Q12–Q13',
    title: 'Notebook 4 · MDA de 1 a dmax y comparación con PCA',
    asks: 'Repetir con **MDA** para d′ de 1 a dmax: ¿cuál es dmax? (Q12), curvas de error, y comparar PCA y MDA en Phoneme entre 1 y dmax features (Q13).',
    did: '$d_{max} = \\min(d, c-1) = \\min(256, 5-1) = 4$. Se ajusta `LinearDiscriminantAnalysis(n_components=d)` sobre train para d = 1…4 y se evalúan LC y QC. Con **4** componentes MDA, LC llega a 7.3 % de error de test, el mismo que con las 256 features, y PCA con 4 se queda en ≈10.8 %: MDA usa las etiquetas y concentra la información discriminante en c − 1 direcciones. Pero con **1–2** dimensiones PCA es mejor (17.7 % frente a 28.3 % con 2): con tan pocas direcciones MDA no puede separar a la vez las cinco clases. Por encima de 4, MDA no ofrece más componentes (la ampliación por grupos de features de las slides está en el bloque correspondiente del notebook 4).',
    data: ['BD_phoneme.csv'], author: 'app',
    source: { notebook: 'lab2_4', doc: 'lab2_guide', section: 'apartado 6, Q12–Q13', origin: 'app' },
    code: LOAD + '\n' + PCA_FIT + '\n' + PCA_LOOP + '\n' + MDA_LOOP + '\n' + MDA_PLOT + `
print("d' | MDA: LC train  LC test  QC train  QC test | PCA con d: LC test  QC test")
for d, a, b, c, e in zip(mda_dims, mlda_tr, mlda_te, mqda_tr, mqda_te):
    k = dims.index(d)
    print(f'{d:2} |      {a:.4f}   {b:.4f}   {c:.4f}   {e:.4f} |           {lda_te[k]:.4f}   {qda_te[k]:.4f}')
print(f'referencia, 256 features (PCA completo): LC test {lda_te[-1]:.4f}, QC test {qda_te[-1]:.4f}')`,
  },
];
