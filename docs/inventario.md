# Inventario de materiales — Machine Learning from Data (LAB)

Inspección realizada el **21/10/2026** sobre
`C:\Users\groli\OneDrive\Documents\MATT\Q2 - Tardor\MLEARN\LAB`.
Los materiales originales **no se modifican**: la app los lee con `npm run import` y guarda una copia de trabajo en `public/materials/` (excluida de git).

Leyenda de origen:

| Etiqueta | Significado |
|---|---|
| **original** | Material de la asignatura (notebooks de V. Vilaplana, guías, reports en blanco, slides de M. Nájar) |
| **modificado** | Celda original con una línea cambiada por ti (ruta, corrección de API, resolución de malla) |
| **solución** | Respuesta tuya o generada en sesiones anteriores (notebooks resueltos, reports rellenados) |
| **ampliación** | Material de estudio añadido que no forma parte del laboratorio (guías explicadas, figuras extra) |

## Calendario de la asignatura (slides `ML_Lab0_Slides_2026.pdf`, p. 6)

| Fecha | Sesión | En la carpeta |
|---|---|---|
| 17/09 y 01/10 | Lab0: Exploratory data analysis (EDA) | Sí |
| 08/10 | Lab1: MAP for Gaussian data (LDA/QDA) | Sí |
| 15/10 | Lab2: Dimensionality reduction: PCA and MDA | Sí (carpeta llamada «Lab2 - Feature Selection») |
| 22/10 | **Lab3: Feature selection** | **No** |
| 29/10 → 17/12 | Lab4 KNN/escalado, Lab5 CV, Lab6 SVM, Lab7 NN, Lab8 árboles, Lab9 ensembles, Lab10 competición | No (futuros) |

Evaluación (misma fuente): quizzes de laboratorio 30 %, competición 25 %, examen final 45 %.

## Lab0 — Exploratory data analysis

| Archivo | Tipo | Origen | Contenido |
|---|---|---|---|
| `Lab0/Lab0/Mlearn_lab0_1.ipynb` | notebook (27 celdas, ejecutado) | original + **ampliación** (celdas 24–26) | Muestras sintéticas Normal, Uniforme, Laplace, Rayleigh (`default_rng(seed=1)`, n=100 000): histogramas, `histplot`+KDE, CDF vs normal, curtosis/asimetría, Q-Q. Las celdas 24–26 («Additional evidence for Q1») son añadidas. |
| `Lab0/Lab0/Mlearn_lab0_2.ipynb` | notebook (53 celdas, ejecutado) | original + **modificado** (7, 32) + **solución** (42) | Iris: `read_csv`, `head/tail/dtypes/info/describe`, histogramas por clase, `FacetGrid`, CDF, curtosis/asimetría por `groupby`, Q-Q, boxplots, violin, scatter, `jointplot`, `pairplot`. Celda 7: ruta absoluta tuya. Celda 32: `skew(axis=0)` corregido a `skew()`. Celda 42: tu solución de la tarea de boxplots. |
| `Lab0/Lab0/Mlearn_lab0_3.ipynb` | notebook (9 celdas, ejecutado) | **solución** | Q6 (IC t de Student, `stats.t.interval`) y Q7 (Shapiro-Wilk, Anderson-Darling). La guía pide crearlo al alumno. |
| `Lab0/Lab0/iris.csv` | dataset 150×5 | original | 4 features en cm + `species` (setosa, versicolor, virginica; 50 c/u). |
| `Lab0/Lab0/Mlearn_lab0_guide_python.pdf` | guía | original | Objetivos y preguntas Q1–Q7. |
| `Lab0/Lab0/Mlearn_lab0_report_python.docx` | report | **solución** | Report rellenado con respuestas y figuras. |
| `Mlearn_EDA_python.pdf` | apuntes | original | Conceptos de EDA (histogramas, momentos, boxplots, Q-Q, IC, contrastes). |
| `ML_Lab0_CI_HT_Review.pdf` | repaso | original | Intervalos de confianza (t de Student) y contrastes de hipótesis (H0/H1, p-valor, errores tipo I/II). |
| `ML_Lab0_Slides_2026.pdf` | slides | original | Organización, calendario, EDA con el dataset EnfX. |
| `Lab0/Lab0/pdf/Lab0_guia_completa_codigo_y_conceptos.pdf` | guía de estudio | **ampliación** | Explicación de los tres notebooks (07/10/2026). |
| `Lab0/Lab0/pdf/Lab0.zip` | — | **ampliación** | **No es un zip**: es un PDF idéntico byte a byte a la guía completa anterior. |
| `Lab0/Lab0/pdf/Mlearn_lab0_2.pdf` | export | original ejecutado | Export a PDF del notebook 2. |

## Lab1 — MAP and Gaussian data

| Archivo | Tipo | Origen | Contenido |
|---|---|---|---|
| `Mlearn_lab1_1.ipynb` | notebook (40 celdas, ejecutado con SNR = 3 dB) | original | Caso 1 (C = σ²I), 2 clases × 3 features, SNR → σ², `multivariate_normal`, histogramas, pairplot, 3D, Estimator API, LDA/QDA, error, matriz de confusión, ROC, Mahalanobis, test independiente. |
| `Mlearn_lab1_2.ipynb` | notebook (21 celdas, ejecutado con SNR = 10 dB, caso 3) | original + **modificado** (19, 20: `h=.02` en vez de `.001`) | QPSK, 4 clases × 2 features, caso 2 (ρ común) y caso 3 (ρ por clase), LDA/QDA, hiperplanos `coef_`/`intercept_`, regiones de decisión con `meshgrid`. |
| `Mlearn_lab1_report.docx` | report en blanco | original | Preguntas Q1–Q12 con tablas. |
| `resuelto_pdf/Mlearn_lab1_guide_python.pdf` | guía | original | Objetivos, modelos, Q1–Q11 (numeración distinta del report, ver abajo). |
| `resuelto_pdf/ML_Lab1.pdf` | slides | original | Discriminantes gaussianos, 3 casos de covarianza, métricas, ROC, Mahalanobis. |
| `resuelto_pdf/Mlearn_lab1_{1,2}_ejecutado.ipynb`, `*.pdf`, `Mlearn_lab1_report_resuelto.docx`, `Mlearn_lab1_report_estudio.pdf`, `Lab1_parte{1,2}_estudio.pdf`, `LEEME.md`, `CAMBIOS.md` | soluciones | **solución / ampliación** | Ejecuciones para todas las SNR y casos, report resuelto, notas sobre erratas. |
| `resuelto_pdf/.work/` | scripts y caché | **ampliación** (interno) | Generación de los PDF anteriores; KaTeX y paquetes vendorizados. No es material de estudio. |
| `__MACOSX/` | metadatos | — | Restos de descompresión en macOS; se ignoran. |

## Lab2 — Dimensionality reduction (la asignatura lo titula «Feature selection using PCA and MDA»)

| Archivo | Tipo | Origen | Contenido |
|---|---|---|---|
| `Lab2/Mlearn_lab2_1_Intro_PCA.ipynb` | notebook (37 celdas, **sin ejecutar**) | original | PCA en 2D (`components_`, `explained_variance_`), reducción 2→1 y `inverse_transform`, digits 64→2, varianza acumulada, t-SNE. |
| `Lab2/Mlearn_lab2_2_Synthetic_PCA_MDA.ipynb` | notebook (30 celdas, **sin ejecutar**) | original | 3 clases × 3 features, covarianzas aleatorias, LDA/QDA en 3D, PCA 2D/1D, MDA (LDA `transform`) 2D/1D, fronteras. |
| `Lab2/Mlearn_lab2_3_Phoneme.ipynb` | notebook (22 celdas, ejecutado) | original | Phoneme 4509×256, centrado, log-periodogramas, selección de las 64 primeras features, `train_test_split` estratificado, LDA/QDA, selección manual de 2 features (`V_coor = [10, 30]`), fronteras. |
| `Lab2/BD_phoneme.csv` | dataset 4509×257 | original | 256 log-periodogramas + etiqueta 1–5 (aa, ao, dcl, iy, sh). |
| `Lab2/Mlearn_Lab2_PCA_MDA_guide_python.pdf` | guía | original | Q1–Q13; pide un notebook propio `Mlearn_lab2_4_Phoneme_PCA_MDA_<apellidos>.ipynb`. |
| `Lab2/Mlearn_Lab2_PCA_MDA_report.docx` | report en blanco | original | Q1–Q13 con tablas. |
| `ML_Lab2_Slides_T2025_PCA_MDA.pdf` | slides | original | Feature selection vs reducción lineal W, matrices de dispersión, PCA, MDA, d' ≤ min(d, c−1), uso en scikit-learn, Phoneme. |

## Otros archivos de la raíz

| Archivo | Comentario |
|---|---|
| `requirements.txt` (UTF-16) | Entorno local: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2. |
| `.venv/`, `iniciar_lab.bat`, `iniciar_lab.ps1`, `debug.log`, `.git/` (sin commits; remoto `machine-learning-course`) | Entorno y utilidades; no son material de estudio. |

## Diferencias y erratas detectadas en los originales

| Dónde | Qué ocurre | Cómo lo trata la app |
|---|---|---|
| lab0_2 celda 7 | Ruta absoluta de Windows (`C:/Users/.../iris.csv`); en Colab era `iris.csv`. | Se muestra la línea original y la app ejecuta `pd.read_csv("iris.csv")` sobre su copia. |
| lab0_2 celda 32 | Original `iris.groupby('species').skew(axis=0)`; ya no es válido en pandas recientes. Tu versión: `.skew()`. | Se explica la corrección. |
| lab0_2 celda 31 | `groupby(...).apply(pd.DataFrame.kurt)` depende de la versión de pandas (columna de agrupación). | Se verifica en el entorno de la app y se explica la alternativa `groupby(...).agg(pd.Series.kurt)`. |
| lab1_1 celda 27 | Imprime `'LDA model means:'` con `ldamodel.priors_` (debería ser `means_`). Igual en QDA. | Errata señalada; ejercicio de corrección. |
| lab1_1 celda 31 | `mahalanobis()` usa `np.mean(data)` (media escalar de toda la matriz) en vez de `np.mean(data, axis=0)`; `if not cov` falla si se pasa una matriz. Devuelve la distancia **al cuadrado**. | Se conserva el original y se explica la versión corregida; ejercicio de corrección. |
| lab1_1 celdas 18, 34; lab1_2 celda 11 | `np.random.permutation` sin semilla: el barajado no es reproducible (no cambia errores ni matrices de confusión). | Explicado; los experimentos de la app usan `rng.permutation`. |
| lab1_2 celdas 13 vs 17 | Colores de clase intercambiados (rojo/verde) entre figuras. | La app usa una paleta fija por clase. |
| lab1_2 celda 17 | Dibuja `w_i·x + b_i = 0` para cada clase; no son las fronteras entre pares de clases. | Nota conceptual y comparación con las regiones de `predict`. |
| lab1 guía vs report | La guía numera Q1–Q11; el report Q1–Q12 (añade la tabla de autovalores como Q10). El report dice «Q8 y Q9» queriendo decir Q6 y Q7. | La app cita ambas numeraciones. |
| lab2_1 celdas 29, 36 | `plt.cm.get_cmap('jet', 10)` se eliminó en Matplotlib 3.11 (tu entorno: AttributeError; la app, con 3.10, solo avisa). | Se ejecuta con `plt.get_cmap('jet', 10)` y se explica. |
| lab2_2 celdas 19, 27 | El texto dice «first two components» al imprimir 1 componente. | Nota. |
| lab2_2 celdas 21, 29 | El «scatter 1D» dibuja la proyección contra sí misma (diagonal). | Explicado; la app ofrece además histogramas 1D por clase. |
| lab2_3 celda 17 | Reasigna `X = X[:, V_coor]` sobre la `X` ya recortada a 64 features (los índices siguen coincidiendo porque son las primeras 64). | Explicado. |
| lab2_3 celda 19 | Igual que lab1_2: líneas `coef_`/`intercept_` por clase. | Nota. |

## Material que falta y partes pendientes

| Falta | Consecuencia en la app |
|---|---|
| **Lab3 – Feature selection** (notebooks, guía, report, slides) | Módulo **pendiente**. Solo se explica la diferencia conceptual selección vs extracción usando lo que ya aparece en Lab2 (`V_coor`, slide 2–3). Al añadir la carpeta y registrarla, se incorpora a los módulos y a «Ponte a prueba». |
| `Mlearn_lab2_4_Phoneme_PCA_MDA` (lo escribe el alumno) | Se trata como **práctica propuesta** a partir de la guía (Q10–Q13). La solución de referencia es de la app, no de la asignatura, y está marcada así. |
| Salidas guardadas de lab2_1 y lab2_2 | Los resultados que se muestran proceden de una **ejecución de la app** (marcada como tal), no del notebook guardado. |
| Notebooks originales sin modificar de lab0_2 / lab1_2 | Solo se dispone de tus copias con cambios; se señalan las líneas cambiadas. |
