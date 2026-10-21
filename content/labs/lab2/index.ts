import type { LabContent } from '../../../src/content/types';
import { lab2_1 } from './lab2_1.ts';
import { lab2_2 } from './lab2_2.ts';
import { lab2_3 } from './lab2_3.ts';
import { lab2_4 } from './lab2_4.ts';

export const lab2: LabContent = {
  id: 'lab2',
  title: 'Lab 2 — Reducción de dimensionalidad: PCA y MDA',
  folder: 'Lab2 - Feature Selection',
  summary:
    'La asignatura lo titula «Feature selection using PCA and MDA»: se reduce el número de features con una transformación lineal aprendida en entrenamiento (PCA, no supervisada; MDA/Fisher, supervisada) y se compara con elegir a mano un subconjunto de features (Phoneme, V_coor). Se mide cómo cambian los errores de LDA y QDA en train y test.',
  topics: ['Maldición de la dimensionalidad', 'PCA: componentes y varianza explicada', 'inverse_transform', 'digits y t-SNE', 'MDA (LDA.transform), d′ ≤ c − 1', 'PCA vs MDA con LDA/QDA', 'Phoneme: log-periodogramas', 'train_test_split estratificado', 'Selección manual de features', 'Curvas de error vs dimensión'],
  notebooks: [lab2_1, lab2_2, lab2_3, lab2_4],
  documents: ['lab2_guide', 'lab2_slides', 'lab2_report'],
  status: 'completo',
  missing: [
    'El notebook `Mlearn_lab2_4_Phoneme_PCA_MDA` lo escribe el alumno (guía, apartado 6): la app lo trata como práctica propuesta con solución de referencia propia, no de la asignatura.',
    'Lab2_1 y Lab2_2 no tienen salidas guardadas: los resultados mostrados son de una ejecución de la app.',
  ],
  overview: `### Selección de características ≠ reducción de dimensionalidad

La guía y las slides llaman a este lab «**Feature selection** using PCA and MDA», pero en él conviven dos ideas distintas:

| | Selección de características (*feature selection*) | Extracción / reducción lineal (*feature extraction*) |
|---|---|---|
| Qué produce | Un **subconjunto** de las features originales | **Nuevas** features $y = W^T x$, combinación lineal de todas |
| Significado | Se conserva (la feature 10 sigue siendo «la frecuencia 10») | Se pierde: cada componente mezcla todas las features |
| En este lab | \`V_coor = [10, 30]\` en Phoneme (selección **manual**), y quedarse con las 64 primeras | PCA (\`PCA(n_components)\`) y MDA (\`LinearDiscriminantAnalysis(n_components).transform\`) |
| ¿Usa etiquetas? | Depende del método (aquí, criterio visual sobre los espectros por clase) | PCA no (no supervisada); MDA sí (supervisada) |
| Número máximo | Hasta d | PCA: hasta d; MDA: hasta min(d, c − 1) |

La slide 2 define *feature selection* como «seleccionar un subconjunto de features relevantes» y remite a \`sklearn.feature_selection\` (slide 3); los métodos automáticos de selección se estudian en el **Lab3 (Feature selection, 22/10)**, cuyos materiales aún no están en la carpeta. En la app, PCA y MDA se tratan como **reducción de dimensionalidad** y se señala cuándo el lab usa la palabra *selección* en sentido amplio.

En ambos casos la regla es la misma: la transformación (o el subconjunto) se decide **solo con el conjunto de entrenamiento** y después se aplica igual al test (slide 4).`,
};
