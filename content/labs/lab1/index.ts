import type { LabContent } from '../../../src/content/types';
import { lab1_1 } from './lab1_1.ts';
import { lab1_2 } from './lab1_2.ts';

export const lab1: LabContent = {
  id: 'lab1',
  title: 'Lab 1 — MAP y datos gaussianos (LDA / QDA)',
  folder: 'Lab1 - MAP - Gauss',
  summary:
    'Clasificación con el criterio de máximo a posteriori sobre datos gaussianos sintéticos: cómo la SNR y la forma de las covarianzas (Case 1, 2 y 3) determinan si la frontera óptima es lineal o cuadrática, y cómo se comparan LDA y QDA con error, matriz de confusión, ROC y distancia de Mahalanobis.',
  topics: ['Regla MAP / Bayes', 'Gaussiana multivariante', 'SNR → σ²', 'LDA y QDA (scikit-learn)', 'Matriz de confusión', 'ROC y AUC', 'Mahalanobis', 'Autovalores y forma de los clusters', 'Fronteras y regiones de decisión'],
  notebooks: [lab1_1, lab1_2],
  documents: ['lab1_guide', 'lab1_slides', 'lab1_report', 'lab1_report_sol'],
  status: 'completo',
  missing: [],
};
