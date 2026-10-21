import type { LabContent } from '../../../src/content/types';
import { lab0_1 } from './lab0_1.ts';
import { lab0_2 } from './lab0_2.ts';
import { lab0_3 } from './lab0_3.ts';

export const lab0: LabContent = {
  id: 'lab0',
  title: 'Lab 0 — Análisis exploratorio de datos (EDA)',
  folder: 'Lab0 - Exploratory data analysis',
  summary:
    'Herramientas de EDA para decidir si una variable puede considerarse gaussiana (supuesto de LDA/QDA) y para explorar la separabilidad de clases: histogramas, CDF, momentos, Q-Q plots, boxplots, scatter y pairplots; después, intervalos de confianza de la media y contrastes de normalidad (Shapiro-Wilk, Anderson-Darling).',
  topics: ['Distribuciones sintéticas', 'Histograma y KDE', 'CDF empírica vs normal', 'Asimetría y curtosis', 'Q-Q plot', 'Iris con pandas y seaborn', 'Boxplot, violin, pairplot', 'IC con t de Student', 'Shapiro-Wilk y Anderson-Darling'],
  notebooks: [lab0_1, lab0_2, lab0_3],
  documents: ['lab0_guide', 'lab0_eda', 'lab0_cuht', 'lab0_slides', 'lab0_report_sol'],
  status: 'completo',
  missing: [],
};
