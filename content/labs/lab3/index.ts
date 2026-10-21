import type { LabContent } from '../../../src/content/types';

export const lab3: LabContent = {
  id: 'lab3',
  title: 'Lab 3 — Feature selection',
  folder: '(no está en la carpeta de materiales)',
  summary:
    'Según el calendario de las slides de Lab0 (22/10), Lab3 trata la selección de características. Sus notebooks, guía, report y slides no están en la carpeta, así que este módulo está pendiente.',
  topics: ['Selección de subconjuntos de features', 'sklearn.feature_selection'],
  notebooks: [],
  documents: [],
  status: 'pendiente',
  missing: [
    'Faltan todos los materiales de Lab3 (notebooks, guía, report, slides). Cópialos en `LAB/Lab3 - …`, añádelos a `content/materials.json`, ejecuta `npm run import` y crea `content/labs/lab3/<notebook>.ts`; el registro los incorpora a «Ponte a prueba» y a la cobertura.',
  ],
};
