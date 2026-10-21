// Registro de laboratorios. Para añadir uno nuevo: crea content/labs/<id>/index.ts y añádelo aquí;
// sus actividades se incorporan automáticamente a «Ponte a prueba», al repaso y a la cobertura.
import type { Activity, LabContent, NotebookContent } from '../src/content/types';
import { lab0 } from './labs/lab0/index.ts';
import { lab1 } from './labs/lab1/index.ts';
import { lab2 } from './labs/lab2/index.ts';
import { lab3 } from './labs/lab3/index.ts';

export const LABS: LabContent[] = [lab0, lab1, lab2, lab3];

export function allNotebooks(): NotebookContent[] {
  return LABS.flatMap((l) => l.notebooks);
}

export function findNotebook(id: string): NotebookContent | undefined {
  return allNotebooks().find((n) => n.id === id);
}

export function findLab(id: string): LabContent | undefined {
  return LABS.find((l) => l.id === id);
}

export interface ActivityRef {
  activity: Activity;
  notebook: NotebookContent;
  lab: LabContent;
}

export function allActivities(): ActivityRef[] {
  const out: ActivityRef[] = [];
  for (const lab of LABS)
    for (const nb of lab.notebooks) {
      for (const a of nb.activities) out.push({ activity: a, notebook: nb, lab });
      for (const b of nb.blocks) if (b.check) out.push({ activity: b.check, notebook: nb, lab });
    }
  return out;
}

export function findActivity(id: string): ActivityRef | undefined {
  return allActivities().find((r) => r.activity.id === id);
}
