// Matriz de cobertura: para cada celda de cada notebook, qué bloque la explica, qué experimento y qué
// actividades la evalúan, y su estado (completo / parcial / pendiente).
import type { Activity, LabContent, NotebookContent } from '../content/types';

export interface CellInfo {
  index: number;
  type: string;
  source: string;
}

export interface CoverageRow {
  lab: string;
  notebook: string;
  cell: number;
  cellType: string;
  section: string;
  block: string | null;
  blockTitle: string;
  concept: string;
  functions: string;
  explanation: boolean;
  experiments: string[];
  activities: string[];
  status: 'completo' | 'parcial' | 'pendiente';
}

function firstLine(src: string) {
  const l = src.split('\n').find((x) => x.trim()) ?? '';
  return l.replace(/^#+\s*/, '').replace(/[*<>]/g, '').slice(0, 70);
}

export function coverageFor(lab: LabContent, nb: NotebookContent, cells: CellInfo[] | null): CoverageRow[] {
  const rows: CoverageRow[] = [];
  const activitiesFor = (blockId: string) =>
    [...nb.activities, ...nb.blocks.filter((b) => b.check).map((b) => b.check as Activity)]
      .filter((a) => a.blocks?.includes(blockId))
      .map((a) => a.id);
  const checkFor = (blockId: string) => {
    const b = nb.blocks.find((x) => x.id === blockId);
    return b?.check ? [b.check.id] : [];
  };
  let section = '';
  const list: CellInfo[] = cells ?? nb.blocks.flatMap((b) => b.cells.map((c) => ({ index: c, type: '?', source: '' })));
  for (const c of list) {
    if (c.type === 'markdown') {
      const h = c.source.split('\n').find((l) => /^#+\s/.test(l.trim()));
      if (h) section = h.replace(/^#+\s*/, '').replace(/[*]/g, '').slice(0, 60);
    }
    const b = nb.blocks.find((x) => x.cells.includes(c.index));
    const exps = b ? nb.experiments.filter((e) => e.blocks.includes(b.id) || e.source.cells?.includes(c.index)).map((e) => e.id) : [];
    const acts = b ? [...new Set([...activitiesFor(b.id), ...checkFor(b.id)])] : [];
    const explanation = !!b && b.problem.length > 0 && b.interpretation.length > 0;
    const isCode = c.type === 'code';
    let status: CoverageRow['status'] = 'pendiente';
    if (b && explanation) {
      // una celda de código está completa si además tiene ejercicio o pregunta; una de texto, con la explicación basta
      status = !isCode || acts.length > 0 ? 'completo' : 'parcial';
    }
    rows.push({
      lab: lab.id,
      notebook: nb.id,
      cell: c.index,
      cellType: c.type,
      section,
      block: b?.id ?? null,
      blockTitle: b?.title ?? '',
      concept: b ? b.problem.replace(/[*`$\\]/g, '').slice(0, 140) : firstLine(c.source),
      functions: b ? b.apis.map((a) => a.name).join(', ') : '',
      explanation,
      experiments: exps,
      activities: acts,
      status,
    });
  }
  return rows;
}

export function summarize(rows: CoverageRow[]) {
  const n = rows.length || 1;
  const c = rows.filter((r) => r.status === 'completo').length;
  const p = rows.filter((r) => r.status === 'parcial').length;
  return { total: rows.length, completo: c, parcial: p, pendiente: rows.length - c - p, pct: Math.round((100 * c) / n) };
}
