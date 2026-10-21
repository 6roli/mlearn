// Genera docs/cobertura.md: la matriz de cobertura completa (misma lógica que la página «Cobertura»).
import fs from 'node:fs';
import path from 'node:path';
import { LABS } from '../content/index.ts';
import { coverageFor, summarize, type CoverageRow } from '../src/lib/coverage.ts';

const root = path.resolve(import.meta.dirname, '..');
const rows: CoverageRow[] = [];
const per: string[] = [];
for (const lab of LABS) {
  for (const nb of lab.notebooks) {
    const f = path.join(root, 'public/materials/notebooks', `${nb.id}.json`);
    const cells = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')).cells : null;
    const r = coverageFor(lab, nb, cells);
    rows.push(...r);
    const s = summarize(r);
    per.push(`| ${lab.id} | ${nb.id}${nb.proposed ? ' (propuesto)' : ''} | ${nb.blocks.length} | ${nb.experiments.length} | ${nb.activities.length} | ${s.total} | ${s.completo} | ${s.parcial} | ${s.pendiente} |`);
  }
}
const esc = (s: string) => s.replace(/\|/g, '\|').replace(/\n/g, ' ');
const s = summarize(rows);
const out = [
  '# Matriz de cobertura',
  '',
  `Generada el ${new Date().toISOString().slice(0, 10)} con \`npm run coverage-matrix\`. También disponible en la app (página «Cobertura»).`,
  '',
  'Estado de una celda: **completo** = explicada en un bloque y (si es de código) evaluada por al menos una actividad; **parcial** = explicada sin actividad; **pendiente** = sin cubrir.',
  '',
  `Total: ${s.total} celdas · ${s.completo} completas · ${s.parcial} parciales · ${s.pendiente} pendientes (${s.pct} %).`,
  '',
  '## Resumen por notebook',
  '',
  '| Lab | Notebook | Bloques | Experimentos | Actividades | Celdas | Completas | Parciales | Pendientes |',
  '|---|---|---|---|---|---|---|---|---|',
  ...per,
  '',
  '## Labs pendientes o incompletos',
  '',
  ...LABS.filter((l) => l.missing.length).flatMap((l) => [`- **${l.title}** (${l.status}):`, ...l.missing.map((m) => `  - ${m}`)]),
  '',
  '## Detalle por celda',
  '',
  '| Lab | Notebook | Celda | Tipo | Sección | Bloque | Concepto / objetivo | Funciones | Explicación | Experimento | Actividades | Estado |',
  '|---|---|---|---|---|---|---|---|---|---|---|---|',
  ...rows.map((r) => `| ${r.lab} | ${r.notebook} | ${r.cell} | ${r.cellType} | ${esc(r.section)} | ${esc(r.blockTitle)} | ${esc(r.concept)} | ${esc(r.functions)} | ${r.explanation ? 'sí' : 'no'} | ${r.experiments.join(', ')} | ${r.activities.length} | ${r.status} |`),
  '',
];
fs.writeFileSync(path.join(root, 'docs/cobertura.md'), out.join('\n'));
console.log(`docs/cobertura.md: ${s.total} celdas, ${s.completo} completas, ${s.parcial} parciales, ${s.pendiente} pendientes`);
