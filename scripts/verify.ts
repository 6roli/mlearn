// Verificación automática de los contenidos con el mismo motor Python que usa la app:
//  - cada ejercicio de código: la solución y las alternativas aprueban; las respuestas erróneas suspenden
//    con la categoría de error esperada; el código inicial no aprueba por sí solo;
//  - cada predicción y cada generador de variantes se ejecutan sin error;
//  - cada experimento se ejecuta y reproduce los valores guardados en los notebooks (fidelity);
//  - las celdas de cada notebook se ejecutan en orden y su salida de texto se compara con la guardada.
// Uso: npm run verify [-- filtro]
import fs from 'node:fs';
import path from 'node:path';
import { createEngine } from './lib/node-engine.ts';
import { LABS } from '../content/index.ts';
import { GLOBAL_CHALLENGES } from '../content/global/index.ts';
import { TASKS } from '../content/tasks.ts';
import { defaults, render } from '../src/lib/template.ts';
import type { Activity, NotebookContent } from '../src/content/types.ts';

const filter = process.argv[2] ?? '';
const e = await createEngine();
let failures = 0;
let checks = 0;
const report: string[] = [];
const log = (ok: boolean, msg: string) => {
  checks++;
  if (!ok) failures++;
  const line = `${ok ? '  ok ' : ' FAIL'} ${msg}`;
  report.push(line);
  if (!ok || process.env.VERBOSE) console.log(line);
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

function firstCategory(r: Any): string | null {
  if (r.run?.error) return r.run.error.category;
  const bad = r.items?.find((i: Any) => !i.ok);
  return bad?.category ?? null;
}

async function verifyActivity(a: Activity, where: string) {
  const tag = `[${where}] ${a.id}`;
  try {
    if (a.kind === 'code') {
      await e.data(a.data ?? []);
      const base = { ex_id: a.id, setup: a.setup, solution: a.solution, checks: a.checks };
      for (const [i, code] of [a.solution, ...(a.alternatives ?? [])].entries()) {
        const r: Any = e.call('grade', { ...base, user_code: code });
        const why = r.passed ? '' : ` → ${JSON.stringify(r.run?.error?.message ?? r.items.filter((x: Any) => !x.ok))}`;
        log(r.passed, `${tag} ${i === 0 ? 'solución' : `alternativa ${i}`} aprueba${why}`);
      }
      for (const [i, w] of (a.wrong ?? []).entries()) {
        const r: Any = e.call('grade', { ...base, user_code: w.code });
        const cat = firstCategory(r);
        log(!r.passed && cat === w.expect, `${tag} errónea ${i + 1} suspende como «${w.expect}» (obtenido: ${r.passed ? 'aprobada' : cat})`);
      }
      if (a.starter.trim()) {
        const r: Any = e.call('grade', { ...base, user_code: a.starter });
        log(!r.passed, `${tag} el código inicial no aprueba por sí solo`);
      }
    } else if (a.kind === 'predict') {
      await e.data(a.data ?? []);
      const r: Any = e.call('evaluate', { setup: a.setup ?? '', code: a.code, expr: a.expr });
      const ok = !r.run.error && r.value !== null && r.value !== undefined;
      log(ok, `${tag} predicción evaluable → ${JSON.stringify(r.value)?.slice(0, 80)}${r.run.error ? ` ${r.run.error.message}` : ''}`);
      if (ok && a.answerKind === 'mcq' && a.options) {
        log(a.options.includes(String(r.value)), `${tag} el valor real «${r.value}» está entre las opciones`);
      }
    } else if (a.kind === 'dynamic') {
      await e.data(a.data ?? []);
      for (const seed of [1, 2, 3]) {
        const r: Any = e.call('generator', { code: a.generator, seed });
        const v = r.variant;
        const ok = !r.error && v && v.prompt && v.answer !== undefined;
        log(ok, `${tag} variante seed=${seed}${r.error ? ` ${r.error.message}` : ` → ${String(v?.answer).slice(0, 40)}`}`);
        if (ok && a.answerKind === 'mcq') log(Array.isArray(v.options) && v.options.map(String).includes(String(v.answer)), `${tag} variante seed=${seed}: la respuesta está entre las opciones`);
      }
    } else if (a.kind === 'mcq') {
      log(a.options.some((o) => o.correct), `${tag} tiene al menos una opción correcta`);
    } else if (a.kind === 'numeric') {
      log(Number.isFinite(a.answer), `${tag} respuesta numérica definida`);
    } else if (a.kind === 'text') {
      log(a.rubric.length > 0 && a.model.length > 0, `${tag} rúbrica y respuesta modelo`);
    }
  } catch (err) {
    log(false, `${tag} error interno: ${(err as Error).message}`);
  }
}

async function verifyExperiments(nb: NotebookContent) {
  for (const exp of nb.experiments) {
    await e.data(exp.data ?? []);
    const runs = exp.fidelity?.length ? exp.fidelity : [{ metrics: {} as Record<string, number>, source: 'ejecución por defecto', tol: 1e-6 }];
    for (const f of runs) {
      const values = { ...defaults(exp), ...(f.params ?? {}) };
      const t0 = Date.now();
      const r: Any = e.call('experiment', { code: render(exp, values) });
      const secs = ((Date.now() - t0) / 1000).toFixed(1);
      if (r.error) {
        log(false, `[${nb.id}] ${exp.id} ${JSON.stringify(f.params ?? {})} error: ${r.error.message}`);
        continue;
      }
      log(!!r.metrics, `[${nb.id}] ${exp.id} ${JSON.stringify(f.params ?? {})} se ejecuta (${secs} s, ${r.figures.length} figuras)`);
      for (const [k, v] of Object.entries(f.metrics)) {
        const got = r.metrics?.[k];
        const tol = f.tol ?? 1e-6;
        const ok = typeof got === 'number' && Math.abs(got - v) <= tol * Math.max(1, Math.abs(v));
        log(ok, `[${nb.id}] ${exp.id} ${k} = ${got} (esperado ${v}; ${f.source})`);
      }
      for (const p of exp.params) {
        if (p.control === 'select') log(!!p.options?.some((o) => o.value === p.default), `[${nb.id}] ${exp.id} el valor por defecto de ${p.id} está entre las opciones`);
      }
    }
  }
}

const norm = (s: string) => s.replace(/\s+/g, ' ').trim();
const NUM = /-?\d+\.\d+(?:e[-+]?\d+)?/gi;
/** Igualdad de salidas de texto: mismo texto y números iguales con tolerancia relativa 1e-7. */
function sameOutput(a: string, b: string) {
  const ta = norm(a), tb = norm(b);
  if (ta.replace(NUM, '#') !== tb.replace(NUM, '#')) return false;
  const na = ta.match(NUM) ?? [], nb = tb.match(NUM) ?? [];
  return na.every((x, i) => Math.abs(Number(x) - Number(nb[i])) <= 1e-7 * Math.max(1, Math.abs(Number(nb[i]))));
}

async function verifyCells(nb: NotebookContent) {
  const file = path.resolve(import.meta.dirname, `../public/materials/notebooks/${nb.id}.json`);
  const exists = fs.existsSync(file);
  if (!exists && !nb.proposed) {
    log(false, `[${nb.id}] materiales importados`);
    return;
  }
  const data = exists ? JSON.parse(fs.readFileSync(file, 'utf8')) : { cells: [] };
  const covered = new Set(nb.blocks.flatMap((b) => b.cells));
  for (const c of data.cells) log(covered.has(c.index), `[${nb.id}] celda ${c.index} asignada a un bloque`);
  const kernel = `verify-${nb.id}`;
  let first = true;
  for (const b of nb.blocks) {
    if (b.noRun) continue;
    await e.data(b.data ?? []);
    const codeCells = b.cells.map((i: number) => data.cells[i]).filter((c: Any) => c?.type === 'code');
    if (!codeCells.length && !b.runCode) continue;
    if (b.runCode) {
      const r: Any = e.call('run', { kernel_id: kernel, code: b.runCode, reset: first });
      first = false;
      log(!r.error, `[${nb.id}] bloque ${b.id} (código adaptado) se ejecuta${r.error ? `: ${r.error.message}` : ''}`);
      continue;
    }
    for (const c of codeCells) {
      const r: Any = e.call('run', { kernel_id: kernel, code: c.source, reset: first });
      first = false;
      if (r.error) {
        log(false, `[${nb.id}] celda ${c.index} (${b.id}) error: ${r.error.type}: ${r.error.message}`);
        continue;
      }
      const saved = c.outputs.filter((o: Any) => o.type === 'stream').map((o: Any) => o.text).join('');
      if (saved.trim()) {
        const same = sameOutput(r.stdout, saved);
        const known = b.versionNotes?.find((v) => v.cell === c.index);
        if (!same && known) {
          log(true, `[${nb.id}] celda ${c.index} difiere por una diferencia de versión documentada`);
          continue;
        }
        log(same, `[${nb.id}] celda ${c.index} salida de texto ${same ? 'idéntica a la guardada' : 'DIFIERE'}${same ? '' : `\n      guardada: ${norm(saved).slice(0, 200)}\n      app:      ${norm(r.stdout).slice(0, 200)}`}`);
      } else {
        log(true, `[${nb.id}] celda ${c.index} se ejecuta`);
      }
    }
  }
}

for (const lab of LABS) {
  for (const nb of lab.notebooks) {
    if (filter && !nb.id.includes(filter)) continue;
    console.log(`\n== ${nb.id}`);
    await verifyCells(nb);
    await verifyExperiments(nb);
    for (const b of nb.blocks) if (b.check) await verifyActivity(b.check, nb.id);
    for (const a of nb.activities) await verifyActivity(a, nb.id);
    const ids = new Set(nb.activities.map((a) => a.id));
    for (const id of [...nb.recall, ...nb.practice.flatMap((p) => p.steps)]) log(ids.has(id), `[${nb.id}] referencia a actividad ${id}`);
    const blockIds = new Set(nb.blocks.map((b) => b.id));
    for (const f of nb.flow) for (const b of f.blocks) log(blockIds.has(b), `[${nb.id}] mapa de flujo → ${b}`);
    for (const a of nb.activities) for (const b of a.blocks ?? []) log(blockIds.has(b), `[${nb.id}] ${a.id} → bloque ${b}`);
  }
}

if (!filter || filter === 'global') {
  console.log('\n== global');
  for (const ch of GLOBAL_CHALLENGES) for (const a of ch.steps) await verifyActivity(a, `global/${ch.id}`);
}

if (!filter || filter === 'tareas') {
  console.log('\n== tareas del enunciado');
  for (const t of TASKS) {
    await e.data(t.data ?? []);
    const t0 = Date.now();
    const r: Any = e.call('experiment', { code: t.code });
    log(!r.error, `[tareas] ${t.id} se ejecuta en Pyodide (${((Date.now() - t0) / 1000).toFixed(1)} s, ${r.figures.length} figuras)${r.error ? ': ' + r.error.message : ''}`);
  }
}

fs.mkdirSync(path.resolve(import.meta.dirname, '../docs'), { recursive: true });
fs.writeFileSync(
  path.resolve(import.meta.dirname, `../docs/verificacion${filter ? '-' + filter : ''}.txt`),
  `Verificación ${new Date().toISOString()}\n${checks} comprobaciones, ${failures} fallos\n\n${report.join('\n')}\n`,
);
console.log(`\n${checks} comprobaciones, ${failures} fallos`);
process.exit(failures ? 1 : 0);
