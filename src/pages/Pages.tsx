import { useEffect, useMemo, useState } from 'react';
import { LABS, allActivities, allNotebooks, findActivity } from '../../content/index';
import type { LabContent, NotebookContent, Skill } from '../content/types';
import { href } from '../router';
import { Md, OriginPill, CategoryTag, useMaterialIndex } from '../ui/common';
import { docUrl, loadNotebook, type NotebookFile } from '../lib/materials';
import { useActivityStatus, ActivityCard } from '../activities/ActivityCard';
import { exportProgress, isDue, useProgress } from '../progress/store';
import { useEngine } from '../engine/client';
import { coverageFor, summarize, type CoverageRow } from '../lib/coverage';
import { CodeView } from '../ui/CodeView';
import { SavedOutputs } from '../ui/Output';
import { ExamRunner } from '../activities/ExamRunner';
import { reviewTargetFor } from './notebook/NotebookPage';
import { MATERIALS_DOCS } from '../../content/docs';

const SKILLS: { id: Skill; label: string }[] = [
  { id: 'concepto', label: 'Comprensión conceptual' },
  { id: 'codigo', label: 'Comprensión del código' },
  { id: 'api', label: 'Funciones y librerías' },
  { id: 'interpretacion', label: 'Interpretación de resultados' },
  { id: 'escritura', label: 'Escritura autónoma' },
  { id: 'metodo', label: 'Aplicación y elección de métodos' },
];

function NotebookProgress({ nb }: { nb: NotebookContent }) {
  const ids = useMemo(() => nb.activities.map((a) => a.id), [nb]);
  const st = useActivityStatus(ids);
  const read = useProgress((s) => nb.blocks.filter((b) => s.read[b.id]).length);
  const t = Math.max(1, st.total);
  return (
    <div>
      <div className="progress-bar" title={`${st.auto} sin ayuda · ${st.hint} con pistas · ${read}/${nb.blocks.length} bloques leídos`}>
        <span className="auto" style={{ width: `${(100 * st.auto) / t}%` }} />
        <span className="hint" style={{ width: `${(100 * st.hint) / t}%` }} />
      </div>
      <div className="faint small">{read}/{nb.blocks.length} bloques leídos · {st.auto} autónomas · {st.hint} con pistas · {st.total} actividades</div>
    </div>
  );
}

export function Home() {
  const items = useProgress((s) => s.items);
  const due = Object.values(items).filter((p) => isDue(p)).length;
  const engine = useEngine();
  return (
    <div className="stack">
      <div>
        <h1>Machine Learning from Data · Laboratorios</h1>
        <p className="muted" style={{ maxWidth: 820 }}>
          Estudia cada notebook bloque a bloque, experimenta con sus parámetros ejecutando Python real, interpreta resultados y escribe el código de memoria. Después, conecta los labs en <a href={href('prueba')}>Ponte a prueba</a>.
        </p>
        <div className="row">
          <a className="btn primary" href={href('repaso')}>Repaso de hoy {due > 0 && <span className="pill warn">{due}</span>}</a>
          <a className="btn" href={href('prueba')}>Ponte a prueba</a>
          <a className="btn ghost" href={href('cobertura')}>Matriz de cobertura</a>
          <span className="faint small">Python: {engine.message}</span>
        </div>
      </div>
      {LABS.map((lab) => (
        <div key={lab.id} className="card">
          <div className="row">
            <h2 style={{ margin: 0 }}><a href={href('lab', lab.id)}>{lab.title}</a></h2>
            <span className={`pill ${lab.status === 'completo' ? 'ok' : lab.status === 'parcial' ? 'warn' : 'bad'}`}>{lab.status}</span>
          </div>
          <p className="muted small" style={{ marginTop: '0.4rem' }}>{lab.summary}</p>
          <div className="grid-3">
            {lab.notebooks.map((nb) => (
              <a key={nb.id} className="card tight link" href={href('lab', lab.id, nb.id)} style={{ color: 'inherit', textDecoration: 'none' }}>
                <div className="row"><strong>{nb.short}</strong><OriginPill origin={nb.origin} /></div>
                <div className="faint small" style={{ margin: '0.2rem 0 0.4rem' }}>{nb.id}.ipynb</div>
                <NotebookProgress nb={nb} />
              </a>
            ))}
          </div>
          {lab.missing.length > 0 && <div className="callout warn small" style={{ marginTop: '0.6rem' }}><Md text={lab.missing.map((m) => `- ${m}`).join('\n')} /></div>}
        </div>
      ))}
    </div>
  );
}

export function LabPage({ lab }: { lab: LabContent }) {
  const idx = useMaterialIndex();
  return (
    <div className="stack">
      <div className="crumbs"><a href={href()}>Inicio</a> › {lab.title}</div>
      <h1>{lab.title}</h1>
      <p className="muted">{lab.summary}</p>
      <div className="row">{lab.topics.map((t) => <span key={t} className="pill">{t}</span>)}</div>
      {lab.overview && <div className="card"><Md text={lab.overview} /></div>}
      <div className="grid-2">
        {lab.notebooks.map((nb) => (
          <div key={nb.id} className="card">
            <div className="row"><h3 style={{ margin: 0 }}><a href={href('lab', lab.id, nb.id)}>{nb.title}</a></h3></div>
            <div className="row" style={{ margin: '0.3rem 0' }}><OriginPill origin={nb.origin} /><span className="faint small">{nb.blocks.length} bloques · {nb.experiments.length} experimentos · {nb.activities.length} actividades</span></div>
            <NotebookProgress nb={nb} />
            <div className="row" style={{ marginTop: '0.5rem' }}>
              <a className="btn sm" href={href('lab', lab.id, nb.id, 'guiado')}>Recorrido guiado</a>
              <a className="btn sm" href={href('lab', lab.id, nb.id, 'experimentar')}>Experimentar</a>
              <a className="btn sm" href={href('lab', lab.id, nb.id, 'examen')}>Práctica de examen</a>
            </div>
          </div>
        ))}
      </div>
      <div className="card">
        <h3>Documentos de apoyo</h3>
        <ul className="md">
          {lab.documents.map((d) => {
            const meta = MATERIALS_DOCS.find((m) => m.id === d);
            const url = docUrl(idx, d);
            return <li key={d}>{url ? <a href={url} target="_blank" rel="noreferrer">{meta?.title ?? d}</a> : meta?.title ?? d} {meta && <OriginPill origin={meta.origin} />}</li>;
          })}
        </ul>
      </div>
      {lab.missing.length > 0 && <div className="callout warn"><h4>Pendiente</h4><Md text={lab.missing.map((m) => `- ${m}`).join('\n')} /></div>}
    </div>
  );
}

export function ReviewPage() {
  const items = useProgress((s) => s.items);
  const errors = useProgress((s) => s.errors);
  const [cat, setCat] = useState('');
  const [mode, setMode] = useState<'due' | 'errors'>('due');
  const due = useMemo(() => allActivities().filter((r) => isDue(items[r.activity.id])).sort((a, b) => (items[a.activity.id]?.due ?? 0) - (items[b.activity.id]?.due ?? 0)), [items]);
  const failedIds = useMemo(() => [...new Set(errors.map((e) => e.itemId))], [errors]);
  const failedPool = failedIds.map((id) => findActivity(id)).filter(Boolean).filter((r) => !items[r!.activity.id]?.lastOk).map((r) => ({ activity: r!.activity, notebook: r!.notebook.id }));
  const filtered = errors.filter((e) => !cat || e.category === cat);
  const target = (id: string) => {
    const r = findActivity(id);
    return r ? reviewTargetFor(r.lab, r.notebook, id) : { label: id, url: href() };
  };
  return (
    <div className="stack">
      <h1>Repaso</h1>
      <p className="muted">Repetición espaciada (cajas de Leitner: 1, 3, 7, 14 y 30 días). Un fallo devuelve la actividad a la primera caja; las preguntas con variantes generan un caso nuevo cada vez.</p>
      <div className="segmented">
        <button className={mode === 'due' ? 'active' : ''} onClick={() => setMode('due')}>Pendientes hoy ({due.length})</button>
        <button className={mode === 'errors' ? 'active' : ''} onClick={() => setMode('errors')}>Repaso de errores ({failedPool.length})</button>
      </div>
      {mode === 'due' && (due.length ? <ExamRunner key={due.length} pool={due.map((r) => ({ activity: r.activity, notebook: r.notebook.id }))} title="Repaso espaciado" mode="repaso" reviewTarget={(id) => target(id)} /> : <div className="card muted">No hay nada pendiente. Las actividades que resuelvas aparecerán aquí cuando toque repasarlas.</div>)}
      {mode === 'errors' && (failedPool.length ? <ExamRunner key={failedPool.length} pool={failedPool} title="Repaso de errores anteriores" mode="repaso" reviewTarget={(id) => target(id)} /> : <div className="card muted">No tienes errores sin corregir.</div>)}
      <div className="card">
        <div className="row"><h3 style={{ margin: 0 }}>Historial de errores</h3>
          <select value={cat} onChange={(e) => setCat(e.target.value)}><option value="">Todas las categorías</option>{['sintaxis', 'dimensiones', 'api', 'razonamiento', 'ejecucion', 'tiempo', 'concepto'].map((c) => <option key={c}>{c}</option>)}</select>
          <span className="faint">{filtered.length} registros</span>
        </div>
        <table style={{ marginTop: '0.5rem' }}>
          <thead><tr><th>Fecha</th><th>Tipo</th><th>Actividad</th><th>Detalle</th><th>Repasar</th></tr></thead>
          <tbody>
            {filtered.slice(0, 200).map((e, i) => {
              const t = target(e.itemId);
              return (
                <tr key={i}>
                  <td className="small faint">{new Date(e.at).toLocaleString()}</td>
                  <td><CategoryTag c={e.category} /></td>
                  <td className="small mono">{e.itemId}</td>
                  <td className="small">{e.message.slice(0, 160)}</td>
                  <td className="small"><a href={t.url}>{t.label}</a></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ProgressPage() {
  const state = useProgress();
  const [importText, setImportText] = useState('');
  const skillScore = (nbs: NotebookContent[], skill: Skill) => {
    const acts = nbs.flatMap((n) => n.activities).filter((a) => a.skills.includes(skill));
    if (!acts.length) return null;
    const s = acts.reduce((acc, a) => {
      const p = state.items[a.id];
      return acc + (!p ? 0 : p.status === 'autonomo' ? 1 : p.status === 'con_pistas' ? 0.6 : 0.2);
    }, 0);
    return { pct: Math.round((100 * s) / acts.length), n: acts.length };
  };
  const download = () => {
    const blob = new Blob([exportProgress()], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `mlearn-progreso-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
  };
  return (
    <div className="stack">
      <h1>Progreso</h1>
      <p className="muted">Puntuación por habilidad: resuelto sin ayuda = 100 %, con pistas = 60 %, intentado = 20 %. Se guarda en este navegador; exporta una copia de seguridad cuando quieras.</p>
      <div className="card">
        <h3>Global</h3>
        {SKILLS.map((s) => {
          const sc = skillScore(allNotebooks(), s.id);
          return <div key={s.id} className="skill-row"><span>{s.label}</span><div className="progress-bar"><span className="auto" style={{ width: `${sc?.pct ?? 0}%` }} /></div><span className="mono small">{sc ? `${sc.pct}%` : '—'}</span></div>;
        })}
      </div>
      <div className="grid-2">
        {allNotebooks().map((nb) => (
          <div key={nb.id} className="card">
            <h3>{nb.short}</h3>
            <NotebookProgress nb={nb} />
            <div style={{ marginTop: '0.5rem' }}>
              {SKILLS.map((s) => {
                const sc = skillScore([nb], s.id);
                return sc ? <div key={s.id} className="skill-row"><span className="small">{s.label}</span><div className="progress-bar"><span className="auto" style={{ width: `${sc.pct}%` }} /></div><span className="mono small">{sc.pct}%</span></div> : null;
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="card">
        <h3>Simulacros y prácticas</h3>
        {state.sessions.length ? (
          <table><thead><tr><th>Fecha</th><th>Modo</th><th>Resultado</th><th>Tiempo</th></tr></thead>
            <tbody>{state.sessions.map((s) => <tr key={s.id}><td className="small faint">{new Date(s.at).toLocaleString()}</td><td className="small">{s.mode}</td><td>{s.correct}/{s.total}</td><td className="small">{Math.floor(s.seconds / 60)} min {s.seconds % 60} s</td></tr>)}</tbody>
          </table>
        ) : <div className="faint">Todavía no has hecho ningún simulacro.</div>}
      </div>
      <div className="card">
        <h3>Copia de seguridad</h3>
        <div className="row">
          <button className="btn" onClick={download}>Exportar progreso (JSON)</button>
          <button className="btn ghost danger" onClick={() => { if (confirm('¿Borrar todo el progreso de este navegador?')) state.reset(); }}>Borrar progreso</button>
        </div>
        <textarea placeholder="Pega aquí un JSON exportado para importarlo" value={importText} onChange={(e) => setImportText(e.target.value)} style={{ marginTop: '0.6rem' }} />
        <button className="btn" disabled={!importText.trim()} onClick={() => { try { state.importAll(importText); setImportText(''); alert('Progreso importado.'); } catch (e) { alert(`JSON no válido: ${(e as Error).message}`); } }}>Importar</button>
      </div>
    </div>
  );
}

export function CoveragePage() {
  const [files, setFiles] = useState<Record<string, NotebookFile | null>>({});
  const [only, setOnly] = useState('');
  useEffect(() => {
    Promise.all(allNotebooks().map((nb) => loadNotebook(nb.id).then((f) => [nb.id, f] as const))).then((xs) => setFiles(Object.fromEntries(xs)));
  }, []);
  const rows: CoverageRow[] = LABS.flatMap((lab) => lab.notebooks.flatMap((nb) => coverageFor(lab, nb, files[nb.id]?.cells ?? null)));
  const shown = rows.filter((r) => !only || r.status === only);
  const s = summarize(rows);
  return (
    <div className="stack">
      <h1>Matriz de cobertura</h1>
      <p className="muted">Cada celda de cada notebook, el bloque que la explica, su concepto, las funciones que usa, el experimento asociado y las actividades que la evalúan. Una celda de código está <strong>completa</strong> si tiene explicación y al menos una actividad; <strong>parcial</strong> si solo tiene explicación; <strong>pendiente</strong> si no está cubierta.</p>
      <div className="row">
        <span className="pill ok">{s.completo} completas</span><span className="pill warn">{s.parcial} parciales</span><span className="pill bad">{s.pendiente} pendientes</span><span className="faint">{s.total} celdas · {s.pct} % completas</span>
        <select value={only} onChange={(e) => setOnly(e.target.value)}><option value="">Todas</option><option value="completo">Completas</option><option value="parcial">Parciales</option><option value="pendiente">Pendientes</option></select>
      </div>
      {LABS.map((lab) => lab.missing.length ? <div key={lab.id} className="callout warn small"><strong>{lab.title}:</strong> <Md text={lab.missing.join(' · ')} inline /></div> : null)}
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Lab</th><th>Notebook</th><th>Celda</th><th>Sección</th><th>Bloque / concepto</th><th>Código y funciones</th><th>Explicación</th><th>Experimento</th><th>Actividades</th><th>Estado</th></tr></thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.notebook + r.cell}>
                <td className="small">{r.lab}</td>
                <td className="small"><a href={href('fuente', r.notebook, String(r.cell))}>{r.notebook}</a></td>
                <td className="small mono">{r.cell} <span className="faint">{r.cellType === 'markdown' ? 'md' : r.cellType}</span></td>
                <td className="small">{r.section}</td>
                <td className="small">{r.block ? <a href={href('lab', r.lab, r.notebook, 'codigo', r.block)}>{r.blockTitle}</a> : <span className="faint">—</span>}<div className="faint">{r.concept}</div></td>
                <td className="small mono">{r.functions}</td>
                <td>{r.explanation ? '✓' : '—'}</td>
                <td className="small">{r.experiments.length ? r.experiments.map((e) => e.split('.exp.')[1]).join(', ') : '—'}</td>
                <td className="small">{r.activities.length || '—'}</td>
                <td><span className={`pill ${r.status === 'completo' ? 'ok' : r.status === 'parcial' ? 'warn' : 'bad'}`}>{r.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function SourcePage({ nbId, cell }: { nbId: string; cell?: string }) {
  const [file, setFile] = useState<NotebookFile | null | undefined>(undefined);
  useEffect(() => { loadNotebook(nbId).then(setFile); }, [nbId]);
  useEffect(() => {
    if (file && cell) setTimeout(() => document.getElementById(`cell-${cell}`)?.scrollIntoView({ block: 'start' }), 50);
  }, [file, cell]);
  const nb = allNotebooks().find((n) => n.id === nbId.replace('_resuelto', ''));
  const lab = LABS.find((l) => l.notebooks.includes(nb!));
  if (file === undefined) return <div className="faint">Cargando…</div>;
  if (!file) return <div className="callout warn">No hay materiales importados para {nbId}. Ejecuta <code>npm run import</code>.</div>;
  return (
    <div className="stack">
      <div className="crumbs"><a href={href()}>Inicio</a> › Fuente</div>
      <h1>{file.path.split('/').pop()}</h1>
      <div className="row"><OriginPill origin={file.origin as 'original'} /><span className="faint small">{file.path} · SHA-256 {file.sha256.slice(0, 12)}…</span></div>
      {nbId.endsWith('_resuelto') && <div className="callout warn">Este archivo es una <strong>solución</strong>: la app lo usa para validar, no como material de enseñanza.</div>}
      {file.cells.map((c) => {
        const b = nb?.blocks.find((x) => x.cells.includes(c.index));
        return (
          <div key={c.index} id={`cell-${c.index}`} className="card tight" style={String(c.index) === cell ? { borderColor: 'var(--warn)' } : undefined}>
            <div className="row small faint" style={{ marginBottom: '0.3rem' }}>
              <span>Celda {c.index} · {c.type}</span>
              {b && lab && <a href={href('lab', lab.id, nb!.id, 'codigo', b.id)}>Explicada en: {b.title}</a>}
            </div>
            {c.type === 'markdown' ? <Md text={c.source} className="nb-md" /> : (
              <>
                <CodeView code={c.source} />
                {c.outputs.length > 0 && <SavedOutputs outputs={c.outputs} />}
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function MaterialsPage() {
  const idx = useMaterialIndex();
  const versions = useEngine((s) => s.versions);
  const LOCAL: Record<string, string> = { python: '3.13', numpy: '2.5.3', pandas: '3.0.6', scipy: '1.18.1', 'scikit-learn': '1.9.1', matplotlib: '3.11.2', seaborn: '0.13.2' };
  return (
    <div className="stack">
      <h1>Materiales y entorno</h1>
      <div className="card">
        <h3>Materiales importados</h3>
        {idx ? (
          <>
            <div className="faint small">Origen: {idx.root}</div>
            <table style={{ marginTop: '0.5rem' }}>
              <thead><tr><th>Archivo</th><th>Tipo</th><th>SHA-256</th></tr></thead>
              <tbody>{idx.files.map((f) => <tr key={f.id}><td className="small">{f.file ? <a href={`/materials/${f.file}`} target="_blank" rel="noreferrer">{f.path}</a> : f.kind === 'notebook' ? <a href={href('fuente', f.id)}>{f.path}</a> : f.path}</td><td className="small">{f.kind}</td><td className="mono small faint">{f.sha256.slice(0, 16)}</td></tr>)}</tbody>
            </table>
          </>
        ) : <div className="callout warn">No hay materiales importados: ejecuta <code>npm run import</code>.</div>}
      </div>
      <div className="card">
        <h3>Entorno de Python</h3>
        <p className="small muted">El código se ejecuta en Pyodide (CPython compilado a WebAssembly) dentro de un Web Worker: sin acceso a tus archivos, a tus credenciales ni a otras webs, con límite de tiempo por ejecución.</p>
        <table>
          <thead><tr><th>Librería</th><th>App (Pyodide)</th><th>Tu entorno local (requirements.txt)</th></tr></thead>
          <tbody>{Object.keys(LOCAL).map((k) => <tr key={k}><td>{k}</td><td className="mono">{versions?.[k] ?? '…'}</td><td className="mono">{LOCAL[k]}</td></tr>)}</tbody>
        </table>
        <div className="faint small" style={{ marginTop: '0.4rem' }}>Diferencias relevantes detectadas: QDA.covariance_ (n vs n − 1, ver Lab1 parte 1, celda 27).</div>
      </div>
    </div>
  );
}

export function ActivityPage({ id }: { id: string }) {
  const r = findActivity(id);
  if (!r) return <div className="callout bad">Actividad no encontrada: {id}</div>;
  return <ActivityCard activity={r.activity} notebook={r.notebook.id} />;
}
