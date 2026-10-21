import { useEffect, useMemo, useRef, useState } from 'react';
import type { Experiment, MetricDef, Param, ParamKind } from '../../content/types';
import { defaults, paramLines, render } from '../../lib/template';
import { engine, useEngine, type RunResult } from '../../engine/client';
import { CodeView } from '../../ui/CodeView';
import { Figures, ErrorBox } from '../../ui/Output';
import { Md, OriginPill, SourceLink } from '../../ui/common';
import { useProgress } from '../../progress/store';

const KIND_LABEL: Record<ParamKind, string> = {
  datos: 'Datos',
  preprocesado: 'Preprocesado',
  modelo: 'Modelo',
  evaluacion: 'Evaluación',
  visualizacion: 'Solo visualización',
};
const KIND_ORDER: ParamKind[] = ['datos', 'preprocesado', 'modelo', 'evaluacion', 'visualizacion'];

interface RunRecord {
  n: number;
  values: Record<string, string>;
  result: RunResult;
  changed: string | null;
  seconds: number;
}

export function fmtMetric(def: MetricDef | undefined, v: unknown): React.ReactNode {
  if (v == null) return '—';
  if (def?.format === 'matrix' && Array.isArray(v)) return <Matrix m={v as number[][]} />;
  if (typeof v !== 'number') return Array.isArray(v) ? JSON.stringify(v) : String(v);
  if (def?.format === 'pct') return `${(v * 100).toFixed(def.digits ?? 2)} %`;
  if (def?.format === 'int') return String(Math.round(v));
  return v.toFixed(def?.digits ?? 4);
}

function Matrix({ m }: { m: number[][] }) {
  if (!Array.isArray(m[0])) return <span className="mono">{JSON.stringify(m)}</span>;
  return (
    <table className="matrix">
      <tbody>
        {m.map((row, i) => (
          <tr key={i}>
            {row.map((x, j) => (
              <td key={j} className={i === j ? 'diag' : ''}>{typeof x === 'number' && !Number.isInteger(x) ? x.toFixed(3) : x}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function delta(def: MetricDef, a: unknown, b: unknown): React.ReactNode {
  if (typeof a !== 'number' || typeof b !== 'number') return null;
  const d = a - b;
  if (Math.abs(d) < 1e-12) return <span className="faint">=</span>;
  const txt = def.format === 'pct' ? `${d > 0 ? '+' : ''}${(d * 100).toFixed(2)} pp` : `${d > 0 ? '+' : ''}${d.toFixed(def.digits ?? 4)}`;
  const worse = def.better ? (def.better === 'low' ? d > 0 : d < 0) : null;
  return <span className={worse == null ? 'muted' : worse ? 'delta-up' : 'delta-down'}>{txt}</span>;
}

function labelOf(p: Param, v: string) {
  return p.options?.find((o) => o.value === v)?.label ?? (p.control === 'toggle' ? (v === 'True' ? 'sí' : 'no') : v);
}

export function ExperimentView({ exp, notebook }: { exp: Experiment; notebook: string }) {
  const def = useMemo(() => defaults(exp), [exp]);
  const [values, setValues] = useState<Record<string, string>>(def);
  const [selected, setSelected] = useState<string>(exp.params[0]?.id);
  const [lastChanged, setLastChanged] = useState<string | null>(null);
  const [history, setHistory] = useState<RunRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<number | null>(null);
  const [name, setName] = useState('');
  const saveExperiment = useProgress((s) => s.saveExperiment);
  const allSaved = useProgress((s) => s.experiments);
  const saved = useMemo(() => allSaved.filter((e) => e.experimentId === exp.id), [allSaved, exp.id]);
  const deleteExperiment = useProgress((s) => s.deleteExperiment);
  const status = useEngine((s) => s.status);
  const counter = useRef(0);
  const autoRan = useRef(false);

  const lines = useMemo(() => paramLines(exp), [exp]);
  const code = useMemo(() => render(exp, values), [exp, values]);
  const current = history[history.length - 1];
  const previous = baseline != null ? history.find((h) => h.n === baseline) : history[history.length - 2];

  const run = async (vals = values, changed = lastChanged) => {
    setBusy(true);
    setError(null);
    const t0 = performance.now();
    try {
      const r = await engine.experiment(render(exp, vals), exp.data);
      counter.current += 1;
      setHistory((h) => [...h.slice(-11), { n: counter.current, values: vals, result: r, changed, seconds: (performance.now() - t0) / 1000 }]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (!autoRan.current && (status === 'ready' || status === 'busy')) {
      autoRan.current = true;
      run(def, null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  const set = (id: string, v: string) => {
    setValues({ ...values, [id]: v });
    setLastChanged(id);
    setSelected(id);
  };

  const changedIds = exp.params.filter((p) => values[p.id] !== def[p.id]).map((p) => p.id);
  const hl = new Set(changedIds.flatMap((id) => lines[id] ?? []));
  const strong = new Set(lastChanged ? lines[lastChanged] ?? [] : []);
  const valueNotes: Record<number, string> = {};
  for (const p of exp.params) for (const ln of lines[p.id] ?? []) valueNotes[ln] = `${p.label}: ${labelOf(p, values[p.id])}${values[p.id] !== def[p.id] ? ` (por defecto ${labelOf(p, def[p.id])})` : ''}`;
  const sel = exp.params.find((p) => p.id === selected);
  const pendingChange = current && JSON.stringify(current.values) !== JSON.stringify(values);

  const changedSummary = (rec: RunRecord, prev?: RunRecord) => {
    if (!prev) return null;
    const diffs = exp.params.filter((p) => rec.values[p.id] !== prev.values[p.id]);
    if (!diffs.length) return <div className="faint">Mismos parámetros que la ejecución de referencia.</div>;
    return diffs.map((p) => (
      <div key={p.id} className="small" style={{ marginBottom: '0.4rem' }}>
        <span className={`pill kind-${p.kind}`}>{KIND_LABEL[p.kind]}</span> <strong>{p.label}</strong>: {labelOf(p, prev.values[p.id])} → {labelOf(p, rec.values[p.id])}
        {' '}(línea{(lines[p.id] ?? []).length > 1 ? 's' : ''} {(lines[p.id] ?? []).join(', ')}, <code>{p.variable}</code>).
        <div className="muted"><Md text={`**Efecto esperado:** ${p.result}`} inline /></div>
      </div>
    ));
  };

  return (
    <div className="stack">
      <div>
        <h2>{exp.title}</h2>
        <Md text={exp.goal} />
      </div>
      <div className="exp-grid">
        {/* ------------------------------------------------ controles */}
        <div className="col">
          {KIND_ORDER.filter((k) => exp.params.some((p) => p.kind === k)).map((k) => (
            <div key={k} className="col" style={{ gap: '0.45rem' }}>
              <h4 style={{ margin: '0.3rem 0 0' }}>{KIND_LABEL[k]}</h4>
              {exp.params.filter((p) => p.kind === k).map((p) => (
                <div key={p.id} className={`param ${values[p.id] !== def[p.id] ? 'changed' : ''}`} onClick={() => setSelected(p.id)}>
                  <div className="head">
                    <label htmlFor={`p-${p.id}`}><strong>{p.label}</strong></label>
                    <span className="row" style={{ gap: '0.3rem' }}>
                      {p.origin && p.origin !== 'original' && <OriginPill origin={p.origin} />}
                      <span className="val">{labelOf(p, values[p.id])}</span>
                    </span>
                  </div>
                  {p.control === 'select' && (
                    <select id={`p-${p.id}`} value={values[p.id]} onChange={(e) => set(p.id, e.target.value)} style={{ width: '100%' }}>
                      {p.options!.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
                    </select>
                  )}
                  {p.control === 'range' && (
                    <input id={`p-${p.id}`} type="range" min={p.min} max={p.max} step={p.step} value={Number(values[p.id])} onChange={(e) => set(p.id, e.target.value)} />
                  )}
                  {p.control === 'toggle' && (
                    <label className="row small"><input id={`p-${p.id}`} type="checkbox" checked={values[p.id] === 'True'} onChange={(e) => set(p.id, e.target.checked ? 'True' : 'False')} /> activar</label>
                  )}
                  <div className="faint small">línea {(lines[p.id] ?? []).join(', ')} · <code>{p.variable}</code></div>
                </div>
              ))}
            </div>
          ))}
          <div className="row" style={{ marginTop: '0.4rem' }}>
            <button className="btn primary" onClick={() => run()} disabled={busy}>{busy ? <span className="spinner" /> : '▶'} Ejecutar</button>
            <button className="btn" onClick={() => { setValues(def); setLastChanged(null); }} disabled={!changedIds.length}>Restaurar valores</button>
            {exp.seedParam && (
              <button className="btn" title="Cambia solo la semilla: nuevas muestras del mismo modelo" onClick={() => {
                const v = { ...values, [exp.seedParam!]: String(Number(values[exp.seedParam!]) + 1) };
                setValues(v); setLastChanged(exp.seedParam!); run(v, exp.seedParam!);
              }} disabled={busy}>🎲 Nueva muestra</button>
            )}
          </div>
          {pendingChange && <div className="callout warn small">Has cambiado parámetros: pulsa <strong>Ejecutar</strong> para ver su efecto. El resto de valores (y la semilla) se mantienen para aislar el cambio.</div>}
          {sel && (
            <div className="card tight">
              <h4>{sel.label} <span className={`pill kind-${sel.kind}`}>{KIND_LABEL[sel.kind]}</span></h4>
              <dl className="explain">
                <dt>Línea y variable</dt><dd>línea {(lines[sel.id] ?? []).join(', ')}: <code>{sel.variable}</code></dd>
                <dt>Efecto sobre el cálculo</dt><dd><Md text={sel.computation} /></dd>
                <dt>Efecto sobre el resultado</dt><dd><Md text={sel.result} /></dd>
                <dt>Por qué</dt><dd><Md text={sel.why} /></dd>
              </dl>
            </div>
          )}
        </div>

        {/* ------------------------------------------------ código */}
        <div className="col">
          <div className="faint small">Código que se ejecuta (líneas amarillas: parámetros cambiados; intenso: último cambio).</div>
          <CodeView code={code} highlight={hl} strong={strong} values={valueNotes} maxHeight={760} />
        </div>

        {/* ------------------------------------------------ resultados */}
        <div className="col">
          {error && <div className="callout bad">{error}</div>}
          {!current && !busy && <div className="faint">Pulsa Ejecutar (o espera a que Python termine de cargar).</div>}
          {busy && <div className="faint row"><span className="spinner" /> Ejecutando en Python…</div>}
          {current && (
            <>
              <div className="card tight">
                <h4>Qué ha cambiado (ejecución #{current.n}{previous ? ` frente a #${previous.n}` : ''})</h4>
                {previous ? changedSummary(current, previous) : <div className="faint">Primera ejecución: valores por defecto del notebook.</div>}
              </div>
              {current.result.error && <ErrorBox error={current.result.error} />}
              <div className="card tight">
                <h4>Métricas</h4>
                <table className="metric-table">
                  <thead><tr><th>Métrica</th><th>#{current.n}</th>{previous && <th>#{previous.n}</th>}{previous && <th>Δ</th>}</tr></thead>
                  <tbody>
                    {exp.metrics.map((m) => (
                      <tr key={m.key}>
                        <td>{m.label}</td>
                        <td className="num">{fmtMetric(m, current.result.metrics?.[m.key])}</td>
                        {previous && <td className="num faint">{m.format === 'matrix' ? fmtMetric(m, previous.result.metrics?.[m.key]) : fmtMetric(m, previous.result.metrics?.[m.key])}</td>}
                        {previous && <td className="num">{m.format === 'matrix' ? null : delta(m, current.result.metrics?.[m.key], previous.result.metrics?.[m.key])}</td>}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="faint small" style={{ marginTop: '0.3rem' }}>Ejecutado en {current.seconds.toFixed(1)} s con Python (Pyodide).</div>
              </div>
              <Figures figures={current.result.figures} />
              {current.result.stdout && (
                <details className="card tight"><summary className="faint" style={{ cursor: 'pointer' }}>Salida de texto (prints del código)</summary><pre className="small">{current.result.stdout}</pre></details>
              )}
            </>
          )}
        </div>
      </div>

      {/* ------------------------------------------------ comparación */}
      {history.length > 1 && (
        <div className="card">
          <h3>Comparar ejecuciones de esta sesión</h3>
          <div style={{ overflowX: 'auto' }}>
            <table className="metric-table">
              <thead>
                <tr>
                  <th>#</th>
                  {exp.params.filter((p) => history.some((h) => h.values[p.id] !== def[p.id])).map((p) => <th key={p.id}>{p.label}</th>)}
                  {exp.metrics.filter((m) => m.format !== 'matrix').map((m) => <th key={m.key}>{m.label}</th>)}
                  <th>Referencia</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h) => (
                  <tr key={h.n} style={h.n === current?.n ? { background: 'var(--accent-soft)' } : undefined}>
                    <td>#{h.n}</td>
                    {exp.params.filter((p) => history.some((x) => x.values[p.id] !== def[p.id])).map((p) => <td key={p.id} className="mono small">{labelOf(p, h.values[p.id])}</td>)}
                    {exp.metrics.filter((m) => m.format !== 'matrix').map((m) => <td key={m.key} className="num">{fmtMetric(m, h.result.metrics?.[m.key])}</td>)}
                    <td><input type="radio" name="baseline" checked={previous?.n === h.n} onChange={() => setBaseline(h.n)} disabled={h.n === current?.n} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <h3>Guardar experimento</h3>
          <div className="row">
            <input type="text" placeholder="Nombre (p. ej. Q2 · −10 dB)" value={name} onChange={(e) => setName(e.target.value)} />
            <button className="btn" disabled={!current} onClick={() => { saveExperiment({ experimentId: exp.id, notebook, name: name || `Ejecución ${new Date().toLocaleString()}`, params: current!.values, metrics: current!.result.metrics ?? null }); setName(''); }}>Guardar la última ejecución</button>
          </div>
          {saved.length > 0 && (
            <table style={{ marginTop: '0.6rem' }}>
              <tbody>
                {saved.map((s) => (
                  <tr key={s.id}>
                    <td>{s.name}<div className="faint small">{new Date(s.at).toLocaleString()}</div></td>
                    <td className="small mono">{exp.params.filter((p) => s.params[p.id] !== def[p.id]).map((p) => `${p.label}=${labelOf(p, s.params[p.id])}`).join(' · ') || 'valores por defecto'}</td>
                    <td className="row">
                      <button className="btn sm" onClick={() => { setValues({ ...def, ...s.params }); setLastChanged(null); run({ ...def, ...s.params }, null); }}>Restaurar y ejecutar</button>
                      <button className="btn sm ghost danger" onClick={() => deleteExperiment(s.id)}>Borrar</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="card">
          <h3>Qué observar</h3>
          <ul className="md">{exp.observe.map((o, i) => <li key={i}><Md text={o} inline /></li>)}</ul>
          <SourceLink source={exp.source} />
        </div>
      </div>
    </div>
  );
}
