import { useEffect, useMemo, useState } from 'react';
import type {
  Activity,
  CodeExercise,
  DynamicQuestion,
  ErrorCategory,
  FigureRef,
  McqQuestion,
  NumericQuestion,
  PredictExercise,
  TextQuestion,
} from '../content/types';
import { activityLevel, LEVEL_LABEL } from '../content/types';
import { CategoryTag, Md, SourceLink } from '../ui/common';
import { CodeView } from '../ui/CodeView';
import { Editor } from '../ui/Editor';
import { ErrorBox, Figures, RunOutput } from '../ui/Output';
import { engine, type GradeResult, type RunResult } from '../engine/client';
import { useProgress, type HelpLevel } from '../progress/store';
import { loadNotebook } from '../lib/materials';
import { findNotebook } from '../../content/index';
import { defaults, render } from '../lib/template';

export interface ActivityProps {
  activity: Activity;
  notebook: string;
  context?: 'notebook' | 'global' | 'repaso' | 'simulacro';
  help?: HelpLevel;
  /** En simulacro no se muestra la corrección hasta el final. */
  deferFeedback?: boolean;
  onResult?: (r: { ok: boolean; category?: ErrorCategory | null; message?: string }) => void;
  compact?: boolean;
}

const SKILL_LABEL: Record<string, string> = {
  concepto: 'Concepto',
  codigo: 'Código',
  api: 'Librerías',
  interpretacion: 'Interpretación',
  escritura: 'Escritura autónoma',
  metodo: 'Elección de método',
};

export function ActivityCard(props: ActivityProps) {
  const { activity: a } = props;
  const status = useProgress((s) => s.items[a.id]);
  const level = activityLevel(a);
  return (
    <div className="activity" id={`act-${a.id}`}>
      <div className="meta">
        <span className="pill accent">
          Nivel {level} · {LEVEL_LABEL[level]}
        </span>
        {a.skills.map((s) => (
          <span key={s} className="pill">
            {SKILL_LABEL[s]}
          </span>
        ))}
        {a.transfer && <span className="pill warn">Transferencia</span>}
        {a.kind === 'code' && a.practice && <span className="pill">Practica: {a.practice}</span>}
        {status && (
          <span className={`pill ${status.status === 'autonomo' ? 'ok' : status.status === 'con_pistas' ? 'warn' : 'bad'} right`}>
            {status.status === 'autonomo' ? 'Resuelto sin ayuda' : status.status === 'con_pistas' ? 'Resuelto con pistas' : 'Intentado'}
          </span>
        )}
      </div>
      {a.kind === 'code' && <h3>{a.title}</h3>}
      {a.kind !== 'code' && a.title && <h3>{a.title}</h3>}
      <Body {...props} />
      {!props.compact && (
        <div style={{ marginTop: '0.6rem' }}>
          <SourceLink source={a.source} />
        </div>
      )}
    </div>
  );
}

function Body(props: ActivityProps) {
  const a = props.activity;
  switch (a.kind) {
    case 'mcq':
      return <McqView {...props} activity={a} />;
    case 'numeric':
      return <NumericView {...props} activity={a} />;
    case 'text':
      return <TextView {...props} activity={a} />;
    case 'dynamic':
      return <DynamicView {...props} activity={a} />;
    case 'predict':
      return <PredictView {...props} activity={a} />;
    case 'code':
      return <CodeView2 {...props} activity={a} />;
  }
}

function useRecorder(props: ActivityProps) {
  const record = useProgress((s) => s.record);
  return (ok: boolean, hintsUsed: number, category?: ErrorCategory | null, message?: string) => {
    record({
      itemId: props.activity.id,
      notebook: props.notebook,
      blocks: props.activity.blocks,
      ok,
      hintsUsed,
      category: ok ? null : category ?? 'concepto',
      message,
      context: props.context ?? 'notebook',
    });
    props.onResult?.({ ok, category, message });
  };
}

// ------------------------------------------------------------------ figuras de apoyo
export function FigureFromRef({ figure }: { figure: FigureRef }) {
  const [png, setPng] = useState<string[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    if (figure.notebook != null && figure.cell != null) {
      loadNotebook(figure.notebook).then((nb) => {
        const outs = nb?.cells[figure.cell!]?.outputs.filter((o) => o.type === 'image' && o.png) ?? [];
        const pick = figure.output != null ? outs.slice(figure.output, figure.output + 1) : outs;
        if (alive) setPng(pick.map((o) => o.png!));
        if (alive && !pick.length) setErr('Figura no disponible: importa los materiales (npm run import).');
      });
    } else if (figure.experiment) {
      const [nbId] = figure.experiment.split('.exp.');
      const exp = findNotebook(nbId)?.experiments.find((x) => x.id === figure.experiment);
      if (!exp) return;
      engine
        .experiment(render(exp, { ...defaults(exp), ...(figure.params ?? {}) }), exp.data)
        .then((r) => {
          if (!alive) return;
          if (r.error) setErr(r.error.message);
          const figs = r.figures.map((f) => f.png);
          setPng(figure.figureIndex != null ? figs.slice(figure.figureIndex, figure.figureIndex + 1) : figs);
        })
        .catch((e) => alive && setErr(String(e.message)));
    }
    return () => {
      alive = false;
    };
  }, [figure]);
  if (err) return <div className="faint">{err}</div>;
  if (!png)
    return (
      <div className="faint row">
        <span className="spinner" /> Generando figura…
      </div>
    );
  return (
    <div>
      <Figures figures={png.map((p) => ({ png: p }))} />
      <div className="faint">
        {figure.notebook ? `Salida guardada del notebook ${figure.notebook}, celda ${figure.cell}` : 'Figura calculada ahora por la app'}
      </div>
    </div>
  );
}

function Prompt({ text, code, figure }: { text: string; code?: string; figure?: FigureRef }) {
  return (
    <div className="col" style={{ marginBottom: '0.6rem' }}>
      <Md text={text} />
      {code && <CodeView code={code} />}
      {figure && <FigureFromRef figure={figure} />}
    </div>
  );
}

// ------------------------------------------------------------------ opción múltiple
function McqView(props: ActivityProps & { activity: McqQuestion }) {
  const a = props.activity;
  const [sel, setSel] = useState<number[]>([]);
  const [done, setDone] = useState(false);
  const recordIt = useRecorder(props);
  const correct = a.options.map((o, i) => (o.correct ? i : -1)).filter((i) => i >= 0);
  const multiple = a.multiple ?? correct.length > 1;
  const submit = () => {
    const ok = sel.length === correct.length && sel.every((i) => correct.includes(i));
    setDone(true);
    const wrongPick = sel.find((i) => !a.options[i].correct);
    recordIt(ok, 0, 'concepto', wrongPick != null ? `Elegido: ${a.options[wrongPick].text}` : undefined);
  };
  const show = done && !props.deferFeedback;
  return (
    <>
      <Prompt text={a.prompt} code={a.code} figure={a.figure} />
      {a.options.map((o, i) => {
        const cls = ['option'];
        if (sel.includes(i)) cls.push('selected');
        if (show && o.correct) cls.push('right');
        if (show && sel.includes(i) && !o.correct) cls.push('wrong');
        return (
          <label key={i} className={cls.join(' ')}>
            <input
              type={multiple ? 'checkbox' : 'radio'}
              checked={sel.includes(i)}
              disabled={done}
              onChange={() => setSel(multiple ? (sel.includes(i) ? sel.filter((x) => x !== i) : [...sel, i]) : [i])}
            />
            <div>
              <Md text={o.text} inline />
              {show && o.feedback && (sel.includes(i) || o.correct) && (
                <div className="small muted" style={{ marginTop: '0.2rem' }}>
                  <Md text={o.feedback} inline />
                </div>
              )}
            </div>
          </label>
        );
      })}
      <div className="row" style={{ marginTop: '0.5rem' }}>
        {!done && (
          <button className="btn primary" disabled={!sel.length} onClick={submit}>
            Comprobar
          </button>
        )}
        {done && !props.deferFeedback && (
          <button className="btn ghost sm" onClick={() => { setDone(false); setSel([]); }}>
            Reintentar
          </button>
        )}
        {done && props.deferFeedback && <span className="faint">Respuesta registrada.</span>}
      </div>
      {show && (
        <div className="callout" style={{ marginTop: '0.6rem' }}>
          <Md text={a.explanation} />
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------ numérica
function parseNum(s: string) {
  return Number(s.replace(',', '.').replace('%', '').trim());
}

function NumericView(props: ActivityProps & { activity: NumericQuestion }) {
  const a = props.activity;
  return (
    <NumericCore
      {...props}
      prompt={<Prompt text={a.prompt} code={a.code} figure={a.figure} />}
      answer={a.answer}
      tolerance={a.tolerance}
      unit={a.unit}
      explanation={a.explanation}
    />
  );
}

function NumericCore(
  props: ActivityProps & { prompt: React.ReactNode; answer: number; tolerance: number; unit?: string; explanation: string },
) {
  const [v, setV] = useState('');
  const [res, setRes] = useState<boolean | null>(null);
  const recordIt = useRecorder(props);
  const submit = () => {
    const x = parseNum(v);
    const ok = Number.isFinite(x) && Math.abs(x - props.answer) <= props.tolerance + 1e-12;
    setRes(ok);
    recordIt(ok, 0, 'razonamiento', `Respuesta ${v}; correcta ${props.answer}`);
  };
  return (
    <>
      {props.prompt}
      <div className="row">
        <input type="text" value={v} onChange={(e) => setV(e.target.value)} placeholder="Tu respuesta" onKeyDown={(e) => e.key === 'Enter' && submit()} />
        {props.unit && <span className="muted">{props.unit}</span>}
        <button className="btn primary" disabled={!v.trim()} onClick={submit}>
          Comprobar
        </button>
      </div>
      {res !== null && !props.deferFeedback && (
        <div className={`callout ${res ? 'ok' : 'bad'}`}>
          <strong>{res ? 'Correcto.' : `No es correcto. Valor esperado: ${props.answer}${props.unit ?? ''} (tolerancia ±${props.tolerance}).`}</strong>
          <Md text={props.explanation} />
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------ respuesta abierta con rúbrica
function TextView(props: ActivityProps & { activity: TextQuestion }) {
  const a = props.activity;
  const [text, setText] = useState('');
  const [stage, setStage] = useState<'write' | 'assess' | 'done'>('write');
  const [marks, setMarks] = useState<boolean[]>(a.rubric.map(() => false));
  const recordIt = useRecorder(props);
  const finish = () => {
    const score = marks.filter(Boolean).length / a.rubric.length;
    const ok = score >= 0.7;
    setStage('done');
    const missing = a.rubric.filter((_, i) => !marks[i]);
    recordIt(ok, 0, 'concepto', missing.length ? `Faltó: ${missing.join(' / ')}` : undefined);
  };
  return (
    <>
      <Prompt text={a.prompt} code={a.code} figure={a.figure} />
      <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Escribe tu respuesta como la escribirías en el examen…" disabled={stage !== 'write'} />
      {stage === 'write' && (
        <div className="row" style={{ marginTop: '0.5rem' }}>
          <button className="btn primary" disabled={text.trim().length < 15} onClick={() => setStage('assess')}>
            Autoevaluar con la rúbrica
          </button>
          <span className="faint">La rúbrica y la respuesta modelo aparecen después de escribir.</span>
        </div>
      )}
      {stage !== 'write' && (
        <div className="card tight" style={{ marginTop: '0.6rem' }}>
          <h4>¿Qué puntos cubre tu respuesta?</h4>
          {a.rubric.map((r, i) => (
            <label key={i} className="check-item">
              <input
                type="checkbox"
                checked={marks[i]}
                disabled={stage === 'done'}
                onChange={() => setMarks(marks.map((m, j) => (j === i ? !m : m)))}
              />
              <Md text={r} inline />
            </label>
          ))}
          {stage === 'assess' && (
            <button className="btn primary" onClick={finish}>
              Registrar autoevaluación
            </button>
          )}
          {stage === 'done' && !props.deferFeedback && (
            <div className="callout" style={{ marginTop: '0.5rem' }}>
              <h4>Respuesta modelo</h4>
              <Md text={a.model} />
            </div>
          )}
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------ variantes generadas
function DynamicView(props: ActivityProps & { activity: DynamicQuestion }) {
  const a = props.activity;
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 10_000));
  const [variant, setVariant] = useState<Record<string, unknown> | null>(null);
  const [figs, setFigs] = useState<{ png: string }[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setVariant(null);
    engine
      .generator(a.generator, seed, a.data)
      .then((r) => {
        if (r.error) setErr(r.error.message);
        setVariant(r.variant);
        setFigs(r.figures);
      })
      .catch((e) => setErr(String(e.message)));
  }, [a, seed]);
  if (err) return <div className="callout bad">No se pudo generar la variante: {err}</div>;
  if (!variant)
    return (
      <div className="faint row">
        <span className="spinner" /> Generando una variante con Python…
      </div>
    );
  const prompt = (
    <div className="col" style={{ marginBottom: '0.6rem' }}>
      <Md text={String(variant.prompt)} />
      <Figures figures={figs} />
      <div>
        <button className="btn ghost sm" onClick={() => setSeed(seed + 1)}>
          ↻ Otra variante
        </button>
      </div>
    </div>
  );
  if (a.answerKind === 'mcq') {
    const opts = (variant.options as string[]).map((o) => ({ text: String(o), correct: String(o) === String(variant.answer) }));
    return (
      <>
        {prompt}
        <McqView
          key={seed}
          {...props}
          activity={{ ...a, kind: 'mcq', prompt: '', options: opts, explanation: String(variant.explanation ?? '') } as McqQuestion}
        />
      </>
    );
  }
  return (
    <NumericCore
      key={seed}
      {...props}
      prompt={prompt}
      answer={Number(variant.answer)}
      tolerance={Number(variant.tolerance ?? 1e-6)}
      unit={variant.unit as string | undefined}
      explanation={String(variant.explanation ?? '')}
    />
  );
}

// ------------------------------------------------------------------ predecir
function normShape(s: string) {
  return s.replace(/[()[\]\s]/g, '').replace(/,$/, '');
}

function PredictView(props: ActivityProps & { activity: PredictExercise }) {
  const a = props.activity;
  const [v, setV] = useState('');
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<{ run: RunResult; value: unknown; ok: boolean } | null>(null);
  const recordIt = useRecorder(props);
  const check = async () => {
    setBusy(true);
    try {
      const r = await engine.evaluate(a.setup ?? '', a.code, a.expr, a.data);
      let ok = false;
      if (a.answerKind === 'shape') ok = normShape(v) === normShape(JSON.stringify(r.value));
      else if (a.answerKind === 'numeric') ok = Math.abs(parseNum(v) - Number(r.value)) <= (a.tolerance ?? 1e-9) + 1e-12;
      else ok = v.trim().toLowerCase() === String(r.value).trim().toLowerCase();
      setOut({ ...r, ok });
      recordIt(ok, 0, a.answerKind === 'shape' ? 'dimensiones' : 'razonamiento', `Predicción ${v}; real ${JSON.stringify(r.value)}`);
    } finally {
      setBusy(false);
    }
  };
  const shown = a.answerKind === 'shape' && Array.isArray(out?.value) ? `(${(out!.value as number[]).join(', ')}${(out!.value as number[]).length === 1 ? ',' : ''})` : JSON.stringify(out?.value);
  return (
    <>
      <Prompt text={a.prompt} code={a.code} />
      <div className="row">
        {a.answerKind === 'mcq' && a.options ? (
          <select value={v} onChange={(e) => setV(e.target.value)}>
            <option value="">Elige…</option>
            {a.options.map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        ) : (
          <input type="text" value={v} onChange={(e) => setV(e.target.value)} placeholder={a.answerKind === 'shape' ? 'p. ej. (150, 4)' : 'Tu predicción'} />
        )}
        <button className="btn primary" disabled={!v.trim() || busy} onClick={check}>
          {busy ? <span className="spinner" /> : '▶'} Comprobar ejecutando
        </button>
      </div>
      {out && !props.deferFeedback && (
        <div className="col" style={{ marginTop: '0.6rem' }}>
          <div className={`callout ${out.ok ? 'ok' : 'bad'}`}>
            <strong>{out.ok ? 'Tu predicción coincide con la ejecución.' : `La ejecución da ${shown}.`}</strong>
            <Md text={a.explanation} />
          </div>
          <RunOutput run={out.run} />
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------ ejercicio de código
function CodeView2(props: ActivityProps & { activity: CodeExercise }) {
  const a = props.activity;
  const help = props.help ?? 'completa';
  const [code, setCode] = useState(a.starter);
  const [busy, setBusy] = useState(false);
  const [grade, setGrade] = useState<GradeResult | null>(null);
  const [trial, setTrial] = useState<RunResult | null>(null);
  const [hints, setHints] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [revealed, setRevealed] = useState<Set<number>>(new Set());
  const [showSolution, setShowSolution] = useState(false);
  const [showSetup, setShowSetup] = useState(false);
  const recordIt = useRecorder(props);
  const penalty = hints + revealed.size + (showSolution ? 10 : 0);

  const run = async () => {
    setBusy(true);
    setGrade(null);
    try {
      const r = await engine.evaluate(a.setup, code, '', a.data);
      setTrial(r.run);
    } catch (e) {
      setTrial({ stdout: '', stderr: String((e as Error).message), display: null, figures: [], error: null });
    } finally {
      setBusy(false);
    }
  };

  const check = async () => {
    setBusy(true);
    setTrial(null);
    try {
      const r = await engine.grade(a, code);
      setGrade(r);
      setAttempts((n) => n + 1);
      const firstBad = r.run.error ?? r.items.find((i) => !i.ok);
      const cat = r.run.error ? r.run.error.category : (firstBad as { category?: ErrorCategory | null } | undefined)?.category;
      const msg = r.run.error ? `${r.run.error.type}: ${r.run.error.message}` : (firstBad as { message?: string } | undefined)?.message;
      recordIt(r.passed, penalty, cat ?? 'razonamiento', msg);
    } catch (e) {
      setGrade({ passed: false, items: [], run: { stdout: '', stderr: '', display: null, figures: [], error: null }, fatal: String((e as Error).message) });
    } finally {
      setBusy(false);
    }
  };

  const canSeeSolution = help !== 'ninguna' && attempts >= 1 && (hints >= a.hints.length || attempts >= 3);
  return (
    <>
      <Md text={a.prompt} />
      <div className="grid-2" style={{ margin: '0.6rem 0' }}>
        <div className="card tight">
          <h4>Disponible</h4>
          <Md text={a.available} />
          {a.setup && help !== 'ninguna' && (
            <button className="btn ghost sm" onClick={() => setShowSetup(!showSetup)}>
              {showSetup ? 'Ocultar' : 'Ver'} código de preparación
            </button>
          )}
          {showSetup && <CodeView code={a.setup} maxHeight={260} />}
        </div>
        <div className="card tight">
          <h4>Debe producir</h4>
          <Md text={a.produce} />
        </div>
      </div>
      <Editor value={code} onChange={setCode} onRun={check} minHeight={a.starter ? 90 : 160} />
      <div className="row" style={{ marginTop: '0.5rem' }}>
        <button className="btn" onClick={run} disabled={busy}>
          ▶ Ejecutar
        </button>
        <button className="btn primary" onClick={check} disabled={busy || !code.trim()}>
          {busy ? <span className="spinner" /> : '✓'} Comprobar
        </button>
        <span className="faint">
          <span className="kbd">Ctrl</span>+<span className="kbd">Enter</span> comprueba
        </span>
        <span className="spacer" />
        {help !== 'ninguna' && hints < a.hints.length && (
          <button className="btn ghost sm" onClick={() => setHints(hints + 1)}>
            💡 Pista {hints + 1}/{a.hints.length}
          </button>
        )}
        <button className="btn ghost sm" onClick={() => { setCode(a.starter); setGrade(null); setTrial(null); }}>
          Reiniciar código
        </button>
        {canSeeSolution && !showSolution && (
          <button className="btn ghost sm" onClick={() => setShowSolution(true)}>
            Ver solución
          </button>
        )}
      </div>
      {a.hints.slice(0, hints).map((h, i) => (
        <div key={i} className="hint">
          <strong>Pista {i + 1}.</strong> <Md text={h} inline />
        </div>
      ))}
      {trial && <div style={{ marginTop: '0.6rem' }}><RunOutput run={trial} title="Ejecución de prueba (sin corrección)" /></div>}
      {grade && (
        <div style={{ marginTop: '0.6rem' }} className="col">
          {grade.fatal && <div className="callout bad">Error interno del ejercicio: {grade.fatal}</div>}
          {grade.run.error && <ErrorBox error={grade.run.error} />}
          {!grade.run.error && !props.deferFeedback && (
            <div className={`callout ${grade.passed ? 'ok' : 'bad'}`}>
              <strong>{grade.passed ? (penalty ? 'Correcto (con ayudas).' : '¡Correcto sin ayudas!') : 'Todavía no: revisa los puntos marcados.'}</strong>
              {grade.items.map((it, i) => (
                <div key={i} className="check-item">
                  <span className="ic">{it.ok ? '✅' : '❌'}</span>
                  <div>
                    {it.category && <CategoryTag c={it.category} />} <Md text={it.message} inline />
                    {it.detail && <div className="small muted"><Md text={it.detail} inline /></div>}
                    {!it.ok && it.reveal && help !== 'ninguna' && (
                      revealed.has(i) ? (
                        <div className="small mono">Esperado: {it.reveal}</div>
                      ) : (
                        <button className="btn ghost sm" onClick={() => setRevealed(new Set(revealed).add(i))}>
                          Mostrar valor esperado (cuenta como ayuda)
                        </button>
                      )
                    )}
                  </div>
                </div>
              ))}
              {grade.passed && a.explanation && <Md text={a.explanation} />}
            </div>
          )}
          {(grade.run.stdout || grade.run.figures.length > 0 || grade.run.display) && <RunOutput run={{ ...grade.run, error: null }} title="Salida de tu código" />}
        </div>
      )}
      {showSolution && (
        <div className="col" style={{ marginTop: '0.6rem' }}>
          <div className="callout warn">Solución de referencia (al verla, el ejercicio cuenta como resuelto con ayuda).</div>
          <CodeView code={a.solution} />
          {a.alternatives?.length ? (
            <details>
              <summary className="faint" style={{ cursor: 'pointer' }}>Otras soluciones válidas ({a.alternatives.length})</summary>
              {a.alternatives.map((s, i) => (
                <div key={i} style={{ marginTop: '0.4rem' }}><CodeView code={s} /></div>
              ))}
            </details>
          ) : null}
        </div>
      )}
    </>
  );
}

export function useActivityStatus(ids: string[]) {
  const items = useProgress((s) => s.items);
  return useMemo(() => {
    let auto = 0, hint = 0, tried = 0;
    for (const id of ids) {
      const p = items[id];
      if (!p) continue;
      if (p.status === 'autonomo') auto++;
      else if (p.status === 'con_pistas') hint++;
      else tried++;
    }
    return { auto, hint, tried, total: ids.length };
  }, [ids, items]);
}
