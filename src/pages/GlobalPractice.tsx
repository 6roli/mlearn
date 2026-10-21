import { useMemo, useState } from 'react';
import { LABS, allActivities, findActivity, type ActivityRef } from '../../content/index';
import { GLOBAL_CHALLENGES } from '../../content/global/index';
import type { Activity, GlobalChallenge } from '../content/types';
import { activityLevel } from '../content/types';
import { ExamRunner } from '../activities/ExamRunner';
import { Md } from '../ui/common';
import { reviewTargetFor } from './notebook/NotebookPage';
import { useProgress } from '../progress/store';
import { href } from '../router';

const MODES: { id: string; label: string; desc: string; pick: (r: ActivityRef) => boolean }[] = [
  { id: 'mezcla', label: 'Preguntas mezcladas', desc: 'De todos los notebooks incorporados, en orden aleatorio.', pick: () => true },
  { id: 'codigo', label: 'Código sin pistas de función', desc: 'Ejercicios de completar, escribir y adaptar sin indicar qué función usar.', pick: (r) => r.activity.kind === 'code' && activityLevel(r.activity) >= 5 },
  { id: 'figuras', label: 'Figuras, tablas y métricas', desc: 'Interpretar salidas reales (figuras guardadas o generadas, tablas, matrices de confusión).', pick: (r) => r.activity.skills.includes('interpretacion') && r.activity.kind !== 'code' },
  { id: 'corregir', label: 'Detectar y corregir errores', desc: 'Errores de sintaxis, de API, de dimensiones y de concepto.', pick: (r) => (r.activity.kind === 'code' && r.activity.level === 4) || /misconception|error/.test(r.activity.id) },
  { id: 'metodos', label: 'Comparar métodos', desc: 'LDA vs QDA, PCA vs MDA, selección vs extracción, contrastes…', pick: (r) => r.activity.skills.includes('metodo') },
  { id: 'prediccion', label: 'Predecir salidas y dimensiones', desc: 'Anticipa formas, valores y efectos de cambios; se comprueba ejecutando.', pick: (r) => r.activity.kind === 'predict' || r.activity.kind === 'dynamic' },
];

export function GlobalPractice({ sub }: { sub?: string }) {
  const [labs, setLabs] = useState<string[]>(LABS.map((l) => l.id));
  const [ctx, setCtx] = useState<'ambos' | 'original' | 'transferencia'>('ambos');
  const [mode, setMode] = useState(sub ?? 'mezcla');
  const [simulacro, setSimulacro] = useState(false);
  const errors = useProgress((s) => s.errors);
  const items = useProgress((s) => s.items);

  const pool = useMemo(() => {
    const m = MODES.find((x) => x.id === mode) ?? MODES[0];
    return allActivities()
      .filter((r) => labs.includes(r.lab.id))
      .filter((r) => ctx === 'ambos' || (ctx === 'transferencia' ? !!r.activity.transfer : !r.activity.transfer))
      .filter(m.pick)
      .map((r) => ({ activity: hidePractice(r.activity), notebook: r.notebook.id }));
  }, [labs, ctx, mode]);

  const challenges = GLOBAL_CHALLENGES.filter((c) => c.requires.every((l) => LABS.some((x) => x.id === l)) && c.requires.some((l) => labs.includes(l)) && (ctx === 'ambos' || c.mode === (ctx === 'original' ? 'contexto' : 'transferencia')));
  const errorPool = [...new Set(errors.map((e) => e.itemId))].map((id) => findActivity(id)).filter((r) => r && !items[r.activity.id]?.lastOk).map((r) => ({ activity: r!.activity, notebook: r!.notebook.id }));
  const target = (id: string) => {
    const r = findActivity(id);
    if (r) return reviewTargetFor(r.lab, r.notebook, id);
    for (const ch of GLOBAL_CHALLENGES) if (ch.review[id]) return { label: ch.review[id].label, url: href('lab', LABS.find((l) => l.notebooks.some((n) => n.id === ch.review[id].notebook))?.id, ch.review[id].notebook, 'codigo', ch.review[id].block) };
    return { label: id, url: href() };
  };

  return (
    <div className="stack">
      <div>
        <h1>Ponte a prueba</h1>
        <p className="muted">Práctica acumulativa con todos los laboratorios incorporados ({LABS.map((l) => l.id).join(', ')}). Crece automáticamente al registrar nuevos labs. Las preguntas son de elaboración propia a partir de los materiales: no reproducen preguntas reales de examen.</p>
      </div>
      <div className="card">
        <div className="grid-3">
          <div>
            <h4>Laboratorios</h4>
            {LABS.map((l) => (
              <label key={l.id} className="row"><input type="checkbox" checked={labs.includes(l.id)} onChange={(e) => setLabs(e.target.checked ? [...labs, l.id] : labs.filter((x) => x !== l.id))} /> {l.title}</label>
            ))}
          </div>
          <div>
            <h4>Contexto</h4>
            {(['ambos', 'original', 'transferencia'] as const).map((c) => (
              <label key={c} className="row"><input type="radio" checked={ctx === c} onChange={() => setCtx(c)} /> {c === 'ambos' ? 'Ambos' : c === 'original' ? 'Contexto original (datasets del lab)' : 'Transferencia (otros escenarios)'}</label>
            ))}
          </div>
          <div>
            <h4>Formato</h4>
            <label className="row"><input type="checkbox" checked={simulacro} onChange={(e) => setSimulacro(e.target.checked)} /> Simulacro: corrección al final, sin ayudas y tiempo opcional</label>
          </div>
        </div>
      </div>
      <div className="segmented" style={{ flexWrap: 'wrap', borderRadius: 14 }}>
        {MODES.map((m) => <button key={m.id} className={mode === m.id ? 'active' : ''} onClick={() => setMode(m.id)}>{m.label}</button>)}
        <button className={mode === 'retos' ? 'active' : ''} onClick={() => setMode('retos')}>Retos multietapa ({challenges.length})</button>
        <button className={mode === 'errores' ? 'active' : ''} onClick={() => setMode('errores')}>Repaso de errores ({errorPool.length})</button>
      </div>
      {MODES.find((m) => m.id === mode) && <p className="faint">{MODES.find((m) => m.id === mode)!.desc} · {pool.length} actividades disponibles.</p>}
      {mode === 'retos' ? (
        <ChallengeList challenges={challenges} target={target} />
      ) : mode === 'errores' ? (
        <ExamRunner key={'err' + errorPool.length} pool={errorPool} title="Repaso de errores" mode="global" reviewTarget={target} />
      ) : (
        <ExamRunner key={mode + ctx + labs.join() + simulacro} pool={pool} title={`${MODES.find((m) => m.id === mode)?.label}${simulacro ? ' · simulacro' : ''}`} mode="global" reviewTarget={target} />
      )}
    </div>
  );
}

function hidePractice(a: Activity): Activity {
  // En la práctica global no se indica de antemano qué función practicar.
  return a.kind === 'code' && a.practice ? { ...a, practice: undefined } : a;
}

function ChallengeList({ challenges, target }: { challenges: GlobalChallenge[]; target: (id: string) => { label: string; url: string } }) {
  const [open, setOpen] = useState<string | null>(null);
  if (!challenges.length) return <div className="card muted">No hay retos para los labs y el contexto elegidos.</div>;
  const ch = challenges.find((c) => c.id === open);
  return (
    <div className="stack">
      {!ch && challenges.map((c) => (
        <div key={c.id} className="card link" onClick={() => setOpen(c.id)}>
          <div className="row"><h3 style={{ margin: 0 }}>{c.title}</h3><span className={`pill ${c.mode === 'transferencia' ? 'warn' : 'accent'}`}>{c.mode}</span><span className="faint small">{c.steps.length} etapas · requiere {c.requires.join(', ')}</span></div>
          <Md text={c.scenario.split('\n')[0]} className="small muted" />
        </div>
      ))}
      {ch && (
        <>
          <button className="btn ghost sm" onClick={() => setOpen(null)}>← Todos los retos</button>
          <div className="card"><h2>{ch.title}</h2><Md text={ch.scenario} /></div>
          <ExamRunner key={ch.id} pool={ch.steps.map((s) => ({ activity: s, notebook: 'global' }))} title={ch.title} ordered mode="global" reviewTarget={(id) => (ch.review[id] ? target(id) : { label: '', url: '' })} />
        </>
      )}
    </div>
  );
}
