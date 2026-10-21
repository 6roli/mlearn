// Recorrido guiado: contexto → cada bloque con su comprobación → experimento → interpretación →
// recuperación → práctica. Para avanzar hay que hacer algo (responder, ejecutar), no solo pulsar «siguiente».
import { useEffect, useMemo, useState } from 'react';
import type { LabContent, NotebookContent } from '../../content/types';
import type { NotebookFile } from '../../lib/materials';
import { BlockView } from './BlockView';
import { ExperimentView } from './ExperimentView';
import { ActivityCard } from '../../activities/ActivityCard';
import { Md } from '../../ui/common';
import { FlowLine } from './FlowMap';
import { useProgress } from '../../progress/store';

type Step =
  | { kind: 'context' }
  | { kind: 'block'; id: string }
  | { kind: 'experiment'; id: string }
  | { kind: 'activity'; id: string; label: string };

export function GuidedRoute({ nb, file }: { lab: LabContent; nb: NotebookContent; file: NotebookFile | null }) {
  const key = `mlearn-guided-${nb.id}`;
  const steps = useMemo<Step[]>(() => {
    const s: Step[] = [{ kind: 'context' }];
    for (const b of nb.blocks) s.push({ kind: 'block', id: b.id });
    for (const e of nb.experiments) s.push({ kind: 'experiment', id: e.id });
    const interp = nb.activities.filter((a) => a.kind !== 'code' && a.kind !== 'predict' && a.skills.includes('interpretacion')).slice(0, 4);
    for (const a of interp) s.push({ kind: 'activity', id: a.id, label: 'Interpretar' });
    for (const id of nb.recall.slice(0, 4)) s.push({ kind: 'activity', id, label: 'Recuperar' });
    return s;
  }, [nb]);
  const [i, setI] = useState(() => {
    try { return Math.min(Number(localStorage.getItem(key) ?? 0), steps.length - 1); } catch { return 0; }
  });
  const [done, setDone] = useState<Record<number, boolean>>({});
  const items = useProgress((s) => s.items);
  const read = useProgress((s) => s.read);
  const markRead = useProgress((s) => s.markRead);
  useEffect(() => { try { localStorage.setItem(key, String(i)); } catch { /* sin almacenamiento */ } }, [i, key]);

  const step = steps[i];
  const unlocked = (() => {
    if (done[i]) return true;
    if (step.kind === 'context') return true;
    if (step.kind === 'block') {
      const b = nb.blocks.find((x) => x.id === step.id)!;
      return b.check ? !!items[b.check.id] : !!read[b.id];
    }
    if (step.kind === 'activity') return !!items[step.id];
    return false;
  })();

  return (
    <div className="stack" style={{ marginTop: '1rem' }}>
      <div className="card tight">
        <div className="row">
          <strong>Recorrido guiado</strong>
          <span className="faint">Paso {i + 1} de {steps.length}</span>
          <span className="spacer" />
          <button className="btn sm" disabled={i === 0} onClick={() => setI(i - 1)}>← Anterior</button>
          <button className="btn sm primary" disabled={!unlocked || i === steps.length - 1} onClick={() => setI(i + 1)} title={unlocked ? '' : 'Completa la acción del paso para continuar'}>Siguiente →</button>
        </div>
        <div className="progress-bar" style={{ marginTop: '0.5rem' }}><span className="auto" style={{ width: `${(100 * (i + 1)) / steps.length}%` }} /></div>
        {!unlocked && (
          <div className="faint small" style={{ marginTop: '0.4rem' }}>
            {step.kind === 'block' ? 'Responde la comprobación (o marca el bloque como leído) para continuar.' : step.kind === 'experiment' ? 'Cambia algún parámetro, ejecuta y pulsa «He experimentado» para continuar.' : 'Intenta la actividad para continuar.'}
          </div>
        )}
      </div>

      {step.kind === 'context' && (
        <div className="stack">
          <div className="card"><h3>Qué vas a aprender</h3><Md text={nb.context.goal} /></div>
          <FlowLine nb={nb} onSelect={(id) => setI(steps.findIndex((s) => s.kind === 'block' && s.id === id))} />
          <div className="card"><h3>Dataset</h3><Md text={nb.context.dataset} /></div>
        </div>
      )}
      {step.kind === 'block' && (
        <BlockView key={step.id} nb={nb} file={file} block={nb.blocks.find((b) => b.id === step.id)!} onNav={(id) => setI(steps.findIndex((s) => s.kind === 'block' && s.id === id))} />
      )}
      {step.kind === 'experiment' && (
        <div>
          <ExperimentView exp={nb.experiments.find((e) => e.id === step.id)!} notebook={nb.id} />
          <button className="btn primary" onClick={() => { setDone({ ...done, [i]: true }); markRead(step.id); }}>He experimentado: continuar</button>
        </div>
      )}
      {step.kind === 'activity' && (() => {
        const a = nb.activities.find((x) => x.id === step.id);
        return a ? (
          <div>
            <h3>{step.label}</h3>
            <ActivityCard activity={a} notebook={nb.id} help={step.label === 'Recuperar' ? 'reducida' : 'completa'} onResult={() => setDone({ ...done, [i]: true })} />
          </div>
        ) : null;
      })()}
      {i === steps.length - 1 && unlocked && (
        <div className="callout ok">Has completado el recorrido. Sigue con la pestaña <strong>Práctica</strong> o con «Ponte a prueba».</div>
      )}
    </div>
  );
}
