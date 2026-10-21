import { useEffect, useMemo, useRef, useState } from 'react';
import type { Activity, ErrorCategory } from '../content/types';
import { ActivityCard } from './ActivityCard';
import { CategoryTag } from '../ui/common';
import { useProgress } from '../progress/store';

export interface PoolItem {
  activity: Activity;
  notebook: string;
}

interface Outcome {
  ok: boolean;
  category?: ErrorCategory | null;
  message?: string;
}

function shuffle<T>(xs: T[]): T[] {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function ExamRunner({
  pool,
  title,
  ordered = false,
  reviewTarget,
  mode = 'practica',
}: {
  pool: PoolItem[];
  title: string;
  ordered?: boolean;
  reviewTarget?: (activityId: string, notebook: string) => { label: string; url: string };
  mode?: string;
}) {
  const [phase, setPhase] = useState<'setup' | 'run' | 'end'>('setup');
  const [count, setCount] = useState(Math.min(10, pool.length));
  const [minutes, setMinutes] = useState(0);
  const [noHelp, setNoHelp] = useState(true);
  const [deferred, setDeferred] = useState(!ordered);
  const [items, setItems] = useState<PoolItem[]>([]);
  const [i, setI] = useState(0);
  const [outcomes, setOutcomes] = useState<Record<string, Outcome>>({});
  const [left, setLeft] = useState(0);
  const started = useRef(0);
  const addSession = useProgress((s) => s.addSession);

  const start = () => {
    const chosen = ordered ? pool : shuffle(pool).slice(0, count);
    setItems(chosen);
    setI(0);
    setOutcomes({});
    setLeft(minutes * 60);
    started.current = Date.now();
    setPhase('run');
  };

  const finish = () => {
    setPhase('end');
    const correct = Object.values(outcomes).filter((o) => o.ok).length;
    addSession({
      mode: `${mode}: ${title}`,
      total: items.length,
      correct,
      seconds: Math.round((Date.now() - started.current) / 1000),
      failed: items.filter((it) => !outcomes[it.activity.id]?.ok).map((it) => it.activity.id),
    });
  };

  useEffect(() => {
    if (phase !== 'run' || !minutes) return;
    const t = setInterval(() => setLeft((s) => s - 1), 1000);
    return () => clearInterval(t);
  }, [phase, minutes]);
  useEffect(() => {
    if (phase === 'run' && minutes && left <= 0) finish();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [left]);

  const byCategory = useMemo(() => {
    const m: Record<string, number> = {};
    for (const o of Object.values(outcomes)) if (!o.ok && o.category) m[o.category] = (m[o.category] ?? 0) + 1;
    return m;
  }, [outcomes]);

  if (!pool.length) return <div className="card muted">No hay actividades disponibles con estos filtros.</div>;

  if (phase === 'setup')
    return (
      <div className="card">
        <h3>{title}</h3>
        <div className="grid-3" style={{ marginTop: '0.6rem' }}>
          {!ordered && (
            <label>
              Número de actividades (máx. {pool.length})
              <input type="number" min={1} max={pool.length} value={count} onChange={(e) => setCount(Math.max(1, Math.min(pool.length, Number(e.target.value))))} style={{ display: 'block', marginTop: 4 }} />
            </label>
          )}
          <label>
            Tiempo (minutos, 0 = sin límite)
            <input type="number" min={0} max={180} value={minutes} onChange={(e) => setMinutes(Math.max(0, Number(e.target.value)))} style={{ display: 'block', marginTop: 4 }} />
          </label>
          <div className="col" style={{ gap: '0.3rem' }}>
            <label className="row"><input type="checkbox" checked={noHelp} onChange={(e) => setNoHelp(e.target.checked)} /> Ayudas desactivadas (sin pistas ni soluciones)</label>
            <label className="row"><input type="checkbox" checked={deferred} onChange={(e) => setDeferred(e.target.checked)} /> Corrección al final (simulacro)</label>
          </div>
        </div>
        <button className="btn primary" style={{ marginTop: '0.8rem' }} onClick={start}>Empezar</button>
      </div>
    );

  if (phase === 'run') {
    const it = items[i];
    const answered = !!outcomes[it.activity.id];
    return (
      <div className="stack">
        <div className="row">
          <strong>{title}</strong>
          <span className="faint">Actividad {i + 1} de {items.length}</span>
          <span className="spacer" />
          {minutes > 0 && <span className={`timer ${left < 60 ? 'delta-up' : ''}`}>⏱ {Math.floor(Math.max(left, 0) / 60)}:{String(Math.max(left, 0) % 60).padStart(2, '0')}</span>}
          <button className="btn ghost sm" onClick={finish}>Terminar</button>
        </div>
        <div className="progress-bar"><span className="read" style={{ width: `${(100 * i) / items.length}%` }} /></div>
        <ActivityCard
          key={it.activity.id + i}
          activity={it.activity}
          notebook={it.notebook}
          help={noHelp ? 'ninguna' : 'completa'}
          deferFeedback={deferred}
          context={mode === 'global' ? 'global' : 'simulacro'}
          compact={deferred}
          onResult={(r) => setOutcomes((o) => ({ ...o, [it.activity.id]: o[it.activity.id]?.ok ? o[it.activity.id] : r }))}
        />
        <div className="row">
          <button className="btn" disabled={i === 0} onClick={() => setI(i - 1)}>← Anterior</button>
          {i < items.length - 1 ? (
            <button className="btn primary" onClick={() => setI(i + 1)}>{answered ? 'Siguiente →' : 'Saltar →'}</button>
          ) : (
            <button className="btn primary" onClick={finish}>Terminar y ver resultados</button>
          )}
        </div>
      </div>
    );
  }

  const correct = items.filter((it) => outcomes[it.activity.id]?.ok).length;
  return (
    <div className="stack">
      <div className="card">
        <h3>Resultado: {correct} / {items.length}</h3>
        <div className="progress-bar" style={{ margin: '0.5rem 0' }}><span className="auto" style={{ width: `${(100 * correct) / Math.max(1, items.length)}%` }} /></div>
        {Object.keys(byCategory).length > 0 && (
          <div className="row">Errores por tipo: {Object.entries(byCategory).map(([c, n]) => <span key={c} className="row" style={{ gap: 4 }}><CategoryTag c={c as ErrorCategory} /> {n}</span>)}</div>
        )}
        <button className="btn" style={{ marginTop: '0.6rem' }} onClick={() => setPhase('setup')}>Repetir</button>
      </div>
      <div className="card">
        <h3>Qué repasar</h3>
        <table>
          <thead><tr><th></th><th>Actividad</th><th>Error</th><th>Repasa</th></tr></thead>
          <tbody>
            {items.map((it) => {
              const o = outcomes[it.activity.id];
              const target = reviewTarget?.(it.activity.id, it.notebook);
              return (
                <tr key={it.activity.id}>
                  <td>{o ? (o.ok ? '✅' : '❌') : '⏭️'}</td>
                  <td className="small">{it.activity.title ?? ('prompt' in it.activity ? String(it.activity.prompt).replace(/[*`$\\]/g, '').slice(0, 110) : it.activity.id)}<div className="faint">{it.notebook}</div></td>
                  <td className="small">{o && !o.ok ? <>{o.category && <CategoryTag c={o.category} />} {o.message?.slice(0, 120)}</> : !o ? 'Sin responder' : ''}</td>
                  <td className="small">{target && (!o || !o.ok) ? <a href={target.url}>{target.label}</a> : ''}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {deferred && (
        <details className="card">
          <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Ver las actividades con su corrección y explicación</summary>
          {items.map((it) => <ActivityCard key={it.activity.id} activity={it.activity} notebook={it.notebook} context="repaso" />)}
        </details>
      )}
    </div>
  );
}
