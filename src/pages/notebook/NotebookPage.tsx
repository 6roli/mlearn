import { useEffect, useMemo, useState } from 'react';
import type { Activity, LabContent, NotebookContent } from '../../content/types';
import { activityLevel, LEVEL_LABEL } from '../../content/types';
import { loadNotebook, type NotebookFile } from '../../lib/materials';
import { go, href } from '../../router';
import { Md, OriginPill, Empty } from '../../ui/common';
import { FlowLine, FlowMap } from './FlowMap';
import { BlockView } from './BlockView';
import { ExperimentView } from './ExperimentView';
import { ActivityCard, useActivityStatus } from '../../activities/ActivityCard';
import { useProgress, type HelpLevel } from '../../progress/store';
import { ExamRunner } from '../../activities/ExamRunner';
import { GuidedRoute } from './GuidedRoute';
import { TaskList } from '../Tasks';
import { TASKS } from '../../../content/tasks';

const TABS = [
  { id: 'contexto', letter: 'A', label: 'Contexto' },
  { id: 'codigo', letter: 'B', label: 'Código' },
  { id: 'experimentar', letter: 'C', label: 'Experimentar' },
  { id: 'interpretar', letter: 'D', label: 'Interpretar' },
  { id: 'recuperar', letter: 'E', label: 'Recuperar de memoria' },
  { id: 'practica', letter: 'F', label: 'Práctica' },
  { id: 'tareas', letter: 'G', label: 'Tareas del enunciado' },
];

export function useNotebookFile(id: string) {
  const [file, setFile] = useState<NotebookFile | null | undefined>(undefined);
  useEffect(() => {
    let alive = true;
    loadNotebook(id).then((f) => alive && setFile(f));
    return () => { alive = false; };
  }, [id]);
  return file;
}

export function NotebookPage({ lab, nb, tab = 'contexto', item }: { lab: LabContent; nb: NotebookContent; tab?: string; item?: string }) {
  const file = useNotebookFile(nb.id);
  const [mode, setMode] = useState<'libre' | 'guiado' | 'examen'>(tab === 'guiado' ? 'guiado' : tab === 'examen' ? 'examen' : 'libre');
  const activityIds = useMemo(() => nb.activities.map((a) => a.id), [nb]);
  const st = useActivityStatus(activityIds);
  const read = useProgress((s) => nb.blocks.filter((b) => s.read[b.id]).length);
  const setTab = (t: string, it?: string) => go('lab', lab.id, nb.id, t, it);

  return (
    <div>
      <div className="crumbs">
        <a href={href()}>Inicio</a> › <a href={href('lab', lab.id)}>{lab.title}</a> › {nb.short}
      </div>
      <div className="nb-head">
        <div>
          <h1>{nb.title}</h1>
          <div className="row">
            <OriginPill origin={nb.origin} />
            {file && <span className="faint small">{file.path} · {file.cells.length} celdas</span>}
            {file === null && !nb.proposed && <span className="pill bad">Materiales sin importar</span>}
            {nb.proposed && <span className="pill warn">Notebook propuesto por la guía (no existe en los materiales)</span>}
            <a className="btn ghost sm" href={href('imprimir', nb.id)} title="Versión para imprimir o guardar en PDF y anotar en la tablet">🖨 PDF para tablet</a>
          </div>
        </div>
        <div className="col" style={{ alignItems: 'flex-end' }}>
          <div className="segmented">
            <button className={mode === 'guiado' ? 'active' : ''} onClick={() => setMode('guiado')}>Recorrido guiado</button>
            <button className={mode === 'libre' ? 'active' : ''} onClick={() => setMode('libre')}>Estudio libre</button>
            <button className={mode === 'examen' ? 'active' : ''} onClick={() => setMode('examen')}>Práctica de examen</button>
          </div>
          <div className="faint small">
            {read}/{nb.blocks.length} bloques leídos · {st.auto} sin ayuda · {st.hint} con pistas · {st.tried} intentados de {st.total} actividades
          </div>
        </div>
      </div>

      {mode === 'guiado' && <GuidedRoute lab={lab} nb={nb} file={file ?? null} />}
      {mode === 'examen' && (
        <div style={{ marginTop: '1rem' }}>
          <ExamRunner pool={nb.activities.map((a) => ({ activity: a, notebook: nb.id }))} title={`Práctica de examen · ${nb.short}`} reviewTarget={(id) => reviewTargetFor(lab, nb, id)} />
        </div>
      )}
      {mode === 'libre' && (
        <>
          <div className="tabs">
            {TABS.map((t) => (
              <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
                <span className="letter">{t.letter}</span>
                {t.label}
              </button>
            ))}
          </div>
          {tab === 'contexto' && <ContextTab nb={nb} onBlock={(id) => setTab('codigo', id)} />}
          {tab === 'codigo' && (
            <div className="layout-code">
              <FlowMap nb={nb} active={item ?? nb.blocks[0]?.id} onSelect={(id) => setTab('codigo', id)} />
              <div>
                {(() => {
                  const b = nb.blocks.find((x) => x.id === (item ?? nb.blocks[0]?.id));
                  return b ? <BlockView key={b.id} nb={nb} file={file ?? null} block={b} onNav={(id) => { setTab('codigo', id); window.scrollTo({ top: 0 }); }} /> : <Empty>Sin bloques.</Empty>;
                })()}
              </div>
            </div>
          )}
          {tab === 'experimentar' && (nb.experiments.length ? <ExperimentsTab nb={nb} item={item} onSelect={(id) => setTab('experimentar', id)} /> : <Empty>Este notebook no tiene parámetros que experimentar.</Empty>)}
          {tab === 'interpretar' && <ActivityList nb={nb} filter={(a) => a.kind !== 'code' && a.kind !== 'predict' && (a.skills.includes('interpretacion') || a.skills.includes('concepto') || a.skills.includes('metodo'))} intro="Preguntas sobre los resultados reales del notebook (o de una ejecución modificada). Justifica y compara antes de mirar la explicación." />}
          {tab === 'recuperar' && <RecallTab nb={nb} />}
          {tab === 'practica' && <PracticeTab lab={lab} nb={nb} />}
          {tab === 'tareas' && (
            <div>
              <p className="muted">Lo que la guía pide <strong>programar</strong> a partir de este notebook: qué se pedía, qué se ha hecho, el código nuevo y su ejecución. Para practicar antes de mirar, usa las pestañas E y F.</p>
              <TaskList tasks={TASKS.filter((t) => t.notebook === nb.id)} />
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function reviewTargetFor(lab: LabContent, nb: NotebookContent, activityId: string) {
  const a = nb.activities.find((x) => x.id === activityId) ?? nb.blocks.find((b) => b.check?.id === activityId)?.check;
  const block = a?.blocks?.[0] ? nb.blocks.find((b) => b.id === a.blocks![0]) : undefined;
  return { label: block ? `${nb.short} › ${block.title}` : nb.short, url: href('lab', lab.id, nb.id, 'codigo', block?.id) };
}

function ContextTab({ nb, onBlock }: { nb: NotebookContent; onBlock: (id: string) => void }) {
  return (
    <div className="stack">
      <div className="card"><h3>Objetivo</h3><Md text={nb.context.goal} /></div>
      <div>
        <h3>Mapa del flujo completo</h3>
        <FlowLine nb={nb} onSelect={onBlock} />
      </div>
      <div className="grid-2">
        <div className="card">
          <h3>Dataset y escenario</h3>
          <Md text={nb.context.dataset} />
          {nb.context.classes && (
            <div className="legend-classes" style={{ marginTop: '0.5rem' }}>
              {nb.context.classes.map((c, i) => <span key={c.id} style={{ ['--dot' as string]: `var(--c${i % 5})` }}>{c.label}</span>)}
            </div>
          )}
        </div>
        <div className="card">
          <h3>Preguntas que responde</h3>
          <table>
            <tbody>
              {nb.context.questions.map((q) => (
                <tr key={q.id}>
                  <td><strong>{q.id}</strong></td>
                  <td><Md text={q.text} inline />{q.blocks.length > 0 && <div className="small">{q.blocks.map((b) => <a key={b} href="#" onClick={(e) => { e.preventDefault(); onBlock(b); }} style={{ marginRight: 8 }}>→ {nb.blocks.find((x) => x.id === b)?.title}</a>)}</div>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="grid-2">
        <div className="card">
          <h3>Variables principales</h3>
          <table>
            <thead><tr><th>Variable</th><th>Significado</th><th>Forma</th></tr></thead>
            <tbody>{nb.context.variables.map((v) => <tr key={v.name}><td><code>{v.name}</code></td><td><Md text={v.meaning} inline /></td><td className="mono small">{v.shape}</td></tr>)}</tbody>
          </table>
        </div>
        <div className="card">
          <h3>Notación y símbolos</h3>
          <table>
            <thead><tr><th>Símbolo</th><th>Significado</th><th>En el código</th></tr></thead>
            <tbody>{nb.context.notation.map((n, i) => <tr key={i}><td><Md text={n.symbol} inline /></td><td><Md text={n.meaning} inline /></td><td>{n.code && <code>{n.code}</code>}</td></tr>)}</tbody>
          </table>
        </div>
      </div>
      {nb.notes?.length ? <div className="callout"><Md text={nb.notes.map((n) => `- ${n}`).join('\n')} /></div> : null}
    </div>
  );
}

function ExperimentsTab({ nb, item, onSelect }: { nb: NotebookContent; item?: string; onSelect: (id: string) => void }) {
  const exp = nb.experiments.find((e) => e.id === item) ?? nb.experiments[0];
  return (
    <div>
      {nb.experiments.length > 1 && (
        <div className="segmented" style={{ marginBottom: '1rem' }}>
          {nb.experiments.map((e) => <button key={e.id} className={e.id === exp.id ? 'active' : ''} onClick={() => onSelect(e.id)}>{e.title}</button>)}
        </div>
      )}
      <ExperimentView key={exp.id} exp={exp} notebook={nb.id} />
    </div>
  );
}

function ActivityList({ nb, filter, intro }: { nb: NotebookContent; filter: (a: Activity) => boolean; intro: string }) {
  const list = nb.activities.filter(filter);
  return (
    <div>
      <p className="muted">{intro}</p>
      {list.length ? list.map((a) => <ActivityCard key={a.id} activity={a} notebook={nb.id} />) : <Empty>Aún no hay actividades de este tipo.</Empty>}
    </div>
  );
}

function RecallTab({ nb }: { nb: NotebookContent }) {
  const [levelFilter, setLevelFilter] = useState(0);
  const helpFor = (i: number, n: number): HelpLevel => (i < n / 3 ? 'completa' : i < (2 * n) / 3 ? 'reducida' : 'ninguna');
  const recall = nb.recall.map((id) => nb.activities.find((a) => a.id === id)).filter(Boolean) as Activity[];
  return (
    <div className="stack">
      <div className="callout">
        <strong>Recorrido de recuperación.</strong> Las ayudas se retiran progresivamente: los primeros ejercicios muestran los datos disponibles y las pistas; los del medio, pistas reducidas; los últimos se resuelven <em>sin ayudas</em>, desde el editor vacío. Un ejercicio resuelto sin pistas cuenta como <strong>autónomo</strong>.
      </div>
      {recall.map((a, i) => {
        const h = helpFor(i, recall.length);
        return (
          <div key={a.id}>
            <div className="faint small" style={{ margin: '0.6rem 0 0.2rem' }}>Paso {i + 1}/{recall.length} · ayudas: {h}</div>
            <ActivityCard activity={a} notebook={nb.id} help={h} />
          </div>
        );
      })}
      <hr />
      <div className="row">
        <h3 style={{ margin: 0 }}>Todos los ejercicios de código por nivel</h3>
        <select value={levelFilter} onChange={(e) => setLevelFilter(Number(e.target.value))}>
          <option value={0}>Todos los niveles</option>
          {[1, 2, 3, 4, 5, 6].map((l) => <option key={l} value={l}>{l}. {LEVEL_LABEL[l]}</option>)}
        </select>
      </div>
      {[1, 2, 3, 4, 5, 6].filter((l) => !levelFilter || l === levelFilter).map((l) => {
        const items = nb.activities.filter((a) => activityLevel(a) === l && (l > 1 || a.skills.includes('codigo') || a.skills.includes('api')));
        return items.length ? (
          <div key={l}>
            <h4 style={{ marginTop: '1rem' }}>{l}. {LEVEL_LABEL[l]}</h4>
            {items.map((a) => <ActivityCard key={a.id} activity={a} notebook={nb.id} />)}
          </div>
        ) : null;
      })}
    </div>
  );
}

function PracticeTab({ lab, nb }: { lab: LabContent; nb: NotebookContent }) {
  const [i, setI] = useState(0);
  if (!nb.practice.length) return <Empty>Este notebook aún no tiene práctica integradora.</Empty>;
  const p = nb.practice[Math.min(i, nb.practice.length - 1)];
  return (
    <div className="stack">
      {nb.practice.length > 1 && (
        <div className="segmented">{nb.practice.map((x, k) => <button key={x.id} className={k === i ? 'active' : ''} onClick={() => setI(k)}>{x.title}</button>)}</div>
      )}
      <div className="card">
        <h2>{p.title}</h2>
        <Md text={p.scenario} />
        <div className="faint small">Combina {p.steps.length} pasos de distintos bloques. Hazlos en orden, sin volver al notebook.</div>
      </div>
      <ExamRunner
        key={p.id}
        pool={p.steps.map((id) => ({ activity: nb.activities.find((a) => a.id === id)!, notebook: nb.id })).filter((x) => x.activity)}
        title={p.title}
        ordered
        reviewTarget={(id) => reviewTargetFor(lab, nb, id)}
      />
    </div>
  );
}
