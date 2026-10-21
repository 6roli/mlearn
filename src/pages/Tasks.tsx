// Tareas de código del enunciado: qué se pedía, qué se ha hecho, el código y sus salidas ejecutadas.
import { useEffect, useState } from 'react';
import type { LabTask } from '../content/types';
import { Md, OriginPill, SourceLink } from '../ui/common';
import { CodeView } from '../ui/CodeView';
import { Figures, RunOutput } from '../ui/Output';
import { engine, type RunResult } from '../engine/client';

export interface TaskOutput {
  stdout: string;
  figures: string[];
  error: string | null;
  seconds: number;
  codeHash: string;
}
interface TaskFile {
  generated: string;
  versions: Record<string, string>;
  tasks: Record<string, TaskOutput>;
}

let cache: Promise<TaskFile | null> | null = null;
export function useTaskOutputs() {
  const [data, setData] = useState<TaskFile | null | undefined>(undefined);
  useEffect(() => {
    cache ??= fetch('/materials/tasks.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
    cache.then(setData);
  }, []);
  return data;
}

async function sha16(text: string) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
}

export function TaskCard({ task, out, generated, print }: { task: LabTask; out?: TaskOutput; generated?: string; print?: boolean }) {
  const [live, setLive] = useState<RunResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [stale, setStale] = useState(false);
  useEffect(() => {
    if (out && crypto?.subtle) sha16(task.code).then((h) => setStale(h !== out.codeHash));
  }, [task, out]);
  const run = async () => {
    setBusy(true);
    try {
      setLive(await engine.experiment(task.code, task.data));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="activity avoid-break" id={`task-${task.id}`}>
      <div className="meta">
        <span className="pill accent">{task.question}</span>
        <OriginPill origin={task.author === 'solucion' ? 'solucion' : 'app'} />
        <span className="faint small">{task.author === 'solucion' ? 'Basada en tu solución' : 'Solución escrita por la app (no es de la asignatura)'}</span>
      </div>
      <h3>{task.title}</h3>
      <div className="grid-2" style={{ margin: '0.4rem 0' }}>
        <div className="card tight"><h4>Qué pide el enunciado</h4><Md text={task.asks} /></div>
        <div className="card tight"><h4>Qué se ha hecho y cómo leerlo</h4><Md text={task.did} /></div>
      </div>
      <CodeView code={task.code} title="Código nuevo" maxHeight={print ? undefined : 520} />
      {out ? (
        <div className="output" style={{ marginTop: '0.5rem' }}>
          <div className="faint" style={{ marginBottom: '0.3rem' }}>
            Salida ejecutada con tu entorno local{generated ? ` (${generated})` : ''}
            {stale && <span className="pill warn" style={{ marginLeft: 6 }}>el código cambió después: ejecuta «npm run tasks»</span>}
          </div>
          {out.stdout && <pre>{out.stdout}</pre>}
          {out.error && <pre className="err">{out.error}</pre>}
          <Figures figures={out.figures.map((png) => ({ png }))} />
        </div>
      ) : (
        <div className="callout warn small" style={{ marginTop: '0.5rem' }}>Aún no hay salida guardada: ejecuta <code>npm run tasks</code> y <code>npm run import</code>, o pulsa «Ejecutar en la app».</div>
      )}
      {!print && (
        <div className="row" style={{ marginTop: '0.5rem' }}>
          <button className="btn" onClick={run} disabled={busy}>{busy ? <span className="spinner" /> : '▶'} Ejecutar en la app</button>
          <SourceLink source={task.source} />
        </div>
      )}
      {live && <div style={{ marginTop: '0.5rem' }}><RunOutput run={live} title="Ejecución en la app (Pyodide)" /></div>}
    </div>
  );
}

export function TaskList({ tasks, print }: { tasks: LabTask[]; print?: boolean }) {
  const data = useTaskOutputs();
  if (!tasks.length) return <div className="card muted">El enunciado no pide código nuevo para este notebook.</div>;
  return (
    <div>
      {tasks.map((t) => <TaskCard key={t.id} task={t} out={data?.tasks[t.id]} generated={data?.generated} print={print} />)}
    </div>
  );
}
