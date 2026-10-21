import { useState } from 'react';
import type { FigureOut, RunResult } from '../engine/client';
import type { Cell, CellOutput, NotebookFile } from '../lib/materials';
import { CategoryTag } from './common';

export function Figures({ figures, thumb }: { figures: { png: string }[]; thumb?: boolean }) {
  const [zoom, setZoom] = useState<string | null>(null);
  if (!figures.length) return null;
  return (
    <>
      <div className="figures">
        {figures.map((f, i) =>
          f.png ? (
            <img
              key={i}
              className={thumb ? 'thumb' : ''}
              src={`data:image/png;base64,${f.png}`}
              alt={`Figura ${i + 1}`}
              onClick={() => setZoom(f.png)}
            />
          ) : null,
        )}
      </div>
      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)}>
          <img src={`data:image/png;base64,${zoom}`} alt="Figura ampliada" />
        </div>
      )}
    </>
  );
}

export function RunOutput({ run, title = 'Salida (ejecución en la app)' }: { run: RunResult | null; title?: string }) {
  if (!run) return null;
  const empty = !run.stdout && !run.display && !run.figures.length && !run.error && !run.stderr;
  return (
    <div className="output">
      <div className="faint" style={{ marginBottom: '0.3rem' }}>{title}</div>
      {empty && <div className="faint">(sin salida visible)</div>}
      {run.stdout && <pre>{run.stdout}</pre>}
      {run.display && <pre>{run.display}</pre>}
      {run.stderr && <pre className="stderr">{run.stderr}</pre>}
      {run.error && <ErrorBox error={run.error} />}
      <Figures figures={run.figures as FigureOut[]} />
    </div>
  );
}

export function ErrorBox({ error }: { error: NonNullable<RunResult['error']> }) {
  return (
    <div className="callout bad">
      <div className="row" style={{ marginBottom: '0.3rem' }}>
        <CategoryTag c={error.category} />
        <strong>
          {error.type}
          {error.line ? ` (línea ${error.line})` : ''}
        </strong>
      </div>
      <div className="mono small" style={{ marginBottom: '0.3rem' }}>{error.message}</div>
      <div>{error.detail}</div>
      <details style={{ marginTop: '0.3rem' }}>
        <summary className="faint" style={{ cursor: 'pointer' }}>Traza completa</summary>
        <pre className="small">{error.traceback}</pre>
      </details>
    </div>
  );
}

/** Salidas guardadas en el notebook original (resultados reales de tu ejecución). */
/** Salida de una celda: la guardada por ti o, si no hay, la de la ejecución completa con tu entorno. */
export function CellOutputs({ cell, file }: { cell: Cell; file: NotebookFile | null }) {
  const patch = file?.execution?.patches?.find((p) => p.cell === cell.index);
  const own = cell.outputs.length > 0;
  const outs = own ? cell.outputs : cell.execOutputs ?? [];
  const label = own
    ? 'Salida guardada en tu notebook'
    : `Salida de la ejecución completa (copia ejecutada con tu entorno local${file?.execution?.generated ? `, ${file.execution.generated}` : ''})`;
  if (!outs.length) return <div className="faint small">Esta celda no produce salida visible.</div>;
  return (
    <div className="col" style={{ gap: '0.3rem' }}>
      {!own && patch && <div className="callout warn small">Para ejecutarla se cambió: <code>{patch.new}</code> — {patch.why}</div>}
      <SavedOutputs outputs={outs} label={label} />
    </div>
  );
}

export function SavedOutputs({ outputs, label = 'Salida guardada en el notebook' }: { outputs: CellOutput[]; label?: string }) {
  if (!outputs.length) return <div className="faint">El notebook original no tiene salidas guardadas para esta celda.</div>;
  const imgs = outputs.filter((o) => o.type === 'image' && o.png).map((o) => ({ png: o.png! }));
  return (
    <div className="output">
      <div className="faint" style={{ marginBottom: '0.3rem' }}>{label}</div>
      {outputs.map((o, i) =>
        o.type === 'stream' || o.type === 'text' ? (
          o.text && !/^<Figure/.test(o.text) ? <pre key={i}>{o.text}</pre> : null
        ) : o.type === 'error' ? (
          <pre key={i} className="err">{o.text}</pre>
        ) : o.type === 'image-omitted' ? (
          <div key={i} className="faint">(figura demasiado grande, omitida en la importación)</div>
        ) : null,
      )}
      <Figures figures={imgs} />
    </div>
  );
}
