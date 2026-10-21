import { useMemo, useState } from 'react';
import type { Block, NotebookContent } from '../../content/types';
import type { NotebookFile } from '../../lib/materials';
import { Md, SourceLink, Collapsible, OriginPill } from '../../ui/common';
import { CodeView } from '../../ui/CodeView';
import { RunOutput, SavedOutputs } from '../../ui/Output';
import { ActivityCard } from '../../activities/ActivityCard';
import { useProgress } from '../../progress/store';
import { runUpTo, resetKernel } from './kernel';
import { engine, type RunResult } from '../../engine/client';
import { GlossaryChip } from '../Glossary';
import { STAGE_LABEL } from './FlowMap';
import { blockDependencies } from '../../lib/deps';

export function BlockView({ nb, file, block, onNav }: { nb: NotebookContent; file: NotebookFile | null; block: Block; onNav: (id: string) => void }) {
  const idx = nb.blocks.findIndex((b) => b.id === block.id);
  const read = useProgress((s) => !!s.read[block.id]);
  const attempted = useProgress((s) => (block.hiddenUntil ? !!s.items[block.hiddenUntil] : true));
  const [forceShow, setForceShow] = useState(false);
  const hidden = !!block.hiddenUntil && !attempted && !forceShow;
  const markRead = useProgress((s) => s.markRead);
  const [note, setNote] = useState<number | null>(null);
  const [runs, setRuns] = useState<RunResult[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [vars, setVars] = useState<{ name: string; type: string; shape: number[] | null; len: number | null }[] | null>(null);

  const cells = useMemo(() => (file ? block.cells.map((i) => file.cells[i]).filter(Boolean) : []), [file, block]);
  const allDeps = useMemo(() => blockDependencies(nb, file), [nb, file]);
  const deps = allDeps[block.id];
  const prev = nb.blocks[idx - 1];
  const next = nb.blocks[idx + 1];

  const run = async () => {
    setBusy('Preparando…');
    try {
      const r = await runUpTo(nb, file, idx, setBusy);
      setRuns(r);
      setVars(await engine.vars(nb.id));
    } catch (e) {
      setRuns([{ stdout: '', stderr: String((e as Error).message), display: null, figures: [], error: null }]);
    } finally {
      setBusy(null);
    }
  };

  const activeLines = (cell: number) => {
    if (note == null) return undefined;
    const n = block.lines[note];
    if (n.cell !== cell) return undefined;
    const s = new Set<number>();
    for (let i = n.lines[0]; i <= n.lines[1]; i++) s.add(i);
    return s;
  };
  const notedLines = (cell: number) => {
    const s = new Set<number>();
    block.lines.filter((n) => n.cell === cell).forEach((n) => { for (let i = n.lines[0]; i <= n.lines[1]; i++) s.add(i); });
    return s;
  };
  const onLine = (cell: number, line: number) => {
    const i = block.lines.findIndex((n) => n.cell === cell && line >= n.lines[0] && line <= n.lines[1]);
    setNote(i >= 0 ? i : null);
  };

  return (
    <div className="stack">
      <div className="nb-head">
        <div>
          <div className="crumbs">
            <span className={`stage-${block.stage} row`} style={{ display: 'inline-flex', gap: '0.35rem' }}>
              <span className="dot" style={{ width: 8, height: 8, borderRadius: '50%', display: 'inline-block' }} /> {STAGE_LABEL[block.stage]}
            </span>{' '}
            · Bloque {idx + 1} de {nb.blocks.length} · celda{block.cells.length > 1 ? 's' : ''} {block.cells.join(', ')}
          </div>
          <h2>{block.title}</h2>
          {block.origin && block.origin !== nb.origin && <OriginPill origin={block.origin} />}
        </div>
        <div className="row">
          {!block.noRun && (block.runCode || (file && cells.some((c) => c.type === 'code'))) && (
            <button className="btn primary" onClick={run} disabled={!!busy}>
              {busy ? <span className="spinner" /> : '▶'} Ejecutar hasta aquí
            </button>
          )}
          <button className={`btn ${read ? '' : 'ghost'}`} onClick={() => markRead(block.id)} disabled={read}>
            {read ? '✓ Leído' : 'Marcar como leído'}
          </button>
        </div>
      </div>

      <div className="block-grid">
        <div className="col">
          {!file && !nb.proposed && <div className="callout warn">No encuentro los materiales importados. Ejecuta <code>npm run import</code> para ver el código original de este bloque.</div>}
          {hidden && (
            <div className="callout warn">
              Este bloque contiene una <strong>solución</strong> (tuya o de una sesión anterior). Se muestra después de intentar la actividad{' '}
              <a href={`#/actividad/${block.hiddenUntil}`}>{block.hiddenUntil}</a>.{' '}
              <button className="btn sm ghost" onClick={() => setForceShow(true)}>Mostrar igualmente</button>
            </div>
          )}
          {!hidden && cells.map((c) =>
            c.type === 'markdown' ? (
              <div key={c.index} className="nb-md">
                <div className="faint" style={{ marginBottom: '0.3rem' }}>Celda {c.index} · texto original del notebook (inglés)</div>
                <Md text={c.source} />
              </div>
            ) : (
              <div key={c.index} className="col">
                <CodeView
                  code={c.source}
                  title={<>Celda {c.index} · código {block.runCode ? 'original (la app ejecuta una versión adaptada)' : 'original'}{c.executionCount ? ` · In [${c.executionCount}]` : ''}</>}
                  highlight={notedLines(c.index)}
                  strong={activeLines(c.index)}
                  noted={notedLines(c.index)}
                  onLine={(l) => onLine(c.index, l)}
                />
                <Collapsible title={<span className="small">Salida guardada en tu notebook</span>}>
                  <SavedOutputs outputs={c.outputs} />
                </Collapsible>
              </div>
            ),
          )}
          {block.runCode && (nb.proposed ? (
            <CodeView code={block.runCode} title={<>Solución de referencia de la app (no es material de la asignatura)</>} />
          ) : (
            <Collapsible title={<span className="small">Código que ejecuta la app (adaptado)</span>}>
              <CodeView code={block.runCode} />
            </Collapsible>
          ))}
          {busy && <div className="faint row"><span className="spinner" /> {busy}</div>}
          {runs?.map((r, i) => <RunOutput key={i} run={r} title={`Salida (ejecución en la app${runs.length > 1 ? `, celda ${i + 1}` : ''})`} />)}
          {vars && (
            <Collapsible title={<span className="small">Variables en el kernel tras este bloque ({vars.length})</span>}>
              <table>
                <thead><tr><th>Variable</th><th>Tipo</th><th>Forma / longitud</th></tr></thead>
                <tbody>
                  {vars.map((v) => (
                    <tr key={v.name}><td className="mono">{v.name}</td><td className="faint">{v.type}</td><td className="mono">{v.shape ? `(${v.shape.join(', ')}${v.shape.length === 1 ? ',' : ''})` : v.len != null ? `len ${v.len}` : '—'}</td></tr>
                  ))}
                </tbody>
              </table>
              <button className="btn ghost sm" onClick={() => { resetKernel(nb.id); setVars(null); setRuns(null); }}>Reiniciar kernel</button>
            </Collapsible>
          )}
        </div>

        <div className="col explain">
          {block.lines.length > 0 && (
            <div className="card tight">
              <h4>Líneas clave (haz clic)</h4>
              <ul className="note-list">
                {block.lines.map((n, i) => (
                  <li key={i} className={note === i ? 'active' : ''} onClick={() => setNote(note === i ? null : i)}>
                    <span className="lnref">c{n.cell}:{n.lines[0]}{n.lines[1] !== n.lines[0] ? `–${n.lines[1]}` : ''}</span>
                    <Md text={n.text} inline />
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="card">
            <dl>
              <dt>Qué problema resuelve</dt>
              <dd><Md text={block.problem} /></dd>
              <dt>Para qué sirve en el notebook</dt>
              <dd><Md text={block.purpose} /></dd>
              <dt>Recibe</dt>
              <dd><Md text={block.inputs} /></dd>
              <dt>Produce</dt>
              <dd><Md text={block.outputs} /></dd>
              {block.prereqs.length > 0 && (
                <>
                  <dt>Conceptos previos</dt>
                  <dd className="row" style={{ gap: '0.35rem' }}>{block.prereqs.map((p) => <GlossaryChip key={p} id={p} />)}</dd>
                </>
              )}
              <dt>Cómo se interpreta</dt>
              <dd><Md text={block.interpretation} /></dd>
            </dl>
          </div>
        </div>
      </div>

      {block.apis.length > 0 && (
        <div className="card">
          <h3>Librerías, funciones y argumentos</h3>
          <table className="api-table">
            <thead><tr><th>Elemento</th><th>Qué hace</th><th>Argumentos usados</th><th>Devuelve / aprende</th></tr></thead>
            <tbody>
              {block.apis.map((a, i) => (
                <tr key={i}>
                  <td><code>{a.name}</code><div className="kind">{a.kind} · {a.lib}</div></td>
                  <td><Md text={a.does} inline /></td>
                  <td>{a.args?.map((x, j) => <div key={j}><code>{x.name}{x.value ? `=${x.value}` : ''}</code> <span className="muted small"><Md text={x.meaning} inline /></span></div>) ?? <span className="faint">—</span>}</td>
                  <td>
                    {a.returns && <div><span className="faint small">devuelve</span> <Md text={a.returns} inline /></div>}
                    {a.learned && <div><span className="faint small">atributos aprendidos</span> {a.learned.map((l) => <code key={l} style={{ marginRight: 4 }}>{l}</code>)}</div>}
                    {!a.returns && !a.learned && <span className="faint">—</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="grid-2">
        {block.shapes && block.shapes.length > 0 && (
          <div className="card">
            <h3>Dimensiones</h3>
            <table>
              <thead><tr><th>Expresión</th><th>Forma</th><th>Significado</th></tr></thead>
              <tbody>{block.shapes.map((s, i) => <tr key={i}><td><code>{s.expr}</code></td><td className="mono">{s.shape}</td><td><Md text={s.meaning} inline /></td></tr>)}</tbody>
            </table>
          </div>
        )}
        <div className="card">
          <h3>Qué pasaría si…</h3>
          {block.whatIf.length ? (
            <ul className="md">{block.whatIf.map((w, i) => <li key={i}><strong><Md text={w.change} inline /></strong>: <Md text={w.effect} inline /></li>)}</ul>
          ) : <div className="faint">Este bloque no tiene variaciones relevantes.</div>}
          <h4 style={{ marginTop: '0.8rem' }}>Si eliminaras este bloque</h4>
          {deps && deps.defines.length ? (
            <div className="small">
              Define <span className="mono">{deps.defines.slice(0, 14).join(', ')}{deps.defines.length > 14 ? '…' : ''}</span>.{' '}
              {deps.usedBy.length ? (
                <>Fallarían con <code>NameError</code> (o usarían valores antiguos):
                  <ul className="md">{deps.usedBy.slice(0, 6).map((u) => <li key={u.blockId}><a href="#" onClick={(e) => { e.preventDefault(); onNav(u.blockId); }}>{u.title}</a> — usa <span className="mono">{u.names.join(', ')}</span></li>)}</ul>
                </>
              ) : 'Ningún bloque posterior usa esos nombres: solo se perdería su salida (tabla, figura o texto).'}
            </div>
          ) : <div className="faint small">No define variables: eliminarlo solo quitaría texto o una salida visual.</div>}
        </div>
      </div>

      {(block.corrections?.length || block.versionNotes?.length) ? (
        <div className="card">
          <h3>Correcciones y diferencias de versión</h3>
          {block.corrections?.map((c, i) => (
            <div key={i} className="callout warn">
              <div className="row"><span className="pill warn">{c.kind}</span><span className="faint">celda {c.cell}</span></div>
              <div className="grid-2" style={{ marginTop: '0.4rem' }}>
                <div><div className="faint small">Original (se conserva)</div><CodeView code={c.original} /></div>
                <div><div className="faint small">Corrección</div><CodeView code={c.fixed} /></div>
              </div>
              <Md text={c.why} />
            </div>
          ))}
          {block.versionNotes?.map((v, i) => (
            <div key={i} className="callout"><span className="faint">celda {v.cell} · versión de librerías</span><Md text={v.text} /></div>
          ))}
        </div>
      ) : null}

      <div className="grid-2">
        <div className="card tight">
          <h4>Viene de</h4>
          {prev ? <a href="#" onClick={(e) => { e.preventDefault(); onNav(prev.id); }}>← {prev.title}</a> : <span className="faint">Inicio del notebook</span>}
          {block.prev && <Md text={block.prev} className="small muted" />}
        </div>
        <div className="card tight">
          <h4>Lleva a</h4>
          {next ? <a href="#" onClick={(e) => { e.preventDefault(); onNav(next.id); }}>{next.title} →</a> : <span className="faint">Fin del notebook</span>}
          {block.next && <Md text={block.next} className="small muted" />}
        </div>
      </div>

      {block.check && (
        <div>
          <h3>Comprobación rápida</h3>
          <ActivityCard activity={block.check} notebook={nb.id} onResult={() => markRead(block.id)} />
        </div>
      )}
      <SourceLink source={{ notebook: nb.id, cells: block.cells, origin: nb.origin }} />
    </div>
  );
}
