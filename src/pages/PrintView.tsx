// Versión imprimible de un notebook (Imprimir → Guardar como PDF) para leer y anotar en tablet.
import { useState } from 'react';
import { findNotebook } from '../../content/index';
import type { Activity, Block } from '../content/types';
import { activityLevel, LEVEL_LABEL } from '../content/types';
import { Md, ORIGIN_LABEL } from '../ui/common';
import { CodeView } from '../ui/CodeView';
import { useNotebookFile } from './notebook/NotebookPage';
import { STAGE_LABEL } from './notebook/FlowMap';
import { glossaryEntry } from '../../content/glossary';
import { href } from '../router';

function promptOf(a: Activity): string {
  if (a.kind === 'code') return `**${a.title}.** ${a.prompt}\n\n*Disponible:* ${a.available}\n\n*Debe producir:* ${a.produce}`;
  if (a.kind === 'dynamic') return '*(Pregunta con variantes generadas: hazla en la app.)*';
  return a.prompt;
}

function answerOf(a: Activity): { md?: string; code?: string } {
  switch (a.kind) {
    case 'mcq':
      return { md: `**Correcta:** ${a.options.filter((o) => o.correct).map((o) => o.text).join(' / ')}\n\n${a.explanation}` };
    case 'numeric':
      return { md: `**${a.answer}${a.unit ?? ''}** (±${a.tolerance}). ${a.explanation}` };
    case 'text':
      return { md: a.model };
    case 'predict':
      return { md: a.explanation };
    case 'code':
      return { code: a.solution, md: a.explanation };
    default:
      return {};
  }
}

export function PrintView({ nbId }: { nbId: string }) {
  const nb = findNotebook(nbId);
  const file = useNotebookFile(nbId);
  const [outputs, setOutputs] = useState(true);
  const [solutions, setSolutions] = useState(false);
  const [questions, setQuestions] = useState(true);
  if (!nb) return <div className="callout bad">Notebook no encontrado.</div>;

  const hiddenBlock = (b: Block) => !!b.hiddenUntil || (nb.proposed && !!b.runCode);

  return (
    <div className="print-doc">
      <div className="no-print card" style={{ marginBottom: '1rem' }}>
        <div className="row">
          <strong>Versión para imprimir o guardar en PDF</strong>
          <span className="faint small">En el diálogo de impresión elige «Guardar como PDF» y ábrelo en tu app de notas de la tablet.</span>
          <span className="spacer" />
          <a className="btn ghost sm" href={href('lab', nb.lab, nb.id)}>← Volver al notebook</a>
        </div>
        <div className="row" style={{ marginTop: '0.5rem' }}>
          <label className="row"><input type="checkbox" checked={outputs} onChange={(e) => setOutputs(e.target.checked)} /> Salidas y figuras guardadas</label>
          <label className="row"><input type="checkbox" checked={questions} onChange={(e) => setQuestions(e.target.checked)} /> Preguntas con espacio para responder</label>
          <label className="row"><input type="checkbox" checked={solutions} onChange={(e) => setSolutions(e.target.checked)} /> Anexo de soluciones (y código de soluciones)</label>
          <button className="btn primary" onClick={() => window.print()}>🖨 Imprimir / PDF</button>
        </div>
      </div>

      <h1>{nb.title}</h1>
      <div className="faint">{file?.path ?? nb.id} · {ORIGIN_LABEL[nb.origin]} · generado por MLEARN el {new Date().toLocaleDateString()}</div>

      <h2>A · Contexto</h2>
      <Md text={nb.context.goal} />
      <h3>Dataset</h3>
      <Md text={nb.context.dataset} />
      <h3>Flujo del notebook</h3>
      <ol>
        {nb.flow.map((f, i) => (
          <li key={i}><strong>{f.label}</strong> — {f.summary} {f.blocks.length > 0 && <span className="faint">({f.blocks.map((id) => nb.blocks.find((b) => b.id === id)?.title).filter(Boolean).join(' · ')})</span>}</li>
        ))}
      </ol>
      {nb.context.questions.length > 0 && (
        <>
          <h3>Preguntas de la guía / report</h3>
          <ul>{nb.context.questions.map((q) => <li key={q.id}><strong>{q.id}.</strong> <Md text={q.text} inline /></li>)}</ul>
        </>
      )}
      <div className="print-cols">
        <div>
          <h3>Variables</h3>
          <table><tbody>{nb.context.variables.map((v) => <tr key={v.name}><td><code>{v.name}</code></td><td><Md text={v.meaning} inline /></td><td className="mono small">{v.shape}</td></tr>)}</tbody></table>
        </div>
        <div>
          <h3>Notación</h3>
          <table><tbody>{nb.context.notation.map((n, i) => <tr key={i}><td><Md text={n.symbol} inline /></td><td><Md text={n.meaning} inline /></td><td>{n.code && <code>{n.code}</code>}</td></tr>)}</tbody></table>
        </div>
      </div>

      <h2 className="page-break">B · Bloques del notebook</h2>
      {nb.blocks.map((b, i) => {
        const cells = file ? b.cells.map((c) => file.cells[c]).filter(Boolean) : [];
        const hide = hiddenBlock(b) && !solutions;
        return (
          <section key={b.id} className="print-block">
            <h3>{i + 1}. {b.title} <span className="faint small">· {STAGE_LABEL[b.stage]}{b.cells.length ? ` · celdas ${b.cells.join(', ')}` : ''}{b.origin && b.origin !== nb.origin ? ` · ${ORIGIN_LABEL[b.origin]}` : ''}</span></h3>
            {hide ? (
              <div className="callout warn small">Bloque con solución: se omite el código (activa «Anexo de soluciones» para incluirlo).</div>
            ) : (
              <>
                {cells.filter((c) => c.type === 'markdown').map((c) => <div key={c.index} className="nb-md"><Md text={c.source} /></div>)}
                {cells.filter((c) => c.type === 'code').map((c) => (
                  <div key={c.index} className="avoid-break">
                    <CodeView code={c.source} title={`Celda ${c.index}`} highlight={new Set(b.lines.filter((n) => n.cell === c.index).flatMap((n) => Array.from({ length: n.lines[1] - n.lines[0] + 1 }, (_, k) => n.lines[0] + k)))} />
                    {outputs && c.outputs.some((o) => (o.type === 'stream' || o.type === 'text') && o.text && !/^<Figure/.test(o.text)) && (
                      <pre className="print-out">{c.outputs.filter((o) => (o.type === 'stream' || o.type === 'text') && !/^<Figure/.test(o.text ?? '')).map((o) => o.text).join('\n').slice(0, 2500)}</pre>
                    )}
                    {outputs && c.outputs.filter((o) => o.type === 'image' && o.png).map((o, k) => <img key={k} className="print-fig" src={`data:image/png;base64,${o.png}`} alt="" />)}
                  </div>
                ))}
                {b.runCode && !cells.some((c) => c.type === 'code') && <CodeView code={b.runCode} title="Código de referencia de la app" />}
              </>
            )}
            {b.lines.length > 0 && (
              <ul className="small">{b.lines.map((n, k) => <li key={k}><span className="mono">c{n.cell}:{n.lines[0]}{n.lines[1] !== n.lines[0] ? `–${n.lines[1]}` : ''}</span> <Md text={n.text} inline /></li>)}</ul>
            )}
            <dl className="print-dl">
              <dt>Problema</dt><dd><Md text={b.problem} /></dd>
              <dt>Para qué sirve</dt><dd><Md text={b.purpose} /></dd>
              <dt>Recibe → produce</dt><dd><Md text={`${b.inputs} → ${b.outputs}`} /></dd>
              {b.prereqs.length > 0 && <><dt>Conceptos previos</dt><dd>{b.prereqs.map((p) => glossaryEntry(p)?.es ?? p).join(' · ')}</dd></>}
              <dt>Interpretación</dt><dd><Md text={b.interpretation} /></dd>
            </dl>
            {b.apis.length > 0 && (
              <table className="small"><thead><tr><th>Función</th><th>Qué hace</th><th>Argumentos</th></tr></thead>
                <tbody>{b.apis.map((a, k) => <tr key={k}><td><code>{a.name}</code></td><td><Md text={a.does} inline /></td><td>{a.args?.map((x) => `${x.name}${x.value ? '=' + x.value : ''}: ${x.meaning}`).join('; ')}{a.learned ? ` · aprende: ${a.learned.join(', ')}` : ''}</td></tr>)}</tbody>
              </table>
            )}
            {b.shapes && b.shapes.length > 0 && <p className="small"><strong>Dimensiones:</strong> {b.shapes.map((s) => `${s.expr} → ${s.shape} (${s.meaning})`).join(' · ')}</p>}
            {b.whatIf.length > 0 && <ul className="small">{b.whatIf.map((w, k) => <li key={k}><strong><Md text={w.change} inline /></strong>: <Md text={w.effect} inline /></li>)}</ul>}
            {b.corrections?.map((c, k) => <div key={k} className="callout warn small"><strong>Corrección ({c.kind}, celda {c.cell}):</strong> <code>{c.original}</code> → <code>{c.fixed}</code>. <Md text={c.why} inline /></div>)}
            {b.versionNotes?.map((v, k) => <div key={k} className="callout small"><Md text={v.text} inline /></div>)}
            <div className="notes-space">Notas:</div>
          </section>
        );
      })}

      {questions && (
        <>
          <h2 className="page-break">Preguntas y ejercicios</h2>
          <p className="faint small">Resuélvelos en papel y compruébalos después en la app (la corrección del código se hace ejecutándolo).</p>
          {nb.activities.map((a, i) => (
            <div key={a.id} className="print-q avoid-break">
              <div className="faint small">{i + 1} · Nivel {activityLevel(a)} · {LEVEL_LABEL[activityLevel(a)]} · {a.id}</div>
              <Md text={promptOf(a)} />
              {'code' in a && a.code && <CodeView code={a.code} />}
              {a.kind === 'code' && a.starter && <CodeView code={a.starter} title="Código de partida" />}
              {a.kind === 'mcq' && <ol type="a">{a.options.map((o, k) => <li key={k}><Md text={o.text} inline /></li>)}</ol>}
              <div className="answer-space" />
            </div>
          ))}
        </>
      )}

      {solutions && (
        <>
          <h2 className="page-break">Anexo: soluciones</h2>
          {nb.activities.map((a, i) => {
            const ans = answerOf(a);
            return (
              <div key={a.id} className="print-q avoid-break">
                <strong>{i + 1}.</strong> <span className="faint small">{a.id}</span>
                {ans.code && <CodeView code={ans.code} />}
                {ans.md && <Md text={ans.md} />}
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
