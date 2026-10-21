import type { NotebookContent, Stage } from '../../content/types';
import { useProgress } from '../../progress/store';

export const STAGE_LABEL: Record<Stage, string> = {
  contexto: 'Contexto',
  datos: 'Datos',
  exploracion: 'Exploración',
  preparacion: 'Preparación',
  metodo: 'Método',
  resultados: 'Resultados',
  interpretacion: 'Interpretación',
};

/** Lista lateral de bloques agrupados por etapa. */
export function FlowMap({ nb, active, onSelect }: { nb: NotebookContent; active?: string; onSelect: (id: string) => void }) {
  const read = useProgress((s) => s.read);
  const byId = Object.fromEntries(nb.blocks.map((b) => [b.id, b]));
  return (
    <nav className="flowmap">
      {nb.flow.map((st, k) => (
        <div key={st.stage + k} className={`stage stage-${st.stage}`}>
          <div className="stage-title">
            <span className="dot" /> {st.label}
          </div>
          {st.blocks.length === 0 && <div className="faint small" style={{ paddingLeft: '0.6rem' }}>{st.summary}</div>}
          {st.blocks.map((id) => {
            const b = byId[id];
            if (!b) return null;
            return (
              <button key={id} className={`blk ${active === id ? 'active' : ''}`} onClick={() => onSelect(id)}>
                <span className="tick">{read[id] ? '✓' : '·'}</span>
                <span>
                  {b.title}
                  {b.cells.length > 0 && <span className="faint"> · c{b.cells[0]}{b.cells.length > 1 ? `–${b.cells[b.cells.length - 1]}` : ''}</span>}
                </span>
              </button>
            );
          })}
        </div>
      ))}
    </nav>
  );
}

/** Mapa horizontal del flujo completo: datos → exploración → … → interpretación. */
export function FlowLine({ nb, onSelect }: { nb: NotebookContent; onSelect: (id: string) => void }) {
  return (
    <div className="flowline">
      {nb.flow.map((st, i) => (
        <div key={st.stage + i} style={{ display: 'contents' }}>
          {i > 0 && <span className="arrow">→</span>}
          <div className={`st stage-${st.stage}`}>
            <h5>
              <span className="dot" /> {st.label}
            </h5>
            <div className="small muted">{st.summary}</div>
            <div className="small" style={{ marginTop: '0.3rem' }}>
              {st.blocks.map((id) => {
                const b = nb.blocks.find((x) => x.id === id);
                return b ? (
                  <div key={id}>
                    <a href="#" onClick={(e) => { e.preventDefault(); onSelect(id); }}>{b.title}</a>
                  </div>
                ) : null;
              })}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
