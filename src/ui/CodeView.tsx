import { useMemo } from 'react';
import { highlightLines } from '../lib/highlight';

export interface CodeViewProps {
  code: string;
  title?: React.ReactNode;
  highlight?: Set<number>; // líneas (1-based) resaltadas suavemente
  strong?: Set<number>; // líneas resaltadas con fuerza
  noted?: Set<number>; // líneas con anotación
  values?: Record<number, string>; // valor actual a la derecha de la línea
  onLine?: (line: number) => void;
  startLine?: number;
  maxHeight?: number;
}

export function CodeView({ code, title, highlight, strong, noted, values, onLine, startLine = 1, maxHeight }: CodeViewProps) {
  const lines = useMemo(() => highlightLines(code.replace(/\s+$/, '')), [code]);
  return (
    <div className="code" style={maxHeight ? { maxHeight } : undefined}>
      {title && <div className="cell-head">{title}</div>}
      <div style={{ padding: '0.45rem 0' }}>
        {lines.map((html, i) => {
          const n = i + startLine;
          const cls = ['ln'];
          if (strong?.has(n)) cls.push('hl-strong');
          else if (highlight?.has(n)) cls.push('hl');
          if (noted?.has(n)) cls.push('noted');
          if (onLine) cls.push('clickable');
          return (
            <div key={i} className={cls.join(' ')} onClick={onLine ? () => onLine(n) : undefined}>
              <span className="no">{n}</span>
              <span className="tx">
                <span dangerouslySetInnerHTML={{ __html: html || ' ' }} />
                {values?.[n] && <span className="val">{values[n]}</span>}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
