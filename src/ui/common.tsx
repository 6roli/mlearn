import { useEffect, useState, type ReactNode } from 'react';
import { renderInline, renderMarkdown } from '../lib/markdown';
import type { ErrorCategory, Origin, SourceRef } from '../content/types';
import { loadIndex, docUrl, type MaterialIndex } from '../lib/materials';
import { href } from '../router';

export function Md({ text, inline, className }: { text: string; inline?: boolean; className?: string }) {
  if (inline) return <span className={className} dangerouslySetInnerHTML={{ __html: renderInline(text) }} />;
  return <div className={`md ${className ?? ''}`} dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }} />;
}

export const ORIGIN_LABEL: Record<Origin, string> = {
  original: 'Original',
  modificado: 'Modificado',
  solucion: 'Solución',
  ampliacion: 'Ampliación',
  app: 'Creado por la app',
};

export function OriginPill({ origin }: { origin: Origin }) {
  return <span className={`pill ${origin}`}>{ORIGIN_LABEL[origin]}</span>;
}

export const CATEGORY_LABEL: Record<ErrorCategory, string> = {
  sintaxis: 'Sintaxis',
  dimensiones: 'Dimensiones',
  api: 'API / librería',
  razonamiento: 'Razonamiento',
  ejecucion: 'Ejecución',
  tiempo: 'Tiempo',
  concepto: 'Concepto',
};

export function CategoryTag({ c }: { c: ErrorCategory }) {
  return <span className={`cat cat-${c}`}>{CATEGORY_LABEL[c]}</span>;
}

let indexCache: MaterialIndex | null = null;
export function useMaterialIndex() {
  const [idx, setIdx] = useState<MaterialIndex | null>(indexCache);
  useEffect(() => {
    if (!indexCache) loadIndex().then((i) => { indexCache = i; setIdx(i); });
  }, []);
  return idx;
}

/** Enlace a la fuente de una explicación o ejercicio. */
export function SourceLink({ source, compact }: { source: SourceRef; compact?: boolean }) {
  const idx = useMaterialIndex();
  const parts: ReactNode[] = [];
  if (source.notebook) {
    const nb = source.notebook.replace('_resuelto', '');
    const cells = source.cells?.length ? ` · celda${source.cells.length > 1 ? 's' : ''} ${source.cells.join(', ')}` : '';
    parts.push(
      <a key="nb" href={href('fuente', source.notebook, String(source.cells?.[0] ?? ''))}>
        {nb}.ipynb{cells}
      </a>,
    );
  }
  if (source.doc) {
    const url = docUrl(idx, source.doc);
    parts.push(
      url ? (
        <a key="doc" href={url} target="_blank" rel="noreferrer">
          {source.doc}
        </a>
      ) : (
        <span key="doc">{source.doc}</span>
      ),
    );
  }
  if (source.section) parts.push(<span key="sec">{source.section}</span>);
  return (
    <span className="faint row" style={{ gap: '0.35rem' }}>
      {!compact && <span>Fuente:</span>}
      <OriginPill origin={source.origin} />
      {parts.map((p, i) => (
        <span key={i}>{p}</span>
      ))}
      {source.note && <span>· {source.note}</span>}
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="card muted">{children}</div>;
}

export function Collapsible({ title, children, open = false }: { title: ReactNode; children: ReactNode; open?: boolean }) {
  return (
    <details open={open} className="card tight">
      <summary style={{ cursor: 'pointer', fontWeight: 600 }}>{title}</summary>
      <div style={{ marginTop: '0.6rem' }}>{children}</div>
    </details>
  );
}
