import { useMemo, useState } from 'react';
import { GLOSSARY, glossaryEntry } from '../../content/glossary';
import { Md } from '../ui/common';
import { href } from '../router';

export function GlossaryChip({ id }: { id: string }) {
  const g = glossaryEntry(id);
  if (!g) return <span className="pill">{id}</span>;
  return (
    <a className="pill" href={href('glosario', id)} title={`${g.en} — ${g.def}`}>
      {g.es}
    </a>
  );
}

export function GlossaryPage({ focus }: { focus?: string }) {
  const [q, setQ] = useState('');
  const [lab, setLab] = useState('');
  const list = useMemo(() => {
    const n = q.trim().toLowerCase();
    return GLOSSARY.filter((g) => (!lab || g.labs.includes(lab)) && (!n || `${g.es} ${g.en} ${g.def}`.toLowerCase().includes(n))).sort((a, b) =>
      a.es.localeCompare(b.es, 'es'),
    );
  }, [q, lab]);
  return (
    <div>
      <h1>Glosario español – inglés</h1>
      <p className="muted">Los notebooks están en inglés; la app explica en español. Cada término indica su equivalente y en qué labs aparece.</p>
      <div className="row" style={{ marginBottom: '1rem' }}>
        <input type="text" placeholder="Buscar (p. ej. kurtosis, autovalor, ROC)…" value={q} onChange={(e) => setQ(e.target.value)} style={{ minWidth: 300 }} />
        <select value={lab} onChange={(e) => setLab(e.target.value)}>
          <option value="">Todos los labs</option>
          <option value="lab0">Lab0</option>
          <option value="lab1">Lab1</option>
          <option value="lab2">Lab2</option>
        </select>
        <span className="faint">{list.length} términos</span>
      </div>
      <table>
        <thead>
          <tr><th>Español</th><th>English</th><th>Definición</th><th>Labs</th></tr>
        </thead>
        <tbody>
          {list.map((g) => (
            <tr key={g.id} id={`g-${g.id}`} style={focus === g.id ? { background: 'var(--hl)' } : undefined}>
              <td><strong>{g.es}</strong>{g.symbol && <div><Md text={g.symbol} inline /></div>}</td>
              <td><em>{g.en}</em></td>
              <td><Md text={g.def} inline /></td>
              <td className="faint">{g.labs.join(', ')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
