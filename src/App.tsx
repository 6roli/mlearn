import { useEffect } from 'react';
import { useRoute, href } from './router';
import { findLab, findNotebook } from '../content/index';
import { Home, LabPage, ReviewPage, ProgressPage, CoveragePage, SourcePage, MaterialsPage, ActivityPage } from './pages/Pages';
import { NotebookPage } from './pages/notebook/NotebookPage';
import { GlossaryPage } from './pages/Glossary';
import { GlobalPractice } from './pages/GlobalPractice';
import { PrintView } from './pages/PrintView';
import { startEngine, useEngine, restartEngine } from './engine/client';
import { useProgress } from './progress/store';

const NAV = [
  { id: '', label: 'Laboratorios' },
  { id: 'prueba', label: 'Ponte a prueba' },
  { id: 'repaso', label: 'Repaso' },
  { id: 'progreso', label: 'Progreso' },
  { id: 'glosario', label: 'Glosario' },
  { id: 'cobertura', label: 'Cobertura' },
  { id: 'materiales', label: 'Materiales' },
];

function EnginePill() {
  const { status, message, pending } = useEngine();
  const label = status === 'ready' ? 'Python listo' : status === 'busy' ? `Ejecutando${pending > 1 ? ` (${pending})` : ''}…` : status === 'error' ? 'Error' : message;
  return (
    <span className={`engine-pill ${status}`} title={message}>
      <span className="led" /> {label}
      {status === 'error' && <button className="btn sm ghost" onClick={() => restartEngine()}>Reintentar</button>}
    </span>
  );
}

export function App() {
  const route = useRoute();
  const theme = useProgress((s) => s.theme);
  const setTheme = useProgress((s) => s.setTheme);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(() => {
    startEngine();
  }, []);
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [route[0], route[1], route[2], route[3]]);

  let page: React.ReactNode;
  const [a, b, c, d, e] = route;
  if (!a) page = <Home />;
  else if (a === 'lab' && b && !c) {
    const lab = findLab(b);
    page = lab ? <LabPage lab={lab} /> : <NotFound />;
  } else if (a === 'lab' && b && c) {
    const lab = findLab(b);
    const nb = findNotebook(c);
    page = lab && nb ? <NotebookPage key={nb.id} lab={lab} nb={nb} tab={d} item={e} /> : <NotFound />;
  } else if (a === 'prueba') page = <GlobalPractice sub={b} />;
  else if (a === 'repaso') page = <ReviewPage />;
  else if (a === 'progreso') page = <ProgressPage />;
  else if (a === 'glosario') page = <GlossaryPage focus={b} />;
  else if (a === 'cobertura') page = <CoveragePage />;
  else if (a === 'materiales') page = <MaterialsPage />;
  else if (a === 'fuente' && b) page = <SourcePage nbId={b} cell={c} />;
  else if (a === 'actividad' && b) page = <ActivityPage id={b} />;
  else if (a === 'imprimir' && b) page = <PrintView nbId={b} />;
  else page = <NotFound />;

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href={href()}>
          <img src="/favicon.svg" alt="" /> MLEARN <small>· estudio de laboratorios</small>
        </a>
        <nav className="nav">
          {NAV.map((n) => (
            <a key={n.id} href={href(n.id)} className={(a ?? '') === n.id || (n.id === '' && a === 'lab') ? 'active' : ''}>
              {n.label}
            </a>
          ))}
        </nav>
        <span className="spacer" />
        <EnginePill />
        <button className="btn ghost sm" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title="Cambiar tema">
          {theme === 'dark' ? '☀︎' : '☾'}
        </button>
      </header>
      <main className="main">{page}</main>
    </div>
  );
}

function NotFound() {
  return (
    <div className="card">
      Página no encontrada. <a href={href()}>Volver al inicio</a>
    </div>
  );
}
