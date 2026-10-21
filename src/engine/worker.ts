/// <reference lib="webworker" />
// Web Worker que aloja Pyodide. No tiene acceso al DOM, a localStorage ni a los archivos del equipo:
// solo ve el sistema de archivos virtual en memoria (/data, /tmp) que la app le proporciona.
import { bootPython, callPython, writeDataFile, PYODIDE_CDN, type Loader } from './boot';

declare const self: DedicatedWorkerGlobalScope;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let py: any = null;

function appUrl(p: string) {
  return new URL(p, self.location.origin + '/').href;
}

const loader: Loader = {
  async bytes(p) {
    const r = await fetch(appUrl(p));
    if (!r.ok) throw new Error(`No se pudo cargar ${p} (${r.status}). ¿Has ejecutado «npm run import»?`);
    return new Uint8Array(await r.arrayBuffer());
  },
  async text(p) {
    const r = await fetch(appUrl(p));
    if (!r.ok) throw new Error(`No se pudo cargar ${p} (${r.status})`);
    return r.text();
  },
};

async function pyodideIndexUrl(): Promise<string> {
  try {
    // el servidor de desarrollo devuelve index.html (200) para rutas inexistentes: comprobar que es JSON
    const r = await fetch(appUrl('pyodide/pyodide-lock.json'), { method: 'HEAD' });
    if (r.ok && (r.headers.get('content-type') ?? '').includes('json')) return appUrl('pyodide/');
  } catch {
    /* sin copia local */
  }
  return PYODIDE_CDN;
}

async function init() {
  const indexURL = await pyodideIndexUrl();
  post({ type: 'progress', message: indexURL.startsWith('http') && !indexURL.startsWith(self.location.origin)
    ? 'Descargando Python (Pyodide) desde el CDN…' : 'Cargando Python (Pyodide) local…' });
  const mod = await import(/* @vite-ignore */ `${indexURL}pyodide.mjs`);
  py = await mod.loadPyodide({ indexURL });
  await bootPython(py, loader, (message) => post({ type: 'progress', message }));
  const versions = py.runPython(
    "import json, sys, numpy, pandas, scipy, sklearn, matplotlib, seaborn; json.dumps({'python': sys.version.split()[0], 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__, 'seaborn': seaborn.__version__})",
  );
  post({ type: 'ready', versions: JSON.parse(versions), indexURL });
}

function post(msg: object) {
  self.postMessage(msg);
}

const ready = init().catch((err) => {
  post({ type: 'fatal', error: String(err?.message ?? err) });
  throw err;
});

self.onmessage = async (ev: MessageEvent) => {
  const { id, name, args, data } = ev.data as { id: number; name: string; args: object; data?: string[] };
  try {
    await ready;
    for (const d of data ?? []) await writeDataFile(py, loader, d);
    const result = callPython(py, name, args);
    post({ type: 'result', id, ok: true, result });
  } catch (err) {
    const e = err as Error & { pyTraceback?: string };
    post({ type: 'result', id, ok: false, error: e.message, traceback: e.pyTraceback });
  }
};

