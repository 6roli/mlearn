// Arranque común de Pyodide para el navegador (Web Worker) y para los scripts de verificación (Node).
export const PYODIDE_VERSION = '314.0.7';
export const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
export const PACKAGES = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'micropip'];
export const SEABORN_WHEEL = 'seaborn-0.13.2-py3-none-any.whl';
export const RUNTIME_FILES = ['mlearn_runtime.py', 'grader.py'];

export interface Loader {
  bytes(path: string): Promise<Uint8Array>; // rutas relativas a public/
  text(path: string): Promise<string>;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Py = any;

export async function bootPython(py: Py, load: Loader, log: (msg: string) => void = () => {}): Promise<void> {
  log('Cargando numpy, pandas, scipy, scikit-learn y matplotlib…');
  await py.loadPackage(PACKAGES, { messageCallback: () => {}, errorCallback: (m: string) => log(m) });
  log('Instalando seaborn…');
  py.FS.mkdirTree('/tmp/wheels');
  py.FS.writeFile(`/tmp/wheels/${SEABORN_WHEEL}`, await load.bytes(`wheels/${SEABORN_WHEEL}`));
  await py.runPythonAsync(`import micropip\nawait micropip.install('emfs:/tmp/wheels/${SEABORN_WHEEL}', deps=False)`);
  log('Preparando el entorno aislado…');
  py.FS.mkdirTree('/home/pyodide/mlearn');
  for (const f of RUNTIME_FILES) {
    py.FS.writeFile(`/home/pyodide/mlearn/${f}`, await load.text(`py/${f}`));
  }
  py.runPython(`
import sys, warnings
sys.path.insert(0, '/home/pyodide/mlearn')
warnings.filterwarnings('ignore', category=DeprecationWarning)
import mlearn_runtime, grader
import seaborn
`);
}

export async function writeDataFile(py: Py, load: Loader, name: string): Promise<void> {
  const target = `/data/${name}`;
  try {
    py.FS.stat(target);
    return; // ya cargado
  } catch {
    /* no existe todavía */
  }
  py.FS.writeFile(target, await load.bytes(`materials/data/${name}`));
}

export function callPython<T = unknown>(py: Py, name: string, args: object): T {
  const fn = py.globals.get('grader').call;
  const raw: string = fn(name, JSON.stringify(args));
  const parsed = JSON.parse(raw);
  if (!parsed.ok) {
    const err = new Error(parsed.error);
    (err as Error & { pyTraceback?: string }).pyTraceback = parsed.traceback;
    throw err;
  }
  return parsed.result as T;
}
