// Motor Python para Node (verificación automática). Usa los mismos archivos que la app.
import { loadPyodide } from 'pyodide';
import fs from 'node:fs';
import path from 'node:path';
import { bootPython, callPython, writeDataFile, type Loader } from '../../src/engine/boot.ts';

const PUBLIC = path.resolve(import.meta.dirname, '../../public');

export const nodeLoader: Loader = {
  async bytes(p) { return new Uint8Array(fs.readFileSync(path.join(PUBLIC, p))); },
  async text(p) { return fs.readFileSync(path.join(PUBLIC, p), 'utf8'); },
};

export async function createEngine() {
  const py = await loadPyodide();
  await bootPython(py, nodeLoader);
  return {
    py,
    async data(names: string[]) { for (const n of names) await writeDataFile(py, nodeLoader, n); },
    call<T = any>(name: string, args: object): T { return callPython<T>(py, name, args); },
  };
}
