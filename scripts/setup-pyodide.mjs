// Opcional: descarga Pyodide y los paquetes que usa la app en public/pyodide/ para trabajar sin conexión.
// Sin este paso, la app descarga Pyodide del CDN de jsDelivr la primera vez (y el navegador lo guarda en caché).
import fs from 'node:fs';
import path from 'node:path';

const VERSION = '314.0.7';
const CDN = `https://cdn.jsdelivr.net/pyodide/v${VERSION}/full/`;
const OUT = path.resolve(import.meta.dirname, '../public/pyodide');
const WANT = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'micropip'];
const CORE = ['pyodide.mjs', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json'];

fs.mkdirSync(OUT, { recursive: true });
async function get(name) {
  const dest = path.join(OUT, name);
  if (fs.existsSync(dest)) return;
  const r = await fetch(CDN + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  fs.writeFileSync(dest, Buffer.from(await r.arrayBuffer()));
  console.log('  ', name);
}
for (const f of CORE) await get(f);
const lock = JSON.parse(fs.readFileSync(path.join(OUT, 'pyodide-lock.json'), 'utf8'));
const seen = new Set();
const visit = (n) => {
  const k = n.toLowerCase();
  if (seen.has(k) || !lock.packages[k]) return;
  seen.add(k);
  for (const d of lock.packages[k].depends) visit(d);
};
WANT.forEach(visit);
for (const k of seen) await get(lock.packages[k].file_name);
console.log(`Pyodide ${VERSION} y ${seen.size} paquetes en public/pyodide/`);
