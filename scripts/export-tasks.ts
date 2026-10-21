// Exporta las tareas del enunciado (content/tasks.ts) a executed/tasks.json para ejecutarlas con tu entorno.
import fs from 'node:fs';
import path from 'node:path';
import { TASKS } from '../content/tasks.ts';

const out = path.resolve(import.meta.dirname, '../executed');
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, 'tasks.json'), JSON.stringify(TASKS.map((t) => ({ id: t.id, code: t.code, data: t.data ?? [] })), null, 1));
console.log(`${TASKS.length} tareas exportadas a executed/tasks.json`);
