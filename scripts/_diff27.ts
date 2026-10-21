import { createEngine } from './lib/node-engine.ts';
import fs from 'node:fs';
const e = await createEngine();
const nb = JSON.parse(fs.readFileSync('public/materials/notebooks/lab1_1.json','utf8'));
let first = true;
for (const i of [3,6,7,8,9,10,13,14,18,27]) { const r: any = e.call('run', {kernel_id:'k', code: nb.cells[i].source, reset: first}); first=false; if (i===27) { console.log(r.stdout); console.log('---saved---'); console.log(nb.cells[27].outputs[0].text); } }
