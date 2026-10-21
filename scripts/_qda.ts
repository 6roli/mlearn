import { createEngine } from './lib/node-engine.ts';
import fs from 'node:fs';
const e = await createEngine();
const r: any = e.call('run', { kernel_id: 'q', code: fs.readFileSync(process.argv[2], 'utf8'), reset: true });
console.log(r.stdout, r.error);
