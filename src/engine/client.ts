// Cliente del motor Python: cola de peticiones al Web Worker, límite de tiempo duro y reinicio.
import { create } from 'zustand';
import type { ErrorCategory } from '../content/types';

export interface FigureOut {
  png: string;
  axes: number;
  titles: string[];
  suptitle: string;
  error?: string;
}

export interface PyError {
  type: string;
  message: string;
  line: number | null;
  category: ErrorCategory;
  detail: string;
  traceback: string;
}

export interface RunResult {
  stdout: string;
  stderr: string;
  display: string | null;
  figures: FigureOut[];
  error: PyError | null;
  metrics?: Record<string, unknown> | null;
}

export interface CheckItem {
  ok: boolean;
  message: string;
  category: ErrorCategory | null;
  detail: string | null;
  reveal: string | null;
}

export interface GradeResult {
  passed: boolean;
  items: CheckItem[];
  run: RunResult;
  fatal?: string;
}

type Status = 'idle' | 'loading' | 'ready' | 'busy' | 'error' | 'restarting';

interface EngineState {
  status: Status;
  message: string;
  versions: Record<string, string> | null;
  source: string;
  pending: number;
}

export const useEngine = create<EngineState>(() => ({
  status: 'idle',
  message: 'Python sin cargar',
  versions: null,
  source: '',
  pending: 0,
}));

const HARD_TIMEOUT_MS = 90_000;

let worker: Worker | null = null;
let nextId = 1;
const waiting = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: number }>();
let queue: Promise<unknown> = Promise.resolve();

function spawn() {
  useEngine.setState({ status: 'loading', message: 'Iniciando Python…' });
  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (ev: MessageEvent) => {
    const msg = ev.data;
    if (msg.type === 'progress') useEngine.setState({ message: msg.message });
    else if (msg.type === 'ready')
      useEngine.setState({ status: 'ready', message: 'Python listo', versions: msg.versions, source: msg.indexURL });
    else if (msg.type === 'fatal') useEngine.setState({ status: 'error', message: msg.error });
    else if (msg.type === 'result') {
      const w = waiting.get(msg.id);
      if (!w) return;
      clearTimeout(w.timer);
      waiting.delete(msg.id);
      const left = waiting.size;
      useEngine.setState({ pending: left, status: left ? 'busy' : 'ready' });
      if (msg.ok) w.resolve(msg.result);
      else w.reject(Object.assign(new Error(msg.error), { traceback: msg.traceback }));
    }
  };
  worker.onerror = (e) => useEngine.setState({ status: 'error', message: `Error del motor: ${e.message}` });
}

export function startEngine() {
  if (!worker) spawn();
}

/** Reinicia el motor (se pierde el estado de los kernels). */
export function restartEngine(reason = 'Reiniciando Python…') {
  worker?.terminate();
  worker = null;
  for (const [, w] of waiting) {
    clearTimeout(w.timer);
    w.reject(new Error('Ejecución cancelada: el motor se ha reiniciado.'));
  }
  waiting.clear();
  useEngine.setState({ status: 'restarting', message: reason, pending: 0 });
  spawn();
}

function send<T>(name: string, args: object, data: string[] = [], timeoutMs = HARD_TIMEOUT_MS): Promise<T> {
  startEngine();
  const run = () =>
    new Promise<T>((resolve, reject) => {
      const id = nextId++;
      const timer = window.setTimeout(() => {
        waiting.delete(id);
        restartEngine('La ejecución superó el tiempo máximo; reiniciando Python…');
        reject(new Error(`Tiempo máximo superado (${Math.round(timeoutMs / 1000)} s). El motor se ha reiniciado.`));
      }, timeoutMs + 60_000); // margen para la primera carga de paquetes
      waiting.set(id, { resolve: resolve as (v: unknown) => void, reject, timer });
      useEngine.setState({ pending: waiting.size, status: useEngine.getState().status === 'ready' ? 'busy' : useEngine.getState().status });
      worker!.postMessage({ id, name, args, data });
    });
  const p = queue.then(run, run);
  queue = p.catch(() => undefined);
  return p;
}

export const engine = {
  run(kernelId: string, code: string, opts: { reset?: boolean; data?: string[]; timeLimit?: number } = {}) {
    return send<RunResult>('run', { kernel_id: kernelId, code, reset: !!opts.reset, time_limit: opts.timeLimit ?? 120 }, opts.data);
  },
  vars(kernelId: string) {
    return send<{ name: string; type: string; shape: number[] | null; len: number | null }[]>('vars', { kernel_id: kernelId });
  },
  grade(a: { id: string; setup: string; solution: string; checks: string; data?: string[] }, userCode: string) {
    return send<GradeResult>(
      'grade',
      { ex_id: a.id, setup: a.setup, solution: a.solution, checks: a.checks, user_code: userCode, time_limit: 30 },
      a.data,
    );
  },
  evaluate(setup: string, code: string, expr: string, data?: string[]) {
    return send<{ run: RunResult; value: unknown }>('evaluate', { setup, code, expr }, data);
  },
  experiment(code: string, data?: string[]) {
    return send<RunResult>('experiment', { code }, data);
  },
  generator(code: string, seed: number, data?: string[]) {
    return send<{ error: PyError | null; variant: Record<string, unknown> | null; figures: FigureOut[] }>(
      'generator',
      { code, seed },
      data,
    );
  },
};
