// Progreso del estudio (se guarda en este navegador; se puede exportar e importar en JSON).
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ErrorCategory } from '../content/types';

export type ItemStatus = 'intentado' | 'con_pistas' | 'autonomo';

export interface ItemProgress {
  status: ItemStatus;
  attempts: number;
  hints: number;
  lastAt: number;
  lastOk: boolean;
  box: number; // caja de Leitner 0..5
  due: number; // timestamp del próximo repaso
  solvedAt?: number;
}

export interface ErrorEntry {
  at: number;
  itemId: string;
  notebook: string;
  blocks?: string[];
  category: ErrorCategory;
  message: string;
  context: 'notebook' | 'global' | 'repaso' | 'simulacro';
}

export interface SavedExperiment {
  id: string;
  experimentId: string;
  notebook: string;
  name: string;
  params: Record<string, string>;
  metrics: Record<string, unknown> | null;
  at: number;
}

export interface ExamSession {
  id: string;
  at: number;
  mode: string;
  total: number;
  correct: number;
  seconds: number;
  failed: string[];
}

export type HelpLevel = 'completa' | 'reducida' | 'ninguna';

interface ProgressState {
  items: Record<string, ItemProgress>;
  read: Record<string, number>; // bloques leídos: id → timestamp
  errors: ErrorEntry[];
  experiments: SavedExperiment[];
  sessions: ExamSession[];
  helpLevel: HelpLevel;
  theme: 'dark' | 'light';
  markRead: (blockId: string) => void;
  record: (a: {
    itemId: string;
    notebook: string;
    blocks?: string[];
    ok: boolean;
    hintsUsed: number;
    category?: ErrorCategory | null;
    message?: string;
    context?: ErrorEntry['context'];
  }) => void;
  saveExperiment: (e: Omit<SavedExperiment, 'id' | 'at'>) => void;
  deleteExperiment: (id: string) => void;
  addSession: (s: Omit<ExamSession, 'id' | 'at'>) => void;
  setHelpLevel: (h: HelpLevel) => void;
  setTheme: (t: 'dark' | 'light') => void;
  importAll: (json: string) => void;
  reset: () => void;
}

const DAY = 86_400_000;
export const INTERVALS = [0, 1, 3, 7, 14, 30]; // días por caja

const initial = {
  items: {},
  read: {},
  errors: [],
  experiments: [],
  sessions: [],
  helpLevel: 'completa' as HelpLevel,
  theme: 'light' as 'dark' | 'light',
};

export const useProgress = create<ProgressState>()(
  persist(
    (set) => ({
      ...initial,
      markRead: (blockId) => set((s) => (s.read[blockId] ? s : { read: { ...s.read, [blockId]: Date.now() } })),
      record: ({ itemId, notebook, blocks, ok, hintsUsed, category, message, context = 'notebook' }) =>
        set((s) => {
          const now = Date.now();
          const prev = s.items[itemId];
          const attempts = (prev?.attempts ?? 0) + 1;
          let status: ItemStatus = prev?.status ?? 'intentado';
          let box = prev?.box ?? 0;
          let due = now;
          if (ok) {
            const autonomous = hintsUsed === 0;
            if (autonomous) status = 'autonomo';
            else if (status !== 'autonomo') status = 'con_pistas';
            box = autonomous ? Math.min(box + 1, 5) : Math.max(box, 1);
            due = now + INTERVALS[box] * DAY;
          } else {
            box = 1;
            due = now + 10 * 60_000;
          }
          const items = {
            ...s.items,
            [itemId]: {
              status,
              attempts,
              hints: Math.max(prev?.hints ?? 0, hintsUsed),
              lastAt: now,
              lastOk: ok,
              box,
              due,
              solvedAt: ok ? (prev?.solvedAt ?? now) : prev?.solvedAt,
            },
          };
          const errors =
            !ok && category
              ? [
                  { at: now, itemId, notebook, blocks, category, message: message ?? '', context },
                  ...s.errors,
                ].slice(0, 500)
              : s.errors;
          return { items, errors };
        }),
      saveExperiment: (e) =>
        set((s) => ({ experiments: [{ ...e, id: crypto.randomUUID(), at: Date.now() }, ...s.experiments].slice(0, 200) })),
      deleteExperiment: (id) => set((s) => ({ experiments: s.experiments.filter((x) => x.id !== id) })),
      addSession: (x) =>
        set((s) => ({ sessions: [{ ...x, id: crypto.randomUUID(), at: Date.now() }, ...s.sessions].slice(0, 100) })),
      setHelpLevel: (helpLevel) => set({ helpLevel }),
      setTheme: (theme) => set({ theme }),
      importAll: (json) => {
        const data = JSON.parse(json);
        set({ ...initial, ...data });
      },
      reset: () => set({ ...initial }),
    }),
    {
      name: 'mlearn-progress-v1',
      version: 2,
      // v2: el tema claro pasa a ser el predeterminado
      migrate: (state, version) => (version < 2 ? { ...(state as object), theme: 'light' } : state) as ProgressState,
    },
  ),
);

export function exportProgress(): string {
  const { items, read, errors, experiments, sessions, helpLevel, theme } = useProgress.getState();
  return JSON.stringify({ items, read, errors, experiments, sessions, helpLevel, theme }, null, 1);
}

export function isDue(p: ItemProgress | undefined, now = Date.now()) {
  return !!p && p.due <= now;
}
