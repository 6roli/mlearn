// Acceso a los materiales importados (public/materials, generado por «npm run import»).
export interface CellOutput {
  type: 'stream' | 'text' | 'image' | 'image-omitted' | 'error';
  text?: string;
  png?: string;
}
export interface Cell {
  index: number;
  type: 'code' | 'markdown' | 'raw';
  source: string;
  executionCount: number | null;
  outputs: CellOutput[];
  /** Salidas de la ejecución completa de una copia con tu entorno local (si existe). */
  execOutputs?: CellOutput[];
}
export interface NotebookFile {
  id: string;
  lab: string;
  path: string;
  origin: string;
  sha256: string;
  cells: Cell[];
  execution?: { generated?: string; versions?: Record<string, string>; patches?: { cell: number; why: string; new: string }[]; errors?: { cell: number; error: string }[] };
}
export interface MaterialIndex {
  root: string;
  files: { id: string; kind: 'notebook' | 'dataset' | 'document'; path: string; sha256: string; file?: string }[];
}

const nbCache = new Map<string, Promise<NotebookFile | null>>();
let indexPromise: Promise<MaterialIndex | null> | null = null;

export function loadNotebook(id: string): Promise<NotebookFile | null> {
  if (!nbCache.has(id)) {
    nbCache.set(
      id,
      fetch(`/materials/notebooks/${id}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    );
  }
  return nbCache.get(id)!;
}

export function loadIndex(): Promise<MaterialIndex | null> {
  indexPromise ??= fetch('/materials/index.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  return indexPromise;
}

export function docUrl(index: MaterialIndex | null, docId: string): string | null {
  const f = index?.files.find((x) => x.id === docId && x.kind === 'document');
  return f?.file ? `/materials/${f.file}` : null;
}
