// Ejecución de bloques en un kernel por notebook: para ejecutar un bloque se ejecutan antes, en orden,
// todos los bloques anteriores que aún no se han ejecutado (como «Run all above» en Jupyter).
import type { NotebookContent } from '../../content/types';
import type { NotebookFile } from '../../lib/materials';
import { engine, useEngine, type RunResult } from '../../engine/client';

const executed = new Map<string, number>(); // notebook → índice del último bloque ejecutado

useEngine.subscribe((s, prev) => {
  if (s.status === 'restarting' && prev.status !== 'restarting') executed.clear();
});

export function resetKernel(nbId: string) {
  executed.delete(nbId);
}

export function lastExecuted(nbId: string) {
  return executed.get(nbId) ?? -1;
}

export function blockCode(nb: NotebookContent, file: NotebookFile, blockIdx: number): string[] {
  const b = nb.blocks[blockIdx];
  if (b.noRun) return [];
  if (b.runCode) return [b.runCode];
  return b.cells.map((i) => file.cells[i]).filter((c) => c?.type === 'code').map((c) => c.source);
}

export async function runUpTo(
  nb: NotebookContent,
  file: NotebookFile,
  blockIdx: number,
  onProgress?: (msg: string) => void,
): Promise<RunResult[]> {
  let last = lastExecuted(nb.id);
  if (last >= blockIdx) last = -1; // volver a ejecutar desde el principio para reproducir el estado
  const data = [...new Set(nb.blocks.slice(0, blockIdx + 1).flatMap((b) => b.data ?? []))];
  let results: RunResult[] = [];
  let needReset = last === -1;
  for (let i = last + 1; i <= blockIdx; i++) {
    const codes = blockCode(nb, file, i);
    results = [];
    for (const code of codes) {
      onProgress?.(`Ejecutando «${nb.blocks[i].title}»…`);
      const r = await engine.run(nb.id, code, { reset: needReset, data });
      needReset = false;
      results.push(r);
      if (r.error && i < blockIdx) {
        executed.set(nb.id, i);
        return [r];
      }
    }
    executed.set(nb.id, i);
  }
  return results;
}
