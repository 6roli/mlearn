// Plantillas de experimentos: {{param}} se sustituye por un literal de Python. El código mostrado
// es exactamente el código que se ejecuta.
import type { Experiment } from '../content/types';

export function defaults(exp: Experiment): Record<string, string> {
  return Object.fromEntries(exp.params.map((p) => [p.id, p.default]));
}

export function render(exp: Experiment, values: Record<string, string>): string {
  return exp.template.replace(/\{\{(\w+)\}\}/g, (_m, id) => {
    if (!(id in values)) throw new Error(`Parámetro sin valor: ${id}`);
    return values[id];
  });
}

/** Líneas (1-based) donde aparece cada parámetro. */
export function paramLines(exp: Experiment): Record<string, number[]> {
  const out: Record<string, number[]> = {};
  exp.template.split('\n').forEach((line, i) => {
    for (const m of line.matchAll(/\{\{(\w+)\}\}/g)) (out[m[1]] ??= []).push(i + 1);
  });
  return out;
}
