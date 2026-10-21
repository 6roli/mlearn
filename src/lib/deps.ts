// Análisis estático sencillo de dependencias entre bloques: qué nombres define cada bloque y qué
// bloques posteriores los usan. Sirve para explicar qué fallaría si se eliminara un bloque.
import type { NotebookContent } from '../content/types';
import type { NotebookFile } from './materials';

const KEYWORDS = new Set('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield print range len enumerate zip list dict tuple set str int float abs min max sum sorted'.split(' '));

function stripStringsAndComments(code: string) {
  return code
    .replace(/("""|''')[\s\S]*?\1/g, '""')
    .replace(/(["'])(?:\\.|(?!\1).)*\1/g, '""')
    .replace(/#.*$/gm, '');
}

export function definedNames(code: string): Set<string> {
  const c = stripStringsAndComments(code);
  const out = new Set<string>();
  for (const m of c.matchAll(/^\s*(?:import\s+[\w.]+\s+as\s+(\w+)|from\s+[\w.]+\s+import\s+(.+)|import\s+(\w+))/gm)) {
    if (m[1]) out.add(m[1]);
    if (m[3]) out.add(m[3]);
    if (m[2]) m[2].split(',').forEach((p) => { const n = p.trim().split(/\s+as\s+/).pop()!.replace(/[()]/g, '').trim(); if (n) out.add(n); });
  }
  for (const m of c.matchAll(/^\s*def\s+(\w+)/gm)) out.add(m[1]);
  for (const m of c.matchAll(/^\s*for\s+([\w\s,]+?)\s+in\b/gm)) m[1].split(',').forEach((n) => out.add(n.trim()));
  for (const m of c.matchAll(/^\s*([A-Za-z_][\w\s,]*?)\s*(?:\+|-|\*|\/)?=(?!=)/gm)) {
    m[1].split(',').map((n) => n.trim()).filter((n) => /^[A-Za-z_]\w*$/.test(n)).forEach((n) => out.add(n));
  }
  return out;
}

export function usedNames(code: string): Set<string> {
  const c = stripStringsAndComments(code);
  const out = new Set<string>();
  for (const m of c.matchAll(/(?<![.\w])([A-Za-z_]\w*)/g)) if (!KEYWORDS.has(m[1])) out.add(m[1]);
  return out;
}

export interface BlockDeps {
  defines: string[];
  usedBy: { blockId: string; title: string; names: string[] }[];
}

export function blockDependencies(nb: NotebookContent, file: NotebookFile | null): Record<string, BlockDeps> {
  const codeOf = (i: number) => {
    const b = nb.blocks[i];
    if (b.runCode) return b.runCode;
    if (!file) return '';
    return b.cells.map((k) => file.cells[k]).filter((x) => x?.type === 'code').map((x) => x.source).join('\n');
  };
  const codes = nb.blocks.map((_, i) => codeOf(i));
  const defs = codes.map(definedNames);
  const uses = codes.map(usedNames);
  const out: Record<string, BlockDeps> = {};
  nb.blocks.forEach((b, i) => {
    const usedBy: BlockDeps['usedBy'] = [];
    const alive = new Set(defs[i]);
    for (let j = i + 1; j < nb.blocks.length && alive.size; j++) {
      const names = [...alive].filter((n) => uses[j].has(n));
      if (names.length) usedBy.push({ blockId: nb.blocks[j].id, title: nb.blocks[j].title, names });
      for (const n of defs[j]) alive.delete(n); // redefinido más adelante: deja de depender de este bloque
    }
    out[b.id] = { defines: [...defs[i]], usedBy };
  });
  return out;
}
