// Resaltado de Python línea a línea (mismo analizador que el editor CodeMirror).
import { pythonLanguage } from '@codemirror/lang-python';
import { highlightCode, classHighlighter } from '@lezer/highlight';

function esc(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function highlightLines(code: string): string[] {
  const tree = pythonLanguage.parser.parse(code);
  const lines: string[] = [''];
  highlightCode(
    code,
    tree,
    classHighlighter,
    (text, classes) => {
      lines[lines.length - 1] += classes ? `<span class="${classes}">${esc(text)}</span>` : esc(text);
    },
    () => lines.push(''),
  );
  return lines;
}
