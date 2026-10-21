// Editor de Python (CodeMirror 6). Ctrl/Cmd+Enter ejecuta.
import { useEffect, useRef } from 'react';
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, highlightActiveLine, drawSelection } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { python } from '@codemirror/lang-python';
import { syntaxHighlighting, indentOnInput, bracketMatching } from '@codemirror/language';
import { classHighlighter } from '@lezer/highlight';
import { closeBrackets, autocompletion } from '@codemirror/autocomplete';

export function Editor({
  value,
  onChange,
  onRun,
  minHeight = 140,
  readOnly = false,
}: {
  value: string;
  onChange?: (v: string) => void;
  onRun?: () => void;
  minHeight?: number;
  readOnly?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const cbs = useRef({ onChange, onRun });
  cbs.current = { onChange, onRun };

  useEffect(() => {
    const state = EditorState.create({
      doc: value,
      extensions: [
        lineNumbers(),
        history(),
        drawSelection(),
        highlightActiveLine(),
        indentOnInput(),
        bracketMatching(),
        closeBrackets(),
        autocompletion({ activateOnTyping: false }),
        python(),
        syntaxHighlighting(classHighlighter),
        EditorState.tabSize.of(4),
        EditorState.readOnly.of(readOnly),
        keymap.of([
          { key: 'Mod-Enter', run: () => (cbs.current.onRun?.(), true) },
          indentWithTab,
          ...defaultKeymap,
          ...historyKeymap,
        ]),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) cbs.current.onChange?.(u.state.doc.toString());
        }),
        EditorView.theme({
          '&': { backgroundColor: 'var(--code-bg)', color: 'var(--text)' },
          '.cm-content': { minHeight: `${minHeight}px`, caretColor: 'var(--accent)' },
          '.cm-gutters': { backgroundColor: 'var(--panel-2)', color: 'var(--text-3)', border: 'none' },
          '.cm-activeLine': { backgroundColor: 'var(--accent-soft)' },
          '.cm-activeLineGutter': { backgroundColor: 'var(--accent-soft)' },
          '&.cm-focused .cm-selectionBackground, .cm-selectionBackground': { backgroundColor: 'rgba(124,156,255,0.3)' },
        }),
      ],
    });
    view.current = new EditorView({ state, parent: host.current! });
    return () => view.current?.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [readOnly]);

  useEffect(() => {
    const v = view.current;
    if (v && v.state.doc.toString() !== value) {
      v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: value } });
    }
  }, [value]);

  return <div className="editor" ref={host} />;
}
