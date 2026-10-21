// Markdown + fórmulas (KaTeX). Las fórmulas se extraen antes de procesar el Markdown.
import { marked } from 'marked';
import katex from 'katex';

marked.setOptions({ gfm: true, breaks: false });

const cache = new Map<string, string>();

function tex(src: string, display: boolean) {
  return katex.renderToString(src, { displayMode: display, throwOnError: false, strict: 'ignore' });
}

export function renderMarkdown(md: string): string {
  if (!md) return '';
  const hit = cache.get(md);
  if (hit) return hit;
  const parts: string[] = [];
  const ph = (html: string) => `@@M${parts.push(html) - 1}@@`;
  let s = md
    .replace(/```[\s\S]*?```|`[^`\n]+`/g, (m) => ph(marked.parseInline(m) as string))
    .replace(/\begin\{(split|align|aligned|equation|gather)\*?\}([\s\S]+?)\end\{\1\*?\}/g, (_m, env, body) =>
      ph(tex(env === 'split' || env === 'align' ? `\begin{aligned}${body}\end{aligned}` : `\begin{${env}}${body}\end{${env}}`, true)),
    )
    .replace(/\$\$([\s\S]+?)\$\$/g, (_m, body) => ph(tex(body, true)))
    .replace(/(^|[^\$])\$([^$\n]+?)\$/g, (_m, pre, body) => pre + ph(tex(body, false)));
  s = s.replace(/<script[\s\S]*?<\/script>/gi, '');
  s = s.replace(/^(#{1,6})([^#\s])/gm, '$1 $2'); // Colab acepta «##Título» sin espacio
  let html = marked.parse(s) as string;
  html = html.replace(/@@M(\d+)@@/g, (_m, i) => parts[Number(i)]);
  cache.set(md, html);
  return html;
}

export function renderInline(md: string): string {
  return renderMarkdown(md).replace(/^<p>([\s\S]*)<\/p>\s*$/, '$1');
}
