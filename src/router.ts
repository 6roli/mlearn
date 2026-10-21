// Enrutado por hash (#/lab/lab1/lab1_1/codigo/b_fit): funciona abriendo la app en local sin servidor especial.
import { useSyncExternalStore } from 'react';

function current() {
  return decodeURIComponent(window.location.hash.replace(/^#\/?/, ''))
    .split('?')[0]
    .split('/')
    .filter(Boolean);
}

let snapshot = current();
let key = snapshot.join('/');

function subscribe(cb: () => void) {
  const on = () => {
    const next = current();
    if (next.join('/') !== key) {
      snapshot = next;
      key = next.join('/');
    }
    cb();
  };
  window.addEventListener('hashchange', on);
  return () => window.removeEventListener('hashchange', on);
}

export function useRoute(): string[] {
  return useSyncExternalStore(subscribe, () => snapshot);
}

export function go(...parts: (string | undefined)[]) {
  window.location.hash = '/' + parts.filter(Boolean).join('/');
}

export function href(...parts: (string | undefined)[]) {
  return '#/' + parts.filter(Boolean).join('/');
}
