import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Política de seguridad: el código Python se ejecuta en un Web Worker que solo puede conectar con
// la propia app y con el CDN de Pyodide (paquetes científicos). Sin acceso a otras webs.
const csp = (dev: boolean) => [
  "default-src 'self'",
  // en desarrollo Vite inyecta un script inline (React Fast Refresh)
  `script-src 'self' 'wasm-unsafe-eval' 'unsafe-eval'${dev ? " 'unsafe-inline'" : ''} https://cdn.jsdelivr.net`,
  "connect-src 'self' https://cdn.jsdelivr.net",
  "img-src 'self' data: blob:",
  "style-src 'self' 'unsafe-inline'",
  "font-src 'self' data:",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "frame-src 'self'",
].join('; ');

const headers = (dev: boolean) => ({ 'Content-Security-Policy': csp(dev), 'X-Content-Type-Options': 'nosniff' });

export default defineConfig({
  plugins: [react()],
  server: { port: 5175, headers: headers(true) },
  preview: { port: 5175, headers: headers(false) },
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 2500 },
});
