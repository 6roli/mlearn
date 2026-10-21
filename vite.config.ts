import { defineConfig, type Plugin } from 'vite';
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

// La política se aplica a la app (HTML, scripts, worker) pero no a los materiales (PDF, CSV, JSON):
// con object-src 'none' el visor de PDF del navegador no podría abrirlos.
function securityHeaders(dev: boolean): Plugin {
  const add = (req: { url?: string }, res: { setHeader: (k: string, v: string) => void }, next: () => void) => {
    if (!req.url?.startsWith('/materials/')) {
      for (const [k, v] of Object.entries(headers(dev))) res.setHeader(k, v);
    }
    next();
  };
  return {
    name: 'mlearn-security-headers',
    configureServer: (server) => { server.middlewares.use(add); },
    configurePreviewServer: (server) => { server.middlewares.use(add); },
  };
}

export default defineConfig(({ command, isPreview }) => ({
  plugins: [react(), securityHeaders(command === 'serve' && !isPreview)],
  server: { port: 5175 },
  preview: { port: 5175 },
  worker: { format: 'es' },
  build: { target: 'es2022', chunkSizeWarningLimit: 2500 },
}));
