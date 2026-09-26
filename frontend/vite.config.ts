import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
// @ts-expect-error - node module
import { handleApiRequest } from './server/apiRouter.mjs';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: 'kavach-opt-api-middleware',
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          if (req.url && req.url.startsWith('/api')) {
            try {
              const handled = await handleApiRequest(req, res);
              if (handled) return;
            } catch (err) {
              console.error('API Middleware error:', err);
            }
          }
          next();
        });
      },
      configurePreviewServer(server) {
        server.middlewares.use(async (req, res, next) => {
          if (req.url && req.url.startsWith('/api')) {
            try {
              const handled = await handleApiRequest(req, res);
              if (handled) return;
            } catch (err) {
              console.error('API Preview Middleware error:', err);
            }
          }
          next();
        });
      },
    },
  ],
  server: {
    port: 5173,
  },
});
