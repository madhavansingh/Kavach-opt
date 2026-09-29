/**
 * @file index.mjs
 * @brief Standalone local HTTP server for Kavach-Opt adapter.
 * Can be run via: node frontend/server/index.mjs (default port: 3001)
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { handleApiRequest } from './apiRouter.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DIST_DIR = path.resolve(__dirname, '../dist');

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3001;
const HOST = process.env.HOST || '0.0.0.0';

const server = http.createServer(async (req, res) => {
  // Try API routes first
  if (req.url.startsWith('/api')) {
    const handled = await handleApiRequest(req, res);
    if (handled) return;
    res.statusCode = 404;
    res.setHeader('Content-Type', 'application/json');
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    res.end(JSON.stringify({ error: 'API route not found' }));
    return;
  }

  // If dist/ exists, serve static assets (production mode)
  if (fs.existsSync(DIST_DIR)) {
    const parsedUrl = new URL(req.url, 'http://localhost');
    let reqPath = parsedUrl.pathname;
    if (reqPath === '/' || reqPath === '') reqPath = '/index.html';

    const filePath = path.join(DIST_DIR, reqPath);
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath);
      const mimeTypes = {
        '.html': 'text/html; charset=utf-8',
        '.js': 'text/javascript; charset=utf-8',
        '.mjs': 'text/javascript; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        '.webp': 'image/webp',
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.wasm': 'application/wasm',
      };
      res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
      fs.createReadStream(filePath).pipe(res);
      return;
    }

    // SPA fallback to index.html for all client-side navigation
    const indexPath = path.join(DIST_DIR, 'index.html');
    if (fs.existsSync(indexPath)) {
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      fs.createReadStream(indexPath).pipe(res);
      return;
    }
  }

  res.statusCode = 404;
  res.end('Not Found');
});

server.listen(PORT, HOST, () => {
  console.log(`[Kavach-Opt Adapter] Live HTTP Bridge active on http://${HOST}:${PORT}`);
});
