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

const server = http.createServer(async (req, res) => {
  // Try API routes first
  if (req.url.startsWith('/api')) {
    const handled = await handleApiRequest(req, res);
    if (handled) return;
  }

  // If dist/ exists, serve static assets (production mode)
  if (fs.existsSync(DIST_DIR)) {
    const parsedUrl = new URL(req.url, `http://localhost:${PORT}`);
    let reqPath = parsedUrl.pathname;
    if (reqPath === '/' || reqPath === '') reqPath = '/index.html';

    const filePath = path.join(DIST_DIR, reqPath);
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
      const ext = path.extname(filePath);
      const mimeTypes = {
        '.html': 'text/html',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.svg': 'image/svg+xml',
      };
      res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream');
      fs.createReadStream(filePath).pipe(res);
      return;
    }

    // SPA fallback to index.html
    const indexPath = path.join(DIST_DIR, 'index.html');
    if (fs.existsSync(indexPath)) {
      res.setHeader('Content-Type', 'text/html');
      fs.createReadStream(indexPath).pipe(res);
      return;
    }
  }

  res.statusCode = 404;
  res.end('Not Found');
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[Kavach-Opt Adapter] Live HTTP Bridge active on http://127.0.0.1:${PORT}`);
});
