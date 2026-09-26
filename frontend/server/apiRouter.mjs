/**
 * @file apiRouter.mjs
 * @brief Dispatcher for local REST API endpoints.
 */

import {
  discoverModels,
  executeSolve,
  loadBenchmarks,
  loadMRPLData,
  getSystemHealth,
} from './solverAdapter.mjs';

function sendJson(res, statusCode, data) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.end(JSON.stringify(data));
}

function parseJsonBody(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 10 * 1024 * 1024) {
        req.destroy();
        reject(new Error('Payload too large'));
      }
    });
    req.on('end', () => {
      if (!body) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

export async function handleApiRequest(req, res) {
  const url = new URL(req.url, 'http://localhost');
  const pathname = url.pathname;

  if (req.method === 'OPTIONS') {
    res.statusCode = 204;
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.end();
    return true;
  }

  // Health & System status
  if (req.method === 'GET' && (pathname === '/api/health' || pathname === '/api/system')) {
    try {
      const health = await getSystemHealth();
      sendJson(res, 200, health);
    } catch (err) {
      sendJson(res, 500, { error: err.message });
    }
    return true;
  }

  // Model discovery
  if (req.method === 'GET' && pathname === '/api/models') {
    try {
      const models = await discoverModels();
      sendJson(res, 200, { models });
    } catch (err) {
      sendJson(res, 500, { error: err.message });
    }
    return true;
  }

  // Solve model execution
  if (req.method === 'POST' && pathname === '/api/solve') {
    try {
      const payload = await parseJsonBody(req);
      const { model_path, options } = payload;
      if (!model_path) {
        sendJson(res, 400, { error: 'Missing required field: model_path' });
        return true;
      }

      const solveResult = await executeSolve(model_path, options || {});
      sendJson(res, 200, solveResult);
    } catch (err) {
      sendJson(res, 400, { error: err.message });
    }
    return true;
  }

  // Verification endpoint
  if (req.method === 'POST' && pathname === '/api/verify') {
    try {
      const payload = await parseJsonBody(req);
      const { model_path } = payload;
      if (!model_path) {
        sendJson(res, 400, { error: 'Missing required field: model_path' });
        return true;
      }

      const solveResult = await executeSolve(model_path, { presolve: false });
      sendJson(res, 200, {
        verification: solveResult.raw_output?.verification || null,
        result: solveResult.raw_output?.result || null,
      });
    } catch (err) {
      sendJson(res, 400, { error: err.message });
    }
    return true;
  }

  // Benchmarks
  if (req.method === 'GET' && pathname === '/api/benchmarks') {
    try {
      const benchmarks = await loadBenchmarks();
      sendJson(res, 200, { benchmarks });
    } catch (err) {
      sendJson(res, 500, { error: err.message });
    }
    return true;
  }

  // MRPL Case Study
  if (req.method === 'GET' && pathname === '/api/mrpl') {
    try {
      const mrpl = await loadMRPLData();
      sendJson(res, 200, mrpl);
    } catch (err) {
      sendJson(res, 500, { error: err.message });
    }
    return true;
  }

  return false; // Not handled
}
