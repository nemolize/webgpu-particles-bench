// Static server. COOP/COEP headers make the page cross-origin isolated, which gives performance.now() ~5us resolution.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json', '.svg': 'image/svg+xml' };
const port = Number(process.env.PORT ?? 8080);

createServer(async (req, res) => {
  const raw = normalize(new URL(req.url, 'http://localhost').pathname).replace(/^(\.\.[/\\])+/, '');
  const path = raw === '/' ? '/index.html' : raw;
  try {
    const body = await readFile(join(root, path));
    res.writeHead(200, {
      'content-type': types[extname(path)] ?? 'application/octet-stream',
      'cross-origin-opener-policy': 'same-origin',
      'cross-origin-embedder-policy': 'require-corp',
    });
    res.end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
}).listen(port, '127.0.0.1', () => console.log(`http://localhost:${port}/?run=1`));
