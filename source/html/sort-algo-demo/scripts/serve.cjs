const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const { ALGORITHMS } = require('../frontend/js/source-store.js');

const MAX_SOURCE = 256 * 1024;
const types = { '.html':'text/html', '.js':'text/javascript', '.hpp':'text/plain', '.cpp':'text/plain', '.css':'text/css', '.svg':'image/svg+xml', '.ttf':'font/ttf' };
const hash = source => crypto.createHash('sha256').update(source).digest('hex');
function failure(status, message) { return Object.assign(new Error(message), { status }); }

function createServer({ root = path.resolve(__dirname, '..') } = {}) {
  root = require('node:fs').realpathSync(root);
  const token = crypto.randomBytes(32).toString('hex');
  const queues = new Map();
  const json = (res, status, body) => res.writeHead(status, { 'Content-Type':'application/json', 'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff' }).end(JSON.stringify(body));
  async function algorithmPath(id) {
    if (!ALGORITHMS.includes(id)) throw failure(404, '未知算法');
    const directory = path.join(root, 'algorithms');
    const filename = path.join(directory, `${id}_sort.cpp`);
    if (await fs.realpath(directory) !== directory || !(await fs.lstat(filename)).isFile()) throw failure(403, '不允许写入软链接或非算法文件');
    return filename;
  }
  async function read(id) {
    const source = await fs.readFile(await algorithmPath(id), 'utf8');
    return { source, revision: hash(source) };
  }
  function serialize(id, task) {
    const next = (queues.get(id) || Promise.resolve()).catch(() => {}).then(task);
    queues.set(id, next);
    next.finally(() => { if (queues.get(id) === next) queues.delete(id); }).catch(() => {});
    return next;
  }
  function body(req) {
    return new Promise((resolve, reject) => {
      let size = 0, rejected = false;
      const chunks = [];
      req.on('data', chunk => {
        size += chunk.length;
        if (size > MAX_SOURCE * 6 + 1024) { rejected = true; chunks.length = 0; reject(failure(413, '源码请求过大')); }
        else if (!rejected) chunks.push(chunk);
      });
      req.on('error', reject);
      req.on('end', () => {
        if (rejected) return;
        try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
        catch { reject(failure(400, '请求必须为有效 JSON')); }
      });
    });
  }
  const server = http.createServer(async (req, res) => {
    try {
      const port = server.address().port;
      if (![`127.0.0.1:${port}`, `localhost:${port}`].includes(req.headers.host)) throw failure(403, '无效的本地 Host');
      const url = new URL(req.url, `http://${req.headers.host}`);
      if (url.pathname.startsWith('/api/')) {
        if (req.headers['sec-fetch-site'] === 'cross-site') throw failure(403, '禁止跨站访问本地保存接口');
        if (url.pathname === '/api/capabilities' && req.method === 'GET') { json(res, 200, { protocol:'sort-source-v1', save:true, token }); return; }
        const match = url.pathname.match(/^\/api\/algorithms\/([a-z]+)$/);
        if (!match || !ALGORITHMS.includes(match[1])) throw failure(404, '未知接口或算法');
        const id = match[1];
        if (req.method === 'GET') { json(res, 200, await read(id)); return; }
        if (req.method !== 'PUT') throw failure(405, '不支持的请求方法');
        if (req.headers.origin !== `http://${req.headers.host}` || req.headers['x-sort-save-token'] !== token) throw failure(403, '保存来源或令牌不合法');
        if (!/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] || '')) throw failure(415, '保存需要 JSON Content-Type');
        const data = await body(req);
        if (typeof data?.source !== 'string' || !/^[a-f0-9]{64}$/.test(data.baseRevision || '')) throw failure(400, '缺少源码或源版本');
        if (Buffer.byteLength(data.source, 'utf8') > MAX_SOURCE) throw failure(413, '源码超过 256 KiB');
        const result = await serialize(id, async () => {
          const filename = await algorithmPath(id);
          if ((await read(id)).revision !== data.baseRevision) throw failure(409, '磁盘文件已被修改；当前草稿未覆盖它');
          const temp = path.join(root, 'algorithms', `.${id}-${crypto.randomBytes(12).toString('hex')}.tmp`);
          try {
            await fs.writeFile(temp, data.source, { flag:'wx', mode:0o600 });
            if ((await read(id)).revision !== data.baseRevision) throw failure(409, '磁盘文件已被修改；当前草稿未覆盖它');
            await fs.chmod(temp, (await fs.stat(filename)).mode & 0o777);
            await fs.rename(temp, filename);
            return { revision:hash(data.source) };
          } finally { await fs.unlink(temp).catch(error => { if (error.code !== 'ENOENT') throw error; }); }
        });
        json(res, 200, result); return;
      }
      if (!['GET','HEAD'].includes(req.method)) throw failure(405, '仅允许静态读取');
      const name = decodeURIComponent(url.pathname).replace(/^\/+/, '') || 'index.html';
      const resolved = path.resolve(root, name);
      const relative = path.relative(root, resolved);
      const allowed = relative === 'index.html' || relative.startsWith('frontend/') || ALGORITHMS.some(id => relative === `algorithms/${id}_sort.cpp`);
      if (!allowed || relative.startsWith('..') || !(await fs.lstat(resolved)).isFile()) throw failure(404, 'Not found');
      const real = path.relative(root, await fs.realpath(resolved));
      if (real.startsWith('..') || path.isAbsolute(real)) throw failure(403, 'Not allowed');
      const bytes = await fs.readFile(resolved);
      const wasm = relative.startsWith('frontend/assets/wasm-clang/') && ['clang','lld','memfs'].includes(path.basename(relative));
      res.writeHead(200, { 'Content-Type':wasm ? 'application/wasm' : types[path.extname(resolved)] || 'application/octet-stream', 'Cache-Control':'no-cache', 'X-Content-Type-Options':'nosniff' });
      res.end(req.method === 'HEAD' ? undefined : bytes);
    } catch (error) { json(res, error.status || (error.code === 'ENOENT' ? 404 : 500), { message:error.status ? error.message : error.code === 'ENOENT' ? 'Not found' : '文件操作失败，请检查文件权限或服务日志' }); }
  });
  return server;
}

if (require.main === module) {
  const server = createServer();
  server.on('error', error => { console.error(error.message); process.exitCode = 1; });
  server.listen(Number(process.env.SORT_DEMO_PORT || 3000), '127.0.0.1', () => console.log(`排序动画：http://127.0.0.1:${server.address().port}（本地自动保存已启用，Ctrl+C 停止）`));
}
module.exports = { createServer };
