// Fetch only the pinned upstream compiler artifacts; never execute the download.
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const revision = '648c4a89997a351eef75cdaec3ef5b89d4937dec';
const blobs = {
  clang: '3c0910b3ed586e50c5a7d8d597db94bb80991db7',
  lld: 'c1f29067d9b35cc8a5ca3c921cae419e36f99596',
  memfs: '21455235f3d627e9f749223c29ae0c454855fd40',
  'sysroot.tar': '29235cde546a96dd14ad09a77e6d6dbac0c7d1af',
  'shared.js': '2304ea301c2e1f1cf132340116ca9d5829c187cf',
  LICENSE: 'd645695673349e3947e8e5ae42332d0ac3164cd7',
  'LICENSE.llvm': '24806ab4c9eb291db4d28e159901f7a0901b9fd1'
};
const directory = path.join(__dirname, '../frontend/assets/wasm-clang');
function hash(buffer) {
  return crypto.createHash('sha1').update(`blob ${buffer.length}\0`).update(buffer).digest('hex');
}
async function main() {
  await fs.mkdir(directory, { recursive: true });
  await Promise.all(Object.entries(blobs).map(async ([name, expected]) => {
    const target = path.join(directory, name);
    try {
      if (hash(await fs.readFile(target)) === expected) return;
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const response = await fetch(`https://raw.githubusercontent.com/binji/wasm-clang/${revision}/${name}`, {
      signal: AbortSignal.timeout(180_000)
    });
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    const buffer = Buffer.from(await response.arrayBuffer());
    if (hash(buffer) !== expected) throw new Error(`${name}: Git blob integrity mismatch`);
    await fs.writeFile(target, buffer);
    console.log(`Verified ${name}: ${buffer.length} bytes`);
  }));
  console.log(`WASM compiler ready: binji/wasm-clang@${revision}`);
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
