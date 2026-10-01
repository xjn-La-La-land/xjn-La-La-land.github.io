const esbuild = require("esbuild");
const fs = require("node:fs/promises");
const path = require("node:path");
process.chdir(path.resolve(__dirname, '..'));

esbuild.build({
  entryPoints: {
    monaco: "scripts/monaco-entry.js",
    "editor.worker": "node_modules/monaco-editor/esm/vs/editor/editor.worker.js"
  },
  outdir: "frontend/assets",
  bundle: true,
  format: "iife",
  target: "es2022",
  minify: true,
  loader: { ".ttf": "file" },
  assetNames: "[name]-[hash]",
  logLevel: "info"
}).then(async () => {
  await fs.copyFile("node_modules/monaco-editor/LICENSE", "frontend/assets/Monaco-LICENSE.txt");
  await fs.copyFile("node_modules/monaco-editor/ThirdPartyNotices.txt", "frontend/assets/Monaco-ThirdPartyNotices.txt");
}).catch(error => { console.error(error.message); process.exitCode = 1; });
