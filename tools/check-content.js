'use strict';

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const sourceRoot = path.join(repoRoot, 'source');
const markdownFiles = [];
const problems = [];

function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath);
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      markdownFiles.push(fullPath);
    }
  }
}

function addProblem(file, line, message) {
  problems.push(`${path.relative(repoRoot, file)}:${line}: ${message}`);
}

function lineNumber(text, index) {
  return text.slice(0, index).split('\n').length;
}

function checkImage(file, text, alt, target, index) {
  const line = lineNumber(text, index);

  if (/feishu\.cn\/space\/api\/box\/stream\/download\/asynccode/.test(target)) {
    addProblem(file, line, `temporary Feishu image URL: ${target}`);
    return;
  }

  if (/^(https?:)?\/\//.test(target) || target.startsWith('data:') || target.startsWith('#')) {
    return;
  }

  const cleanTarget = decodeURIComponent(target.split(/[?#]/)[0]);
  const localPath = cleanTarget.startsWith('/')
    ? path.join(sourceRoot, cleanTarget.slice(1))
    : path.resolve(path.dirname(file), cleanTarget);

  if (!fs.existsSync(localPath)) {
    addProblem(file, line, `missing local image "${target}" used by "${alt || 'image'}"`);
  }
}

walk(sourceRoot);

for (const file of markdownFiles) {
  const text = fs.readFileSync(file, 'utf8');

  for (const match of text.matchAll(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)) {
    checkImage(file, text, match[1], match[2], match.index);
  }

  for (const match of text.matchAll(/<img\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi)) {
    checkImage(file, text, 'html img', match[1], match.index);
  }
}

if (problems.length) {
  console.error(`Content check failed with ${problems.length} problem(s):`);
  for (const problem of problems) console.error(`- ${problem}`);
  process.exit(1);
}

console.log(`Content check passed for ${markdownFiles.length} Markdown file(s).`);
