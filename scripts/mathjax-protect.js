'use strict';

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function encodeTex(tex) {
  return Buffer.from(tex.trim(), 'utf8').toString('base64url');
}

function decodeTex(encoded) {
  return Buffer.from(encoded, 'base64url').toString('utf8');
}

function isEscaped(text, index) {
  let count = 0;
  for (let i = index - 1; i >= 0 && text[i] === '\\'; i--) count++;
  return count % 2 === 1;
}

function findUnescaped(text, needle, start) {
  let index = text.indexOf(needle, start);
  while (index !== -1) {
    if (!isEscaped(text, index)) return index;
    index = text.indexOf(needle, index + needle.length);
  }
  return -1;
}

function renderMath(tex, display) {
  return `@@MATH_${display ? 'DISPLAY' : 'INLINE'}_${encodeTex(tex)}@@`;
}

function restoreMath(html) {
  const restored = html.replace(/@@MATH_(INLINE|DISPLAY)_([A-Za-z0-9_-]+)@@/g, (_, type, encoded) => {
    const escaped = escapeHtml(decodeTex(encoded));
    return type === 'DISPLAY'
      ? `<div class="math math-display">\\[${escaped}\\]</div>`
      : `<span class="math math-inline">\\(${escaped}\\)</span>`;
  });

  return restored.replace(/<p>\s*(<div class="math math-display">[\s\S]*?<\/div>)\s*<\/p>/g, '$1');
}

function startsFence(text, index) {
  const match = text.slice(index).match(/^( {0,3})(```+|~~~+)/);
  return match ? { raw: match[0], marker: match[2] } : null;
}

function startsIndentedCode(text, index) {
  const line = text.slice(index, text.indexOf('\n', index) === -1 ? text.length : text.indexOf('\n', index));
  if (/^[ \t]*(?:[-+*]|\d+\.)\s+/.test(line)) return false;
  return text[index] === '\t' || /^( {4,})/.test(line);
}

function isValidInlineDollar(text, start, end) {
  const before = text[start - 1] || '';
  const after = text[end + 1] || '';
  const first = text[start + 1] || '';
  const last = text[end - 1] || '';

  return (
    start !== end &&
    !/[A-Za-z0-9_]/.test(before) &&
    !/[A-Za-z0-9_]/.test(after) &&
    !/\s/.test(first) &&
    !/\s/.test(last)
  );
}

function protectMath(text) {
  let output = '';
  let i = 0;
  let atLineStart = true;
  let fence = null;
  let inlineTicks = 0;

  while (i < text.length) {
    if (fence) {
      const closingFence = atLineStart ? startsFence(text, i) : null;
      if (closingFence && closingFence.marker[0] === fence[0] && closingFence.marker.length >= fence.length) {
        output += closingFence.raw;
        i += closingFence.raw.length;
        fence = null;
        atLineStart = false;
        continue;
      }
      output += text[i];
      atLineStart = text[i] === '\n';
      i++;
      continue;
    }

    if (inlineTicks) {
      if (text.startsWith('`'.repeat(inlineTicks), i)) {
        output += '`'.repeat(inlineTicks);
        i += inlineTicks;
        inlineTicks = 0;
        atLineStart = false;
        continue;
      }
      output += text[i];
      atLineStart = text[i] === '\n';
      i++;
      continue;
    }

    const openingFence = atLineStart ? startsFence(text, i) : null;
    if (openingFence) {
      fence = openingFence.marker;
      output += openingFence.raw;
      i += openingFence.raw.length;
      atLineStart = false;
      continue;
    }

    if (atLineStart && startsIndentedCode(text, i)) {
      const nextLine = text.indexOf('\n', i);
      if (nextLine === -1) {
        output += text.slice(i);
        break;
      }
      output += text.slice(i, nextLine + 1);
      i = nextLine + 1;
      atLineStart = true;
      continue;
    }

    if (text[i] === '`') {
      const match = text.slice(i).match(/^`+/);
      inlineTicks = match[0].length;
      output += match[0];
      i += inlineTicks;
      atLineStart = false;
      continue;
    }

    if (text.startsWith('$$', i) && !isEscaped(text, i)) {
      const end = findUnescaped(text, '$$', i + 2);
      if (end !== -1) {
        output += renderMath(text.slice(i + 2, end), true);
        i = end + 2;
        atLineStart = false;
        continue;
      }
    }

    if (text.startsWith('\\[', i) && !isEscaped(text, i)) {
      const end = findUnescaped(text, '\\]', i + 2);
      if (end !== -1) {
        output += renderMath(text.slice(i + 2, end), true);
        i = end + 2;
        atLineStart = false;
        continue;
      }
    }

    if (text.startsWith('\\(', i) && !isEscaped(text, i)) {
      const end = findUnescaped(text, '\\)', i + 2);
      if (end !== -1) {
        output += renderMath(text.slice(i + 2, end), false);
        i = end + 2;
        atLineStart = false;
        continue;
      }
    }

    if (text[i] === '$' && !isEscaped(text, i) && text[i + 1] !== '$' && !/\s/.test(text[i + 1] || '')) {
      const end = findUnescaped(text, '$', i + 1);
      const tex = end === -1 ? '' : text.slice(i + 1, end);
      if (end !== -1 && tex.trim() && !tex.includes('\n') && isValidInlineDollar(text, i, end)) {
        output += renderMath(tex, false);
        i = end + 1;
        atLineStart = false;
        continue;
      }
    }

    output += text[i];
    atLineStart = text[i] === '\n';
    i++;
  }

  return output;
}

hexo.extend.filter.register('before_post_render', data => {
  if (data.mathjax || data.math) {
    data.content = protectMath(data.content);
  }
  return data;
});

hexo.extend.filter.register('after_post_render', data => {
  if (data.mathjax || data.math) {
    for (const key of ['content', 'excerpt', 'more']) {
      if (typeof data[key] === 'string') data[key] = restoreMath(data[key]);
    }
  }
  return data;
});
