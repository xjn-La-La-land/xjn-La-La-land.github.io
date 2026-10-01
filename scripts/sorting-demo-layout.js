'use strict';

const fs = require('node:fs');
const path = require('node:path');

hexo.extend.filter.register('before_generate', () => {
  hexo.theme.setView('sorting-demo.ejs', fs.readFileSync(
    path.join(hexo.base_dir, 'layouts/sorting-demo.ejs'), 'utf8'
  ));
});

// A full-page demo needs a native visit, including links created by site search.
const entryScript = `<script id="sorting-demo-native-entry">
document.addEventListener('click', function (event) {
  var link = event.target.closest && event.target.closest('a[href]');
  if (link && new URL(link.href).pathname === '/2026/10/01/sorting-algorithm-visualizer/') {
    link.setAttribute('data-no-swup', '');
  }
}, true);
</script>`;

hexo.extend.filter.register('after_render:html', html => {
  if (!html.includes('</head>') || html.includes('id="sorting-demo-native-entry"')) return html;
  return html.replace('</head>', entryScript + '\n</head>');
});
