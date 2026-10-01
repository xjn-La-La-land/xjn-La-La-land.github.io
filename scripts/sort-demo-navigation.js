'use strict';

// This standalone app does not use the blog theme's Swup page container.
hexo.extend.filter.register('after_render:html', html =>
  html.replace(/<a\b[^>]*>/gi, tag => {
    if (!/\bhref\s*=\s*(["'])\/html\/sort-algo-demo\/\1/i.test(tag)
        || /\bdata-no-swup\b/i.test(tag)) return tag;
    return tag.replace(/>$/, ' data-no-swup>');
  })
);
