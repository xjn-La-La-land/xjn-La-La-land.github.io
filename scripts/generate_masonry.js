const fs = require('fs');
const path = require('path');

hexo.extend.generator.register('masonry', function(locals) {
  const albumDir = path.join(hexo.source_dir, 'images/album');
  const outputPath = path.join(hexo.source_dir, '_data/masonry.yml');
  
  if (!fs.existsSync(albumDir)) {
    console.warn('Album directory not found:', albumDir);
    return;
  }

  const files = fs.readdirSync(albumDir)
    .filter(file => /\.(jpg|jpeg|png|gif)$/i.test(file))  // 仅匹配图片
    .map(file => `- image: /images/album/${file}`)
    .join('\n');

  fs.writeFileSync(outputPath, files, 'utf8');
  console.log('Masonry configuration generated:', outputPath);
});
