const fs = require("fs");
const path = require("path");

const albumDir = path.join(__dirname, "source/images/album");
const files = fs.readdirSync(albumDir);

const lines = files
  .filter(f => !f.startsWith(".")) // 忽略隐藏文件
  .map(f => `- image: /images/album/${f}`)
  .join("\n");

fs.writeFileSync("./source/_data/masonry.yml", lines);
console.log("masonry.yaml generated!");