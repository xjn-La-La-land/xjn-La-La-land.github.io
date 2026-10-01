const fs = require('node:fs');
const path = require('node:path');
const { ALGORITHMS } = require('../frontend/js/source-store.js');
module.exports = Object.fromEntries(ALGORITHMS.map(id => [id, fs.readFileSync(path.join(__dirname, '..', 'algorithms', `${id}_sort.cpp`), 'utf8')]));
