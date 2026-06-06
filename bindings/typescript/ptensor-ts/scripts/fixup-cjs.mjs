// The package is "type": "module", so a bare `.js` under dist/cjs would be
// parsed as ESM. Drop a package.json that flips just that subtree to CommonJS,
// so the `require` export condition loads real CJS.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const cjsDir = join(here, '..', 'dist', 'cjs');
writeFileSync(join(cjsDir, 'package.json'), JSON.stringify({ type: 'commonjs' }, null, 2) + '\n');
console.log('wrote dist/cjs/package.json (type: commonjs)');
