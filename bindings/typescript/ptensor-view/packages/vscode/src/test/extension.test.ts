import * as assert from 'assert';

import { fileNameFor } from '../tensorFileName';

suite('fileNameFor', () => {
	const name = (key: string) => fileNameFor(key);

	test('keeps a plain label readable', () => {
		assert.match(name('Tensor: frame'), /^Tensor-frame-[0-9a-f]{8}$/);
	});

	test('cannot escape its directory', () => {
		for (const key of [
			'../../etc/passwd',
			'..',
			'./.././x',
			'a/b/c',
			'C:\\Windows\\system32',
			'tensor\u0000.json',
		]) {
			const file = name(key);
			assert.ok(!file.includes('/'), `${key} -> ${file}`);
			assert.ok(!file.includes('\\'), `${key} -> ${file}`);
			assert.ok(!file.includes('..'), `${key} -> ${file}`);
			assert.ok(!file.startsWith('.'), `${key} -> ${file}`);
			assert.ok(!file.startsWith('-'), `${key} -> ${file}`);
		}
	});

	test('never runs a label out of characters', () => {
		// Everything readable stripped: the hash still names a file.
		assert.match(name('///'), /^tensor-[0-9a-f]{8}$/);
		assert.match(name(''), /^tensor-[0-9a-f]{8}$/);
	});

	test('separates labels that differ only in dropped characters', () => {
		// Both sanitize to the same readable part, so the hash is what keeps
		// them on different files.
		assert.notStrictEqual(name('a/b'), name('a:b'));
		assert.notStrictEqual(name('frame 1'), name('frame-1'));
	});

	test('is stable and bounded', () => {
		assert.strictEqual(name('Tensor: frame'), name('Tensor: frame'));
		assert.ok(name('x'.repeat(500)).length <= 64 + 9);
	});
});
