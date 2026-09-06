import { describe, expect, test } from 'bun:test';
import { lineSplitter } from '../lineSplitter';

describe('lineSplitter', () => {
    test('emits only complete lines and keeps the remainder', () => {
      const splitter = lineSplitter(1024);
      const encoder = new TextEncoder();
      expect(splitter(encoder.encode('{"a":1}\n{"b":'))).toEqual(['{"a":1}']);
      expect(splitter(encoder.encode('2}\n'))).toEqual(['{"b":2}']);
    });

    test('joins a chunk split inside a multi-byte character', () => {
        const splitter = lineSplitter(1024);
        const bytes = new TextEncoder().encode('"héllo"\n');
        const cut = 2; // splits the two-byte 'é'
        expect(splitter(bytes.slice(0, cut))).toEqual([]);
        expect(splitter(bytes.slice(cut))).toEqual(['"héllo"']);
    });

    test('reports overflow once one line passes the cap', () => {
        const splitter = lineSplitter(8);
        expect(splitter(new TextEncoder().encode('0123456789'))).toBe('overflow');
    });
});
