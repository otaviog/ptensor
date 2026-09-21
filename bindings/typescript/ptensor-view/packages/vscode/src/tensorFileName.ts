// Turning a tensor's label into a filename.
//
// Its own module, with no `vscode` import, so it can be reasoned about and
// tested on its own: the label comes from the debuggee -- an expression the user
// typed, or whatever a producer called the tensor -- and it ends up as a path.

/** Longest readable part of a name kept; the hash follows it. */
const READABLE_LIMIT = 64;

/**
 * A filename that stays inside its directory whatever the tensor was called.
 *
 * Everything outside `[A-Za-z0-9._-]` becomes `-`, so no separator survives;
 * leading dots and dashes go, so nothing becomes `..` or looks like a flag. The
 * FNV-1a suffix is what makes it unique: two labels that differ only in the
 * characters just dropped must not collide on the same file.
 */
export function fileNameFor(key: string): string {
    const readable = key
        .replace(/[^A-Za-z0-9._-]+/g, '-')
        .replace(/^[-.]+/, '')
        .slice(0, READABLE_LIMIT);
    let hash = 0x811c9dc5;
    for (let i = 0; i < key.length; i++) {
        hash = ((hash ^ key.charCodeAt(i)) * 0x01000193) >>> 0;
    }
    const suffix = hash.toString(16).padStart(8, '0');
    return readable.length > 0 ? `${readable}-${suffix}` : `tensor-${suffix}`;
}
