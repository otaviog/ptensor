type SplitterOutput = string[] | 'overflow';
export type LineSplitter =  (chunk: Uint8Array) => SplitterOutput;

const NEWLINE = 0x0a;

/**
 * Splits a byte stream into newline delimited lines.
 *
 * The newline is looked for in the raw chunk, and a line is decoded once, when
 * it is complete. Decoding every chunk into a string and re-scanning the whole
 * accumulated string instead makes the cost quadratic in the number of chunks,
 * which a multi-megabyte tensor line arrives in.
 */
export function lineSplitter(maxLineSizeBytes: number) : LineSplitter {
  const decoder = new TextDecoder('utf-8')
  let pending: Uint8Array[] = [];
  let pendingBytes = 0;

  // Decodes the chunks held so far plus `last`, which completes the line.
  const takeLine = (last: Uint8Array): string => {
    if (pending.length === 0) {
      return decoder.decode(last);
    }
    const joined = new Uint8Array(pendingBytes + last.length);
    let at = 0;
    for (const part of pending) {
      joined.set(part, at);
      at += part.length;
    }
    joined.set(last, at);
    pending = [];
    pendingBytes = 0;
    return decoder.decode(joined);
  };

  return (chunk: Uint8Array): SplitterOutput  => {
    const lines = [];
    let start = 0;

    for (;;) {
      const nl = chunk.indexOf(NEWLINE, start);
      if (nl === -1) {
        break;
      }
      const line = takeLine(chunk.subarray(start, nl)).trim();
      if (line.length > 0) {
        lines.push(line);
      }
      start = nl + 1;
    }

    if (start < chunk.length) {
      // Copied, not referenced: the caller owns `chunk` and may reuse it.
      const tail = chunk.slice(start);
      pending.push(tail);
      pendingBytes += tail.length;
      if (pendingBytes > maxLineSizeBytes) {
        return 'overflow';
      }
    }
    return lines;
  }
}
