type SplitterOutput = string[] | 'overflow';
export type LineSplitter =  (chunk: Uint8Array) => SplitterOutput;

export function lineSplitter(maxLineSizeBytes: number) : LineSplitter {
  const decoder = new TextDecoder('utf-8')
  let buffer = '';
  
  return (chunk: Uint8Array): SplitterOutput  => {
    buffer += decoder.decode(chunk, {stream: true});
    const lines = [];
    let start = 0;
    
    for (;;) {
      const nl = buffer.indexOf('\n', start);
      if (nl === -1) {
        break;
      }
      const line = buffer.slice(start, nl).trim();
      if (line.length > 0) {
        lines.push(line);
      }
      start = nl + 1;
    }
    buffer = buffer.slice(start);
    if (buffer.length > maxLineSizeBytes) {
      return 'overflow';
    }
    return lines;
  }
}
