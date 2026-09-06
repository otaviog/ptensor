// `tsc --noEmit` over the app sources. Electrobun ships its SDK as raw .ts
// (not .d.ts), so tsc also checks the SDK itself and reports pre-existing
// errors from it; those are filtered out and only our own are fatal.

const proc = Bun.spawn(['bunx', 'tsc', '--noEmit', '--pretty', 'false'], {
    stdout: 'pipe',
    stderr: 'pipe',
});
const [out, err] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
]);
await proc.exited;

const lines = `${out}${err}`.split('\n').filter((line) => line.trim().length > 0);
// Diagnostics start at a path; their continuation lines are indented.
const ours: string[] = [];
let keeping = false;
for (const line of lines) {
    if (/^\s/.test(line)) {
        if (keeping) {
            ours.push(line);
        }
        continue;
    }
    keeping = !line.startsWith('node_modules/');
    if (keeping) {
        ours.push(line);
    }
}

if (ours.length > 0) {
    console.error(ours.join('\n'));
    process.exit(1);
}
console.log('typecheck: no errors in app sources');
