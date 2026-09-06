// Runs the window against a Vite dev server: UI edits reach the open window
// through React Fast Refresh, and the bun process -- with the socket feed and
// every tensor it has taken -- keeps running underneath.
//
//   bun run dev/ui
//
// `bun run dev` stays the plain path: one bundle, no dev server.

const DEV_URL = 'http://localhost:5174';

function run(command: string[], env: Record<string, string> = {}) {
    return Bun.spawn(command, {
        stdio: ['inherit', 'inherit', 'inherit'],
        env: { ...process.env, ...env },
    });
}

/**
 * Resolves once the dev server answers. The window loads its URL once and does
 * not retry, so starting the app first would leave it on an error page.
 */
async function waitForViteServer(timeoutMs: number): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        try {
            await fetch(DEV_URL, { signal: AbortSignal.timeout(500) });
            return true;
        } catch {
            await Bun.sleep(200);
        }
    }
    return false;
}

// The app bundle needs its icons cut before electrobun packages it.
await run(['bun', 'scripts/makeIcons.ts']).exited;

const vite = run(['bun', 'x', 'vite']);
if (!(await waitForViteServer(20_000))) {
    console.error(`the vite dev server did not come up at ${DEV_URL}`);
    vite.kill();
    process.exit(1);
}

const app = run(['bun', 'x', 'electrobun', 'dev'], { PTENSOR_VIEW_DEV_URL: DEV_URL });

// Vite holds :5174 with strictPort, so leaving it behind breaks the next run.
function shutdown() {
    app.kill();
    vite.kill();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

await app.exited;
shutdown();
