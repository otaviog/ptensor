// The node:net socket that drives this package's framing and protocol.
// Imported as `ptensor-tlog/node`, so a Bun consumer never loads it.
//
// Unlike the Bun server, this one binds an ephemeral port: the VS Code
// extension listens on whatever the kernel gives it and hands the address to
// the debugger, which calls `p10::tlog::log_to(<address>, ...)` in the stopped
// frame.

import * as net from 'net';
import { handleMessage, type TensorSink } from '../connectionHandler';
import { lineSplitter } from '../lineSplitter';
import { parseIncoming } from '../protocol';

/** Largest single JSON line accepted, in bytes. */
const MAX_LINE_BYTES = 256 * 1024 * 1024;

export interface TensorFeedOptions {
    /** Called once per accepted tensor, in arrival order. */
    onTensor: TensorSink;
    /** Where the feed reports what happened; nothing is logged without one. */
    log?: (message: string) => void;
}

/**
 * Owns the listening socket. One feed per extension host, started on the first
 * visualize command and closed on deactivate.
 */
export class TensorFeed {
    private server: net.Server | undefined;
    private port: number | undefined;

    constructor(private readonly options: TensorFeedOptions) {}

    /** Starts listening if it is not already, and returns the "host:port". */
    async address(): Promise<string> {
        if (this.port !== undefined) {
            return `127.0.0.1:${this.port}`;
        }
        const server = net.createServer((socket) => this.onConnection(socket));
        this.server = server;

        const port = await new Promise<number>((resolve, reject) => {
            server.once('error', reject);
            // Port 0: the kernel picks a free one, which is why the debuggee
            // has to be told the address rather than reading it from the env.
            server.listen(0, '127.0.0.1', () => {
                const address = server.address();
                if (address === null || typeof address === 'string') {
                    reject(new Error('the tensor feed did not bind a TCP port'));
                    return;
                }
                resolve(address.port);
            });
        });
        server.on('error', (err) => this.options.log?.(`feed error: ${err.message}`));

        this.port = port;
        this.options.log?.(`Tensor feed listening on 127.0.0.1:${port}.`);
        return `127.0.0.1:${port}`;
    }

    dispose(): void {
        this.server?.close();
        this.server = undefined;
        this.port = undefined;
    }

    private onConnection(socket: net.Socket): void {
        const clientAddress = socket.remoteAddress ?? 'unknown';
        const split = lineSplitter(MAX_LINE_BYTES);
        const handler = handleMessage({ clientAddress }, this.options.onTensor, {
            info: (message) => this.options.log?.(message),
            warn: (message) => this.options.log?.(message),
        });

        socket.on('data', (chunk: Buffer) => {
            const lines = split(new Uint8Array(chunk));
            if (lines === 'overflow') {
                this.options.log?.(
                    `Line over ${MAX_LINE_BYTES} bytes from ${clientAddress}, closing connection.`
                );
                socket.end();
                return;
            }
            for (const line of lines) {
                this.applyLine(line, handler.onMessage);
            }
        });
        socket.on('error', (err) => this.options.log?.(`connection error: ${err.message}`));
        socket.on('close', () => handler.onConnectionClose());
    }

    /** Decodes and applies one line. A bad line is logged and skipped. */
    private applyLine(line: string, onMessage: (msg: ReturnType<typeof parseIncoming>) => void) {
        let decoded: unknown;
        try {
            decoded = JSON.parse(line);
        } catch (err) {
            this.options.log?.(
                `Malformed line: ${err instanceof Error ? err.message : String(err)}`
            );
            return;
        }
        try {
            onMessage(parseIncoming(decoded));
        } catch (err) {
            this.options.log?.(
                `Invalid message: ${err instanceof Error ? err.message : String(err)}`
            );
        }
    }
}
