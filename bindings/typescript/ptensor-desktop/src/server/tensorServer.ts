// Local TCP feed. Producers connect to 127.0.0.1 and write newline-delimited
// JSON; each accepted line either stores a tensor under its session id or
// clears a session. Nothing is executed from the wire, and a line that exceeds
// the size cap drops the connection instead of growing the buffer forever.

import { getAppLogger } from '../shared/logging';
import { lineSplitter, LineSplitter } from './lineSplitter';
import { DEFAULT_PORT } from './constants';
import { ServerInfo } from '../shared/rpc';
import { IncomingMessage, parseIncoming } from './protocol';
import { MessageHandler, handleMessage } from './connectionHandler';

export interface TensorServerOptions {
  port?: number;
  hostname?: string;
  /** Largest single JSON line accepted, in bytes. Default 256 MiB. */
  maxLineBytes?: number;
  /** Called after a batch of lines has been applied to the store. */
  onChange: () => void;
}

interface ConnectionState {
  splitter: LineSplitter;
  handler: MessageHandler;
}

/** Owns the listening socket and applies incoming messages to the store. */
export class TensorServer {
  private readonly log = getAppLogger('tensor-feed');
  private server: Bun.TCPSocketListener<ConnectionState> | null = null;
  private info: ServerInfo;
  private readonly maxLineBytes: number;

  constructor(
    private readonly options: TensorServerOptions
  ) {
    this.maxLineBytes = options.maxLineBytes ?? 256 * 1024 * 1024;
    this.info = {
      host: options.hostname ?? '127.0.0.1',
      port: options.port ?? DEFAULT_PORT,
      listening: false,
    };
  }

  /** Starts listening. A bind failure is reported through `serverInfo()`. */
  start(): ServerInfo {
    try {
      this.server = Bun.listen<ConnectionState>({
        hostname: this.info.host,
        port: this.info.port,
        socket: {
          open: (socket) => {
            socket.data = {
              splitter: lineSplitter(this.maxLineBytes),
              handler: handleMessage({ clientAddress: socket.remoteAddress }),
            };
          },
          data: (socket, chunk) => {
            const lines = socket.data.splitter(chunk);
            if (lines === 'overflow') {
              this.log.error(
                'Line over {maxLineBytes} bytes, closing connection.',
                { maxLineBytes: this.maxLineBytes }
              );
              socket.end();
              return;
            }
            if (lines.length === 0) {
              return;
            }
            for (const line of lines) {
              this.apply(line);
            }
            this.options.onChange();
          },
          error: (_socket, error) => {
            this.log.error('Socket error: {message}', { message: error.message });
          },
        },
      });
      this.info = { ...this.info, listening: true, error: undefined };
      this.log.info('Listening on {host}:{port}.', {
        host: this.info.host,
        port: this.info.port,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.info = { ...this.info, listening: false, error: message };
      this.log.error('Failed to listen: {message}', { message });
    }
    return this.info;
  }

  stop(): void {
    this.server?.stop(true);
    this.server = null;
    this.info = { ...this.info, listening: false };
  }

  serverInfo(): ServerInfo {
    return this.info;
  }

  /** Decodes and applies one line. Bad lines are logged and skipped. */
  private apply(line: string, handler: MessageHandler): void {
    let decoded: unknown;
    try {
      decoded = JSON.parse(line);
    } catch (error) {
      this.log.error('Bad formed line: {message}', {
        message: error instanceof Error ? error.message : String(error),
      });
      return;
    }

    try {
      handler(parseIncoming(decoded));
    } catch (error) {
      this.log.error('Invalid message: {decoded}', {
        decoded,
        error
      });
    }
  }
}
