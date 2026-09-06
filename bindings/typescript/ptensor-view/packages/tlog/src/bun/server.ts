// The Bun socket that drives this package's framing and protocol. Imported as
// `ptensor-tlog/bun`, so a Node consumer never loads it.
//
// Producers connect to 127.0.0.1 and write newline-delimited JSON: a session
// line naming the session, then the tensors. Each connection keeps its own
// handler and accepted tensors leave through `onTensor`. Nothing is executed
// from the wire, and a line over the size cap drops the connection instead of
// growing the buffer forever.

import { handleMessage, type MessageHandler, type TensorSink } from '../connectionHandler';
import { DEFAULT_PORT } from '../constants';
import type { FeedInfo } from '../feedInfo';
import { lineSplitter, type LineSplitter } from '../lineSplitter';
import { parseIncoming } from '../protocol';

/** Where the server reports what happened; the host routes it to its own log. */
export interface ServerLogger {
  info?(message: string): void;
  warn?(message: string): void;
  error?(message: string): void;
}

export interface TensorServerOptions {
  port?: number;
  hostname?: string;
  /** Largest single JSON line accepted, in bytes. Default 256 MiB. */
  maxLineBytes?: number;
  /** Called once per accepted tensor, in arrival order. */
  onTensor: TensorSink;
  /** Optional; nothing is logged without one. */
  logger?: ServerLogger;
}

interface ConnectionState {
  splitter: LineSplitter;
  handler: MessageHandler;
}

type Socket = Bun.Socket<ConnectionState>;

/** Owns the listening socket and applies incoming messages to the store. */
export class TensorServer {
  private readonly log: ServerLogger;
  private server: Bun.TCPSocketListener<ConnectionState> | null = null;
  private info: FeedInfo;
  private readonly maxLineBytes: number;

  constructor(
    private readonly options: TensorServerOptions
  ) {
    this.log = options.logger ?? {};
    this.maxLineBytes = options.maxLineBytes ?? 256 * 1024 * 1024;
    this.info = {
      host: options.hostname ?? '127.0.0.1',
      port: options.port ?? DEFAULT_PORT,
      listening: false,
    };
  }

  /** Starts listening. A bind failure is reported through `serverInfo()`. */
  start(): FeedInfo {
    try {
      this.server = Bun.listen<ConnectionState>({
        hostname: this.info.host,
        port: this.info.port,
        socket: {
          open: (socket) => this.onConnectionOpen(socket),
          data: (socket, chunk) => this.onData(socket, chunk),
          error: (_socket, error) => this.log.error?.(`Socket error: ${error.message}`),
          close: (socket) => socket.data.handler.onConnectionClose()
        },
      });
      this.info = { ...this.info, listening: true, error: undefined };
      this.log.info?.(`Listening on ${this.info.host}:${this.info.port}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.info = { ...this.info, listening: false, error: message };
      this.log.error?.(`Failed to listen: ${message}`);
    }
    return this.info;
  }

  stop(): void {
    this.server?.stop(true);
    this.server = null;
    this.info = { ...this.info, listening: false };
  }

  serverInfo(): FeedInfo {
    return this.info;
  }

  private onConnectionOpen(socket: Socket) {
    socket.data = {
      splitter: lineSplitter(this.maxLineBytes),
      handler: handleMessage({ clientAddress: socket.remoteAddress }, this.options.onTensor, this.log),
    };
  }
  private onData(socket: Socket, chunk: Buffer<ArrayBufferLike>) {
    const lines = socket.data.splitter(chunk);

    if (lines === 'overflow') {
      this.log.error?.(`Line over ${this.maxLineBytes} bytes, closing connection.`);
      socket.end();
      return;
    }
    for (const line of lines) {
      this.decodeLine(line, socket.data.handler);
    }
  }

  /** Decodes and applies one line. Bad lines are logged and skipped. */
  private decodeLine(line: string, handler: MessageHandler): void {
    let decoded: unknown;
    try {
      decoded = JSON.parse(line);
    } catch (error) {
      this.log.error?.(
        `Malformed line: ${error instanceof Error ? error.message : String(error)}`
      );
      return;
    }

    try {
      handler.onMessage(parseIncoming(decoded));
    } catch (error) {
      this.log.error?.(
        `Invalid message: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }
}
