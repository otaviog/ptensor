// Local TCP feed: the Bun socket that drives ptensor-tlog's framing and
// protocol. Producers connect to 127.0.0.1 and write newline-delimited JSON, a
// session line naming the session then the tensors; each connection keeps its
// own handler and accepted tensors leave through `onTensor`. Nothing is
// executed from the wire, and a line that exceeds the size cap drops the
// connection instead of growing the buffer forever.

import {
  DEFAULT_PORT,
  handleMessage,
  lineSplitter,
  type LineSplitter,
  type MessageHandler,
  parseIncoming,
  type TensorSink,
} from 'ptensor-tlog';
import { getAppLogger } from '../../shared/logging';
import { FeedInfo } from '../../shared/rpc';

export interface TensorServerOptions {
  port?: number;
  hostname?: string;
  /** Largest single JSON line accepted, in bytes. Default 256 MiB. */
  maxLineBytes?: number;
  /** Called once per accepted tensor, in arrival order. */
  onTensor: TensorSink;
}

interface ConnectionState {
  splitter: LineSplitter;
  handler: MessageHandler;
}

type Socket = Bun.Socket<ConnectionState>;

/** Owns the listening socket and applies incoming messages to the store. */
export class TensorServer {
  private readonly log = getAppLogger('tensor-feed');
  private server: Bun.TCPSocketListener<ConnectionState> | null = null;
  private info: FeedInfo;
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
  start(): FeedInfo {
    try {
      this.server = Bun.listen<ConnectionState>({
        hostname: this.info.host,
        port: this.info.port,
        socket: {
          open: (socket) => this.onConnectionOpen(socket),
          data: (socket, chunk) => this.onData(socket, chunk),
          error: (_socket, error) => this.log.error('Socket error: {message}', { message: error.message }),
          close: (socket) => socket.data.handler.onConnectionClose()
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

  serverInfo(): FeedInfo {
    return this.info;
  }

  private onConnectionOpen(socket: Socket) {
    socket.data = {
      splitter: lineSplitter(this.maxLineBytes),
      handler: handleMessage({ clientAddress: socket.remoteAddress }, this.options.onTensor, {
        info: (message) => this.log.info(message),
        warn: (message) => this.log.warn(message),
      }),
    };
  }
  private onData(socket: Socket, chunk: Buffer<ArrayBufferLike>) {
    const lines = socket.data.splitter(chunk);
    this.log.info(`Received ${chunk.byteLength} bytes from ${socket.remoteAddress}`);

    if (lines === 'overflow') {
      this.log.error(
        'Line over {maxLineBytes} bytes, closing connection.',
        { maxLineBytes: this.maxLineBytes }
      );
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
      this.log.info(`Decoding line with size ${line.length}`);
      decoded = JSON.parse(line);
    } catch (error) {
      this.log.error('Bad formed line: {message}', {
        message: error instanceof Error ? error.message : String(error),
      });
      return;
    }

    try {
      handler.onMessage(parseIncoming(decoded));
    } catch (error) {
      this.log.error('Invalid message: {decoded}', {
        decoded,
        error
      });
    }
  }
}
