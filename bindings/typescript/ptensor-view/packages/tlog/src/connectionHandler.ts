// One handler per connected producer. It holds the connection's session id --
// sent once, before the tensors -- and hands finished payloads to the sink.
//
// Transport-free: the caller feeds it decoded messages and decides what a
// payload is for (a window, a webview, a file).

import type { TensorJson } from 'ptensor-ts';
import { DEFAULT_SESSION } from './constants';
import type { IncomingMessage } from './protocol';

/** What the server knows about the peer, for the log lines. */
export type ConnectionInfo = {
  clientAddress: string;
};

/** One tensor as it left the feed, on its way to whatever shows it. */
export type TensorPayload = {
  /** Session the producing connection announced, or the default one. */
  sessionId: string;
  /** Label shown in the tensor list. */
  name: string;
  tensor: TensorJson;
  receivedAt: number;
};

/** Where accepted tensors go. */
export type TensorSink = (payload: TensorPayload) => void;

/**
 * Where the handler reports what happened. Every method is optional: a host
 * with no logger of its own passes nothing and the events are dropped.
 */
export interface ConnectionLogger {
  info?(message: string): void;
  warn?(message: string): void;
}

export interface MessageHandler {
  onMessage: (msg: IncomingMessage) => void;
  onConnectionClose: () => void;
}

export function handleMessage(
  connectionInfo: ConnectionInfo,
  onTensor: TensorSink,
  logger: ConnectionLogger = {}
): MessageHandler {
  let sessionId: string | undefined;

  return {
    onMessage: (msg: IncomingMessage): void => {
      switch (msg.kind) {
        case 'session-message':
          if (sessionId === undefined) {
            sessionId = msg.sessionId;
            logger.info?.(
              `Client (${connectionInfo.clientAddress}) joined session ${sessionId}.`
            );
          } else if (sessionId !== msg.sessionId) {
            // The id is fixed for the life of a connection: a producer that
            // wants another session opens another connection.
            logger.warn?.(
              `Client (${connectionInfo.clientAddress}) attempted to change session ID ` +
                `from ${sessionId} to ${msg.sessionId}`
            );
          }
          break;
        case 'tensor-message':
          onTensor({
            sessionId: sessionId ?? DEFAULT_SESSION,
            name: msg.name,
            tensor: msg.tensor,
            receivedAt: Date.now(),
          });
          break;
      }
    },
    onConnectionClose: (): void => {
      logger.info?.(
        `Client (${connectionInfo.clientAddress}) closed connection for session ${sessionId}.`
      );
    },
  };
}
