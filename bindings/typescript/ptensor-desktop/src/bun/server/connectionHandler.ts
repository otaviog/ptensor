// One handler per connected producer. It holds the connection's session id --
// sent once, before the tensors -- and hands finished payloads to the sink.

import { getAppLogger } from "../../shared/logging";
import type { TensorPayload } from "../../shared/rpc";
import { DEFAULT_SESSION } from "./constants";
import { IncomingMessage } from "./protocol";

// export type MessageHandler = (msg: IncomingMessage) => void;
export type ConnectionInfo = {
  clientAddress: string
}

/** Where accepted tensors go, usually a push to the webview. */
export type TensorSink = (payload: TensorPayload) => void;

export interface MessageHandler {
  onMessage: (msg: IncomingMessage) => void;
  onConnectionClose: () => void;
}


export function handleMessage(connectionInfo: ConnectionInfo, onTensor: TensorSink): MessageHandler {
  let sessionId: string | undefined;
  const logger = getAppLogger("client-handler");

  return {
    onMessage: (msg: IncomingMessage): void => {
      switch (msg.kind) {
        case 'session-message':
          if (sessionId === undefined) {
            sessionId = msg.sessionId;
            logger.info('Client ({connection}) joined session {sessionId}.', {
              connection: connectionInfo.clientAddress,
              sessionId
            });
          } else if (sessionId !== msg.sessionId) {
            // The id is fixed for the life of a connection: a producer that wants
            // another session opens another connection.
            logger.warn('Client ({connection}) attempted to change session ID from {oldSessionId} to {newSessionId}', {
              oldSessionId: sessionId,
              newSessionId: msg.sessionId,
              connection: connectionInfo.clientAddress
            });
          }
          break;
        case 'tensor-message':
          onTensor({
            sessionId: sessionId ?? DEFAULT_SESSION,
            name: msg.name,
            tensor: msg.tensor,
            receivedAt: Date.now()
          });
          break;
      }
    },
    onConnectionClose: (): void => {
      logger.info('Client ({connection}) closed connection for session {sessionId}.', {
        connection: connectionInfo.clientAddress, sessionId
      })
    }
  }
}
