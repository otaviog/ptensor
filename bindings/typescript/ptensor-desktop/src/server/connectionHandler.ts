import { IncomingMessage } from "./protocol";
import { getAppLogger } from "../shared/logging";

export type MessageHandler = (msg: IncomingMessage) => void;
export type ConnectionInfo = {
  clientAddress: string
}

export function handleMessage(connectionInfo: ConnectionInfo): MessageHandler {
  let sessionId: string | undefined;
  const logger = getAppLogger("client-handler");
  
  return (msg: IncomingMessage): void => {
    switch (msg.kind) {
      case 'session-message':
        if (sessionId !== undefined) {
          sessionId = msg.sessionId;
        } else {
          logger.warn('Client ({connection}) attempted to change session ID from {oldSessionId} to {newSessionId}', {
            oldSessionId: sessionId,
            newSessionId: msg.sessionId,
            connection: connectionInfo.clientAddress
          });
        }
        break;
      case 'tensor-message':
        
        break;
    }
  }
  
}
}
