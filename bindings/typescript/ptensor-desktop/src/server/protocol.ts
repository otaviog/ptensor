import { TensorJson, P10Error } from "ptensor-ts";
import { PdError } from "../shared/pdError";
import { getAppLogger } from "../shared/logging";
import { validateTensorJson } from "../../../ptensor-ts/dist/esm/tensorJson";

export type SessionMessage = {
  sessionId: string;
}

/** A tensor pushed by a producer. `type` may be omitted; it is the default. */
export type TensorMessage = {
  /** Label shown in the tensor list; defaults to `tensor <n>`. */
  name: string;
  tensor: TensorJson;
}

export type IncomingMessage =
  { kind: 'tensor-message' } & TensorMessage | { kind: 'session-message' } & SessionMessage;

/**
 * Validates one decoded line. Returns the message, or an error string naming
 * what is wrong (producers are external, so nothing here is trusted).
 */
export function parseIncoming(value: unknown): IncomingMessage {
  if (typeof value !== 'object' || value === null) {
    throw new PdError('message is not a JSON object');
  }

  const valueDict = value as Record<string, unknown>;
  if (typeof valueDict.sessionId === 'string') {
    return {
      kind: 'session-message',
      sessionId: valueDict.sessionId
    }
  }

  if (typeof valueDict.name === 'string' && typeof valueDict.tensor === 'object') {
    try {
      return {
        kind: 'tensor-message',
        name: valueDict.name,
        tensor: validateTensorJson(valueDict.tensor)
      };
    } catch (error) {
      if (error instanceof P10Error) {
        getAppLogger("protocol").error('Invalid tensor JSON: {message}', { message: error.message });
        throw new PdError(error);
      }
      throw error;
    }
  }

  throw new PdError('message is neither a session message nor a tensor message');
}

