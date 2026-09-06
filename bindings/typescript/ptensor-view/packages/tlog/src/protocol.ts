// One line of the tlog wire format, decoded and checked. Producers are external
// programs (the C++ `p10::tlog` client, the test drivers), so nothing that
// arrives here is trusted: a bad line is rejected with a reason, never assumed.

import { P10Error, type TensorJson, validateTensorJson } from 'ptensor-ts';

/** Opens a connection: every tensor after it belongs to this session. */
export type SessionMessage = {
  sessionId: string;
};

/** A tensor pushed by a producer. */
export type TensorMessage = {
  /** Label shown in the tensor list. */
  name: string;
  tensor: TensorJson;
};

export type IncomingMessage =
  | ({ kind: 'tensor-message' } & TensorMessage)
  | ({ kind: 'session-message' } & SessionMessage);

/** Thrown by `parseIncoming` for a line that is not a message it accepts. */
export class TlogProtocolError extends Error {
  constructor(message: string | P10Error, options?: ErrorOptions) {
    super(message instanceof P10Error ? `P10Error: ${message.message}` : message, options);
    this.name = 'TlogProtocolError';
  }
}

/**
 * Validates one decoded line. Throws `TlogProtocolError` naming what is wrong.
 * The two message kinds are told apart by their fields, not by a tag.
 */
export function parseIncoming(value: unknown): IncomingMessage {
  if (typeof value !== 'object' || value === null) {
    throw new TlogProtocolError('message is not a JSON object');
  }

  const valueDict = value as Record<string, unknown>;
  if (typeof valueDict.sessionId === 'string') {
    return { kind: 'session-message', sessionId: valueDict.sessionId };
  }

  if (typeof valueDict.name === 'string' && typeof valueDict.tensor === 'object') {
    try {
      return {
        kind: 'tensor-message',
        name: valueDict.name,
        tensor: validateTensorJson(valueDict.tensor),
      };
    } catch (error) {
      if (error instanceof P10Error) {
        throw new TlogProtocolError(error);
      }
      throw error;
    }
  }

  throw new TlogProtocolError('message is neither a session message nor a tensor message');
}
