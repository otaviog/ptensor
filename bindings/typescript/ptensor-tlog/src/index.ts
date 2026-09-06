// Server side of the p10::tlog wire protocol: producers connect over TCP and
// write newline-delimited JSON, a session line naming the session followed by
// the tensors. Everything here is transport-free -- feed `lineSplitter` the
// bytes your socket gives you, whatever socket that is.

export { DEFAULT_PORT, DEFAULT_SESSION } from './constants';
export { lineSplitter, type LineSplitter } from './lineSplitter';
export {
  type IncomingMessage,
  parseIncoming,
  type SessionMessage,
  type TensorMessage,
  TlogProtocolError,
} from './protocol';
export {
  type ConnectionInfo,
  type ConnectionLogger,
  handleMessage,
  type MessageHandler,
  type TensorPayload,
  type TensorSink,
} from './connectionHandler';
