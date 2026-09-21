// The p10::tlog wire protocol: producers connect over TCP and write
// newline-delimited JSON, a session line naming the session followed by the
// tensors.
//
// This entry is what a *host* needs: where to listen, what comes out, and the
// shape of a line for anything that writes one. The machinery that turns bytes
// into payloads -- the line splitter, the parser, the per-connection handler --
// is what `./bun` and `./node` are built from, and stays inside.

export { DEFAULT_PORT } from './constants';
export type { FeedInfo } from './feedInfo';
export type { SessionMessage, TensorMessage } from './protocol';
export type { TensorPayload, TensorSink } from './connectionHandler';

// What a host keeps between the feed and its viewer: the tensors, as the JSON
// they arrived as, under a byte budget.
export { DEFAULT_BUDGET_BYTES, TensorStore } from './store';
export type { AddResult, StoreLogger, TensorMeta, TensorStoreOptions } from './store';
