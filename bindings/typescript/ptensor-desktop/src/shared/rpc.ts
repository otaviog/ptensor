// RPC contract between the bun process (owns the socket feed and the store)
// and the mainview webview (renders the panel). Summaries are pushed; tensor
// payloads are pulled on selection.

import type { RPCSchema } from 'electrobun';
import type { TensorJson } from 'ptensor-ts';


/** A tensor pushed by a producer. `type` may be omitted; it is the default. */
export type TensorPayload = {
  /** Label shown in the tensor list; defaults to `tensor <n>`. */
  sessionId: string
  name: string;
  tensor: TensorJson;
  receivedAt: number;
}

/** Where the feed listens, and whether it came up. */
export interface ServerInfo {
  host: string;
  port: number;
  listening: boolean;
  error?: string;
}

export type ViewerRPC = {
  server: RPCSchema<{
    requests: {
      /** Server host/port and whether the feed is up. */
      getServerInfo: { params: {}; response: ServerInfo };
    };
    messages: {};
  }>;
  renderer: RPCSchema<{
    requests: {};
    messages: {
      newTensor: TensorPayload
    };
  }>;
};
