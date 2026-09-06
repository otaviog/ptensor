// RPC contract between the bun process (src/bun, owns the socket feed) and the
// webview (src/webview, renders the panel). Every accepted tensor is pushed as
// it arrives; the webview keeps whatever history it shows.
//
// The `bun` / `webview` key names are electrobun's, not ours: the two sides
// index this schema by those literals (see ElectrobunRPCSchema). Each side
// declares what it answers.

import type { RPCSchema } from 'electrobun';
import type { TensorJson } from 'ptensor-ts';


/** One tensor as it left the feed, on its way to the panel. */
export type TensorPayload = {
  /** Session the producing connection announced, or the default one. */
  sessionId: string
  /** Label shown in the tensor list. */
  name: string;
  tensor: TensorJson;
  receivedAt: number;
}

/** Where the feed listens, and whether it came up. Owned by the feed itself. */
export interface FeedInfo {
  host: string;
  port: number;
  listening: boolean;
  error?: string;
}

/**
 * What `getServerInfo` answers: the feed's state plus the settings the window
 * cannot read for itself, since they come from the main process' environment.
 */
export interface ServerInfo extends FeedInfo {
  /** Tensors the window keeps per session (PTENSOR_VIEW_HISTORY). */
  maxTensorsPerSession: number;
}

export type ViewerRPC = {
  bun: RPCSchema<{
    requests: {
      /** Server host/port and whether the feed is up. */
      getServerInfo: { params: {}; response: ServerInfo };
    };
    messages: {};
  }>;
  webview: RPCSchema<{
    requests: {};
    messages: {
      newTensor: TensorPayload
    };
  }>;
};
