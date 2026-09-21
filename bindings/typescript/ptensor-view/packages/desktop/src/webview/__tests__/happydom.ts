// Test preload: gives `bun test` a DOM so the React panel can render
// headlessly, and tells React that `act()` is legitimate here.
//
// happy-dom also installs its own `fetch`, `WebSocket` and `Worker`, emulations
// meant for a simulated browser: the first cannot talk to a real server (a 204
// answer comes back as `HPE_UNEXPECTED_CONTENT_LENGTH`) and the last does not
// run anything. The feed and decoder tests need the real ones, and `bun test`
// shares one process across files, so they are put back after registering.
import { GlobalRegistrator } from '@happy-dom/global-registrator';

const realFetch = globalThis.fetch;
const realWebSocket = globalThis.WebSocket;
const realRequest = globalThis.Request;
const realResponse = globalThis.Response;
const realHeaders = globalThis.Headers;
// The decode worker is started from a blob URL, which bun can actually run.
const realWorker = globalThis.Worker;
const realBlob = globalThis.Blob;
const realURL = globalThis.URL;

GlobalRegistrator.register();

globalThis.fetch = realFetch;
globalThis.WebSocket = realWebSocket;
globalThis.Request = realRequest;
globalThis.Response = realResponse;
globalThis.Headers = realHeaders;
globalThis.Worker = realWorker;
globalThis.Blob = realBlob;
globalThis.URL = realURL;

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
