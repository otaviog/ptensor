// Test preload: gives `bun test` a DOM so the React panel can render
// headlessly, and tells React that `act()` is legitimate here.
//
// happy-dom also installs its own `fetch` and `WebSocket`, emulations meant for
// a simulated browser, which cannot talk to a real server -- a 204 answer comes
// back as `HPE_UNEXPECTED_CONTENT_LENGTH`. The feed tests do talk to a real
// server, and `bun test` shares one process across files, so the real ones are
// put back after registering.
import { GlobalRegistrator } from '@happy-dom/global-registrator';

const realFetch = globalThis.fetch;
const realWebSocket = globalThis.WebSocket;
const realRequest = globalThis.Request;
const realResponse = globalThis.Response;
const realHeaders = globalThis.Headers;

GlobalRegistrator.register();

globalThis.fetch = realFetch;
globalThis.WebSocket = realWebSocket;
globalThis.Request = realRequest;
globalThis.Response = realResponse;
globalThis.Headers = realHeaders;

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
