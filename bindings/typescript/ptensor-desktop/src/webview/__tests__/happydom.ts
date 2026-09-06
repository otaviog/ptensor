// Test preload: gives `bun test` a DOM so the React panel can render
// headlessly, and tells React that `act()` is legitimate here.
import { GlobalRegistrator } from '@happy-dom/global-registrator';

GlobalRegistrator.register();
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
