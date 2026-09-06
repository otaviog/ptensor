// Webview entry for the shipped app: electrobun bundles this into one JS file,
// so the stylesheets are imported as text and injected by hand (ptensor-view's
// CSS lives in its own package). The RPC bridge and the mount live in ./mount,
// which the Vite dev entry shares.

import appCss from './app.css' with { type: 'text' };
import tensorViewCss from '@ptensor/tensor-view/styles.css' with { type: 'text' };
import { mount } from './mount';

function injectStyles(): void {
    const style = document.createElement('style');
    style.textContent = `${tensorViewCss}\n${appCss}`;
    document.head.appendChild(style);
}

injectStyles();
mount();
