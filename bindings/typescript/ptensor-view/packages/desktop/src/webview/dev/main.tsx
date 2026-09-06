// Webview entry for `bun run dev:ui`. Vite owns the stylesheets here, so
// editing app.css or ptensor-view's CSS updates the running window without a
// reload; the app itself comes from ../mount, exactly as in the shipped build.

import '@ptensor/tensor-view/styles.css';
import '../app.css';
import { mount } from '../mount';

mount();
