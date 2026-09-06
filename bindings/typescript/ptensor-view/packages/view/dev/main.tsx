// Dev playground entry. `vite dev` serves index.html -> this file with HMR, so
// the React panel can be developed against the SAMPLES fixtures with no
// debugger or extension host. Uses the same SampleBrowser the in-editor demo
// command renders.

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { SampleBrowser } from '../src/components/SampleBrowser';
import { SAMPLES } from '../src/samples';
import '../src/styles.css';
import './playground.css';

createRoot(document.getElementById('root') as HTMLElement).render(
    <StrictMode>
        <SampleBrowser samples={SAMPLES} />
    </StrictMode>
);
