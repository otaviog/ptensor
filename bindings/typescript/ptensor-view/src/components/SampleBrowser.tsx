import { useState } from 'react';
import type { TensorView } from '../types';
import { TensorViewer } from './TensorViewer';

/** Sidebar of tensors + the viewer for the selected one. Shared by the dev
 * playground and the extension's in-editor demo command. */
export function SampleBrowser({
    samples,
    tableThreshold,
}: {
    samples: TensorView[];
    tableThreshold?: number;
}) {
    const [index, setIndex] = useState(0);
    return (
        <div className="ptv-browser">
            <nav className="ptv-browser-nav">
                <h3>samples</h3>
                {samples.map((s, i) => (
                    <button
                        key={s.name ?? i}
                        type="button"
                        className={i === index ? 'active' : ''}
                        onClick={() => setIndex(i)}
                    >
                        {s.name} [{s.shape.map(Number).join(', ')}]
                    </button>
                ))}
            </nav>
            <main className="ptv-browser-main">
                <TensorViewer tensor={samples[index]} tableThreshold={tableThreshold} />
            </main>
        </div>
    );
}
