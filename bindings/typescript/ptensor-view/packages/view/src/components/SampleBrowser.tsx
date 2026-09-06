import { useState } from 'react';
import type { Tensor } from 'ptensor-ts';
import { TensorViewer } from './TensorViewer';

/** Sidebar of tensors + the viewer for the selected one. Shared by the dev
 * playground and the extension's in-editor demo command. */
export function SampleBrowser({
    samples,
    tableThreshold,
}: {
    samples: Record<string, Tensor>;
    tableThreshold?: number;
}) {
    const entries = Object.entries(samples);
    const [index, setIndex] = useState(0);
    const [name, tensor] = entries[index] ?? ['', undefined];
    return (
        <div className="ptv-browser">
            <nav className="ptv-browser-nav">
                <h3>samples</h3>
                {entries.map(([sampleName, sample], i) => (
                    <button
                        key={sampleName}
                        type="button"
                        className={i === index ? 'active' : ''}
                        onClick={() => setIndex(i)}
                    >
                        {sampleName} [{sample.shape.join(', ')}]
                    </button>
                ))}
            </nav>
            <main className="ptv-browser-main">
                {tensor && (
                    <TensorViewer
                        tensor={tensor}
                        name={name}
                        tableThreshold={tableThreshold}
                    />
                )}
            </main>
        </div>
    );
}
