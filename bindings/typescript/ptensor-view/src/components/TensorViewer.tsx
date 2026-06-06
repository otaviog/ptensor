import { useMemo } from 'react';
import { computeStats } from '../stats';
import { resolveView } from '../resolveView';
import type { TensorView } from '../types';
import { StatsBar } from './StatsBar';
import { ImageView } from './ImageView';
import { LargeTablePreview, TableView } from './TableView';

export interface TensorViewerProps {
    tensor: TensorView;
    /** Element count at or below which the tensor renders as a full table. */
    tableThreshold?: number;
}

/** Top-level panel: header + stats + the resolved table/image body. */
export function TensorViewer({ tensor, tableThreshold = 256 }: TensorViewerProps) {
    const shape = useMemo(() => tensor.shape.map(Number), [tensor.shape]);
    const stats = useMemo(() => computeStats(tensor.array), [tensor.array]);
    const view = useMemo(() => resolveView(shape, tableThreshold), [shape, tableThreshold]);

    return (
        <div className="ptv-root">
            <h2 className="ptv-title">{tensor.name ?? 'tensor'}</h2>
            <div className="ptv-meta">
                shape=[{shape.join(', ')}] dtype={tensor.dtype} elems={stats.count}
            </div>
            <StatsBar stats={stats} />
            {view.mode === 'table' && <TableView tensor={tensor} />}
            {view.mode === 'image' && view.image && (
                <ImageView
                    tensor={tensor}
                    plane={view.image}
                    batch={view.batch}
                    stats={stats}
                />
            )}
            {view.mode === 'large-table' && <LargeTablePreview tensor={tensor} />}
        </div>
    );
}
