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
    /** When set, a refresh button is shown that re-reads the tensor from its source. */
    onRefresh?: () => void;
}

/** Row-major (C-contiguous) layout: each stride is the product of the trailing dims. */
function isContiguous(shape: bigint[], stride: bigint[]): boolean {
    if (shape.length !== stride.length) {
        return false;
    }
    let expected = 1n;
    for (let i = shape.length - 1; i >= 0; i--) {
        // A length-0/1 dim's stride is irrelevant to the layout; skip it.
        if (shape[i] > 1n && stride[i] !== expected) {
            return false;
        }
        expected *= shape[i];
    }
    return true;
}

/** Top-level panel: header + stats + the resolved table/image body. */
export function TensorViewer({ tensor, tableThreshold = 256, onRefresh }: TensorViewerProps) {
    const shape = useMemo(() => tensor.shape.map(Number), [tensor.shape]);
    const stride = useMemo(() => tensor.stride.map(Number), [tensor.stride]);
    const contiguous = useMemo(
        () => isContiguous(tensor.shape, tensor.stride),
        [tensor.shape, tensor.stride]
    );
    const stats = useMemo(() => computeStats(tensor.array), [tensor.array]);
    const view = useMemo(() => resolveView(shape, tableThreshold), [shape, tableThreshold]);

    return (
        <div className="ptv-root">
            <div className="ptv-header">
                <h2 className="ptv-title">{tensor.name ?? 'tensor'}</h2>
                {onRefresh && (
                    <button type="button" className="ptv-refresh" onClick={onRefresh}>
                        ↻ Refresh
                    </button>
                )}
            </div>
            <div className="ptv-meta">
                shape=[{shape.join(', ')}] stride=[{stride.join(', ')}] elems={stats.count}{' '}
                <span className="ptv-badge ptv-badge-dtype">{tensor.dtype}</span>{' '}
                <span
                    className={`ptv-badge ${contiguous ? 'ptv-badge-ok' : 'ptv-badge-warn'}`}
                >
                    {contiguous ? 'contiguous' : 'non-contiguous'}
                </span>
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
