import { useMemo } from 'react';
import { computeStats } from '../stats';
import { resolveView } from '../resolveView';
import type { Tensor } from 'ptensor-ts';
import { StatsBar } from './StatsBar';
import { ImageView } from './ImageView';
import { LargeTablePreview, TableView } from './TableView';

export interface TensorViewerProps {
    tensor: Tensor;
    /** Label for the panel header (e.g. the debugger expression, a log entry). */
    name?: string;
    /** Element count at or below which the tensor renders as a full table. */
    tableThreshold?: number;
    /** When set, a refresh button is shown that re-reads the tensor from its source. */
    onRefresh?: () => void;
}

/** Row-major (C-contiguous) layout: each stride is the product of the trailing dims. */
function isContiguous(shape: number[], stride: number[]): boolean {
    if (shape.length !== stride.length) {
        return false;
    }
    let expected = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
        // A length-0/1 dim's stride is irrelevant to the layout; skip it.
        if (shape[i] > 1 && stride[i] !== expected) {
            return false;
        }
        expected *= shape[i];
    }
    return true;
}

/** Top-level panel: header + stats + the resolved table/image body. */
export function TensorViewer({
    tensor,
    name,
    tableThreshold = 256,
    onRefresh,
}: TensorViewerProps) {
    const { shape, stride } = tensor;
    const contiguous = useMemo(() => isContiguous(shape, stride), [shape, stride]);
    const stats = useMemo(() => computeStats(tensor.data), [tensor.data]);
    const view = useMemo(() => resolveView(shape, tableThreshold), [shape, tableThreshold]);

    return (
        <div className="ptv-root">
            <div className="ptv-header">
                <h2 className="ptv-title">{name ?? 'tensor'}</h2>
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
