import { formatNumber } from '../format';
import { elementAt, type TensorView } from '../types';

/** Renders the tensor as a 2D grid using the last dim as columns. */
export function TableView({ tensor }: { tensor: TensorView }) {
    const shape = tensor.shape.map(Number);
    const data = tensor.array;
    const n = data.length;

    if (shape.length === 0 || n === 1) {
        return (
            <table className="ptv-table">
                <tbody>
                    <tr>
                        <td>{formatNumber(elementAt(data, 0))}</td>
                    </tr>
                </tbody>
            </table>
        );
    }

    const cols = shape.length === 1 ? n : shape[shape.length - 1];
    const rows = n / cols;
    return (
        <table className="ptv-table">
            <tbody>
                {Array.from({ length: rows }, (_, r) => (
                    <tr key={r}>
                        {Array.from({ length: cols }, (_, c) => (
                            <td key={c}>{formatNumber(elementAt(data, r * cols + c))}</td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

/** Truncated single-row preview for large, non-image tensors. */
export function LargeTablePreview({ tensor }: { tensor: TensorView }) {
    const data = tensor.array;
    const count = Math.min(256, data.length);
    return (
        <>
            <p className="ptv-warn">
                Tensor too large for a full table and shape is not image-like — showing first {count}{' '}
                elements.
            </p>
            <table className="ptv-table">
                <tbody>
                    <tr>
                        {Array.from({ length: count }, (_, i) => (
                            <td key={i}>{formatNumber(elementAt(data, i))}</td>
                        ))}
                    </tr>
                </tbody>
            </table>
        </>
    );
}
