import { useEffect, useRef, useState } from 'react';
import type { ImagePlane } from '../resolveView';
import { imageMapping, planeToRgba } from '../imageData';
import type { TensorStats } from '../stats';
import type { Tensor } from 'ptensor-ts';

interface Props {
    tensor: Tensor;
    plane: ImagePlane;
    batch: number;
    stats: TensorStats;
}

/** Renders one or more image planes; batched tensors get a tab per image. */
export function ImageView({ tensor, plane, batch, stats }: Props) {
    const [active, setActive] = useState(0);
    return (
        <div>
            {batch > 1 && (
                <div className="ptv-tabs">
                    {Array.from({ length: batch }, (_, n) => (
                        <button
                            key={n}
                            type="button"
                            className={n === active ? 'active' : ''}
                            onClick={() => setActive(n)}
                        >
                            image {n}
                        </button>
                    ))}
                </div>
            )}
            <ImageCanvas tensor={tensor} plane={plane} index={active} stats={stats} />
        </div>
    );
}

function ImageCanvas({
    tensor,
    plane,
    index,
    stats,
}: {
    tensor: Tensor;
    plane: ImagePlane;
    index: number;
    stats: TensorStats;
}) {
    const ref = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = ref.current;
        if (!canvas) {
            return;
        }
        const ctx = canvas.getContext('2d');
        if (!ctx) {
            return;
        }
        const planeElements = plane.width * plane.height * plane.channels;
        const rgba = planeToRgba(
            tensor.data,
            index * planeElements,
            plane,
            imageMapping(tensor.dtype, stats)
        );
        // Wrapped, not copied into a `createImageData` buffer: the RGBA is
        // already the right shape, and the copy doubled it.
        ctx.putImageData(new ImageData(rgba, plane.width, plane.height), 0, 0);
    }, [tensor, plane, index, stats]);

    return <canvas className="ptv-canvas" width={plane.width} height={plane.height} ref={ref} />;
}
