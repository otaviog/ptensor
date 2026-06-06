import { useEffect, useRef, useState } from 'react';
import type { ImagePlane } from '../resolveView';
import { imageMapping, planeToRgba } from '../imageData';
import type { TensorStats } from '../stats';
import type { TensorView } from '../types';

interface Props {
    tensor: TensorView;
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
    tensor: TensorView;
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
            tensor.array,
            index * planeElements,
            plane,
            imageMapping(tensor.dtype, stats)
        );
        const img = ctx.createImageData(plane.width, plane.height);
        img.data.set(rgba);
        ctx.putImageData(img, 0, 0);
    }, [tensor, plane, index, stats]);

    return <canvas className="ptv-canvas" width={plane.width} height={plane.height} ref={ref} />;
}
