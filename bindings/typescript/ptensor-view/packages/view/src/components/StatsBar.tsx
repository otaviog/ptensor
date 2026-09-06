import { formatNumber } from '../format';
import type { TensorStats } from '../stats';

export function StatsBar({ stats }: { stats: TensorStats }) {
    return (
        <div className="ptv-stats">
            <Stat label="min" value={formatNumber(stats.min)} />
            <Stat label="max" value={formatNumber(stats.max)} />
            <Stat label="mean" value={formatNumber(stats.mean)} />
            <Stat label="count" value={String(stats.count)} />
        </div>
    );
}

function Stat({ label, value }: { label: string; value: string }) {
    return (
        <div className="ptv-stat">
            <div className="ptv-stat-label">{label}</div>
            {value}
        </div>
    );
}
