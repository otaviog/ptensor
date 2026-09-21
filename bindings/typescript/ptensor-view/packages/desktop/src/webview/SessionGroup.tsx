import type { TensorMeta } from '../shared/feed';
import { shownTensor, type NameGroup, type Positions, type Selection, type Session } from './types';
import { positionKey } from './types';

export function SessionGroup({
    session,
    selected,
    positions,
    onSelect,
    onScrub,
    onClear,
}: {
    session: Session;
    selected: Selection | null;
    positions: Positions;
    onSelect: (sessionId: string, name: string) => void;
    onScrub: (sessionId: string, name: string, index: number) => void;
    onClear: () => void;
}) {
    return (
        <section className="session">
            <header className="session-head">
                <span className="session-id" title={session.id}>
                    {session.id}
                </span>
                <button type="button" className="session-clear" onClick={onClear}>
                    ✕
                </button>
            </header>
            <ul className="tensor-list">
                {session.groups.map((group) => (
                    <li key={group.name}>
                        <NameRow
                            group={group}
                            active={
                                selected?.sessionId === session.id && selected.name === group.name
                            }
                            pinned={positions[positionKey(session.id, group.name)]}
                            onSelect={() => onSelect(session.id, group.name)}
                            onScrub={(index) => onScrub(session.id, group.name, index)}
                        />
                    </li>
                ))}
            </ul>
        </section>
    );
}

/**
 * One name, and a slider across the times it arrived. The slider is only there
 * once there is something to move between.
 */
function NameRow({
    group,
    active,
    pinned,
    onSelect,
    onScrub,
}: {
    group: NameGroup;
    active: boolean;
    pinned: string | undefined;
    onSelect: () => void;
    onScrub: (index: number) => void;
}) {
    const shown = shownTensor(group, pinned);
    const index = group.tensors.indexOf(shown);
    const count = group.tensors.length;

    return (
        <div className={`name-group${active ? ' active' : ''}`}>
            <button type="button" className="tensor-item" onClick={onSelect}>
                <span className="tensor-name">{group.name}</span>
                <span className="tensor-meta">
                    {shown.dtype} [{shown.shape.join('×')}]
                </span>
            </button>
            {count > 1 && (
                <div className="tensor-scrub">
                    <input
                        type="range"
                        className="tensor-slider"
                        min={0}
                        max={count - 1}
                        step={1}
                        value={index}
                        aria-label={`${group.name} over time`}
                        onChange={(event) => onScrub(Number(event.target.value))}
                    />
                    <span className="tensor-pos">
                        {index + 1}/{count}
                    </span>
                </div>
            )}
            <span className="tensor-time">{timeOf(shown)}</span>
        </div>
    );
}

function timeOf(meta: TensorMeta): string {
    return new Date(meta.receivedAt).toLocaleTimeString();
}
