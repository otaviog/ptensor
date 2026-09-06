import type { Selection, Session } from './types';


export function SessionGroup({
    session,
    selected,
    onSelect,
    onClear,
}: {
    session: Session;
    selected: Selection | null;
    onSelect: (sessionId: string, tensorId: string) => void;
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
                {[...session.tensors].reverse().map((entry) => {
                    const active =
                        selected?.sessionId === session.id && selected.tensorId === entry.id;
                    return (
                        <li key={entry.id}>
                            <button
                                type="button"
                                className={`tensor-item${active ? ' active' : ''}`}
                                onClick={() => onSelect(session.id, entry.id)}
                            >
                                <span className="tensor-name">{entry.payload.name}</span>
                                <span className="tensor-meta">
                                    {entry.payload.tensor.dtype} [
                                    {entry.payload.tensor.shape.join('×')}]
                                </span>
                                <span className="tensor-time">
                                    {new Date(entry.payload.receivedAt).toLocaleTimeString()}
                                </span>
                            </button>
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}
