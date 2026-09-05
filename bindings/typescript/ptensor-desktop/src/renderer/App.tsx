import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TensorViewer, type TensorView } from '@ptensor/tensor-view';
import type { ServerInfo, SessionSummary } from '../shared/protocol';

export interface AppProps {
    /** Fetches a tensor payload from the bun process. */
    loadTensor: (sessionId: string, tensorId: string) => Promise<TensorView | null>;
    /** Clears one session, or every session when the id is omitted. */
    clearSession: (sessionId?: string) => void;
    /** Subscribes to session pushes; returns the current list synchronously. */
    subscribe: (onSessions: (s: SessionSummary[]) => void, onInfo: (i: ServerInfo) => void) => void;
}

interface Selection {
    sessionId: string;
    tensorId: string;
}

export function App({ loadTensor, clearSession, subscribe }: AppProps) {
    const [sessions, setSessions] = useState<SessionSummary[]>([]);
    const [info, setInfo] = useState<ServerInfo | null>(null);
    const [selected, setSelected] = useState<Selection | null>(null);
    const [tensor, setTensor] = useState<TensorView | null>(null);
    const [error, setError] = useState<string | null>(null);
    // Follow mode keeps the newest tensor of the active session on screen.
    const [follow, setFollow] = useState(true);
    const followRef = useRef(follow);
    followRef.current = follow;

    useEffect(() => {
        subscribe(
            (next) => {
                setSessions(next);
                setSelected((current) => reselect(current, next, followRef.current));
            },
            (next) => setInfo(next)
        );
    }, [subscribe]);

    useEffect(() => {
        if (!selected) {
            setTensor(null);
            return;
        }
        let cancelled = false;
        loadTensor(selected.sessionId, selected.tensorId)
            .then((next) => {
                if (cancelled) {
                    return;
                }
                setTensor(next);
                setError(next ? null : 'Tensor is no longer in the session history.');
            })
            .catch((err: unknown) => {
                if (!cancelled) {
                    setTensor(null);
                    setError(err instanceof Error ? err.message : String(err));
                }
            });
        return () => {
            cancelled = true;
        };
    }, [selected, loadTensor]);

    const activeSession = useMemo(
        () => sessions.find((session) => session.id === selected?.sessionId) ?? null,
        [sessions, selected]
    );

    const onSelect = useCallback((sessionId: string, tensorId: string) => {
        setSelected({ sessionId, tensorId });
    }, []);

    return (
        <div className="app">
            <aside className="sidebar">
                <div className="sidebar-head">
                    <span className="app-name">ptensor View</span>
                    <StatusDot info={info} />
                </div>
                <div className="feed-line">{describeFeed(info)}</div>
                <label className="follow">
                    <input
                        type="checkbox"
                        checked={follow}
                        onChange={(event) => setFollow(event.target.checked)}
                    />
                    Follow newest
                </label>
                {sessions.length === 0 && (
                    <p className="empty">
                        No tensors yet. Send one to the socket and it shows up here.
                    </p>
                )}
                {sessions.map((session) => (
                    <SessionGroup
                        key={session.id}
                        session={session}
                        selected={selected}
                        onSelect={onSelect}
                        onClear={() => clearSession(session.id)}
                    />
                ))}
                {sessions.length > 0 && (
                    <button
                        type="button"
                        className="clear-all"
                        onClick={() => clearSession(undefined)}
                    >
                        Clear all sessions
                    </button>
                )}
            </aside>
            <main className="panel">
                {tensor ? (
                    <TensorViewer tensor={tensor} />
                ) : (
                    <div className="placeholder">
                        {error ?? 'Select a tensor on the left.'}
                    </div>
                )}
                {activeSession && tensor && (
                    <div className="panel-foot">
                        session <strong>{activeSession.id}</strong> ·{' '}
                        {activeSession.tensors.length} kept / {activeSession.totalReceived}{' '}
                        received
                    </div>
                )}
            </main>
        </div>
    );
}

function SessionGroup({
    session,
    selected,
    onSelect,
    onClear,
}: {
    session: SessionSummary;
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
                                <span className="tensor-name">{entry.name}</span>
                                <span className="tensor-meta">
                                    {entry.dtype} [{entry.shape.join('×')}]
                                </span>
                                <span className="tensor-time">
                                    {new Date(entry.receivedAt).toLocaleTimeString()}
                                </span>
                            </button>
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}

function StatusDot({ info }: { info: ServerInfo | null }) {
    const state = !info ? 'pending' : info.listening ? 'up' : 'down';
    return <span className={`status status-${state}`} title={describeFeed(info)} />;
}

function describeFeed(info: ServerInfo | null): string {
    if (!info) {
        return 'connecting to the feed…';
    }
    if (!info.listening) {
        return `feed down: ${info.error ?? 'not listening'}`;
    }
    return `listening on ${info.host}:${info.port}`;
}

/**
 * Keeps the selection valid across pushes: in follow mode the newest tensor of
 * the active session wins; otherwise the current one is kept while it exists.
 */
function reselect(
    current: Selection | null,
    sessions: SessionSummary[],
    follow: boolean
): Selection | null {
    if (sessions.length === 0) {
        return null;
    }
    const session =
        sessions.find((candidate) => candidate.id === current?.sessionId) ?? sessions[0];
    if (session.tensors.length === 0) {
        return null;
    }
    if (!follow && current) {
        const stillThere = session.tensors.some((entry) => entry.id === current.tensorId);
        if (stillThere) {
            return current;
        }
    }
    const newest = session.tensors[session.tensors.length - 1];
    if (current && current.sessionId === session.id && current.tensorId === newest.id) {
        return current;
    }
    return { sessionId: session.id, tensorId: newest.id };
}
