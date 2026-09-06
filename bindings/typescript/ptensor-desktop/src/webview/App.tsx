// The panel. Tensors arrive one push at a time and are kept here, grouped by
// the session their producer announced; the bun process stores nothing. Only
// the selected tensor is decoded into a TensorView.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TensorViewer, fromTensorJson, type TensorView } from '@ptensor/tensor-view';
import type { ServerInfo, TensorPayload } from '../shared/rpc';

/** Cap used until `getServerInfo` answers with the one the app was started with. */
const DEFAULT_HISTORY = 100;

export interface AppProps {
    /** Subscribes to the feed. Called once, on mount. */
    subscribe: (onTensor: (payload: TensorPayload) => void) => void;
    /** Asks the bun process for the feed state and the history cap. */
    getServerInfo: () => Promise<ServerInfo>;
    /** Overrides the cap the bun process reports. Mostly for tests. */
    maxTensorsPerSession?: number;
}

interface ReceivedTensor {
    id: string;
    payload: TensorPayload;
}

interface Session {
    id: string;
    tensors: ReceivedTensor[];
    updatedAt: number;
    totalReceived: number;
}

interface Selection {
    sessionId: string;
    tensorId: string;
}

export function App({ subscribe, getServerInfo, maxTensorsPerSession }: AppProps) {
    const [sessions, setSessions] = useState<Session[]>([]);
    const [info, setInfo] = useState<ServerInfo | null>(null);
    const [selected, setSelected] = useState<Selection | null>(null);
    const [error, setError] = useState<string | null>(null);
    // Follow mode keeps the newest tensor of the active session on screen.
    const [follow, setFollow] = useState(true);
    const followRef = useRef(follow);
    followRef.current = follow;
    // PTENSOR_VIEW_HISTORY lives in the bun process' environment and reaches the
    // window with the server info, so the first pushes may use the default cap.
    const history = maxTensorsPerSession ?? info?.maxTensorsPerSession ?? DEFAULT_HISTORY;
    const historyRef = useRef(history);
    historyRef.current = history;

    useEffect(() => {
        let nextId = 1;
        subscribe((payload) => {
            const entry: ReceivedTensor = { id: `t${nextId++}`, payload };
            setSessions((current) => append(current, entry, historyRef.current));
            setSelected((current) => reselect(current, entry, followRef.current));
        });
    }, [subscribe]);

    useEffect(() => {
        let cancelled = false;
        getServerInfo()
            .then((next) => {
                if (!cancelled) {
                    setInfo(next);
                }
            })
            .catch((err: unknown) => {
                if (!cancelled) {
                    setError(err instanceof Error ? err.message : String(err));
                }
            });
        return () => {
            cancelled = true;
        };
    }, [getServerInfo]);

    // A cap that arrives (or shrinks) after tensors did applies to them too.
    useEffect(() => {
        setSessions((current) =>
            current.map((session) =>
                session.tensors.length <= history
                    ? session
                    : { ...session, tensors: session.tensors.slice(-history) }
            )
        );
    }, [history]);

    const activeSession = useMemo(
        () => sessions.find((session) => session.id === selected?.sessionId) ?? null,
        [sessions, selected]
    );

    // Decoding is deferred to selection: the history holds base64 blobs only.
    const tensor = useMemo<TensorView | null>(() => {
        const entry = activeSession?.tensors.find((item) => item.id === selected?.tensorId);
        if (!entry) {
            return null;
        }
        return fromTensorJson(entry.payload.tensor, entry.payload.name);
    }, [activeSession, selected]);

    const onSelect = useCallback((sessionId: string, tensorId: string) => {
        setSelected({ sessionId, tensorId });
    }, []);

    const clearSession = useCallback((sessionId?: string) => {
        setSessions((current) =>
            sessionId === undefined
                ? []
                : current.filter((session) => session.id !== sessionId)
        );
        setSelected((current) =>
            sessionId === undefined || current?.sessionId === sessionId ? null : current
        );
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

/** Files one received tensor under its session, newest session first. */
function append(sessions: Session[], entry: ReceivedTensor, maxTensors: number): Session[] {
    const { sessionId, receivedAt } = entry.payload;
    const current =
        sessions.find((session) => session.id === sessionId) ??
        ({ id: sessionId, tensors: [], updatedAt: receivedAt, totalReceived: 0 } as Session);
    const tensors = [...current.tensors, entry];
    const next: Session = {
        id: sessionId,
        tensors: tensors.slice(Math.max(0, tensors.length - maxTensors)),
        updatedAt: receivedAt,
        totalReceived: current.totalReceived + 1,
    };
    return [next, ...sessions.filter((session) => session.id !== sessionId)];
}

/**
 * Keeps the selection valid as tensors arrive: in follow mode the new tensor
 * wins, otherwise the current one is kept and the first tensor seeds an empty
 * panel.
 */
function reselect(
    current: Selection | null,
    entry: ReceivedTensor,
    follow: boolean
): Selection | null {
    if (current !== null && !follow) {
        return current;
    }
    return { sessionId: entry.payload.sessionId, tensorId: entry.id };
}
