// The panel. What it keeps is metadata -- what tensors exist, under which
// session and name -- and it fetches the one tensor on screen from the bun
// process, which holds them all.
//
// That split is the whole point: a tensor is hundreds of megabytes of base64,
// and keeping a session's worth of them in window state, or passing them across
// as messages, costs gigabytes. Decoded tensors are kept in a byte-capped cache
// (./tensorCache) so moving back and forth does not re-fetch, and does not grow
// without bound either.
//
// The sidebar is a row per name, not per arrival: a producer logging in a loop
// sends one name many times, and what you want then is a slider over it.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TensorViewer, type Tensor } from '@ptensor/tensor-view';
import type { FeedEvent, FeedInfo, TensorMeta } from '../shared/feed';
import type { FeedClient } from './feedClient';
import { TensorCache } from './tensorCache';
import {
    positionKey,
    shownTensor,
    type NameGroup,
    type Positions,
    type Selection,
    type Session,
} from './types';
import { SessionGroup } from './SessionGroup';

export interface AppProps {
    client: FeedClient;
    /** Decoded tensors the window holds. Injected by tests. */
    cache?: TensorCache;
}

export function App({ client, cache }: AppProps) {
    const decoded = useMemo(() => cache ?? new TensorCache(), [cache]);
    const [sessions, setSessions] = useState<Session[]>([]);
    const [info, setInfo] = useState<FeedInfo | null>(null);
    const [selected, setSelected] = useState<Selection | null>(null);
    const [positions, setPositions] = useState<Positions>({});
    const [error, setError] = useState<string | null>(null);
    // Follow mode keeps each group on its newest arrival, and moves the panel
    // to whichever one just arrived.
    const [follow, setFollow] = useState(true);
    const followRef = useRef(follow);
    followRef.current = follow;
    // The tensor on screen, and which id it is: a fetch that lands after the
    // selection moved on must not overwrite what is shown.
    const [shown, setShown] = useState<{ id: string; tensor: Tensor } | null>(null);
    const [shownError, setShownError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    // Read by the event handler, which needs the current list without being
    // re-subscribed every time the list changes.
    const sessionsRef = useRef(sessions);
    sessionsRef.current = sessions;

    useEffect(() => {
        const forget = (ids: Iterable<string>): void => {
            for (const id of ids) {
                decoded.delete(id);
            }
        };

        return client.subscribe((event: FeedEvent) => {
            switch (event.type) {
                case 'hello':
                    // A reconnect re-states everything, so the panel follows the
                    // store rather than accumulating its own idea of it.
                    setInfo(event.info);
                    setError(null);
                    setSessions(group(event.tensors));
                    break;
                case 'tensor':
                    forget(event.dropped);
                    setSessions((current) => append(current, event.tensor, event.dropped));
                    if (followRef.current) {
                        // Unpinned, so the group tracks its newest arrival.
                        setPositions((current) =>
                            without(current, positionKey(event.tensor.sessionId, event.tensor.name))
                        );
                        setSelected({
                            sessionId: event.tensor.sessionId,
                            name: event.tensor.name,
                        });
                    } else {
                        setSelected((current) => current ?? {
                            sessionId: event.tensor.sessionId,
                            name: event.tensor.name,
                        });
                    }
                    break;
                case 'cleared':
                    forget(idsOf(sessionsRef.current, event.sessionId));
                    setSessions((current) =>
                        event.sessionId === undefined
                            ? []
                            : current.filter((session) => session.id !== event.sessionId)
                    );
                    setSelected((current) =>
                        event.sessionId === undefined || current?.sessionId === event.sessionId
                            ? null
                            : current
                    );
                    setPositions((current) => forSessions(current, event.sessionId));
                    break;
            }
        });
    }, [client, decoded]);

    // The socket's hello already carries the listing, so this is really the
    // startup probe: a refused connection or a rejected token surfaces here as
    // a message, where a socket that just keeps retrying would leave the panel
    // saying "connecting" forever.
    useEffect(() => {
        let cancelled = false;
        client
            .listTensors()
            .then((tensors) => {
                if (!cancelled) {
                    setSessions((current) => (current.length === 0 ? group(tensors) : current));
                }
            })
            .catch((err: unknown) => {
                if (!cancelled) {
                    setError(messageOf(err));
                }
            });
        return () => {
            cancelled = true;
        };
    }, [client]);

    const activeSession = useMemo(
        () => sessions.find((session) => session.id === selected?.sessionId) ?? null,
        [sessions, selected]
    );
    const activeGroup = useMemo(
        () => activeSession?.groups.find((entry) => entry.name === selected?.name) ?? null,
        [activeSession, selected]
    );
    const activeTensor = useMemo(
        () =>
            activeGroup === null || selected === null
                ? null
                : shownTensor(
                      activeGroup,
                      positions[positionKey(selected.sessionId, activeGroup.name)]
                  ),
        [activeGroup, positions, selected]
    );

    // Fetching and decoding is deferred to what is on screen. It can fail -- an
    // evicted tensor, a blob the panel cannot read -- and a throw from a render
    // would take the whole window down, so the message goes to the panel.
    const selectedId = activeTensor?.id;
    useEffect(() => {
        if (selectedId === undefined) {
            setShown(null);
            setShownError(null);
            return;
        }
        const held = decoded.get(selectedId);
        if (held !== undefined) {
            setShown({ id: selectedId, tensor: held });
            setShownError(null);
            return;
        }

        let cancelled = false;
        setLoading(true);
        client
            .loadTensor(selectedId)
            .then((tensor) => {
                decoded.put(selectedId, tensor);
                if (!cancelled) {
                    setShown({ id: selectedId, tensor });
                    setShownError(null);
                }
            })
            .catch((err: unknown) => {
                if (!cancelled) {
                    setShown(null);
                    setShownError(messageOf(err));
                }
            })
            .finally(() => {
                if (!cancelled) {
                    setLoading(false);
                }
            });
        return () => {
            cancelled = true;
        };
    }, [client, decoded, selectedId]);

    const onSelect = useCallback((sessionId: string, name: string) => {
        setSelected({ sessionId, name });
    }, []);

    /**
     * Moves one group's slider. Scrubbing off the newest arrival turns follow
     * off: otherwise the next tensor to land would yank the panel straight back
     * to the end.
     */
    const onScrub = useCallback((sessionId: string, name: string, index: number) => {
        const entry = sessionsRef.current
            .find((session) => session.id === sessionId)
            ?.groups.find((candidate) => candidate.name === name);
        const target = entry?.tensors[index];
        if (entry === undefined || target === undefined) {
            return;
        }
        setSelected({ sessionId, name });
        setPositions((held) => ({ ...held, [positionKey(sessionId, name)]: target.id }));
        if (index < entry.tensors.length - 1) {
            setFollow(false);
        }
    }, []);

    // The store is the bun process', so clearing is a request. What it drops
    // comes back as a `cleared` event, which is what updates the panel.
    const clearSession = useCallback(
        (sessionId?: string) => {
            client.clear(sessionId).catch((err: unknown) => setError(messageOf(err)));
        },
        [client]
    );

    const tensor = shown !== null && shown.id === selectedId ? shown.tensor : null;

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
                        positions={positions}
                        onSelect={onSelect}
                        onScrub={onScrub}
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
                    <TensorViewer tensor={tensor} name={activeTensor?.name} />
                ) : (
                    <div className="placeholder">
                        {shownError ??
                            error ??
                            (loading ? 'Loading the tensor…' : 'Select a tensor on the left.')}
                    </div>
                )}
                {activeSession && tensor && (
                    <div className="panel-foot">
                        session <strong>{activeSession.id}</strong> ·{' '}
                        {countOf(activeSession)} held · {formatBytes(activeSession.heldBytes)} in
                        the store
                    </div>
                )}
            </main>
        </div>
    );
}

function StatusDot({ info }: { info: FeedInfo | null }) {
    const state = !info ? 'pending' : info.listening ? 'up' : 'down';
    return <span className={`status status-${state}`} title={describeFeed(info)} />;
}

function describeFeed(info: FeedInfo | null): string {
    if (!info) {
        return 'connecting to the feed…';
    }
    if (!info.listening) {
        return `feed down: ${info.error ?? 'not listening'}`;
    }
    return `listening on ${info.host}:${info.port}`;
}

function messageOf(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function countOf(session: Session): number {
    return session.groups.reduce((total, entry) => total + entry.tensors.length, 0);
}

/** Bytes as the sidebar shows them: no more precision than is readable. */
function formatBytes(bytes: number): string {
    if (bytes < 1024) {
        return `${bytes} B`;
    }
    const units = ['KB', 'MB', 'GB'];
    let value = bytes / 1024;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit++;
    }
    return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

/**
 * Groups a listing into sessions. The store lists newest tensor first; this
 * walks it oldest-first so the arrivals within a name end up in the order they
 * came, and the names in the order they first appeared.
 */
function group(tensors: TensorMeta[]): Session[] {
    let sessions: Session[] = [];
    for (const meta of [...tensors].reverse()) {
        sessions = append(sessions, meta, []);
    }
    return sessions;
}

/** Files one tensor under its session and name, newest session first. */
function append(sessions: Session[], meta: TensorMeta, dropped: string[]): Session[] {
    const gone = new Set(dropped);
    const current = sessions.find((session) => session.id === meta.sessionId);
    const groups = addToGroups(current?.groups ?? [], meta, gone);
    const next: Session = {
        id: meta.sessionId,
        groups,
        updatedAt: meta.receivedAt,
        heldBytes: groups.reduce((total, entry) => total + entry.heldBytes, 0),
    };
    const others = sessions
        .filter((session) => session.id !== meta.sessionId)
        .map((session) => withoutDropped(session, gone))
        .filter((session) => session.groups.length > 0);
    return [next, ...others];
}

/**
 * Appends `meta` to its name's group, keeping the groups in the order their
 * names first appeared -- a recency order would reshuffle the sidebar every
 * time two producers alternated.
 */
function addToGroups(groups: NameGroup[], meta: TensorMeta, gone: Set<string>): NameGroup[] {
    const pruned = groups
        .map((entry) => withoutDroppedFromGroup(entry, gone))
        .filter((entry) => entry.tensors.length > 0 || entry.name === meta.name);
    const existing = pruned.find((entry) => entry.name === meta.name);
    if (existing === undefined) {
        return [...pruned, groupOf(meta.name, [meta])];
    }
    return pruned.map((entry) =>
        entry.name === meta.name ? groupOf(entry.name, [...entry.tensors, meta]) : entry
    );
}

function groupOf(name: string, tensors: TensorMeta[]): NameGroup {
    return {
        name,
        tensors,
        heldBytes: tensors.reduce((total, meta) => total + meta.storedBytes, 0),
    };
}

function withoutDroppedFromGroup(entry: NameGroup, gone: Set<string>): NameGroup {
    if (gone.size === 0) {
        return entry;
    }
    const tensors = entry.tensors.filter((meta) => !gone.has(meta.id));
    return tensors.length === entry.tensors.length ? entry : groupOf(entry.name, tensors);
}

function withoutDropped(session: Session, gone: Set<string>): Session {
    if (gone.size === 0) {
        return session;
    }
    const groups = session.groups
        .map((entry) => withoutDroppedFromGroup(entry, gone))
        .filter((entry) => entry.tensors.length > 0);
    return groups.length === session.groups.length &&
        groups.every((entry, at) => entry === session.groups[at])
        ? session
        : {
              ...session,
              groups,
              heldBytes: groups.reduce((total, entry) => total + entry.heldBytes, 0),
          };
}

/** Ids held for one session, or for all of them. */
function idsOf(sessions: Session[], sessionId?: string): string[] {
    return sessions
        .filter((session) => sessionId === undefined || session.id === sessionId)
        .flatMap((session) => session.groups.flatMap((entry) => entry.tensors.map((m) => m.id)));
}

function without(positions: Positions, key: string): Positions {
    if (!(key in positions)) {
        return positions;
    }
    const next = { ...positions };
    delete next[key];
    return next;
}

/** Drops the slider positions of one cleared session, or of all of them. */
function forSessions(positions: Positions, sessionId: string | undefined): Positions {
    if (sessionId === undefined) {
        return {};
    }
    const prefix = `${sessionId}\n`;
    return Object.fromEntries(
        Object.entries(positions).filter(([key]) => !key.startsWith(prefix))
    );
}
