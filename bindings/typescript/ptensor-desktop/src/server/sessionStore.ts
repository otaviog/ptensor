// In-memory store of everything the socket feed received, grouped by session.
// Tensor payloads stay here; the webview only gets summaries until it selects
// one, so a fast producer does not push megabytes through the RPC channel.

import {
    type SessionSummary,
    type TensorMessage,
    type TensorPayload,
    type TensorSummary,
} from '../shared/protocol';

interface StoredTensor {
    summary: TensorSummary;
    payload: TensorPayload;
}

interface StoredSession {
    id: string;
    tensors: StoredTensor[];
    updatedAt: number;
    totalReceived: number;
}

export interface SessionStoreOptions {
    /** Tensors kept per session; the oldest are dropped past this. */
    maxTensorsPerSession?: number;
}

export class SessionStore {
    private readonly sessions = new Map<string, StoredSession>();
    private readonly maxTensors: number;
    private nextId = 1;

    constructor(options: SessionStoreOptions = {}) {
        this.maxTensors = options.maxTensorsPerSession ?? 100;
    }

    /** Stores a received tensor and returns its summary. */
    add(sessionId: string, msg: TensorMessage): TensorSummary {
        const session = this.session(sessionId);
        const id = `t${this.nextId++}`;
        const shape = msg.tensor.shape.map(Number);
        const summary: TensorSummary = {
            id,
            name: msg.name && msg.name.length > 0 ? msg.name : `tensor ${id.slice(1)}`,
            receivedAt: Date.now(),
            dtype: msg.tensor.dtype,
            shape,
            elems: shape.reduce((acc, dim) => acc * dim, 1),
            bytes: base64Bytes(msg.tensor.blob),
        };
        session.tensors.push({
            summary,
            payload: { sessionId, id, name: summary.name, tensor: msg.tensor },
        });
        if (session.tensors.length > this.maxTensors) {
            session.tensors.splice(0, session.tensors.length - this.maxTensors);
        }
        session.updatedAt = summary.receivedAt;
        session.totalReceived++;
        return summary;
    }

    /** Newest session first, and newest tensor last within a session. */
    summaries(): SessionSummary[] {
        return [...this.sessions.values()]
            .sort((a, b) => b.updatedAt - a.updatedAt)
            .map((session) => ({
                id: session.id,
                tensors: session.tensors.map((entry) => entry.summary),
                updatedAt: session.updatedAt,
                totalReceived: session.totalReceived,
            }));
    }

    payload(sessionId: string, tensorId: string): TensorPayload | null {
        const session = this.sessions.get(sessionId);
        const found = session?.tensors.find((entry) => entry.summary.id === tensorId);
        return found ? found.payload : null;
    }

    /** Drops one session, or every session when `sessionId` is undefined. */
    clear(sessionId?: string): void {
        if (sessionId === undefined) {
            this.sessions.clear();
            return;
        }
        this.sessions.delete(sessionId);
    }

    private session(id: string): StoredSession {
        let session = this.sessions.get(id);
        if (!session) {
            session = { id, tensors: [], updatedAt: Date.now(), totalReceived: 0 };
            this.sessions.set(id, session);
        }
        return session;
    }
}

/** Decoded byte length of a base64 string, without decoding it. */
function base64Bytes(blob: string): number {
    const padding = blob.endsWith('==') ? 2 : blob.endsWith('=') ? 1 : 0;
    return Math.max(0, Math.floor((blob.length * 3) / 4) - padding);
}
