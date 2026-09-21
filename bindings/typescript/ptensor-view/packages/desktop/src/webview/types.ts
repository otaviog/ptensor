import type { TensorMeta } from '../shared/feed';

/**
 * Every arrival that carried one name. A producer logging in a loop sends the
 * same name over and over, so this is what the sidebar lists: one row per name,
 * with a slider across its arrivals.
 */
export interface NameGroup {
    name: string;
    /** Oldest first, the order they arrived in. */
    tensors: TensorMeta[];
    /** JSON the store is holding for this name, summed. */
    heldBytes: number;
}

/**
 * What the window knows about one producer's tensors. The tensors themselves
 * stay in the bun process; this is the metadata the sidebar lists.
 */
export interface Session {
    id: string;
    /**
     * In the order the names first appeared. Not by recency: two producers
     * alternating would otherwise reshuffle the sidebar on every arrival.
     */
    groups: NameGroup[];
    updatedAt: number;
    heldBytes: number;
}

/** Which name's row is on screen. Where its slider sits is kept separately. */
export interface Selection {
    sessionId: string;
    name: string;
}

/**
 * Slider positions, by session and name: a group keeps its own place even while
 * another one is on screen. An entry is the pinned tensor's id; no entry means
 * the group follows its newest arrival.
 */
export type Positions = Record<string, string>;

export function positionKey(sessionId: string, name: string): string {
    // Newline: a session id is a uuid and a name cannot contain one, having
    // arrived as a single JSON line.
    return `${sessionId}\n${name}`;
}

/** The arrival a group is showing: its pinned one, or the newest. */
export function shownTensor(group: NameGroup, pinned: string | undefined): TensorMeta {
    const found = pinned === undefined ? undefined : group.tensors.find((m) => m.id === pinned);
    return found ?? group.tensors[group.tensors.length - 1];
}
