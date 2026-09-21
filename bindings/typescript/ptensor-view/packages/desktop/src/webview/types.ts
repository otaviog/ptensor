import type { TensorMeta } from '../shared/feed';

/**
 * What the window knows about one producer's tensors. The tensors themselves
 * stay in the bun process; this is the metadata the sidebar lists.
 */
export interface Session {
    id: string;
    /** Oldest first, the order they arrived in. The sidebar reverses it. */
    tensors: TensorMeta[];
    updatedAt: number;
    /** JSON the store is holding for this session, summed. */
    heldBytes: number;
}

export interface Selection {
    sessionId: string;
    tensorId: string;
}
