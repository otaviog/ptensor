import { TensorPayload } from "../shared/rpc";

export interface ReceivedTensor {
    id: string;
    payload: TensorPayload;
}

export interface Session {
    id: string;
    tensors: ReceivedTensor[];
    updatedAt: number;
    totalReceived: number;
}

export interface Selection {
    sessionId: string;
    tensorId: string;
}
