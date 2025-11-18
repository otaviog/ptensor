import { MODULE } from './module-init.js';

export interface Dtype {
    toString(): string;
    delete(): void;
}

export type DTypeString = 'float32' | 'float64' | 'uint8' | 'uint16' | 'uint32' | 'int8' | 'int16' | 'int32' | 'int64';

export const createDtype = (dtype: DTypeString): Dtype => {
    return MODULE.Dtype.fromString(dtype);
}
