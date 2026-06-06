/** Tensor element types, mirroring `p10::Dtype` / `P10DTypeEnum`. */
export type DTypeString =
  | 'float32'
  | 'float64'
  | 'float16'
  | 'uint8'
  | 'uint16'
  | 'uint32'
  | 'int8'
  | 'int16'
  | 'int32'
  | 'int64';

/** Numeric values matching the C `P10DTypeEnum` ordering. */
export const dtypeToNumber: Record<DTypeString, number> = {
  float32: 0,
  float64: 1,
  float16: 2,
  uint8: 3,
  uint16: 4,
  uint32: 5,
  int8: 6,
  int16: 7,
  int32: 8,
  int64: 9,
};

export const numberToDtype: Record<number, DTypeString> = Object.fromEntries(
  Object.entries(dtypeToNumber).map(([k, v]) => [v, k]),
) as Record<number, DTypeString>;

/** Size of a single element in bytes. */
export const dtypeSizeBytes: Record<DTypeString, number> = {
  float32: 4,
  float64: 8,
  float16: 2,
  uint8: 1,
  uint16: 2,
  uint32: 4,
  int8: 1,
  int16: 2,
  int32: 4,
  int64: 8,
};

const DTYPE_SET = new Set<string>(Object.keys(dtypeToNumber));

/** Narrows an arbitrary string to a DTypeString, or undefined if unknown. */
export function asDType(value: string): DTypeString | undefined {
  return DTYPE_SET.has(value) ? (value as DTypeString) : undefined;
}
