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
  Object.entries(dtypeToNumber).map(([k, v]) => [v, k])
) as Record<number, DTypeString>;
