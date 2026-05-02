import { p10_get_last_error_message } from './backends/bun/ffi';

export const P10ErrorCode = {
  OK: 0,
  UNKNOWN_ERROR: 1,
  ASSERTION_ERROR: 2,
  INVALID_ARGUMENT: 3,
  INVALID_OPERATION: 4,
  OUT_OF_MEMORY: 5,
  OUT_OF_RANGE: 6,
  NOT_IMPLEMENTED: 7,
  OS_ERROR: 8,
  IO_ERROR: 9,
  INFER_ERROR: 10,
} as const;

export type P10ErrorCodeValue = (typeof P10ErrorCode)[keyof typeof P10ErrorCode];

export class P10Error extends Error {
  readonly code: number;

  constructor(code: number, message?: string) {
    const msg = message ?? p10_get_last_error_message() ?? 'Unknown error';
    super(`P10Error(${code}): ${msg}`);
    this.name = 'P10Error';
    this.code = code;
  }

  /** Throws P10Error if code is not OK. */
  static check(code: number): void {
    if (code !== P10ErrorCode.OK) {
      throw new P10Error(code);
    }
  }
}

