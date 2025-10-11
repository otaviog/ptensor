import { ErrorCode, ERROR_MESSAGES } from './enums';
import * as ffi from './ffi';

/**
 * Base error class for Ptensor errors
 */
export class PtensorError extends Error {
  constructor(
    public readonly code: ErrorCode,
    message?: string
  ) {
    super(message || ERROR_MESSAGES[code]);
    this.name = 'PtensorError';
    Object.setPrototypeOf(this, PtensorError.prototype);
  }
}

/**
 * Check error code and throw if not OK
 */
export function checkError(errorCode: ErrorCode): void {
  if (errorCode !== ErrorCode.OK) {
    const lastError = ffi.p10_get_last_error_message();
    throw new PtensorError(errorCode, lastError || ERROR_MESSAGES[errorCode]);
  }
}
