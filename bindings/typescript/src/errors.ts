import { ErrorCode, ERROR_MESSAGES } from './enums';
import * as ffi from './ffi';

/**
 * Base error class for P10 errors
 */
export class P10Error extends Error {
  constructor(
    public readonly code: ErrorCode,
    message?: string
  ) {
    super(message || ERROR_MESSAGES[code]);
    this.name = 'P10Error';
    Object.setPrototypeOf(this, P10Error.prototype);
  }
}

/**
 * Check error code and throw if not OK
 */
export function checkError(errorCode: ErrorCode): void {
  if (errorCode !== ErrorCode.OK) {
    const lastError = ffi.p10_get_last_error_message();
    throw new P10Error(errorCode, lastError || ERROR_MESSAGES[errorCode]);
  }
}
