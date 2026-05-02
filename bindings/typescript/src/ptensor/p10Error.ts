export type P10ErrorCode = 'ok' |
    'unknown_error' |
    'assertion_error' |
    'invalid_argument' |
    'invalid_operation' |
    'out_of_memory' |
    'out_of_range' |
    'not_implemented' |
    'os_error' |
    'io_error' |
    'infer_error';

export class P10Error extends Error {
  constructor(message: string, errorCode?: P10ErrorCode) {
    if (errorCode) {      
      super(`(${errorCode}) ${message}`)
    } else {
      super(message);
    }
  }
}
