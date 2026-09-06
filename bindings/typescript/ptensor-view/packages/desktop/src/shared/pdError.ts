import { P10Error } from "ptensor-ts";

export class PdError extends Error {
  constructor(message: string | P10Error, options?: ErrorOptions) {
    if (message instanceof P10Error) {
      super(`P10Error: ${message.message}`, options);
    } else {
      super(message as string, options)
      this.name = "PdError";
    }
  }
}
