export class P10Error extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "P10Error";
  }
}
