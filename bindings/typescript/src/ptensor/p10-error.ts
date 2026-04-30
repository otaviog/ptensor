export interface P10Error {
  code(): number;
  toString(): string;
  isOk(): boolean;
  isError(): boolean;
  delete(): void;
}