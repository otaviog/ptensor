import { MODULE } from "./module-init.js";

export interface Shape {
  toArray(): number[];
  dims(): number;
  count(): number;
  empty(): boolean;
  delete(): void;
}

export const createShape = (shape: number[]) => {
  return (MODULE.Shape as any).fromArray(shape);
}
