import { initP10 } from './module-init.js';
import { fromArray, zeros } from './tensor.js';


export type {  Tensor } from './tensor.js';

const p10 = {
    init: initP10,
    zeros,
    fromArray
}
export default p10;