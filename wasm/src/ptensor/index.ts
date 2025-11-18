import { initP10 } from './module-init';
import { fromArray, zeros } from './tensor';


export type {  Tensor } from './tensor';

const p10 = {
    init: initP10,
    zeros,
    fromArray
}
export default p10;