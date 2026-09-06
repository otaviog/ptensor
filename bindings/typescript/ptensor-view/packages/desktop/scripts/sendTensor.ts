// Test producer: streams synthetic tensors into a running ptensor View.
//
//   bun scripts/sendTensor.ts --session demo --count 20 --interval 200

import { Command, EnumType } from '@cliffy/command';
import { bytesToBase64 } from 'ptensor-ts';
import { DEFAULT_PORT, type SessionMessage, type TensorMessage } from 'ptensor-tlog';

const kindType = new EnumType(['rgb', 'gray', 'table'] as const);

type Kind = typeof kindType extends EnumType<infer T> ? T : never;

function contiguousStride(shape: number[]): number[] {
    const stride = new Array<number>(shape.length).fill(1);
    for (let i = shape.length - 2; i >= 0; i--) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
}

/** A moving gradient/checker so consecutive frames differ visibly. */
function makeTensor(kind: Kind, frame: number): TensorMessage['tensor'] {
    if (kind === 'table') {
        const shape = [4, 5];
        const data = new Float32Array(20);
        for (let i = 0; i < data.length; i++) {
            data[i] = Math.sin((i + frame) * 0.3);
        }
        return json('float32', shape, new Uint8Array(data.buffer));
    }
    const h = 96;
    const w = 128;
    if (kind === 'gray') {
        const data = new Float32Array(h * w);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                data[y * w + x] = ((x + y + frame * 4) % w) / w;
            }
        }
        return json('float32', [h, w], new Uint8Array(data.buffer));
    }
    const data = new Uint8Array(h * w * 3);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 3;
            data[i] = (x * 2 + frame * 5) % 256;
            data[i + 1] = (y * 2 + frame * 3) % 256;
            data[i + 2] = (x + y + frame) % 256;
        }
    }
    return json('uint8', [h, w, 3], data);
}

function json(dtype: string, shape: number[], bytes: Uint8Array): TensorMessage['tensor'] {
    return {
        dtype,
        shape,
        stride: contiguousStride(shape),
        size_bytes: bytes.byteLength,
        encoding: 'base64',
        blob: bytesToBase64(bytes),
    };
}

const { options: args } = await new Command()
    .name('sendTensor')
    .description('Stream synthetic tensors into a running ptensor View.')
    .type('kind', kindType)
    .option('--host <host:string>', 'View host to connect to.', { default: '127.0.0.1' })
    .option('--port <port:integer>', 'View port to connect to.', {
        default: Number(process.env.PTENSOR_VIEW_PORT ?? DEFAULT_PORT),
    })
    .option('--session <session:string>', 'Session id to file the tensors under.', {
        default: 'demo',
    })
    .option('--count <count:integer>', 'Number of frames to send.', { default: 10 })
    .option('--interval <interval:integer>', 'Delay between frames in ms.', { default: 250 })
    .option('--kind <kind:kind>', 'Tensor flavor to generate.', { default: 'rgb' as const })
    .parse(Bun.argv.slice(2));
const socket = await Bun.connect({
    hostname: args.host,
    port: args.port,
    socket: {
        data: () => {},
        error: (_socket, error) => console.error('send failed:', error.message),
    },
});

// The session is announced once, before the tensors: the viewer files
// everything this connection sends under it.
const session: SessionMessage = { sessionId: args.session };
console.log('Starting session with', session)
const sent = socket.write(`${JSON.stringify(session)}\n`);
if (sent <= 0) {
  console.log('Fail to send')
}

for (let frame = 0; frame < args.count; frame++) {
    const message: TensorMessage = {
        name: `${args.kind} ${frame}`,
        tensor: makeTensor(args.kind, frame),
    };
    socket.write(`${JSON.stringify(message)}\n`);
    console.log(`sent ${message.name} to session '${args.session}'`);
    if (args.interval > 0 && frame + 1 < args.count) {
        await Bun.sleep(args.interval);
    }
}

// Give the kernel a moment to flush the last write before the socket closes.
await Bun.sleep(50);
socket.end();
