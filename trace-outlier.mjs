// Replays the JS reference for one particle of the 1M-particle agreement check and prints each wall bounce.
import { initialState, stepJS, PARAMS } from './bench.mjs';

const n = 1000000;
const k = Number(process.argv[2] ?? 580018);
const { pos, vel } = initialState(n);
const p = Float32Array.of(pos[2 * k], pos[2 * k + 1]);
const v = Float32Array.of(vel[2 * k], vel[2 * k + 1]);
for (let step = 0; step < 100; step++) {
  const [px, py, vx, vy] = [p[0], p[1], v[0], v[1]];
  stepJS(p, v, 1, PARAMS);
  if (Math.sign(v[0]) !== Math.sign(vx) || Math.sign(v[1]) !== Math.sign(vy)) {
    console.log(`bounce at step ${step + 1}: before (${px}, ${py}) after (${p[0]}, ${p[1]})`);
  }
}
