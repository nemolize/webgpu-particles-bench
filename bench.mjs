// Particle step benchmark: JS vs WASM vs WebGPU compute. Entry: window.runBench(opts).
const f = Math.fround;
export const PARAMS = { dt: f(1 / 60), g: f(0.5), eps: f(0.01), damp: f(0.9995) };
export const COUNTS = [100, 316, 1000, 3162, 10000, 31623, 100000, 316228, 1000000, 3162278, 10000000];
const WARMUP_FRAMES = 30;
const WARMUP_MS = 300;
const RENDER_SIZE = 1024;
const TOLERANCE = 1e-3;
const WASM_BASE = 4 << 20;

const now = () => performance.now();

function pct(sorted, p) {
  return sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)];
}
function summarize(values) {
  const s = Float64Array.from(values).sort();
  return { p50: pct(s, 0.5), p99: pct(s, 0.99), n: s.length };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function initialState(n) {
  const rnd = mulberry32(12345);
  const pos = new Float32Array(2 * n);
  const vel = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    const x = (rnd() * 2 - 1) * 0.9;
    const y = (rnd() * 2 - 1) * 0.9;
    pos[2 * i] = x;
    pos[2 * i + 1] = y;
    vel[2 * i] = -y * 0.6;
    vel[2 * i + 1] = x * 0.6;
  }
  return { pos, vel };
}

// Reference implementation. Math.fround after every operation reproduces f32 arithmetic.
export function stepJS(pos, vel, n, P) {
  const { dt, g, eps, damp } = P;
  for (let i = 0; i < n; i++) {
    let px = pos[2 * i], py = pos[2 * i + 1];
    let vx = vel[2 * i], vy = vel[2 * i + 1];
    const dx = -px, dy = -py;
    const r2 = f(f(f(dx * dx) + f(dy * dy)) + eps);
    const inv = f(1 / f(Math.sqrt(r2)));
    const k = f(f(f(g * inv) * inv) * inv);
    vx = f(f(vx + f(f(dx * k) * dt)) * damp);
    vy = f(f(vy + f(f(dy * k) * dt)) * damp);
    px = f(px + f(vx * dt));
    py = f(py + f(vy * dt));
    if (px > 1) { px = f(2 - px); vx = -vx; }
    if (px < -1) { px = f(-2 - px); vx = -vx; }
    if (py > 1) { py = f(2 - py); vy = -vy; }
    if (py < -1) { py = f(-2 - py); vy = -vy; }
    pos[2 * i] = px; pos[2 * i + 1] = py;
    vel[2 * i] = vx; vel[2 * i + 1] = vy;
  }
}

const COMPUTE_WGSL = /* wgsl */ `
struct Params { n: u32, dt: f32, g: f32, eps: f32, damp: f32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec2<f32>>;
@compute @workgroup_size(256)
fn step(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= P.n) { return; }
  var p = pos[i];
  var v = vel[i];
  let d = -p;
  let r2 = dot(d, d) + P.eps;
  let inv = 1.0 / sqrt(r2);
  let k = P.g * inv * inv * inv;
  v = (v + d * k * P.dt) * P.damp;
  p = p + v * P.dt;
  if (p.x > 1.0) { p.x = 2.0 - p.x; v.x = -v.x; }
  if (p.x < -1.0) { p.x = -2.0 - p.x; v.x = -v.x; }
  if (p.y > 1.0) { p.y = 2.0 - p.y; v.y = -v.y; }
  if (p.y < -1.0) { p.y = -2.0 - p.y; v.y = -v.y; }
  pos[i] = p;
  vel[i] = v;
}`;

const RENDER_WGSL = /* wgsl */ `
@vertex fn vs(@location(0) p: vec2<f32>) -> @builtin(position) vec4<f32> {
  return vec4<f32>(p * 0.95, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.6, 0.2, 1.0);
}`;

async function setupGPU() {
  if (!navigator.gpu) throw new Error('no navigator.gpu');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no adapter');
  const hasTS = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: hasTS ? ['timestamp-query'] : [],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  const info = adapter.info ?? {};
  const computeModule = device.createShaderModule({ code: COMPUTE_WGSL });
  const renderModule = device.createShaderModule({ code: RENDER_WGSL });
  const computePipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module: computeModule, entryPoint: 'step' },
  });
  const format = 'rgba8unorm';
  const renderPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: renderModule,
      entryPoint: 'vs',
      buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }] }],
    },
    fragment: { module: renderModule, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'point-list' },
  });
  const target = device.createTexture({
    size: [RENDER_SIZE, RENDER_SIZE],
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return { adapter, device, hasTS, info, computePipeline, renderPipeline, target };
}

function uniformData(n, P) {
  const b = new ArrayBuffer(32);
  new Uint32Array(b, 0, 1)[0] = n;
  new Float32Array(b, 4, 4).set([P.dt, P.g, P.eps, P.damp]);
  return b;
}

function renderPass(gpu, enc, vbuf, n, timestampWrites) {
  const pass = enc.beginRenderPass({
    colorAttachments: [{
      view: gpu.target.createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'clear',
      storeOp: 'store',
    }],
    ...(timestampWrites ? { timestampWrites } : {}),
  });
  pass.setPipeline(gpu.renderPipeline);
  pass.setVertexBuffer(0, vbuf);
  pass.draw(n);
  pass.end();
}

function makeGpuState(gpu, n, P, init) {
  const { device } = gpu;
  const size = 8 * n;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const pos = device.createBuffer({ size, usage });
  const vel = device.createBuffer({ size, usage });
  const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(pos, 0, init.pos);
  device.queue.writeBuffer(vel, 0, init.vel);
  device.queue.writeBuffer(params, 0, uniformData(n, P));
  const bind = device.createBindGroup({
    layout: gpu.computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: pos } },
      { binding: 2, resource: { buffer: vel } },
    ],
  });
  return { pos, vel, params, bind, destroy() { pos.destroy(); vel.destroy(); params.destroy(); } };
}

function encodeCompute(gpu, enc, st, n, timestampWrites) {
  const pass = enc.beginComputePass(timestampWrites ? { timestampWrites } : undefined);
  pass.setPipeline(gpu.computePipeline);
  pass.setBindGroup(0, st.bind);
  pass.dispatchWorkgroups(Math.ceil(n / 256));
  pass.end();
}

async function readBack(gpu, buf, byteLength) {
  const { device } = gpu;
  const rb = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return out;
}

// Offscreen target, no presentation: frame time is not capped by vsync.
async function timeFrames(frame, samples) {
  const start = now();
  let seen = 0;
  const rows = [];
  while (rows.length < samples) {
    const t0 = now();
    const parts = await frame();
    const total = now() - t0;
    seen++;
    if (seen > WARMUP_FRAMES && now() - start > WARMUP_MS) rows.push({ total, ...parts });
  }
  return { rows, discarded: seen - rows.length };
}

function summarizeRows(rows, discarded) {
  const out = { samples: rows.length, warmupDiscarded: discarded, frame: summarize(rows.map((r) => r.total)) };
  for (const key of Object.keys(rows[0]).filter((k) => k !== 'total')) {
    out[key] = summarize(rows.map((r) => r[key]));
  }
  return out;
}

async function runCPU(gpu, kind, n, P, init, samples, wasm) {
  const { device } = gpu;
  const vbuf = device.createBuffer({ size: 8 * n, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  let pos, vel, stepFn;
  if (kind === 'js') {
    pos = Float32Array.from(init.pos);
    vel = Float32Array.from(init.vel);
    stepFn = () => stepJS(pos, vel, n, P);
  } else {
    const bytes = 8 * n;
    const need = WASM_BASE + 2 * bytes;
    const have = wasm.memory.buffer.byteLength;
    if (need > have) wasm.memory.grow(Math.ceil((need - have) / 65536));
    pos = new Float32Array(wasm.memory.buffer, WASM_BASE, 2 * n);
    vel = new Float32Array(wasm.memory.buffer, WASM_BASE + bytes, 2 * n);
    pos.set(init.pos);
    vel.set(init.vel);
    stepFn = () => wasm.step(WASM_BASE, WASM_BASE + bytes, n, P.dt, P.g, P.eps, P.damp);
  }
  const frame = async () => {
    const t0 = now();
    stepFn();
    const t1 = now();
    device.queue.writeBuffer(vbuf, 0, pos);
    const t2 = now();
    const enc = device.createCommandEncoder();
    renderPass(gpu, enc, vbuf, n);
    device.queue.submit([enc.finish()]);
    const t3 = now();
    await device.queue.onSubmittedWorkDone();
    return { cpuStep: t1 - t0, upload: t2 - t1, renderSubmit: t3 - t2 };
  };
  const { rows, discarded } = await timeFrames(frame, samples);
  vbuf.destroy();
  return summarizeRows(rows, discarded);
}

async function runGPU(gpu, n, P, init, samples) {
  const { device } = gpu;
  const st = makeGpuState(gpu, n, P, init);
  const frame = async () => {
    const t0 = now();
    const enc = device.createCommandEncoder();
    encodeCompute(gpu, enc, st, n);
    renderPass(gpu, enc, st.pos, n);
    device.queue.submit([enc.finish()]);
    const t1 = now();
    await device.queue.onSubmittedWorkDone();
    return { dispatchSubmit: t1 - t0 };
  };
  const { rows, discarded } = await timeFrames(frame, samples);
  const result = summarizeRows(rows, discarded);

  if (gpu.hasTS) {
    const querySet = device.createQuerySet({ type: 'timestamp', count: 4 });
    const resolve = device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rb = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const compute = [];
    const render = [];
    const reps = Math.min(samples, 200);
    for (let i = 0; i < 30 + reps; i++) {
      const enc = device.createCommandEncoder();
      encodeCompute(gpu, enc, st, n, { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 });
      renderPass(gpu, enc, st.pos, n, { querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 });
      enc.resolveQuerySet(querySet, 0, 4, resolve, 0);
      enc.copyBufferToBuffer(resolve, 0, rb, 0, 32);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(rb.getMappedRange().slice(0));
      rb.unmap();
      if (i >= 30) {
        compute.push(Number(t[1] - t[0]) / 1e6);
        render.push(Number(t[3] - t[2]) / 1e6);
      }
    }
    result.gpuComputeTs = summarize(compute);
    result.gpuRenderTs = summarize(render);
    querySet.destroy();
    resolve.destroy();
    rb.destroy();
  }

  const rbBuf = device.createBuffer({ size: 8 * n, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const hostPos = new Float32Array(2 * n);
  const readInto = async () => {
    await rbBuf.mapAsync(GPUMapMode.READ);
    hostPos.set(new Float32Array(rbBuf.getMappedRange()));
    rbBuf.unmap();
  };
  const readbackOnly = async () => {
    await device.queue.onSubmittedWorkDone();
    const t0 = now();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(st.pos, 0, rbBuf, 0, 8 * n);
    device.queue.submit([enc.finish()]);
    await readInto();
    return { readback: now() - t0 };
  };
  const computeReadback = async () => {
    const enc = device.createCommandEncoder();
    encodeCompute(gpu, enc, st, n);
    enc.copyBufferToBuffer(st.pos, 0, rbBuf, 0, 8 * n);
    device.queue.submit([enc.finish()]);
    await readInto();
    return {};
  };
  const computeOnly = async () => {
    const enc = device.createCommandEncoder();
    encodeCompute(gpu, enc, st, n);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return {};
  };
  const lightSamples = n >= 1000000 ? 100 : samples;
  const rb = await timeFrames(readbackOnly, lightSamples);
  result.readback = summarize(rb.rows.map((r) => r.readback));
  result.computeOnlyFrame = summarizeRows((await timeFrames(computeOnly, lightSamples)).rows, 0).frame;
  result.computeReadbackFrame = summarizeRows((await timeFrames(computeReadback, lightSamples)).rows, 0).frame;
  result.lightSamples = lightSamples;
  rbBuf.destroy();
  st.destroy();
  return result;
}

async function checkAgreement(gpu, wasm, n, P, steps) {
  const init = initialState(n);
  const refPos = Float32Array.from(init.pos);
  const refVel = Float32Array.from(init.vel);
  for (let s = 0; s < steps; s++) stepJS(refPos, refVel, n, P);

  const bytes = 8 * n;
  const need = WASM_BASE + 2 * bytes;
  if (need > wasm.memory.buffer.byteLength) wasm.memory.grow(Math.ceil((need - wasm.memory.buffer.byteLength) / 65536));
  const wp = new Float32Array(wasm.memory.buffer, WASM_BASE, 2 * n);
  const wv = new Float32Array(wasm.memory.buffer, WASM_BASE + bytes, 2 * n);
  wp.set(init.pos);
  wv.set(init.vel);
  for (let s = 0; s < steps; s++) wasm.step(WASM_BASE, WASM_BASE + bytes, n, P.dt, P.g, P.eps, P.damp);

  const st = makeGpuState(gpu, n, P, init);
  const enc = gpu.device.createCommandEncoder();
  for (let s = 0; s < steps; s++) {
    encodeCompute(gpu, enc, st, n);
  }
  gpu.device.queue.submit([enc.finish()]);
  const gp = await readBack(gpu, st.pos, bytes);
  const gv = await readBack(gpu, st.vel, bytes);
  st.destroy();

  const diff = (a, b) => {
    let max = 0, sum = 0, over = 0, at = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      if (d > max) { max = d; at = i; }
      if (d > TOLERANCE) over++;
      sum += d;
    }
    return { max, mean: sum / a.length, over, of: a.length, at, refAtWorst: a[at], otherAtWorst: b[at] };
  };
  return {
    n,
    steps,
    wasmVsJs: { pos: diff(refPos, wp), vel: diff(refVel, wv) },
    gpuVsJs: { pos: diff(refPos, gp), vel: diff(refVel, gv) },
  };
}

export async function runBench({ counts = COUNTS, log = console.log, checkSteps = 100, checkCounts = [10000, 1000000] } = {}) {
  const gpu = await setupGPU();
  const wasm = (await WebAssembly.instantiate(await (await fetch('sim.wasm')).arrayBuffer())).instance.exports;
  const out = {
    env: {
      userAgent: navigator.userAgent,
      gpu: { vendor: gpu.info.vendor, architecture: gpu.info.architecture, device: gpu.info.device, description: gpu.info.description },
      timestampQuery: gpu.hasTS,
      crossOriginIsolated: self.crossOriginIsolated,
      params: P2(PARAMS),
      warmup: { minFrames: WARMUP_FRAMES, minMs: WARMUP_MS },
      renderSize: RENDER_SIZE,
      tolerance: TOLERANCE,
    },
    agreement: [],
    cells: [],
  };
  for (const n of checkCounts) {
    log(`agreement n=${n}`);
    out.agreement.push(await checkAgreement(gpu, wasm, n, PARAMS, checkSteps));
  }
  for (const n of counts) {
    const samples = n < 1000000 ? 500 : 200;
    const init = initialState(n);
    const cell = { n };
    for (const kind of ['js', 'wasm']) {
      log(`n=${n} ${kind}`);
      cell[kind] = await runCPU(gpu, kind, n, PARAMS, init, samples, wasm);
    }
    log(`n=${n} gpu`);
    cell.gpu = await runGPU(gpu, n, PARAMS, init, samples);
    out.cells.push(cell);
  }
  gpu.device.destroy();
  return out;
}

function P2(p) { return { ...p }; }
