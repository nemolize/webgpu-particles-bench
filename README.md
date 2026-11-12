# webgpu-particles-bench

One particle simulation, implemented three ways with the same update rule: plain JS, WASM (Rust, scalar f32, one thread) and a WebGPU compute shader. `bench.mjs` times each across particle counts from 100 to 10,000,000 and reports frame-time p50 and p99 per count.

`results.json` holds two sweeps on one machine (Apple M2 Max, macOS, Chrome). The numbers do not transfer to other hardware; rerun on yours.

## Run

Needs Node and a browser with WebGPU (Chrome). No install step.

```sh
node serve.mjs          # serves this directory on http://localhost:8080
# open http://localhost:8080/?run=1 and wait for "done" in the page log
# the result object is window.__result (JSON)
```

The server sends cross-origin-isolation headers so `performance.now()` has about 5 µs resolution. Run with the machine otherwise idle, on AC power.

`?run=1&counts=1000,100000` runs a subset. `charts.html` draws the two charts from `results.json`.

## How `results.json` was produced

- `cells`, `agreement`: one `?run=1` sweep in Chrome.
- `agreementOneStep`: the same check with one step, from the page console: `await window.runBench({counts: [], checkSteps: 1, checkCounts: [1000000]})` (read `.agreement`).
- `timestampsUnquantized`: a second `?run=1` sweep in Chrome launched with `--enable-webgpu-developer-features`, which removes the 65.536 µs quantization of `timestamp-query`. Its `gpuComputeTs` and `gpuRenderTs` are the values the article reports; its frame summaries show run-to-run spread against `cells`.
- `node trace-outlier.mjs` replays the JS reference for the one particle that disagrees with the GPU at 1,000,000 particles and prints its wall bounce.

## Rebuild the WASM

`sim.wasm` is committed so no toolchain is needed to run. To rebuild it:

```sh
rustup target add wasm32-unknown-unknown
rustc --target wasm32-unknown-unknown -C opt-level=3 -C panic=abort -C strip=symbols --crate-type cdylib sim.rs -o sim.wasm
```

## Files

- `bench.mjs`: the three implementations, timing loop, agreement check
- `trace-outlier.mjs`: JS-side trace of the one disagreeing particle
- `sim.rs`, `sim.wasm`: the WASM step
- `index.html`, `serve.mjs`: page and static server
- `results.json`: the measurements
- `charts.html`: draws the charts from `results.json`
