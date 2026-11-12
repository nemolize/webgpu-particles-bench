// One update step of the particle simulation (single thread, scalar f32).
// Build: rustc --target wasm32-unknown-unknown -C opt-level=3 -C panic=abort -C strip=symbols --crate-type cdylib sim.rs -o sim.wasm
#[no_mangle]
pub unsafe extern "C" fn step(pos: *mut f32, vel: *mut f32, n: usize, dt: f32, g: f32, eps: f32, damp: f32) {
    for i in 0..n {
        let (mut px, mut py) = (*pos.add(2 * i), *pos.add(2 * i + 1));
        let (mut vx, mut vy) = (*vel.add(2 * i), *vel.add(2 * i + 1));
        let (dx, dy) = (-px, -py);
        let r2 = dx * dx + dy * dy + eps;
        let inv = 1.0 / r2.sqrt();
        let k = g * inv * inv * inv;
        vx = (vx + dx * k * dt) * damp;
        vy = (vy + dy * k * dt) * damp;
        px += vx * dt;
        py += vy * dt;
        if px > 1.0 { px = 2.0 - px; vx = -vx; }
        if px < -1.0 { px = -2.0 - px; vx = -vx; }
        if py > 1.0 { py = 2.0 - py; vy = -vy; }
        if py < -1.0 { py = -2.0 - py; vy = -vy; }
        *pos.add(2 * i) = px;
        *pos.add(2 * i + 1) = py;
        *vel.add(2 * i) = vx;
        *vel.add(2 * i + 1) = vy;
    }
}
