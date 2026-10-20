# GPU clock governor, driver probes, two CPU trims, full-coverage first draws and the new touch controls (2026-10-20)

中文：[2026-10-20-probes-touch.zh-CN.md](2026-10-20-probes-touch.zh-CN.md)

Measured on 2026-10-20 (NZDT) on the Redmi K50 Ultra (Adreno 730, Vulkan 1.1, Android 15), camp, cold starts, unless noted otherwise. GPU times are the per-pass timer (`gfx-pass-profile`), frame totals from `NVN PERF v=1`.

## First: at a steady 60 fps, GPU time per frame follows the clock governor

With the eight main-pass fragment shaders replaced by ones that write a constant (a timing probe; the picture is wrong), the main pass fell from 5,465 to 2,984 µs, but the frame total only from 14.02 to 13.55 ms: every other pass got slower (shadow map 678 → 884 µs, the rest 6,852 → 8,364 µs). The GPU's clock governor lowered the clock when the work shrank. At a steady 60 fps it seems to settle near 14 ms of the 16.7 ms period, which is why the camp lands there whatever changes. Neither phone exposes the GPU clock or busy counters without root.

So at 60 fps, per-frame GPU milliseconds understate both savings and costs: a saving shows up as a lower clock (less power and heat), a cost as a higher one, until the clock is at its maximum. GPU comparisons need a saturated GPU, where frame time is the work at the top clock. The sections below that compare GPU times at 60 fps carry that caveat.

Two ways to see the work anyway:

- **Saturate the GPU.** At a fixed 3D scale of 1.0 the camp runs at 56.8–57.2 fps with GPU 17.3 ms a frame, over the 16.7 ms period, while the render thread stays at 9.6 ms: the GPU is the limit and runs at its top clock. That is the bench for GPU A/Bs (merged `linux-build`, APK `2faa2215…`).
- **Use the shadow map as a clock meter.** In the camp it is a fixed 1024 × 1024 pass with fixed geometry, so its time tracks the clock: 678 µs at 0.75 (N0), 651 µs at 0.8, 653 µs at 1.0 (top clock), 884 µs with the trivial fragment shaders (clock about 23% lower). Dividing pass times by it compares work at equal clock.

At equal clock, 0.8 costs about 5% more GPU work than 0.75 and 1.0 about 28% more. The main pass at 0.75 fits about 3.1 ms that does not depend on resolution plus 2.1 ms that scales with pixels, and its fragment shaders are about 3 ms of it.

## What a higher 3D scale costs (read with the caveat above)

| Scale | Build | GPU per frame | Main pass | fps |
|---|---|---|---|---|
| 0.75 | `526f03f` | 14.11 ms | 5,232 µs | 60.0 |
| 0.80 | `d20ca16` | 14.24 ms | 5,302 µs | 60.0 |
| 0.85 | `d20ca16` | 14.93 ms | 5,546 µs | 60.0 |

(Main pass: median over the profiled frames whose main pass has the same 547 draws and 601,923 vertices.)

From 0.75 to 0.8 the full-screen post passes grow about 10% for 13.8% more pixels, while the fixed-size passes (UI, shadow map, compute) come out about 3–4% faster, and the main pass grows only about 1%: the governor raised the clock to absorb the extra work. Corrected for clock (above), 0.8 costs about 5% more GPU work and 1.0 about 28%. A higher scale is not free: it costs clock, power and the headroom heavier scenes need, and at 1.0 the camp no longer holds 60 fps. The dynamic scale steps down from the configured value to 0.70, 0.65 and 0.60, so a higher default would also need finer levels.

## Small Vulkan probes run from adb shell

- **Memory types for buffers.** For every buffer usage the driver offers only types 0 (device-local) and 6 (host-visible, coherent and cached; `memoryTypeBits` 0x41). GPU reads are equally fast from both: 23 GB/s through a compute shader and 13.6 GB/s through vertex fetch, 64 MB each. Moving vertex data to device-local memory would gain nothing. CPU reads from type 6 run at 0.94 GB/s.
- **Descriptor-set binds.** With 6 sampled images and 9 dynamic uniform buffers, pinned to the A710 cores: about 0.7 µs to rebind the same set with the same offsets (the driver does not skip it), and about 1.5–1.9 µs once any offset changes, whether 1 or all 9 change. Binding the uniform buffers as a second set costs about the same (1.6–1.7 µs), so splitting the set would not help. The camp does about 1,356 binds a frame with 9.1 offsets each, 508 of them rebinding the same set with new offsets. `0e00099` adds which offsets change in those rebinds to `NVN PERF binds`, to size a push-constant path for the uniform block that changes on every draw.
- **Uniform-only arithmetic.** The Adreno 730 compiler does not hoist it: a 1,500-op chain on uniform-buffer values costs the same per vertex as the same chain seeded by a vertex input (about 13 ps per vertex per op). But after the engine's SPIR-V clean-ups the game has little of it: 45 ops in the main mesh vertex shader, 38 in its pre-pass shader, and in the main fragment shaders mostly bitcasts and negations. Doubling the main and pre-pass vertex shaders' uniform work added 72 µs to the pre-pass and nothing to the main pass. Not worth moving to the CPU.

## Two CPU trims (`1ca118c`, `d19b6d3`)

- **`thread_info` through `getrusage`.** The compat layer's `thread_info(THREAD_BASIC_INFO)` opened `/proc/self/task/<tid>/stat` on every call, and `hle_nvn.c` asks for the calling thread twice per submit. For the calling thread it is now one `getrusage(RUSAGE_THREAD)` call. In the call graph of the render thread, `thread_info` went from 0.72% to 0.04%. PR #32 brings it to `main`.
- **No condition signal for a spinning waiter.** With `DM_HLE_SPIN_US` the guest main thread polls its woke flag before a short wait sleeps; a waker that finds it spinning now skips `pthread_cond_signal`, which on bionic makes a futex system call whether anyone waits or not. In 60 s of camp, 21.8 of 67.6 guest wakes per frame (32%) skipped the signal. These wakes come from the job workers, so the render thread's own share did not change.

## Depth attachments nothing uses (`a0b961c`, diagnostic)

`NVN PERF pass_ends depth_idle` and `dz=1` on `NVN GPASS` lines mark passes that carry a depth attachment no draw tests and no clear touches. The camp has about 7 per frame, but they are tiny (draw-less clears and small blits, 7–36 µs each, about 117 µs a frame together). The large full-screen passes that carry depth do use it. No change follows.

## Full-coverage first draws (`93b7ca7`, `56ada6d`; off by default)

The upstream no-load path was removed because a vertex count and a full viewport do not prove that a draw covers every pixel. `56ada6d` brings back the Metal side's proof (`first_draw_covers` in `nvn_mtl.m`, the pure function `nvn_cover.h` and its unit test `t_cover.c`) for the Vulkan engine. The vertex position must be one input as it is: `93b7ca7` finds that input in the SPIR-V, with a unit test, and on the game's library it accepts the four full-screen vertex shaders and rejects the mesh shaders. The positions are read from the vertex and index snapshots the draw hands to the GPU. A proven first draw begins its pass with a twin render pass whose color `loadOp` is `DONT_CARE`.

In the camp the proof holds for 13 passes a frame (9 more fail on blending, 2 on shaders, 1 on depth). Culling had to be passed to `nvn_cover` as the Metal side does: with culling required off, none qualified. In an in-process A/B over 12 adjacent 10 s phase pairs, GPU on − off was −0.19 ± 0.27 ms, the 13 passes themselves differed by under 5 µs, and the other passes did not move either, so the governor does not explain it away: the driver evidently renders these single-draw passes in place. Screenshots with it always on show the camp unchanged. It stays behind marker `gfx-noload` (env `DM_GFX_NOLOAD`, A/B target `noload`) for GPUs where the load does cost.

## The new touch controls (`android/touch-port`, merged as `c5ba12a`)

`main`'s Android touch series is now on `linux-build`: the community design from upstream #74 (dark socket buttons with a metal rim, the right-thumb skill fan with the game's own skill icons, the I/II skill-set switch on release with slide-out cancel), the native bottom HUD, the overlay contract and its startup check. `linux-build`'s four original touch fixes (`2fa66d5`, `6f37d61`, `722f0db`, `cfd8f56`) gave way to `main`'s versions of them. Two additions: the map button draws its original icon in a socket, as on iOS, and the joystick image carries no caption; `stage_touch_assets.py` now stages the repository's `dm_touch_joy.png`, which nothing put into the overlay before. Tests: `test_touch_snapshot` 13 cases, `t_touch_controls` 48/48, `t_skill_aim` 82/82, the asset and overlay contract checks.

Builds now need `dm_touch_native_art.c` in `DM_GEN_DIR` and an overlay staged from the same source fingerprint (here `2457068d…`); `README.md` describes the generation. On the Redmi the overlay was verified file by file after import, startup logged `overlay verified`, and the camp ran at 60.0 fps with render-thread CPU 9.4–9.7 ms, as before.

## A short free-play session

The user played 2.5 minutes on `a0b961c` with production defaults (dynamic scale on, per-pass profile on). 150 one-second windows: fps median 60.0, 10th percentile 59.9; the 4 windows under 58 fps are all in the first 7 seconds of loading in. GPU 13.9 ms median (maximum 14.6), render-thread CPU 9.0 ms median (maximum 10.3 after loading). The scale stayed at 0.75; the battery went from 35.3 to 37.0 °C.

## A second phone: OnePlus 13 (Adreno 830)

PJZ110, Snapdragon 8 Elite (SM8750), Android 17, 2376 × 1080 at 120 Hz, 4 KB pages. It runs the game from the same files as the Redmi (`06ac8b59…`, pre-touch overlay): character select at 60 fps with GPU about 10.5 ms and render-thread CPU about 5.5 ms; camp at fixed 0.75 at 59.1–59.7 fps, render-thread CPU 10.8–11.2 ms, GPU 13.3–14.2 ms. ColorOS caps its CPUs when the app is not in its Games app (the 4.32 GHz cores at 1.2–1.7 GHz during the run), so its CPU numbers are not yet comparable. ColorOS also refuses `adb install`, so APKs are installed by hand there.
