# Review fixes, 32-lane warps and texture checks on the Redmi K50 Ultra (2026-10-18/19)

中文：[2026-10-19-review-fixes.zh-CN.md](2026-10-19-review-fixes.zh-CN.md)

Measured 21:30 on 10-18 to 06:45 on 10-19 NZDT on the Redmi K50 Ultra (Adreno 730, Vulkan 1.1, Android 15), with Game Turbo on. Build `android/linux-build` with the Android defaults, production dynamic scale, unless noted. The other agent's read-only review (kept in the local handoff file) found the first four items below.

## Two lines of code that had become comments

Both came from my own earlier edits. Appending a trailing comment to a long line swallowed the statements after it on the same line. A scan of every branch for code after a `//` found no other case.

| Fix | Commit, PR | What was lost |
|---|---|---|
| Depth bias | f5a7c75, #3 | `rs.depthBiasSlopeFactor`, `rs.depthBiasClamp` and the clamp feature check. Pipelines got only the constant factor; the slope and clamp the game sets were zero. |
| Snapshot writeback check | 78c2dd7, #10 | `cpu_access(address,bytes,1)` in the snapshot slow path. A pending GPU writeback covering a snapshot only in part, or misaligned for a view, was not landed before the copy. |

No logged window ever needed the second one: every camp and fight log shows 0 access-triggered syncs. So no earlier measurement changes.

## One full barrier per game barrier (PR #20)

`OP_BARRIER` recorded the same `ALL_COMMANDS` memory barrier three times with nothing between them. It now records one: 198 → 66 `vkCmdPipelineBarrier` calls per camp frame. In-process A/B in the camp (target `barrier3`), CPU 10.22 ms with the change against 10.21 ms without, GPU 15.80 against 15.78 ms. No measurable difference: the driver evidently collapses them. It stays as a cleanup.

The same PR fills the descriptor-key tags (they feed only the `keydiff` census) on timed frames only.

## 32-lane warps inside 64-lane subgroups (PR #21)

The shader library translates for Maxwell's 32-lane warp, and Adreno runs 64-lane subgroups. In the warp-aggregated append of D2R's fluid, clutter, grass and PPR compute shaders, lanes 32–63 were missing from the count and all wrote one entry past the reserved range. In the fluid shader (8×8 workgroups, one shared list per workgroup) every list lost its upper 32 lanes' entries.

The engine now rewrites these compute modules as the vendored Ryujinx translator does for hosts wider than 32 lanes, so every 32 lanes act as their own warp (`xbg_spirv_warp.cpp`). Over the 6,931 modules of the library it changes 5 compute modules, and all 5 pass `spirv-val`.

- **A dynamic index into a ballot.** The first version picked the lane's ballot word with `OpVectorExtractDynamic`. The Adreno 730 compiler then failed the fluid pipeline with `VK_ERROR_UNKNOWN`, and the game aborted at start. Constant extracts and a select, as Ryujinx emits, work.
- **What runs.** At the Android settings the camp and the Blood Moor use the fluid and light-bin modules (the light-bin shader's 32-wide workgroups were never affected). Grass, clutter and PPR did not load in either place.
- **Cost.** Per-pass GPU profile, Blood Moor, `DM_GAME_SEED=12345`:

| Fluid accumulation dispatch (32×32 groups) | µs |
|---|---|
| Before (upper lanes lost) | 96 |
| Rewritten | 739 |
| Rewritten, shuffle left 32-lane (diagnostic, wrong) | 468 |

About 370 µs is the shader processing its full lists, as on the Switch. About 270 µs is the leader broadcast in the append: each half now has its own leader, so the shuffle index is no longer uniform across the subgroup, and a non-uniform shuffle costs much more on this GPU. The camp stays at 60.0 fps at scale 0.75 (GPU 15.8 ms). The Blood Moor screenshots look the same by eye. Marker `gfx-warp32` = 0 turns the rewrite off.

## Texture checks through the dirty-page tracker (PR #22)

`get_image` re-hashed every CPU-uploaded texture a draw bound: on its first 4 checks, then on every 16th submission. In a Den of Evil fight `hash_memory` was 2.2% of the game thread, about 0.27 ms a frame. A streamed write into a texture still being sampled could also show stale texels for up to 16 submissions.

The iOS backend (`nvn_mtl.m` `tex_mtl`) already uses the write-protect tracker for this, and the engine now does the same. An upload write-protects the texture's pages and keeps the tracker's stamp. A check is one generation compare when nothing faulted since, and hashes only after a guest or host write to its pages. Untracked pools keep the old hash checks. In the steady camp no check needed a hash (`NVN PERF texdirty checks=0`).

Den of Evil fight (`DM_GAME_SEED=12345`), in-process A/B (target `texdirty`), steady windows of the same scene: render thread 8.47 ms with the tracker against 8.59 ms with the hash checks, so about 0.1 ms less per frame. The `hash` part on timed frames roughly halves. The rest of `hash_memory` is the texel-buffer verification, which this does not change. PR #22.

## Command pools reset on another thread (PR #23)

`vkResetCommandPool` ran on the render thread right before each slot was recorded again. Now a reset thread waits for the slot's fence after the submission and resets the pool. Retiring the slot waits for that job first, so the fence reset never overlaps the thread's wait.

Den of Evil fight, in-process A/B (target `poolreset`), steady windows:

| | Windows | Render thread CPU ms | Present ms | GPU ms |
|---|---|---|---|---|
| Reset on the thread | 40 | 7.83 | 1.80 | 14.70 |
| Reset in place (before) | 50 | 8.05 | 1.95 | 14.77 |

No retirement had to wait for a reset (`NVN PERF poolreset waits=0`). The fight is at the 60 fps pacing cap, so the 0.22 ms comes off the render thread's frame (where the long frames are) rather than showing in fps.

## Pinning the game thread to the A710 cores: cooler, but slower

Soaks in the camp with the production defaults, from a cool phone until 40 °C or 12 minutes:

| Pin | fps | CPU ms | GPU ms | Battery temperature |
|---|---|---|---|---|
| X2 (cpu7, the default) | 60.0 | 10.3–10.9 | 15.3–15.6 | 30 → 40 °C in 423 s |
| A710 (cpu4–6, `gfx-pin-cpu` = 70) | 50.1–52.2 | 13.4–13.9 | 16.3–16.7 | 28 → 39.6 °C in 727 s |

The A710 pin heats about 20% more slowly but costs 8–10 fps, so the X2 pin stays the default.

## Save commits are off the game thread

An fs commit clones the save tree, 17–25 ms each, about every 5 minutes in the camp. No long frame lines up with one, so they run on another thread and are not a source of hitches.
