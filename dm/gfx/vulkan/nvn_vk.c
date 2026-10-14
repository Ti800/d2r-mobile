// NVN 空设备的 Vulkan 执行层（安卓）：实现与 Metal 回放层 nvn/nvn_mtl.m 同一套 mtl_* 接口（声明在 nvn/nvn_mtl.h、nvn/nvn_obj.h），接口层 nvn/hle_nvn.c 原样复用。
// 当前阶段 = 无图形的「CPU 可见语义」执行层：命令流照 nvn_mtl.m 的 mtl_exec 走一遍，只做游戏读得到的效果——
//   缓冲拷贝 / 常数缓冲更新按提交顺序执行；事件写值、计数器记录在「提交完成」时按顺序写（与 Metal 层同一约定：GPU 完成的效果只在完成之后可见）；
//   栅栏在提交时即完成；出图只计数；遮挡查询报「有像素通过」（不画图时让游戏把物体都当可见，避免按全遮挡裁掉逻辑）。
// 用于在没有 GPU 后端时先验证运行库（装载、线程、文件、音频、游戏逻辑）；真正的绘制在后续阶段按 nvn_mtl.m 逐项补上。
// Vulkan execution layer of the NVN null device (Android): implements the same mtl_* interface as the Metal replay layer nvn/nvn_mtl.m (declared in nvn/nvn_mtl.h, nvn/nvn_obj.h); the interface layer nvn/hle_nvn.c is reused as is.
// Current stage = a graphics-less "CPU-visible semantics" execution layer: the command stream is walked as in nvn_mtl.m's mtl_exec, doing only the effects the game can read --
//   buffer copies / constant buffer updates run in submission order; event writes and counter records are written in order at "submission complete" (the same convention as the Metal layer: effects of GPU completion are visible only after completion);
//   fences complete at submit; presents are only counted; occlusion queries report "samples passed" (when nothing is drawn, the game treats objects as visible instead of culling logic as fully occluded).
// Used to validate the runtime (loading, threads, files, audio, game logic) before the GPU backend exists; real drawing is added item by item following nvn_mtl.m in later stages.
#include "../../host/dm.h"
#include "../nvn/nvn_obj.h"
#include "../nvn/nvn_mtl.h"
#include "../../host/front.h"
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>

#define VLOG(...) LOG("[vk] " __VA_ARGS__)

static uint64_t vk_now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// 计数器时间戳：与 Metal 层同一换算（Switch GPU 计时器 19.2 MHz：ns × 12 / 625）
// Counter timestamp: the same conversion as the Metal layer (Switch GPU timer 19.2 MHz: ns × 12 / 625)
static uint64_t gpu_ticks(void) { return vk_now_ns() * 12 / 625; }

void mtl_init(void) { static int done; if (done) return; done = 1; VLOG("执行层：无图形（只执行 CPU 可见效果）"); }
void bf_want_note(const char *key) { (void)key; }

// ---------- 内存池：GPU 地址 = CPU 地址；只记范围供缓冲拷贝核对 ----------
// ---------- Memory pools: GPU address = CPU address; only the ranges are recorded so buffer copies can be checked ----------
#define POOL_MAX 4096
static struct { uint64_t lo, hi; } g_pools[POOL_MAX];
static int g_npool;
static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
void mtl_pool_add(const void *mem, uint64_t size) {
    pthread_mutex_lock(&g_pool_mu);
    if (g_npool == POOL_MAX) die("内存池超过 %d 个", POOL_MAX);
    g_pools[g_npool].lo = (uint64_t)mem; g_pools[g_npool].hi = (uint64_t)mem + size; g_npool++;
    pthread_mutex_unlock(&g_pool_mu);
}
void mtl_pool_del(const void *mem) {
    pthread_mutex_lock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].lo == (uint64_t)mem) { g_pools[i] = g_pools[--g_npool]; break; }
    pthread_mutex_unlock(&g_pool_mu);
}
static int in_pool(uint64_t a, uint64_t n) {
    int ok = 0; pthread_mutex_lock(&g_pool_mu);
    for (int i = 0; i < g_npool && !ok; i++) ok = a >= g_pools[i].lo && a + n <= g_pools[i].hi;
    pthread_mutex_unlock(&g_pool_mu); return ok;
}
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) { *buf = NULL; *off = addr; return in_pool(addr, len); }

// ---------- 提交：操作按录制顺序排进本次提交，提交完成时依次生效 ----------
// ---------- Submission: operations are queued into this submission in recording order and take effect in turn when it completes ----------
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t a, b, n; } post_op;   // kind：1 事件 2 计数器 3 拷贝 | kind: 1 event 2 counter 3 copy
static post_op *g_post; static int g_npost, g_postcap, g_in_sub;
static uint64_t g_fence, g_done;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_cnt_report[32], g_cnt_reset[32];
static uint64_t g_stat_draw, g_stat_dispatch;
static void post(post_op o) {
    if (g_npost == g_postcap) { g_postcap = g_postcap ? g_postcap * 2 : 64; g_post = realloc(g_post, (size_t)g_postcap * sizeof *g_post); if (!g_post) die("提交操作表"); }
    g_post[g_npost++] = o;
}
void mtl_sub_begin(void) { mtl_init(); if (g_in_sub) die("提交重入"); g_in_sub = 1; g_npost = 0; }
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    if (!in_pool(src, n) || !in_pool(dst, n)) die("缓冲拷贝地址不在任何内存池内 src=%#llx dst=%#llx n=%#llx", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n);
    post((post_op){3, NULL, 0, 0, src, dst, n});
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0, 0}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { post((post_op){2, NULL, 0, 0, addr, type, 0}); }
uint64_t mtl_sub_commit(void) {
    uint64_t fence = ++g_fence;
    for (int i = 0; i < g_npost; i++) {
        post_op *o = &g_post[i];
        if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
        else if (o->kind == 3) memmove((void *)o->b, (const void *)o->a, (size_t)o->n);
        else {   // 计数器记录 {值, 时间戳}：通过采样数报 1（可见），时间戳类报当前时刻 | Counter record {value, timestamp}: samples passed reports 1 (visible), timestamp types report the current time
            uint64_t t = gpu_ticks(), rec[2] = {o->b == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 1 : (o->b == 0 || o->b == 0x10) ? t : 0, t};
            memcpy((void *)o->a, rec, sizeof rec);
        }
    }
    g_npost = 0; g_in_sub = 0;
    pthread_mutex_lock(&g_mu); g_done = fence; pthread_cond_broadcast(&g_cv); pthread_mutex_unlock(&g_mu);
    return fence;
}
uint64_t mtl_last_fence(void) { return g_fence; }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) { (void)timeout_ns; pthread_mutex_lock(&g_mu); int ok = g_done >= fence; pthread_mutex_unlock(&g_mu); return ok; }
void mtl_finish(void) {}
void mtl_drain(void) {}

// ---------- 命令流：与 nvn_mtl.m mtl_exec 同一套记录格式 ----------
// ---------- Command stream: the same record format as nvn_mtl.m mtl_exec ----------
void mtl_exec(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const uint8_t *a = p + 8;
        switch (h.op) {
            case OP_END: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); mtl_exec((const uint8_t *)t); break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); mtl_sub_event(val, m, v); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); g_cnt_report[t & 31]++; mtl_sub_counter(ad, (uint32_t)t); break; }
            case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); g_cnt_reset[t & 31]++; break; }
            case OP_COPY_B2B: { uint64_t s, d, n; memcpy(&s, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); mtl_sub_copy(s, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); post((post_op){3, NULL, 0, 0, (uint64_t)(a + 16), d, n}); break; }
            case OP_DRAW: case OP_DRAW_IDX: g_stat_draw++; break;
            case OP_DISPATCH: g_stat_dispatch++; break;
            default: break;   // 绘制状态、贴图拷贝 / 清除：无图形时没有游戏可见效果 | Draw state, texture copies / clears: no game-visible effect without graphics
        }
        p += h.size;
    }
}
void mtl_exec_rest(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }

// ---------- 贴图 / 采样器 / 程序：无图形时不建对象 ----------
// ---------- Textures / samplers / programs: no objects are created without graphics ----------
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) { (void)id; (void)t; (void)v; }
void mtl_reg_samp(int id, const samp_par *p) { (void)id; (void)p; }
void mtl_set_tex_pool_n(int n) { (void)n; }
void mtl_prog_release(prog_t *p) { (void)p; }
void mtl_tex_release(tex_t *t) { (void)t; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    (void)t; (void)level; (void)layer0; (void)img;
    static int once; if (!once++) VLOG("【红】CPU 读贴图（无图形，读出全 0）");
    if (r && r->height > 0) memset(dst, 0, row * (size_t)r->height * (size_t)(r->depth > 0 ? r->depth : 1));
}

// ---------- 出图：只计数；票号即刻完成 ----------
// ---------- Present: counted only; tickets complete immediately ----------
static uint64_t g_present_n;
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) { (void)t; (void)period; (void)crop; return ++g_present_n; }
void mtl_present_wait(uint64_t tk) { (void)tk; }
void mtl_present_flush(void) {}
double mtl_presented_time(uint64_t n) { (void)n; return 0; }
uint64_t mtl_present_count(void) { return g_present_n; }
int mtl_has_window(void) { return front_metal_layer() != NULL; }
int mtl_display_hz(void) { return 0; }
__attribute__((weak)) int dm_fps_setting(void) { return 60; }
int mtl_fps_target(void) { return dm_fps_setting() == 30 ? 30 : 60; }
int mtl_link_active(void) { return 0; }
int mtl_link_wait_tick(uint64_t timeout_ns) { (void)timeout_ns; return 0; }
int mtl_warm_wait(int timeout_ms) { (void)timeout_ms; return 0; }

// ---------- 计时与统计（无图形时全 0） ----------
// ---------- Timing and statistics (all 0 without graphics) ----------
static _Atomic uint64_t g_cur_frame;
void mtl_set_frame(uint64_t f) { atomic_store(&g_cur_frame, f); }
void mtl_ub_present(double iv) { (void)iv; }
double mtl_gpu_ms(uint64_t f) { (void)f; return -1; }
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = 0; *draws = g_stat_draw; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = *reuse = 0; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = *skip = 0; }
uint64_t mtl_ds_omit(void) { return 0; }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = *store = *full = 0; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = *reuploads = 0; }
uint64_t mtl_pkg_miss(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return 0; }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = *pso = 0; }
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { *lib_ns = *pso_ns = 0; }
void mtl_alloc_stats(uint64_t *waits, uint64_t *grown, int *n) { *waits = *grown = 0; *n = 0; }
void mtl_counter_stats(uint64_t rep[32], uint64_t rst[32]) { memcpy(rep, g_cnt_report, sizeof g_cnt_report); memcpy(rst, g_cnt_reset, sizeof g_cnt_reset); }
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb, int heap_ok, unsigned long long heap_mb, unsigned long long heap_free_mb, unsigned long long heap_maxblk_kb) {
    (void)frame; (void)rss_mb; (void)heap_ok; (void)heap_mb; (void)heap_free_mb; (void)heap_maxblk_kb;
}
void mtl_res_snapshot(mtl_res_count *out) { memset(out, 0, sizeof *out); out->pools = (uint32_t)g_npool; }
void mtl_res_counts(uint32_t tick, const char *why) { VLOG("资源 拍 %u（%s）：内存池 %d，绘制 %llu，派发 %llu", tick, why, g_npool, (unsigned long long)g_stat_draw, (unsigned long long)g_stat_dispatch); }
void mtl_ui_ghost(int on, float px, float py, const float grid[4], float cx, float cy) { (void)on; (void)px; (void)py; (void)grid; (void)cx; (void)cy; }
int mtl_ui_ghost_rect(float r[4]) { (void)r; return 0; }
int dm_nvn_ruler_truth(void) { return 0; }
void dm_gfx_shot_at(long k, const char *path) { (void)k; VLOG("截图（无图形，跳过）：%s", path); }
