// App 形态的 JNI 入口（Java 侧 app/src/main/java/com/xiaoboluo/d2r/DmActivity.java）：对应 iOS 的 main.m 启动链 + front_ios.m 的生命周期 / 触屏转发。
//   nativeStart：备好目录与环境变量（同 iOS boot_prepare：DM_ROMFS / DM_SAVE / DM_TEMP、cwd），stderr 接到日志文件 + logcat，起宿主线程跑 dm_main。
//   nativeSurface：交来 / 收回 ANativeWindow（front_android.c）；nativeTouch：整组触点（front.h dm_touch_set）；nativeFocus：场景生命周期（同 iOS 1 / 2 / 3 + 后台挂起）。
// 目录约定（都不进 APK，见 dm/android/README）：
//   <外部私有目录>/dm/out/   改写产物（*.patched.bin 与各表，patch_nso.py DM_TARGET=android 出），开发时 adb push
//   资源目录（缺省 /sdcard/Download/data）  玩家的 18 个 .data 档案 + .build.config；App 在内部目录建 romfs/data/ 逐个链接过去
//   <内部目录>/save、temp            存档与临时文件
// JNI entry of the App form (Java side app/src/main/java/com/xiaoboluo/d2r/DmActivity.java): counterpart of iOS's main.m startup chain + front_ios.m's lifecycle / touch forwarding.
//   nativeStart: prepare directories and environment variables (as iOS boot_prepare: DM_ROMFS / DM_SAVE / DM_TEMP, cwd), connect stderr to a log file + logcat, start the host thread running dm_main.
//   nativeSurface: hand over / take back the ANativeWindow (front_android.c); nativeTouch: the whole set of touch points (front.h dm_touch_set); nativeFocus: scene lifecycle (as iOS 1 / 2 / 3 + background suspend).
// Directory conventions (none of these go into the APK, see dm/android/README):
//   <external private dir>/dm/out/   rewriter products (*.patched.bin and tables, from patch_nso.py DM_TARGET=android), adb push during development
//   resource dir (default /sdcard/Download/data)  the player's 18 .data archives + .build.config; the App creates romfs/data/ in the internal dir linking to each of them
//   <internal dir>/save, temp        saves and temporary files
#include "../host/dm.h"
#include "../host/front.h"
#include "../host/hle.h"
#include <android/log.h>
#include <android/native_window_jni.h>
#include <dirent.h>
#include <errno.h>
#include <jni.h>
#include "vulkan_present.h"
#include "touch_android.h"
#include "lifecycle_dispatch.h"
#include <sys/stat.h>
#include <unistd.h>

void front_android_set_window(void *win, int w, int h);
void front_android_set_refresh(float hz);
int dm_main(int argc, char **argv);
void front_android_boot_failed(int code);

static char g_files[1024], g_ext[1024], g_src[1024], g_runtime[1100];
static FILE *g_logf;

// 挂起只能在原生pthread栈上执行；UI只登记最新状态，不能等正在等待Surface的呈现调用。
// Suspend only on a native pthread stack; UI posts latest state without waiting for a present that needs a Surface.
static dm_lifecycle_dispatch g_lifecycle;
static pthread_once_t g_lifecycle_once = PTHREAD_ONCE_INIT;
static int g_lifecycle_error;
static void lifecycle_graphics(int on) {
    __android_log_print(ANDROID_LOG_INFO, "dm-lifecycle", "graphics suspend=%d begin", on);
    dm_gfx_suspend(on);
    __android_log_print(ANDROID_LOG_INFO, "dm-lifecycle", "graphics suspend=%d complete", on);
}
static void lifecycle_init(void) {
    dm_lifecycle_ops ops = {dm_focus_set, dm_audio_suspend, lifecycle_graphics};
    g_lifecycle_error = dm_lifecycle_start(&g_lifecycle, ops);
    if (!g_lifecycle_error) pthread_setname_np(g_lifecycle.thread, "dm-lifecycle");
    if (g_lifecycle_error) __android_log_print(ANDROID_LOG_ERROR, "dm-lifecycle", "worker start failed rc=%d", g_lifecycle_error);
}
static int lifecycle_ready(void) { pthread_once(&g_lifecycle_once, lifecycle_init); return !g_lifecycle_error; }

// stderr → 管道 → 本线程：逐行写日志文件（adb pull 取）并转 logcat（标签 dm）
// stderr → pipe → this thread: write line by line to the log file (fetched with adb pull) and forward to logcat (tag dm)
static void *log_pump(void *arg) {
    int fd = (int)(intptr_t)arg; char buf[4096]; size_t have = 0; ssize_t k;
    while ((k = read(fd, buf + have, sizeof buf - 1 - have)) > 0) {
        have += (size_t)k; buf[have] = 0; char *s = buf, *nl;
        while ((nl = strchr(s, '\n'))) { *nl = 0; __android_log_write(ANDROID_LOG_INFO, "dm", s); if (g_logf) { fputs(s, g_logf); fputc('\n', g_logf); fflush(g_logf); } s = nl + 1; }
        have = strlen(s); memmove(buf, s, have);
        if (have == sizeof buf - 1) { __android_log_write(ANDROID_LOG_INFO, "dm", buf); if (g_logf) { fputs(buf, g_logf); fflush(g_logf); } have = 0; }
    }
    return NULL;
}
static void log_init(void) {
    static int initialized; if (initialized) return; initialized = 1;
    char p[1200]; snprintf(p, sizeof p, "%s/boot.log", g_runtime); g_logf = fopen(p, "a");
    int fds[2]; if (pipe(fds)) return;
    dup2(fds[1], 2); close(fds[1]); setvbuf(stderr, NULL, _IOLBF, 0);
    pthread_t th; if (!pthread_create(&th, NULL, log_pump, (void *)(intptr_t)fds[0])) pthread_detach(th);
}

static void mkdirs(const char *p) {
    char b[1100]; snprintf(b, sizeof b, "%s", p);
    for (char *s = b + 1; *s; s++) if (*s == '/') { *s = 0; mkdir(b, 0700); *s = '/'; }
    mkdir(b, 0700);
}
// 内部目录 romfs/data/ 下逐个链接到资源目录的文件（游戏按 rom:/data/… 读；链接让资源留在玩家放的位置，不复制 5 GB）
// Under the internal dir romfs/data/, link each file of the resource dir (the game reads rom:/data/...; links keep the resources where the player put them instead of copying 5 GB)
// 剔零档案的「.holes」区段表也要链过去：hle_fs.c 在档案旁边（即 romfs/data/ 下）找它；缺了就按物理偏移读剔过零的档案，第一个零段之后读到的全错位（解压失败、随后崩溃）
// The ".holes" extent tables of zero-stripped archives are linked too: hle_fs.c looks for them next to the archive (in romfs/data/); without them a zero-stripped archive is read at physical offsets, and everything after its first stripped range comes from the wrong place (decompression failures, then a crash)
static int link_romfs(const char *romfs) {
    char data[1100]; snprintf(data, sizeof data, "%s/data", romfs); mkdirs(data);
    DIR *d = opendir(g_src); if (!d) { LOG("【红】资源目录打不开 %s：errno %d（没给「所有文件访问」权限？）", g_src, errno); return -1; }
    struct dirent *e; int n = 0;
    while ((e = readdir(d))) {
        const char *nm = e->d_name; size_t l = strlen(nm);
        if (!(l > 5 && !strcmp(nm + l - 5, ".data")) && !(l > 11 && !strcmp(nm + l - 11, ".data.holes")) && strcmp(nm, ".build.config") && strcmp(nm, "build.config")) continue;
        char from[1100], to[1100]; snprintf(from, sizeof from, "%s/%s", g_src, nm);
        snprintf(to, sizeof to, "%s/%s", data, strcmp(nm, "build.config") ? nm : ".build.config");   // 电脑上常被存成无点名 | often saved without the leading dot on a computer
        unlink(to); if (!symlink(from, to)) n++;
    }
    closedir(d);
    return n;
}

static void *host_thread(void *arg) {
    (void)arg;
    char out[1200]; snprintf(out, sizeof out, "%s/out", g_runtime);
    // 图形初始化在宿主线程进行；着色器由玩家在包外提供，失败时停止启动。
    // Initialize graphics on the host thread; shaders stay external, and failure stops startup.
    char shaders[1200], error[512] = {0};
    snprintf(shaders, sizeof shaders, "%s/shaders", g_runtime);
    int graphics_rc = dm_android_vk_initialize(shaders, error, sizeof error);
    if (graphics_rc) { LOG("boot: Vulkan initialization failed %d: %s", graphics_rc, error); front_android_boot_failed(-6); return NULL; }
    char *argv[] = {"dmhost", out, "main", "sdk", NULL};
    LOG("boot: dm_main %s", out);
    int rc = dm_main(4, argv);
    LOG("boot: dm_main 返回 %d", rc);
    front_android_boot_failed(-7);
    return NULL;
}

// 测试用：files/dm/env 每行 KEY=VALUE（# 开头为注释），在下面的缺省值之前设进环境，让 run-as 能改 DM_* 开关（iOS / Mac 用启动环境变量做同样的事）；没有这个文件就什么都不做
// For testing: each KEY=VALUE line of files/dm/env (# starts a comment) is set before the defaults below, so run-as can change DM_* switches (iOS / Mac do the same with launch environment variables); without the file nothing happens
static void load_env_file(const char *dir) {
    char path[1300]; snprintf(path, sizeof path, "%s/env", dir);
    FILE *fp = fopen(path, "r"); if (!fp) return;
    char line[512];
    while (fgets(line, sizeof line, fp)) {
        line[strcspn(line, "\r\n")] = 0;
        char *eq = strchr(line, '=');
        if (!line[0] || line[0] == '#' || !eq || eq == line) continue;
        *eq = 0; setenv(line, eq + 1, 1); LOG("boot: 环境 %s=%s（来自 %s）", line, eq + 1, path);
    }
    fclose(fp);
}

JNIEXPORT jint JNICALL Java_com_xiaoboluo_d2r_DmActivity_nativeStart(JNIEnv *env, jclass cls, jstring files, jstring ext, jstring src, jstring lang) {
    (void)cls;
    static int started; if (started) return 0;
    const char *f = (*env)->GetStringUTFChars(env, files, NULL), *x = (*env)->GetStringUTFChars(env, ext, NULL), *s = (*env)->GetStringUTFChars(env, src, NULL), *lg = (*env)->GetStringUTFChars(env, lang, NULL);
    snprintf(g_files, sizeof g_files, "%s", f); snprintf(g_ext, sizeof g_ext, "%s", x); snprintf(g_src, sizeof g_src, "%s", s);
    // 启动时间线零点：Process.getStartUptimeMillis() 是 SystemClock.uptimeMillis（CLOCK_MONOTONIC），与 now_ns 同一时钟（见 main.c proc_start_mono_ns）
    // Boot timeline zero: Process.getStartUptimeMillis() is SystemClock.uptimeMillis (CLOCK_MONOTONIC), the same clock as now_ns (see main.c proc_start_mono_ns)
    jclass proc = (*env)->FindClass(env, "android/os/Process");
    g_proc_start_ns = (uint64_t)(*env)->CallStaticLongMethod(env, proc, (*env)->GetStaticMethodID(env, proc, "getStartUptimeMillis", "()J")) * 1000000ull;
    // 调试包可用 run-as 导入内部目录；有完整主程序时优先选内部目录。
    // Debug builds can import via run-as; prefer internal storage when the main module exists.
    char p[1200]; snprintf(g_runtime, sizeof g_runtime, "%s/dm", g_files);
    snprintf(p, sizeof p, "%s/out/main.patched.bin", g_runtime);
    if (access(p, R_OK)) snprintf(g_runtime, sizeof g_runtime, "%s/dm", g_ext);
    mkdirs(g_runtime);
    log_init();
    load_env_file(g_runtime);
    LOG("boot: 内部目录 %s，外部目录 %s，资源 %s，系统语言 %s", g_files, g_ext, g_src, lg);
    char romfs[1100], save[1100], temp[1100];
    snprintf(romfs, sizeof romfs, "%s/romfs", g_files); snprintf(save, sizeof save, "%s/save", g_files); snprintf(temp, sizeof temp, "%s/temp", g_files);
    mkdirs(save); mkdirs(temp);
    int n = link_romfs(romfs);
    LOG("boot: 资源链接 %d 个文件", n);
    setenv("DM_ROMFS", romfs, 0); setenv("DM_SAVE", save, 0); setenv("DM_TEMP", temp, 0);
    // 被更新的 GPU 结果覆盖、又只有 GPU 读的存储写回不做（引擎的 gfx-wb-skip；标记文件仍优先）：营地每帧省 1.37 MB、约 1 ms 的回读。
    // Storage writebacks that a newer GPU result supersedes and only the GPU reads are dropped (the engine's gfx-wb-skip; a
    // marker file still wins): 1.37 MB and about 1 ms of readback per camp frame.
    setenv("DM_GFX_WB_SKIP", "1", 0);
    // 绘制状态与上一次绘制相同时直接沿用上一次的管线，不再序列化建管线信息去查（引擎的 gfx-pipe-memo；标记文件仍优先）。
    // A draw whose pipeline state equals the previous draw's reuses its pipeline without serializing the create info for a
    // lookup (the engine's gfx-pipe-memo; a marker file still wins).
    setenv("DM_GFX_PIPE_MEMO", "1", 0);
    // Vulkan 适配器先沿用同步提交；上游新增翻译线程需独立验证对象生命周期。
    // Keep synchronous submission until the new upstream translation thread's object lifetimes are verified with Vulkan.
    setenv("DM_NVN_SYNC", "1", 0);
    // BC 贴图按原格式采样（引擎的 gfx-native-bc；设备支持时；标记文件仍优先）：Redmi K50 Ultra 营地 GPU 少 0.5 ms，图像内存 827 → 320 MB。
    // BC textures sampled in their own formats where the device can (the engine's gfx-native-bc; a marker file still wins):
    // on the Redmi K50 Ultra camp GPU -0.5 ms and live image memory 827 -> 320 MB.
    setenv("DM_GFX_NATIVE_BC", "1", 0);
    // 片元着色器的四元组邻居交换改用精细导数（引擎的 gfx-quad-derivatives；标记文件仍优先）：Adreno 上辅助调用在子组洗牌里取到垃圾，
    // 罪恶洞穴打斗时单位周围出现黑方块。
    // Fragment-shader quad-neighbour shuffles go through fine derivatives (the engine's gfx-quad-derivatives; a marker file
    // still wins): on Adreno helper invocations read garbage in subgroup shuffles, and black squares appeared around units in
    // the Den of Evil fight.
    setenv("DM_GFX_QUAD_DERIVATIVES", "1", 0);
    // 按 32 道线程束翻译的计算着色器，在 64 道的子组里每 32 道各当一个线程束（引擎的 gfx-warp32；标记文件仍优先）：Adreno 上
    // 杂物、草地、流体与 PPR 着色器的线程束聚合追加丢了 32–63 号车道、条目互相覆盖。
    // Compute shaders translated for 32-lane warps run every 32 lanes of a 64-lane subgroup as their own warp (the engine's
    // gfx-warp32; a marker file still wins): on Adreno the clutter, grass, fluid and PPR shaders' warp-aggregated appends lost
    // lanes 32-63 and overwrote entries.
    setenv("DM_GFX_WARP32", "1", 0);
    // 帧重叠缺省开：命令缓冲里的 SignalEvent 不提交、留到所在槽退役时再公布（引擎的 gfx-defer-events = 2；标记文件仍优先）。
    // 与呈现端的 gfx-present-overlap 一起，Redmi K50 Ultra 营地从 7.7 fps 到 19.5 fps（PR #9）。
    // Frame overlap on by default: command-buffer SignalEvents do not submit and are published when their slot retires (the
    // engine's gfx-defer-events = 2; a marker file still wins). With the presenter's gfx-present-overlap the Redmi K50 Ultra
    // camp went from 7.7 to 19.5 fps (PR #9).
    setenv("DM_GFX_DEFER_EVENTS", "2", 0);
    // 顶点与索引的跨帧副本缺省开（引擎的 gfx-vcache；靠 nvn_dirty.c 的写保护脏页跟踪，标记文件仍优先）：营地 CPU 少约 0.7 ms。
    // Cross-frame vertex and index copies on by default (the engine's gfx-vcache, through nvn_dirty.c's write-protect
    // tracking; a marker file still wins): about 0.7 ms less CPU per camp frame on the Redmi K50 Ultra.
    setenv("DM_GFX_VCACHE", "1", 0);
    if (!getenv("DM_LANG") && *lg) setenv("DM_LANG", lg, 0);   // 首次安装的语言按系统首选（同 iOS CFLocaleCopyPreferredLanguages，见 dm_i18n.c） | first-install language follows the system preference (as iOS CFLocaleCopyPreferredLanguages, see dm_i18n.c)
    if (chdir(g_files)) LOG("【红】chdir %s 失败 errno %d", g_files, errno);
    (*env)->ReleaseStringUTFChars(env, files, f); (*env)->ReleaseStringUTFChars(env, ext, x); (*env)->ReleaseStringUTFChars(env, src, s); (*env)->ReleaseStringUTFChars(env, lang, lg);
    if (n <= 0) return -2;
    char required[1300];
    const char *modules[] = {"main.patched.bin", "main.fixups.tsv", "main.gates.tsv", "sdk.patched.bin", "sdk.fixups.tsv", "sdk.gates.tsv"};
    for (unsigned i = 0; i < sizeof modules / sizeof modules[0]; i++) {
        snprintf(required, sizeof required, "%s/out/%s", g_runtime, modules[i]);
        if (access(required, R_OK)) { LOG("boot: missing %s", required); return -3; }
    }
    snprintf(required, sizeof required, "%s/data/.build.config", romfs);
    if (access(required, R_OK)) { LOG("boot: missing build configuration"); return -5; }
    snprintf(required, sizeof required, "%s/shaders", g_runtime);
    DIR *shaders = opendir(required); int have_shader = 0;
    if (shaders) { struct dirent *entry; while ((entry = readdir(shaders))) {
        size_t len = strlen(entry->d_name);
        if (len > 4 && !strcmp(entry->d_name + len - 4, ".xgs")) { have_shader = 1; break; }
    } closedir(shaders); }
    if (!have_shader) { LOG("boot: no shader bundles in %s", required); return -4; }
    snprintf(required, sizeof required, "%s/overlay", g_runtime);
    struct stat overlay_stat;
    if (!stat(required, &overlay_stat) && S_ISDIR(overlay_stat.st_mode)) setenv("DM_ROMFS_OVERLAY", required, 1);
    if (!lifecycle_ready()) return -8;
    pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setstacksize(&a, 8u << 20);
    pthread_t th; int rc = pthread_create(&th, &a, host_thread, NULL); pthread_attr_destroy(&a);
    if (!rc) { started = 1; pthread_detach(th); }
    return rc ? -1 : n;
}

static ANativeWindow *g_win;
JNIEXPORT void JNICALL Java_com_xiaoboluo_d2r_DmActivity_nativeRefresh(JNIEnv *env, jclass cls, jfloat hz) {
    (void)env; (void)cls;
    front_android_set_refresh(hz);
}
JNIEXPORT void JNICALL Java_com_xiaoboluo_d2r_DmActivity_nativeSurface(JNIEnv *env, jclass cls, jobject surface, jint w, jint h) {
    (void)cls;
    ANativeWindow *old = g_win;
    g_win = surface ? ANativeWindow_fromSurface(env, surface) : NULL;
    dm_android_vk_set_window(g_win);
    front_android_set_window(g_win, w, h);
    if (old) ANativeWindow_release(old);
    LOG("前端：画面 %s %dx%d", g_win ? "就绪" : "收回", w, h);
}

// 触点：Java 按手指槽号（最小空槽，抬起前不变）交整组归一化坐标，n = 0 = 全抬起
// Touch points: Java hands over the whole set of normalized coordinates by finger slot number (lowest free slot, unchanged until lifted), n = 0 = all lifted
JNIEXPORT void JNICALL Java_com_xiaoboluo_d2r_DmActivity_nativeTouch(JNIEnv *env, jclass cls, jint n, jintArray ids, jfloatArray xs, jfloatArray ys) {
    (void)cls;
    dm_touch t[16]; if (n > 16) n = 16; if (n < 0) n = 0;
    jint id[16]; jfloat x[16], y[16];
    if (n) { (*env)->GetIntArrayRegion(env, ids, 0, n, id); (*env)->GetFloatArrayRegion(env, xs, 0, n, x); (*env)->GetFloatArrayRegion(env, ys, 0, n, y); }
    for (int i = 0; i < n; i++) t[i] = (dm_touch){(uint32_t)id[i], x[i], y[i]};
    dm_touch_set(t, n);
}

// UI只入队：0失焦，1后台，2前台，3激活；激活也声明前台，合并2/3时不会漏恢复。
// UI posts only: 0 unfocused, 1 background, 2 foreground, 3 active; active also declares foreground so merging 2/3 cannot lose resume.
JNIEXPORT void JNICALL Java_com_xiaoboluo_d2r_DmActivity_nativeFocus(JNIEnv *env, jclass cls, jint ev) {
    (void)env; (void)cls;
    dm_android_touch_focus(ev == 2 || ev == 3);
    if (lifecycle_ready()) {
        dm_lifecycle_post(&g_lifecycle, ev);
        __android_log_print(ANDROID_LOG_INFO, "dm-lifecycle", "UI posted event=%d", ev);
    }
}
