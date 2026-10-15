// 系统输入框桥：只阻塞客体线程；主线程确认、取消或离开页面都会解除等待。
// System text-input bridge: only the guest waits; UI completion or leaving the page releases it.
#include "../host/dm.h"
#include "../host/dm_actions.h"
#include <jni.h>
#include <stdatomic.h>

static JavaVM *vm;
static jclass activity_class;
static jmethodID show_input, report_failure;
static pthread_mutex_t input_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t input_done = PTHREAD_COND_INITIALIZER;
static jlong next_request, active_request;
static int pending, result, output_capacity;
static char *output;

extern int g_swkbd_name_panel, g_swkbd_numeric_max;
extern _Atomic int g_fe_name_no_accept;
size_t swkbd_u8_to_u16(const char *, size_t, uint16_t *, size_t);
size_t swkbd_u16_to_u8(const uint16_t *, size_t, char *, size_t);

JNIEXPORT jint JNI_OnLoad(JavaVM *java_vm, void *reserved) {
    (void)reserved;
    JNIEnv *env;
    if ((*java_vm)->GetEnv(java_vm, (void **)&env, JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;
    jclass local = (*env)->FindClass(env, "com/xiaoboluo/d2r/DmActivity");
    if (!local) return JNI_ERR;
    activity_class = (*env)->NewGlobalRef(env, local);
    (*env)->DeleteLocalRef(env, local);
    if (!activity_class) return JNI_ERR;
    show_input = (*env)->GetStaticMethodID(env, activity_class, "showNativeInput",
        "(JLjava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;III)V");
    report_failure = (*env)->GetStaticMethodID(env, activity_class, "showNativeFailure", "(I)V");
    if (!show_input || !report_failure) return JNI_ERR;
    vm = java_vm;
    return JNI_VERSION_1_6;
}

// JNI 的 NewStringUTF 不是标准 UTF-8；沿用宿主转换，保留中文及代理对。
// NewStringUTF uses modified UTF-8; use host conversion to preserve CJK and surrogate pairs.
static jstring utf8_string(JNIEnv *env, const char *text) {
    if (!text) text = "";
    size_t capacity = strlen(text) + 1;
    uint16_t *wide = calloc(capacity, sizeof *wide);
    if (!wide) return NULL;
    size_t count = swkbd_u8_to_u16(text, capacity, wide, capacity);
    jstring value = (*env)->NewString(env, wide, (jsize)count);
    free(wide);
    return value;
}

static JNIEnv *thread_env(int *attached) {
    JNIEnv *env = NULL; *attached = 0;
    if (!vm) return NULL;
    jint status = (*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6);
    if (status == JNI_EDETACHED) {
        if ((*vm)->AttachCurrentThread(vm, &env, NULL) != JNI_OK) return NULL;
        *attached = 1;
    } else if (status != JNI_OK) return NULL;
    return env;
}

void front_android_boot_failed(int code) {
    int attached; JNIEnv *env = thread_env(&attached);
    if (!env) return;
    (*env)->CallStaticVoidMethod(env, activity_class, report_failure, code);
    if ((*env)->ExceptionCheck(env)) { (*env)->ExceptionDescribe(env); (*env)->ExceptionClear(env); }
    if (attached) (*vm)->DetachCurrentThread(vm);
}

int front_android_text_input(const char *title, const char *sub, const char *guide,
                             const char *initial, int max_chars, char *out, int out_cap) {
    if (!out || out_cap <= 0) return -1;
    int attached; JNIEnv *env = thread_env(&attached);
    if (!env) return -1;
    pthread_mutex_lock(&input_lock);
    if (active_request) { pthread_mutex_unlock(&input_lock); if (attached) (*vm)->DetachCurrentThread(vm); return 0; }
    jlong id = active_request = ++next_request;
    pending = 1; result = 0; output = out; output_capacity = out_cap; out[0] = 0;
    pthread_mutex_unlock(&input_lock);
    jstring strings[] = {utf8_string(env, title), utf8_string(env, sub), utf8_string(env, guide), utf8_string(env, initial)};
    int valid = strings[0] && strings[1] && strings[2] && strings[3];
    if (valid) (*env)->CallStaticVoidMethod(env, activity_class, show_input, id,
        strings[0], strings[1], strings[2], strings[3], max_chars, g_swkbd_name_panel, g_swkbd_numeric_max);
    if ((*env)->ExceptionCheck(env)) { (*env)->ExceptionDescribe(env); (*env)->ExceptionClear(env); valid = 0; }
    for (unsigned i = 0; i < 4; i++) if (strings[i]) (*env)->DeleteLocalRef(env, strings[i]);
    pthread_mutex_lock(&input_lock);
    if (!valid) pending = 0;
    while (pending) pthread_cond_wait(&input_done, &input_lock);
    int accepted = result;
    output = NULL; active_request = 0;
    pthread_mutex_unlock(&input_lock);
    if (attached) (*vm)->DetachCurrentThread(vm);
    if (accepted && g_swkbd_name_panel && !out[0]) {
        accepted = dm_name_gen(out, out_cap, dm_name_taken);
        if (accepted) atomic_store(&g_fe_name_no_accept, 1);
    }
    return accepted;
}

JNIEXPORT void JNICALL Java_com_xiaoboluo_d2r_DmActivity_nativeTextResult(
    JNIEnv *env, jclass cls, jlong id, jboolean accepted, jstring text) {
    (void)cls;
    pthread_mutex_lock(&input_lock);
    // 延迟到达的旧对话框回调不能写入下一次请求的缓冲区。
    // A stale dialog callback must never write into the next request's buffer.
    if (pending && active_request == id) {
        result = accepted && text;
        if (result) {
            const jchar *wide = (*env)->GetStringChars(env, text, NULL);
            if (wide) { swkbd_u16_to_u8(wide, (*env)->GetStringLength(env, text), output, output_capacity); (*env)->ReleaseStringChars(env, text, wide); }
            else result = 0;
        }
        pending = 0;
        pthread_cond_signal(&input_done);
    }
    pthread_mutex_unlock(&input_lock);
}
