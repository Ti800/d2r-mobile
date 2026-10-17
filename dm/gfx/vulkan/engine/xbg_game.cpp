#include "../../nvn/nvn_copy_pitch.h"
#include <pthread.h>
#include "xbg_game_internal.h"
#include "xbg_cpu_timing.h"
#include "xbg_descriptor_timing.h"
#include "xbg_draw_timing.h"
#include "xbg_provenance.h"
#include "xbg_pipeline_key.h"
#include "xbg_vulkan.h"
#include "xbg_depth.h"
#include "xbg_sampler.h"
#include "xbg_framebuffer.h"
#include "xbg_clear.h"
#include "xbg_blit.h"
#include "xbg_clear_spv.h"
extern "C" {
#include "../../tex/nvn_deswizzle.h"
}
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/mman.h>
#include <sys/resource.h>
#include <signal.h>
#include <cerrno>
#include <unistd.h>
#include <dlfcn.h>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <functional>
#include <deque>
#include <condition_variable>
#include <set>

namespace {
xbg_game_config cfg{};
std::recursive_mutex mutex;
uint32_t deferred_live_profile=0;
// ---- Asynchronous translation (Claude, C076): the game thread queues submits/presents/resource updates;
// one graphics thread translates NVN -> Vulkan and owns the VkQueue, in order. Calls that need results
// (fence waits, readback, finish, command-memory allocation) drain the queue first. Like a real GPU,
// guest memory is read at execution; the game only reuses it after the fence (waits drain).
// files/gfx-sync-submit containing '1' restores synchronous translation.
bool gfx_async=false;std::atomic<uint64_t> gfx_enqueued{0},gfx_executed{0};std::atomic<int> gfx_presents_queued{0};
std::mutex gfx_job_mutex;std::condition_variable gfx_job_cv,gfx_idle_cv;std::deque<std::function<void()>> gfx_jobs;bool gfx_busy=false,gfx_stop=false;std::thread gfx_worker;std::thread::id gfx_worker_id;
void gfx_worker_loop(){
    gfx_worker_id=std::this_thread::get_id();
    pthread_setname_np(pthread_self(),"xbl-gfx");
    // Graphics thread on performance cores (QoS USER_INTERACTIVE, as for the audio thread).
    {void *h=dlopen("libqos.so",RTLD_NOW);auto set=h?reinterpret_cast<int(*)(int)>(dlsym(h,"OH_QoS_SetThreadQoS")):nullptr;int r=set?set(5):-1;xbg_logf("NVN async graphics thread QoS=USER_INTERACTIVE rc=%d",r);}
    for(;;){std::function<void()> job;
        {std::unique_lock<std::mutex> l(gfx_job_mutex);gfx_job_cv.wait(l,[]{return gfx_stop||!gfx_jobs.empty();});if(gfx_jobs.empty())return;job=std::move(gfx_jobs.front());gfx_jobs.pop_front();gfx_busy=true;}
        job();
        {std::lock_guard<std::mutex> l(gfx_job_mutex);gfx_busy=false;}gfx_idle_cv.notify_all();
    }
}
void gfx_enqueue(std::function<void()> job){
    if(!gfx_async || std::this_thread::get_id()==gfx_worker_id){job();return;}
    {std::lock_guard<std::mutex> l(gfx_job_mutex);gfx_jobs.push_back(std::move(job));}gfx_job_cv.notify_one();
}
std::atomic<uint64_t> gfx_drain_calls{0},gfx_drain_ns{0},gfx_pace_ns{0},gfx_wait_calls{0};
void gfx_drain(){
    if(!gfx_async || std::this_thread::get_id()==gfx_worker_id)return;
    uint64_t t0=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    {std::unique_lock<std::mutex> l(gfx_job_mutex);gfx_idle_cv.wait(l,[]{return gfx_jobs.empty() && !gfx_busy;});}
    ++gfx_drain_calls;gfx_drain_ns+=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())-t0;
}
void gfx_async_stop(){if(!gfx_worker.joinable())return;{std::lock_guard<std::mutex> l(gfx_job_mutex);gfx_stop=true;}gfx_job_cv.notify_all();gfx_worker.join();gfx_stop=false;gfx_async=false;}
char error_text[1024]{};
bool initialized=false,failed=false;
VkBool32 sampler_mirror_clamp_enabled=VK_FALSE;
// Formatless storage access per format needs VK_KHR_format_feature_flags2 (core in Vulkan 1.3). Without it (Vulkan 1.1 drivers
// such as Adreno 730 512.615) the spec's "Formats Without Shader Storage Format" rule applies: the shaderStorageImage{Read,Write}
// WithoutFormat features cover exactly these formats, where the format also supports VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT.
bool format_feature_flags2=false;
// NVN DEPTH16 (0x32) is stored as X8_D24 when the driver cannot linearly filter D16_UNORM but can filter X8_D24: the game samples
// shadow maps with linear depth comparison, and Adreno 730 512.615 reports no SAMPLED_IMAGE_FILTER_LINEAR for D16_UNORM (or
// D32_SFLOAT). Transfers convert 16 <-> 24-bit depth with exact rounding; constant depth bias is rescaled to the finer depth unit.
bool depth16_as_d24=false;
uint32_t d16_to_d24(uint32_t k){return uint32_t((uint64_t(k)*2*16777215u+65535u)/(2*65535u));}
uint32_t d24_to_d16(uint32_t v){return uint32_t((uint64_t(v&0xffffffu)*2*65535u+16777215u)/(2*16777215u));}
bool storage_without_format_listed(VkFormat f){
    switch(f){
    case VK_FORMAT_R8G8B8A8_UNORM:case VK_FORMAT_R8G8B8A8_SNORM:case VK_FORMAT_R8G8B8A8_UINT:case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R32_UINT:case VK_FORMAT_R32_SINT:case VK_FORMAT_R32_SFLOAT:case VK_FORMAT_R32G32_UINT:case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:case VK_FORMAT_R32G32B32A32_UINT:case VK_FORMAT_R32G32B32A32_SINT:case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UINT:case VK_FORMAT_R16G16B16A16_SINT:case VK_FORMAT_R16G16B16A16_SFLOAT:case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:case VK_FORMAT_R16_SFLOAT:case VK_FORMAT_R16G16B16A16_UNORM:case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
    case VK_FORMAT_R16G16_UNORM:case VK_FORMAT_R8G8_UNORM:case VK_FORMAT_R16_UNORM:case VK_FORMAT_R8_UNORM:case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16_SNORM:case VK_FORMAT_R8G8_SNORM:case VK_FORMAT_R16_SNORM:case VK_FORMAT_R8_SNORM:case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R8G8_SINT:case VK_FORMAT_R16_SINT:case VK_FORMAT_R8_SINT:case VK_FORMAT_A2B10G10R10_UINT_PACK32:case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R8G8_UINT:case VK_FORMAT_R16_UINT:case VK_FORMAT_R8_UINT:return true;
    default:return false;
    }
}
uint64_t last_attachment_trace=UINT64_MAX;
struct Capture {
    std::string directory;FILE *trace=nullptr;uint64_t frame=0,bytes=0;
    unsigned events=0;bool active=false;
    ~Capture(){if(trace)fclose(trace);}
} capture;
constexpr uint64_t capture_limit=64ull*1024*1024;
void capture_log(const char *format,...){
    if(!capture.trace)return;va_list args;va_start(args,format);vfprintf(capture.trace,format,args);va_end(args);fputc('\n',capture.trace);fflush(capture.trace);
}
bool capture_begin(const char *kind,uint64_t present){
    capture.active=capture.trace && capture.frame==present+1 && capture.events<16;
    if(capture.active){++capture.events;capture_log("EVENT frame=%llu event=%u kind=%s",(unsigned long long)capture.frame,capture.events,kind);}
    return capture.active;
}
void capture_data(const char *label,const void *data,size_t bytes){
    if(!capture.active || !bytes)return;
    if(bytes>capture_limit-capture.bytes){capture_log("SKIP label=%s bytes=%zu reason=64MiB-budget",label,bytes);return;}
    char leaf[256];snprintf(leaf,sizeof(leaf),"f%llu-e%u-%s.bin",(unsigned long long)capture.frame,capture.events,label);
    std::string path=capture.directory+"/"+leaf;FILE *file=fopen(path.c_str(),"wb");
    if(!file){capture_log("IO_ERROR file=%s",leaf);return;}
    size_t written=fwrite(data,1,bytes,file);int closed=fclose(file);capture.bytes+=written;
    capture_log("FILE name=%s bytes=%zu requested=%zu closed=%d",leaf,written,bytes,closed);
}
// Diagnostic reads must remain available after runtime's fatal callback performs
// a nonlocal exit with the backend mutex held. Counters do not own GPU state.
struct Stats {
    std::atomic<uint64_t> submissions{0},draws{0},dispatches{0},presents{0},shader_loads{0};
    void reset(){submissions=0;draws=0;dispatches=0;presents=0;shader_loads=0;}
} stats;

// All access is under the existing recursive gfx mutex. No resource ownership.
XbgProvenance provenance;
FILE *provenance_file=nullptr;
void provenance_close(const char *reason){
    if(!provenance_file){provenance.armed=false;return;}
    FILE *f=provenance_file;provenance_file=nullptr;provenance.armed=false;
    char tail[128];int n=snprintf(tail,sizeof(tail),"STOP reason=%s events=%u draws=%u\n",reason,provenance.events,provenance.draws);
    bool bad=n<0||size_t(n)>=sizeof(tail);
    if(!bad)bad=fwrite(tail,1,size_t(n),f)!=size_t(n);
    if(fflush(f))bad=true;if(fclose(f))bad=true;
    xbg_logf("NVN PROVENANCE_CLOSE reason=%s io=%s",reason,bad?"FAIL":"OK");
}
void provenance_log(const char *format,...){
    if(!provenance_file || !provenance.window(stats.presents.load()))return;
    char line[2048];va_list a;va_start(a,format);int n=vsnprintf(line,sizeof(line)-1,format,a);va_end(a);
    if(n<0||size_t(n)>=sizeof(line)-1){provenance_close("FORMAT");return;}
    line[n++]='\n';if(!provenance.accept(unsigned(n))){provenance_close("LIMIT");return;}
    if(fwrite(line,1,size_t(n),provenance_file)!=size_t(n)||fflush(provenance_file))provenance_close("IO_FAIL");
}
void provenance_draw(){
    if(!provenance_file || !provenance.window(stats.presents.load()))return;
    if(!provenance.draw()){provenance_close("DRAW_LIMIT");return;}
    provenance_log("DRAW ordinal=%u frame=%llu submit=%llu",provenance.draws,(unsigned long long)provenance.target,(unsigned long long)stats.submissions.load());
}

uint64_t timing_clock(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
XbgCpuTiming timing(timing_clock);
thread_local XbgCpuTiming *timing_current=nullptr;
uint64_t timing_first=0,timing_last=0,timing_start_ns=0;
xbg_game_stats timing_before{};
bool timing_active=false;
bool perf_timing=true; // always-on wall-time breakdown for NVN PERF (exclusive deltas)
// Every scope reads the clock twice (~95k reads per D2R camp frame; ~105 ns each through the vDSO on Adreno 730
// phones, ~19% of the render thread). Time one frame in perf_sample (marker gfx-perf-sample = N; 1 = every frame,
// 0 = off); parts and descriptor lines are averaged over the timed frames.
unsigned perf_sample=8;bool perf_frame_timed=true;uint64_t perf_timed_frames=0;
XbgCpuTiming::Counter perf_timing_before[XbgCpuTiming::Count];
bool descriptor_detail=false;
XbgDescriptorTiming descriptor_timing;
thread_local XbgCpuTiming::Scope *descriptor_root=nullptr;
struct DescriptorRoot {
    XbgCpuTiming::Scope *previous=descriptor_root;
    DescriptorRoot(XbgCpuTiming::Scope *scope){descriptor_root=descriptor_detail?scope:nullptr;}
    ~DescriptorRoot(){descriptor_root=previous;}
};
#define DESC_SCOPE(kind) XbgDescriptorTiming::Scope descriptor_scope(descriptor_detail?&descriptor_timing:nullptr,descriptor_root,XbgDescriptorTiming::kind)
#define DESC_CALL(kind, expr) [&](){DESC_SCOPE(kind);return (expr);}()

bool draw_detail=false;
XbgDrawTiming draw_timing;
thread_local XbgCpuTiming::Scope *draw_root=nullptr;
struct DrawRoot {
    XbgCpuTiming::Scope *previous=draw_root;
    DrawRoot(XbgCpuTiming::Scope *scope){draw_root=draw_detail?scope:nullptr;}
    ~DrawRoot(){draw_root=previous;}
};
#define DRAW_SCOPE(kind) XbgDrawTiming::Scope draw_scope(draw_detail?&draw_timing:nullptr,draw_root,XbgDrawTiming::kind)
bool live_profile_used=false;
struct PipelineCounts {uint64_t hits=0,misses=0,evictions=0,bypass=0;} pipeline_counts,pipeline_before;
void report_pipeline_cache();
struct TimingRoot {
    XbgCpuTiming *previous=timing_current;
    TimingRoot(){timing_current=(timing_active||(perf_timing&&perf_frame_timed))?&timing:nullptr;}
    ~TimingRoot(){timing_current=previous;}
};
#define CPU_SCOPE(kind) XbgCpuTiming::Scope cpu_scope(timing_current,XbgCpuTiming::kind)
inline uint64_t perf_clock(){return timing_current?timing_clock():0;}   // hot-path sub-timers: free on untimed frames
void timing_start(){
    timing_active=true;timing_start_ns=timing_clock();pipeline_before=pipeline_counts;
    timing_before={stats.submissions.load(),stats.draws.load(),stats.dispatches.load(),stats.presents.load(),stats.shader_loads.load()};
}
void descriptor_report();
void draw_report();
void timing_report(const char *reason){
    if(!timing_active)return;
    uint64_t elapsed=timing_clock()-timing_start_ns;timing_active=false;timing_first=0;
    uint64_t measured=0;for(const auto &c:timing.counters)measured+=c.exclusive_ns;
    xbg_logf("NVN CPU_TIMING v=1 reason=%s first=%llu last=%llu frames=%llu wall_ns=%llu measured_ns=%llu uncovered_ns=%llu draws=%llu dispatches=%llu submissions=%llu shaders=%llu",reason,
        (unsigned long long)(timing_before.presents+1),(unsigned long long)stats.presents.load(),(unsigned long long)(stats.presents-timing_before.presents),
        (unsigned long long)elapsed,(unsigned long long)measured,(unsigned long long)(elapsed>=measured?elapsed-measured:0),
        (unsigned long long)(stats.draws-timing_before.draws),(unsigned long long)(stats.dispatches-timing_before.dispatches),
        (unsigned long long)(stats.submissions-timing_before.submissions),(unsigned long long)(stats.shader_loads-timing_before.shader_loads));
    report_pipeline_cache();
    static const char *names[]={"other","draw","dispatch","descriptor","pass","buffer","pipeline","submit","fence","present","cleanup","hash","image","log","program","cpuwrite","snapshot"};
    for(unsigned i=0;i<XbgCpuTiming::Count;i++){const auto &c=timing.counters[i];xbg_logf("NVN CPU_TIMING part=%s calls=%llu exclusive_ns=%llu inclusive_ns=%llu max_ns=%llu",names[i],(unsigned long long)c.calls,(unsigned long long)c.exclusive_ns,(unsigned long long)c.inclusive_ns,(unsigned long long)c.max_ns);}
    descriptor_report();
    draw_report();
}
void descriptor_report(){
    if(!descriptor_detail)return;
    const uint64_t total=timing.counters[XbgCpuTiming::Descriptor].exclusive_ns;
    const uint64_t tagged=descriptor_timing.sum();
    xbg_logf("NVN DESC_TIMING v=2 first=%llu last=%llu descriptor_ns=%llu tagged_ns=%llu remaining_ns=%llu coverage=%s clock_reads=%llu",(unsigned long long)(timing_before.presents+1),(unsigned long long)stats.presents.load(),(unsigned long long)total,(unsigned long long)tagged,(unsigned long long)(tagged<=total?total-tagged:0),tagged<=total && !descriptor_timing.invalid && !descriptor_timing.top?"OK":"INVALID",(unsigned long long)descriptor_timing.reads);
    const char *names[]={"setup","buffer_bind","image_lookup","image_transition","image_view","sampler_bind","element_rest"};
    for(unsigned i=0;i<XbgDescriptorTiming::Count;++i)xbg_logf("NVN DESC_TIMING part=%s calls=%llu exclusive_ns=%llu",names[i],(unsigned long long)descriptor_timing.counters[i].calls,(unsigned long long)descriptor_timing.counters[i].ns);
    xbg_logf("NVN DESC_TIMING part=remaining calls=0 exclusive_ns=%llu",(unsigned long long)(tagged<=total?total-tagged:0));
}
void draw_report(){
    if(!draw_detail)return;
    const uint64_t total=timing.counters[XbgCpuTiming::Draw].exclusive_ns;
    const uint64_t tagged=draw_timing.finish(total);
    const uint64_t remaining=tagged<=total?total-tagged:0;
    xbg_logf("NVN DRAW_TIMING v=1 first=%llu last=%llu draw_ns=%llu tagged_ns=%llu remaining_ns=%llu coverage=%s clock_reads=%llu flags=%u",(unsigned long long)(timing_before.presents+1),(unsigned long long)stats.presents.load(),(unsigned long long)total,(unsigned long long)tagged,(unsigned long long)remaining,draw_timing.flags?"INVALID":"OK",(unsigned long long)draw_timing.reads,draw_timing.flags);
    const char *names[]={"writeback","vertex_prepare"};
    for(unsigned i=0;i<XbgDrawTiming::Count;++i)xbg_logf("NVN DRAW_TIMING part=%s calls=%llu exclusive_ns=%llu",names[i],(unsigned long long)draw_timing.counters[i].calls,(unsigned long long)draw_timing.counters[i].ns);
    xbg_logf("NVN DRAW_TIMING part=remaining calls=0 exclusive_ns=%llu",(unsigned long long)remaining);
    xbg_logf("NVN DRAW_TIMING counters storage_ranges=%llu images_visited=%llu writeback_bytes=%llu index_expand_calls=%llu index_expand_bytes=%llu",(unsigned long long)draw_timing.storage_ranges,(unsigned long long)draw_timing.images_visited,(unsigned long long)draw_timing.writeback_bytes,(unsigned long long)draw_timing.index_expand_calls,(unsigned long long)draw_timing.index_expand_bytes);
}
void timing_boundary(){
    if(timing_active){if(stats.presents>=timing_last)timing_report("frames");else if(timing_clock()-timing_start_ns>=60000000000ull)timing_report("60s-next-present");}
    else if(timing_first && stats.presents+1==timing_first)timing_start();
}
bool sync_mode=false; // diagnostic: marker files/gfx-sync-mode resolves all GPU results at each submit end
std::atomic<uint64_t> completed{0};uint64_t submitted=0; // guest fence values: submitted xbg_submit calls / GPU-complete ones (see Slot::covers)
VkCommandPool command_pool{};
VkPipelineCache pipeline_cache{};
VkPhysicalDeviceProperties properties{};
VkPhysicalDeviceMemoryProperties memories{};
uint32_t timestamp_bits=0;
VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
void need(bool condition,const char *text) { if (!condition) xbg_fatalf("%s",text); }
void check(VkResult r,const char *call) { if(r!=VK_SUCCESS) xbg_fatalf("%s failed: %d",call,int(r)); }
#define VK(call) check((call),#call)
uint32_t memory_type(uint32_t bits,VkMemoryPropertyFlags flags) {
    for(uint32_t i=0;i<memories.memoryTypeCount;i++)
        if((bits&(1u<<i)) && (memories.memoryTypes[i].propertyFlags&flags)==flags) return i;
    xbg_fatalf("No Vulkan memory type for flags %#x",flags);
}
// ---------------------------------------------------------------------------
// Deferred submission. All GPU work of a guest submission is recorded into one
// command buffer (per frame slot) instead of submit+wait per operation. Every
// operation still starts with a full memory barrier, so GPU-side ordering and
// visibility match the former one-submit-per-operation model. The CPU waits
// only where it needs GPU results (storage writeback, readback, queries).
// Guest memory read by the GPU (uniform/storage/vertex/index data, uploads) is
// snapshotted into host arenas at record time, exactly as before.
// ---------------------------------------------------------------------------
uint32_t find_memory(uint32_t bits,VkMemoryPropertyFlags required,VkMemoryPropertyFlags preferred){
    for(uint32_t i=0;i<memories.memoryTypeCount;i++)
        if((bits&(1u<<i)) && (memories.memoryTypes[i].propertyFlags&(required|preferred))==(required|preferred)) return i;
    return memory_type(bits,required);
}
std::map<VkFormat,VkFormatProperties> format_property_cache;
// Core formats (< 256) answer from a flat table: format_properties is asked for every vertex attribute of every draw.
VkFormatProperties format_flat[256];bool format_flat_known[256];
const VkFormatProperties &format_properties(VkFormat f){
    if(uint32_t(f)<256){if(!format_flat_known[f]){vkGetPhysicalDeviceFormatProperties(cfg.physical,f,&format_flat[f]);format_flat_known[f]=true;}return format_flat[f];}
    auto it=format_property_cache.find(f);if(it!=format_property_cache.end())return it->second;
    VkFormatProperties p{};vkGetPhysicalDeviceFormatProperties(cfg.physical,f,&p);return format_property_cache.emplace(f,p).first->second;
}
struct ArenaChunk{VkBuffer buffer{};VkDeviceMemory memory{};uint8_t *mapped{};VkDeviceSize size=0,used=0;bool coherent=true;};
constexpr VkBufferUsageFlags arena_usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
constexpr VkDeviceSize arena_chunk_bytes=32ull*1024*1024;
constexpr VkDeviceSize arena_flush_bytes=96ull*1024*1024;
VkDeviceSize arena_alignment=256;
struct Buffer {
    // A view into an arena chunk. Arena memory lives until the owning frame slot
    // is retired after its fence, so no per-buffer destruction is needed.
    VkBuffer buffer{}; VkDeviceMemory memory{}; VkDeviceSize offset=0; size_t size{}; void *mapped{}; bool coherent=true;
    void flush(){if(coherent)return;VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};r.memory=memory;r.size=VK_WHOLE_SIZE;VK(vkFlushMappedMemoryRanges(cfg.device,1,&r));}
    void invalidate(){if(coherent)return;VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};r.memory=memory;r.size=VK_WHOLE_SIZE;VK(vkInvalidateMappedMemoryRanges(cfg.device,1,&r));}
};
struct Arena {
    std::vector<ArenaChunk> chunks;size_t current=0;bool cached=false;VkDeviceSize used_total=0,peak=0;
    ArenaChunk make(VkDeviceSize bytes){
        CPU_SCOPE(Buffer);ArenaChunk c;c.size=bytes;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=bytes;ci.usage=arena_usage;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK(vkCreateBuffer(cfg.device,&ci,nullptr,&c.buffer));
        VkMemoryRequirements r;vkGetBufferMemoryRequirements(cfg.device,c.buffer,&r);
        uint32_t type=cached?find_memory(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT):find_memory(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if(cached && !(memories.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT))type=find_memory(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        c.coherent=(memories.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)!=0;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=r.size;ai.memoryTypeIndex=type;
        VK(vkAllocateMemory(cfg.device,&ai,nullptr,&c.memory));VK(vkBindBufferMemory(cfg.device,c.buffer,c.memory,0));
        void *p=nullptr;VK(vkMapMemory(cfg.device,c.memory,0,VK_WHOLE_SIZE,0,&p));c.mapped=static_cast<uint8_t*>(p);return c;
    }
    Buffer alloc(size_t n){
        need(n && n<=256ull*1024*1024,"Invalid or excessive buffer size");
        for(size_t i=current;;++i){
            if(i==chunks.size())chunks.push_back(make(std::max<VkDeviceSize>(arena_chunk_bytes,(n+arena_alignment-1)&~(arena_alignment-1))));
            ArenaChunk &c=chunks[i];VkDeviceSize at=(c.used+arena_alignment-1)&~(arena_alignment-1);
            if(at+n>c.size)continue;
            current=i; // leftovers in skipped chunks are abandoned until reset
            c.used=at+n;used_total+=n;peak=std::max(peak,used_total);
            Buffer b;b.buffer=c.buffer;b.memory=c.memory;b.offset=at;b.size=n;b.mapped=c.mapped+at;b.coherent=c.coherent;return b;
        }
    }
    // reset() keeps all memory mapped (a synchronous reader may still copy out
    // of it); trim() releases oversized/extra chunks before the next recording.
    size_t keep_chunks=3;
    void reset(){for(auto &c:chunks)c.used=0;current=0;used_total=0;}
    void trim(){
        size_t kept=0;std::vector<ArenaChunk> keep;
        for(auto &c:chunks){if(c.size==arena_chunk_bytes && kept<keep_chunks){keep.push_back(c);++kept;}else destroy(c);}
        chunks.swap(keep);current=0;
    }
    static void destroy(ArenaChunk &c){if(c.mapped)vkUnmapMemory(cfg.device,c.memory);if(c.buffer)vkDestroyBuffer(cfg.device,c.buffer,nullptr);if(c.memory)vkFreeMemory(cfg.device,c.memory,nullptr);c=ArenaChunk{};}
    void release(){for(auto &c:chunks)destroy(c);chunks.clear();current=0;used_total=0;}
};
// Deferred destruction until the GPU finished the slot that referenced objects.
struct Retire {
    std::vector<std::function<void()>> handles;std::vector<std::shared_ptr<void>> objects;
    void run(){for(auto &h:handles)h();handles.clear();objects.clear();}
};
// Deferred GPU->CPU results. Storage writebacks and counter reports are
// resolved at the end of each guest submission (when the guest may observe
// completion) or earlier if anything touches the affected guest memory.
struct EpochQuery {const void *slot;VkQueryPool pool;uint32_t index;uint64_t value;bool known;};
// Occlusion counters accumulate from the last reset; the game may never reset, so queries are kept
// incrementally: known prefix and folded prefix sum advance once, consumed entries are dropped (C103).
struct EpochState{std::deque<EpochQuery> q;uint64_t base=0,known_upto=0,summed_upto=0,sum=0;uint64_t size()const{return base+q.size();}
    void advance_known(){while(known_upto<size() && q[known_upto-base].known)++known_upto;}
    uint64_t prefix(uint64_t count){while(summed_upto<count && summed_upto<size()){sum+=q[summed_upto-base].value;++summed_upto;}return sum;}
    void trim(){while(base<summed_upto && !q.empty()){q.pop_front();++base;}}};
uint64_t counter_epoch=0;std::map<uint64_t,EpochState> epochs;
struct PendingReport {uint64_t addr,type,epoch;size_t count;const void *slot;VkQueryPool ts;uint32_t ts_index;uint64_t ts_value;bool ts_known;};
std::vector<PendingReport> pending_reports;
// Command-buffer SignalEvent (4 per frame in D2R) completes when the GPU reaches it. Completing all work at the signal stalls
// the CPU on the GPU at every signal; marker file gfx-defer-events = '1' instead submits the work so far without waiting and
// publishes the value when that submission's slot retires (as the Metal backend writes it from a completion handler).
// '2' does not submit at the signal: the value waits in the slot being recorded and is published when that slot retires,
// so a frame stays one submission and the 2-slot ring does not wrap inside it.
// 客体可能直接读取池；未证明全路径前只允许显式实验，产品默认同步。
// Guest code may read the pool directly; until all paths are proven, deferral is experimental and the product stays synchronous.
struct PendingEvent {uint64_t addr;uint32_t mode,v;const void *slot;};
std::vector<PendingEvent> pending_events,recording_events;   // recording_events: mode 2, slot not yet submitted
int defer_events=0;uint64_t events_deferred=0,event_reads=0;
struct PendingWriteback {uint64_t addr,bytes;Buffer buffer;const void *slot;};
std::vector<PendingWriteback> pending_writebacks;
void cpu_access(uint64_t address,uint64_t bytes,unsigned why=0);   // resolve pending GPU results overlapping a CPU access
void resolve_pending();
void resolve_slot_writebacks(const void *slot);
void resolve_known_reports();
extern bool resolving_pending;
uint64_t snap_site_bytes[6]={};unsigned snap_site=0; // perf: copied bytes by caller (0 other,1 vertex,2 index,3 storage,4 texel,5 uniform)
std::unordered_set<uint64_t> storage_pages;uint64_t storage_snap_count=0;
namespace wtrack{extern uint64_t copied_bytes,dirty_pages,mismatches,fallbacks;}
extern std::atomic<uint64_t> memo_epoch;extern uint64_t key_diff[14];extern uint64_t memo_bump[5];extern uint64_t xmemo_hits,xmemo_drops;struct PinRef{VkDescriptorSetLayout layout;uint32_t refs;};extern std::unordered_map<VkDescriptorSet,PinRef> memo_pinned[2];
extern uint64_t desc_memo_hits,desc_memo_misses,desc_memo_why[6],desc_ns[6],noload_passes,img_cache_hits,img_cache_misses,noload_why[8],binds_skipped;
namespace warm{extern uint64_t hits,creates,create_ns,create_max_ns;extern std::atomic<uint64_t> built,failed,build_ns;void flush();void register_module(VkShaderModule m,const std::string &key);}
uint64_t cleanup_ns[5]={},writeback_bytes=0; // perf: cleanup phase split
// Driver/support constant buffers are rebuilt only when their inputs change (iOS: reuse parameter
// blocks for identical inputs). Generations bump on binding ops; slot_serial on every frame-slot switch.
uint64_t tex_gen[6]={},buf_gen[6]={},slot_serial=1;
struct DriverCache{uint64_t gen=0,slot=0;Buffer buffer;};DriverCache driver_cache[6][2];
struct SupportCache{float w=-1,h=-1;uint64_t slot=0;Buffer buffer;};SupportCache support_cache;
struct SetKeyHash{size_t operator()(const std::vector<uint64_t> &k)const{uint64_t h=1469598103934665603ull;for(uint64_t v:k){h^=v;h*=1099511628211ull;h^=h>>32;}return size_t(h);}};
uint64_t set_cache_hits=0,set_cache_misses=0,uniform_trimmed=0;bool template_enabled=true;uint64_t template_updates=0,plain_updates=0;bool set_cache_enabled=true; // same-process A/B showed no gain on Maleoon
// In-process A/B (iOS method: alternate in the same process, same thermal state). files/gfx-ab-toggle holding
// N (seconds) flips ab_on every N seconds; ab_on=false disables the optimization under test. Perf lines carry ab=.
int ab_period=0;bool ab_on=true;
bool ab_frame_on=true;
bool ab_enabled(){if(ab_period<=0)return true;ab_on=((std::chrono::steady_clock::now().time_since_epoch().count()/1000000000ll/ab_period)&1)==0;return ab_on;}
struct Slot {
    VkCommandBuffer cmd{};VkCommandPool pool{};VkFence fence{};VkQueryPool timestamps{};bool recording=false,pending=false;uint64_t ops=0,covers=0; // covers: guest fence value complete once this slot retires
    std::vector<VkQueryPool> occlusion_pools,timestamp_pools;uint32_t occlusion_used=0,timestamp_used=0;
    Retire retire;Arena stream,readback;std::vector<VkDescriptorPool> pools;size_t pool_next=0;
    std::map<std::vector<uint64_t>,VkFramebuffer> framebuffers;
    std::map<std::array<uint64_t,4>,VkBufferView> texel_views; // per-frame texel views (same snapshot -> one view)
    Buffer query_results;size_t query_copied=0; // occlusion results copied by the GPU at the end of the frame
    std::vector<std::pair<uint64_t,uint64_t>> query_refs; // (epoch, absolute index) per used occlusion query, in result order
    std::unordered_map<std::vector<uint64_t>,VkDescriptorSet,SetKeyHash> set_cache; // identical descriptor contents -> one set per frame
    std::unordered_map<VkDescriptorSetLayout,std::vector<VkDescriptorSet>> spare_sets; // batch-allocated, reset with the pools
    struct PassRec {
        uint32_t q0=0,q1=0,w=0,h=0,att=0,fmt=0,draws0=0,draws1=0,x=0,y=0,z=0;
        bool compute=false,complete=false,mixed=false;std::string first,last;
    };
    bool prof=false;VkQueryPool prof_pool{};uint32_t prof_used=0,prof_slot=0,prof_skipped=0;uint64_t prof_frame=0;
    std::vector<PassRec> prof_recs;int prof_open=-1;
    std::unordered_map<VkDescriptorSetLayout,std::vector<VkDescriptorSet>> used_sets;
    std::vector<struct Descriptors*> desc_used; // pooled per-draw descriptor records, returned after this slot's fence // handed out this frame; recycled into spare_sets after the slot fence (no per-frame pool reset)
};
Slot slots[2];unsigned slot_index=0;
// Where the CPU blocks on a busy slot fence, and why slots are submitted (NVN PERF block_ms).
// The outermost site wins, except that a ring wrap in recording_slot is always counted as ring.
enum SyncSite{SITE_OTHER,SITE_RING,SITE_PRESENT,SITE_EVENT,SITE_ARENA,SITE_GPU_SYNC,SITE_COMPLETE,SITE_POLL,SITE_LEGACY_WAIT,SITE_COUNT};
int sync_site=SITE_OTHER;uint64_t block_ns[SITE_COUNT],block_calls[SITE_COUNT],submit_by[SITE_COUNT],legacy_wait_ns=0;
struct SiteScope{int old;explicit SiteScope(int site,bool force=false):old(sync_site){if(force || sync_site==SITE_OTHER)sync_site=site;}~SiteScope(){sync_site=old;}};
uint64_t write_seq=1,barrier_seq=0; // image writes vs. last full barrier (render-pass merging)
// Render-pass merging: consecutive draws into the same framebuffer share one
// render pass. Any non-draw operation, a texture read of something written
// since the last barrier, or a layout change ends the open pass first.
struct Image;
void bind_cache_reset();
// Attachment images of a pass (8 color + depth): fixed capacity, so building a Pass per draw does not allocate.
struct ImageList{Image *v[9];uint32_t n=0;void push_back(Image *i){if(n<9)v[n++]=i;}Image **begin(){return v;}Image **end(){return v+n;}Image *const *begin()const{return v;}Image *const *end()const{return v+n;}
    size_t size()const{return n;}bool empty()const{return !n;}Image *operator[](size_t i)const{return v[i];}};
struct OpenPass {VkRenderPass pass{};VkFramebuffer framebuffer{};VkCommandBuffer cmd{};ImageList images;} open_pass;
// Framebuffers of cached render passes, kept across frames (creating and destroying one per pass per frame was ~5% of the
// render thread on Adreno 730). Each attachment Image lists the keys of the framebuffers that use its views; the first of
// them to be destroyed (all GPU work that used it is complete by then) takes the framebuffer along. gfx-no-persistent-fb
// keeps the former per-slot cache.
std::map<std::vector<uint64_t>,VkFramebuffer> persistent_framebuffers;bool persistent_fb_enabled=true;uint64_t fb_hits=0,fb_creates=0;
void forget_framebuffers(Image &im,std::vector<VkFramebuffer> *deferred=nullptr);
// Per-slot command pools reset whole (vkResetCommandPool) instead of resetting the command buffer: the buffer reset was
// 3.6% of the render thread on Adreno 730. gfx-no-slot-pools keeps the shared pool. Readback arenas in host-cached memory
// (gfx-readback-cached = '1'): GPU results read back by the CPU (storage writebacks) come from uncached memory otherwise.
bool slot_pools_enabled=true,readback_cached=false;
// Resident uniform texel copies (texel_acquire, before descriptors()).
struct TexelKey{uint64_t base,bytes;uint32_t format;bool operator==(const TexelKey &o)const{return base==o.base&&bytes==o.bytes&&format==o.format;}};
struct TexelKeyHash{size_t operator()(const TexelKey &k)const{uint64_t x=k.base*0x9E3779B97F4A7C15ull^k.bytes*0xC2B2AE3D27D4EB4Full^k.format;x^=x>>29;return size_t(x*0xBF58476D1CE4E5B9ull);}};
struct TexelResident{Buffer buffer;VkBufferView view{};uint64_t hash=0,used=0,gen=0,checked=0,check_epoch=0;};   // checked: verified in this submission
std::unordered_map<TexelKey,TexelResident,TexelKeyHash> texel_resident[2];Arena texel_arena[2];uint64_t texel_hole_bytes[2]={};
bool texel_memo_enabled=false;uint64_t texel_gen=0,texel_refreshes=0,texel_fallbacks=0,texel_evictions=0,texel_hits=0;
// A copy verified earlier in this submission stays valid until an in-stream CPU write overlaps it (snapshot_invalidate
// bumps texel_write_epoch), so a ~1 MB range bound by 558 draws is hashed once per frame, not per draw.
uint64_t texel_write_epoch=1,texel_lo=UINT64_MAX,texel_hi=0;void texel_touch(uint64_t address,uint64_t bytes);
void texel_evict(unsigned si,bool all);
// 显式gfx-pass-profile：每600次呈现采完整一帧的所有slot；整帧上限1024查询/32slot。
// Opt-in gfx-pass-profile: sample all slots of one frame every 600 presents,
// bounded to 1024 timestamp queries and 32 slots per selected frame.
constexpr uint32_t pass_prof_query_limit=1024,pass_prof_slot_limit=32;
bool pass_prof_enabled=false;uint64_t pass_prof_next=300,pass_prof_draws=0,pass_prof_frame=UINT64_MAX;
uint32_t pass_prof_queries=0,pass_prof_slots=0,pass_prof_skipped_slots=0;
void pass_prof_end(VkCommandBuffer cmd);
// 同一渲染段内连续游戏 draw 共用精确样本计数；清屏、报告、重置和结束 pass 均截断。
// Consecutive game draws share a precise count. Clears, reports, resets and pass ends split it.
struct OpenOcclusion {VkCommandBuffer cmd{};VkQueryPool pool{};uint32_t index=0;} open_occlusion;
bool occlusion_batch_enabled=true;uint64_t occlusion_queries_started=0;
std::string occlusion_batch_marker;
// 仅自然结束的pass可提早提交；最多两次/帧，不拆pass或借用跨slot写回。
// Submit only naturally closed passes, at most twice/frame, with no borrowed writebacks.
bool early_submit_enabled=false;std::string early_submit_marker;
uint64_t early_submit_frame=UINT64_MAX,early_submit_window=0,early_blocked_window=0;
unsigned early_submit_count=0;
void refresh_early_submit(){FILE *f=early_submit_marker.empty()?nullptr:fopen(early_submit_marker.c_str(),"rb");early_submit_enabled=f!=nullptr;if(f)fclose(f);}
void refresh_occlusion_batch(){
    FILE *f=occlusion_batch_marker.empty()?nullptr:fopen(occlusion_batch_marker.c_str(),"rb");
    occlusion_batch_enabled=!f;if(f)fclose(f);
}
void end_occlusion_query(){
    if(open_occlusion.cmd){vkCmdEndQuery(open_occlusion.cmd,open_occlusion.pool,open_occlusion.index);open_occlusion=OpenOcclusion{};}
}
void begin_occlusion_query(VkCommandBuffer cmd,VkQueryPool pool,uint32_t index){
    need(!open_occlusion.cmd,"Nested occlusion query");
    vkCmdBeginQuery(cmd,pool,index,VK_QUERY_CONTROL_PRECISE_BIT);open_occlusion={cmd,pool,index};++occlusion_queries_started;
}
void end_open_pass(){if(open_pass.cmd){end_occlusion_query();vkCmdEndRenderPass(open_pass.cmd);pass_prof_end(open_pass.cmd);open_pass=OpenPass{};}}
void full_barrier(VkCommandBuffer cmd){VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&b,0,nullptr,0,nullptr);barrier_seq=write_seq;}
struct PerfCounts{size_t pipelines,layouts,samplers,passes,images,host_segments,host_bytes;};
PerfCounts perf_counts();
struct Perf {
    uint64_t frames=0,submits=0,waits=0,sync_writeback=0,sync_readback=0,sync_query=0,sync_other=0,gpu_ns=0,gpu_samples=0,
        submit_cpu_ns=0,present_cpu_ns=0,passes=0,draws=0,dispatches=0,snapshot_hits=0,snapshot_misses=0,pending_views=0,access_by[12]={},folded_clears=0,why_wait=0,why_poll=0,why_access=0,why_cmdwait=0,why_finish=0,why_slot=0,snapshot_bytes=0,arena_peak=0,window_start=0,ops=0;
} perf;
void snapshot_reset();
void release_descriptors(Slot &s);
void snapshot_invalidate(uint64_t address,uint64_t bytes);
void apply_event(uint64_t addr,uint32_t mode,uint32_t v){
    snapshot_invalidate(addr,4);auto ptr=reinterpret_cast<uint32_t*>(addr);
    if(mode==NVN_EVENT_SIGNAL_MODE_ADD)__atomic_fetch_add(ptr,v,__ATOMIC_SEQ_CST);else{need(mode==NVN_EVENT_SIGNAL_MODE_WRITE,"Unknown event mode");__atomic_store_n(ptr,v,__ATOMIC_SEQ_CST);}
}
// Deferred events in recording order; stop at the first whose slot is still in flight (slots retire in submission order).
void apply_ready_events(){
    size_t k=0;for(;k<pending_events.size();++k){const PendingEvent &e=pending_events[k];if(static_cast<const Slot*>(e.slot)->pending)break;apply_event(e.addr,e.mode,e.v);}
    pending_events.erase(pending_events.begin(),pending_events.begin()+k);
}
void retire_slot(Slot &s){
    if(!s.pending)return;
    {CPU_SCOPE(Fence);bool busy=vkGetFenceStatus(cfg.device,s.fence)==VK_NOT_READY;uint64_t b0=busy?timing_clock():0;
     VK(vkWaitForFences(cfg.device,1,&s.fence,VK_TRUE,std::max<uint64_t>(cfg.fence_timeout_ns,10000000000ull)));
     if(busy){block_ns[sync_site]+=timing_clock()-b0;++block_calls[sync_site];}}
    ++perf.waits;
    if(s.prof){
        s.prof=false;
        // 栅栏不替代查询可用性检查；按有效位数计算环绕差，只报告完整查询对。
        // Check availability as well as the fence; mask timestamp wrap to valid bits.
        struct Stamp{uint64_t value=0,available=0;};std::vector<Stamp> t(s.prof_used);
        VkResult status=s.prof_used?vkGetQueryPoolResults(cfg.device,s.prof_pool,0,s.prof_used,t.size()*sizeof(Stamp),t.data(),sizeof(Stamp),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WITH_AVAILABILITY_BIT):VK_SUCCESS;
        const uint64_t mask=timestamp_bits<64?(uint64_t(1)<<timestamp_bits)-1:~0ull;
        double per=properties.limits.timestampPeriod/1000.0,pass_us=0,compute_us=0,span_us=-1;size_t complete=0;
        uint64_t first=0,last=0;
        if(status==VK_SUCCESS || status==VK_NOT_READY)for(size_t i=0;i<s.prof_recs.size();++i){const auto &r=s.prof_recs[i];
            if(!r.complete || r.q0>=t.size() || r.q1>=t.size() || !t[r.q0].available || !t[r.q1].available)continue;
            double us=double((t[r.q1].value-t[r.q0].value)&mask)*per;
            if(!complete)first=t[r.q0].value;last=t[r.q1].value;++complete;
            if(r.compute){compute_us+=us;xbg_logf("NVN GPASS frame=%llu slot=%u ring=%u i=%zu kind=compute us=%.0f cs=%s groups=%u,%u,%u",(unsigned long long)s.prof_frame,s.prof_slot,unsigned(&s-slots),i,us,r.first.c_str(),r.x,r.y,r.z);}
            else{pass_us+=us;xbg_logf("NVN GPASS frame=%llu slot=%u ring=%u i=%zu kind=pass us=%.0f %ux%u att=%u fmt=%u draws=%u first=%s last=%s mixed=%u",(unsigned long long)s.prof_frame,s.prof_slot,unsigned(&s-slots),i,us,r.w,r.h,r.att,r.fmt,r.draws1-r.draws0,r.first.empty()?"builtin-clear/no-game-draw":r.first.c_str(),r.last.empty()?"builtin-clear/no-game-draw":r.last.c_str(),unsigned(r.mixed));}
        }
        if(complete)span_us=double((last-first)&mask)*per;
        xbg_logf("NVN GPASS frame=%llu slot=%u ring=%u summary recorded=%zu complete=%zu skipped=%u queries=%u result=%d pass_us=%.0f compute_us=%.0f span_us=%.0f",(unsigned long long)s.prof_frame,s.prof_slot,unsigned(&s-slots),s.prof_recs.size(),complete,s.prof_skipped,s.prof_used,int(status),pass_us,compute_us,span_us);
    }
    if(s.timestamps){uint64_t t[2]{};if(vkGetQueryPoolResults(cfg.device,s.timestamps,0,2,sizeof(t),t,8,VK_QUERY_RESULT_64_BIT)==VK_SUCCESS && t[1]>=t[0]){uint64_t mask=timestamp_bits<64?(uint64_t(1)<<timestamp_bits)-1:~0ull;perf.gpu_ns+=uint64_t(double((t[1]-t[0])&mask)*properties.limits.timestampPeriod);++perf.gpu_samples;}}
    CPU_SCOPE(Cleanup);uint64_t c0=timing_clock();
    // One readback per occlusion pool (the fence has signalled, so every used query is available).
    {static std::vector<uint64_t> results;size_t used=s.occlusion_used;results.assign(used,0);
     if(s.query_copied==used && used){s.query_results.invalidate();memcpy(results.data(),s.query_results.mapped,used*8);}
     else for(size_t i=0;i*256<used;i++){uint32_t n=uint32_t(std::min<size_t>(256,used-i*256));if(vkGetQueryPoolResults(cfg.device,s.occlusion_pools[i],0,n,n*sizeof(uint64_t),results.data()+i*256,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT)!=VK_SUCCESS)for(uint32_t k=0;k<n;k++)results[i*256+k]=0;}
     s.query_copied=0;
     for(size_t i=0;i<s.query_refs.size();i++){auto e=epochs.find(s.query_refs[i].first);if(e==epochs.end())continue;uint64_t a=s.query_refs[i].second;
         if(a>=e->second.base && a<e->second.size()){auto &q=e->second.q[a-e->second.base];q.value=i<used?results[i]:0;q.known=true;}}
     s.query_refs.clear();for(auto &e:epochs)e.second.advance_known();}
    for(auto &r:pending_reports)if(r.slot==&s && !r.ts_known){uint64_t v=0;if(vkGetQueryPoolResults(cfg.device,r.ts,r.ts_index,1,sizeof(v),&v,sizeof(v),VK_QUERY_RESULT_64_BIT)==VK_SUCCESS)r.ts_value=v;r.ts_known=true;}
    s.occlusion_used=0;s.timestamp_used=0;uint64_t c1=timing_clock();
    if(!resolving_pending){resolve_slot_writebacks(&s);resolve_known_reports();
        bool waiting=false;for(auto &r:pending_reports)if(r.slot==&s)waiting=true;if(!waiting)completed=std::max(completed.load(),s.covers);}
    uint64_t c2=timing_clock();VK(vkResetFences(cfg.device,1,&s.fence));
    for(auto &f:s.framebuffers)vkDestroyFramebuffer(cfg.device,f.second,nullptr);s.framebuffers.clear();
    for(auto &v:s.texel_views)vkDestroyBufferView(cfg.device,v.second,nullptr);s.texel_views.clear();
    uint64_t c3=timing_clock();s.retire.run();release_descriptors(s);uint64_t c4=timing_clock();
    perf.arena_peak=std::max<uint64_t>(perf.arena_peak,s.stream.peak+s.readback.peak);s.stream.peak=s.readback.peak=0;
    s.stream.reset();s.readback.reset();
    // Every set is fully rewritten before use, so sets are recycled instead of resetting pools (vkAllocateDescriptorSets cost ~15 us/set on Maleoon).
    for(auto &u:s.used_sets){auto &sp=s.spare_sets[u.first];sp.insert(sp.end(),u.second.begin(),u.second.end());u.second.clear();}s.set_cache.clear();
    cleanup_ns[0]+=c1-c0;cleanup_ns[1]+=c2-c1;cleanup_ns[2]+=c3-c2;cleanup_ns[3]+=c4-c3;cleanup_ns[4]+=timing_clock()-c4;
    s.pending=false;
    if(!resolving_pending)apply_ready_events();   // after this slot's writebacks; inside resolve_pending its own copy runs first
}
Slot &recording_slot(){
    Slot &s=slots[slot_index];
    if(!s.recording){
        {SiteScope site(SITE_RING,true);retire_slot(s);}
        // Pending writebacks still read this slot's readback arena; finish them before trimming.
        for(auto &w:pending_writebacks)if(w.slot==&s){++perf.why_slot;resolve_pending();break;}
        s.stream.trim();s.readback.keep_chunks=1;s.readback.trim();
        if(texel_memo_enabled && (slot_serial&15)<2)texel_evict(slot_index,false);   // this slot's copies: its previous submission retired
        if(!s.cmd){if(slot_pools_enabled){VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=cfg.queue_family;VK(vkCreateCommandPool(cfg.device,&cp,nullptr,&s.pool));}
            VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};a.commandPool=s.pool?s.pool:command_pool;a.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;a.commandBufferCount=1;VK(vkAllocateCommandBuffers(cfg.device,&a,&s.cmd));}
        s.readback.cached=readback_cached;
        if(!s.fence){VkFenceCreateInfo f{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};VK(vkCreateFence(cfg.device,&f,nullptr,&s.fence));}
        if(!s.timestamps && timestamp_bits){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=2;VK(vkCreateQueryPool(cfg.device,&q,nullptr,&s.timestamps));}
        bind_cache_reset();if(s.pool)VK(vkResetCommandPool(cfg.device,s.pool,0));else VK(vkResetCommandBuffer(s.cmd,0));VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};b.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;VK(vkBeginCommandBuffer(s.cmd,&b));
        if(s.timestamps){vkCmdResetQueryPool(s.cmd,s.timestamps,0,2);vkCmdWriteTimestamp(s.cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,s.timestamps,0);}
        s.recording=true;s.ops=0;
        const uint64_t frame=stats.presents.load()+1;
        if(pass_prof_enabled && timestamp_bits){
            if(stats.presents.load()>=pass_prof_next && pass_prof_frame!=frame){pass_prof_frame=frame;pass_prof_next=stats.presents.load()+600;pass_prof_queries=pass_prof_slots=pass_prof_skipped_slots=0;}
            if(frame==pass_prof_frame){
                if(pass_prof_slots<pass_prof_slot_limit){
                    if(!s.prof_pool){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=pass_prof_query_limit;VK(vkCreateQueryPool(cfg.device,&q,nullptr,&s.prof_pool));}
                    vkCmdResetQueryPool(s.cmd,s.prof_pool,0,pass_prof_query_limit);s.prof=true;s.prof_used=0;s.prof_recs.clear();s.prof_open=-1;s.prof_skipped=0;s.prof_frame=frame;s.prof_slot=pass_prof_slots++;
                }else ++pass_prof_skipped_slots;
            }
        }
        // Order against work of earlier submissions on this queue.
        {VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;vkCmdPipelineBarrier(s.cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&b,0,nullptr,0,nullptr);barrier_seq=write_seq;}
        for(auto q:s.occlusion_pools)vkCmdResetQueryPool(s.cmd,q,0,256);
        for(auto q:s.timestamp_pools)vkCmdResetQueryPool(s.cmd,q,0,64);
    }
    return s;
}
// Submit the recording slot. With wait, block until all submitted work is done.
void submit_slots(bool wait){
    Slot &s=slots[slot_index];
    if(s.recording){
        if(open_pass.cmd==s.cmd)end_open_pass();barrier_seq=0;
        // Occlusion results: GPU copies them into mapped memory (vkGetQueryPoolResults is slow on this driver).
        s.query_copied=0;if(s.occlusion_used){size_t used=s.occlusion_used;s.query_results=s.readback.alloc(used*8);
            for(size_t i=0;i*256<used;i++){uint32_t n=uint32_t(std::min<size_t>(256,used-i*256));vkCmdCopyQueryPoolResults(s.cmd,s.occlusion_pools[i],0,n,s.query_results.buffer,s.query_results.offset+i*256*8,8,VK_QUERY_RESULT_64_BIT);}
            VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};hb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;hb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(s.cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&hb,0,nullptr,0,nullptr);
            s.query_copied=used;}
        if(s.timestamps)vkCmdWriteTimestamp(s.cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,s.timestamps,1);
        VK(vkEndCommandBuffer(s.cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&s.cmd;
        {CPU_SCOPE(Submit);VK(vkQueueSubmit(cfg.queue,1,&si,s.fence));}++submit_by[sync_site];
        s.recording=false;s.pending=true;s.covers=submitted;slot_index^=1;++perf.submits;snapshot_reset();++slot_serial;
        pending_events.insert(pending_events.end(),recording_events.begin(),recording_events.end());recording_events.clear();
    }
    if(wait){retire_slot(slots[slot_index]);retire_slot(slots[slot_index^1]);}
}
// Pipelines persist across launches (the driver validates the header and
// ignores incompatible data), so later launches skip shader compilation.
size_t pipelines_unsaved=0;uint64_t pipeline_save_frame=0;
std::string pipeline_cache_path(){std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');return (cut==std::string::npos?dir:dir.substr(0,cut))+"/vk-pipeline-cache.bin";}
std::vector<uint8_t> load_pipeline_cache(){
    std::vector<uint8_t> data;FILE *f=fopen(pipeline_cache_path().c_str(),"rb");if(!f)return data;
    fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);
    if(n>32 && n<(256l<<20)){data.resize(size_t(n));if(fread(data.data(),1,data.size(),f)!=data.size())data.clear();}
    fclose(f);
    if(data.size()>=32){uint32_t header[4];memcpy(header,data.data(),16);if(header[1]!=1 || header[2]!=properties.vendorID || header[3]!=properties.deviceID || memcmp(data.data()+16,properties.pipelineCacheUUID,16))data.clear();}
    xbg_logf("NVN pipeline cache load bytes=%zu",data.size());return data;
}
void save_pipeline_cache(){
    if(!pipeline_cache || !pipelines_unsaved)return;
    size_t n=0;if(vkGetPipelineCacheData(cfg.device,pipeline_cache,&n,nullptr)!=VK_SUCCESS || !n)return;
    std::vector<uint8_t> data(n);if(vkGetPipelineCacheData(cfg.device,pipeline_cache,&n,data.data())!=VK_SUCCESS)return;
    std::string path=pipeline_cache_path(),tmp=path+".tmp";FILE *f=fopen(tmp.c_str(),"wb");if(!f)return;
    bool ok=fwrite(data.data(),1,n,f)==n;ok=fclose(f)==0&&ok;if(ok)ok=rename(tmp.c_str(),path.c_str())==0;
    warm::flush();xbg_logf("NVN pipeline cache save bytes=%zu new=%zu ok=%d",n,pipelines_unsaved,int(ok));if(ok)pipelines_unsaved=0;
}
void perf_frame(){
    if(perf_frame_timed)++perf_timed_frames;perf_frame_timed=perf_sample && stats.presents.load()%perf_sample==0;   // the next frame
    {bool was=ab_frame_on;ab_frame_on=ab_enabled();if(was!=ab_frame_on && ab_period>0){perf=Perf{};perf.window_start=timing_clock();set_cache_hits=set_cache_misses=0;return;}}
    if(pipelines_unsaved && stats.presents.load()>=pipeline_save_frame+300){pipeline_save_frame=stats.presents.load();save_pipeline_cache();}
    uint64_t now=timing_clock();if(!perf.window_start){perf.window_start=now;perf.frames=0;return;}
    if(++perf.frames<60)return;
    double wall=double(now-perf.window_start)/1e6,f=double(perf.frames);
    const double tf=perf_timed_frames?double(perf_timed_frames):1.0;   // parts and descriptor timers come from the timed frames only
    xbg_logf("NVN PERF v=1 sync=%d frames=%llu wall_ms=%.1f fps=%.2f submit_cpu_ms=%.2f present_cpu_ms=%.2f gpu_ms=%.2f gpu_samples=%llu submits=%.2f waits=%.2f sync_wb=%llu sync_rb=%llu sync_q=%llu ops=%.1f folded=%.1f passes=%.1f draws=%.1f dispatches=%.1f why=w%llu/p%llu/a%llu(%llu,%llu,%llu,%llu,%llu,%llu|%llu,%llu,%llu,%llu)/c%llu/f%llu/s%llu views=%llu snap_hit=%llu snap_miss=%llu snap_mb=%.1f arena_peak_mb=%.1f pipelines=%zu layouts=%zu samplers=%zu passes=%zu images=%zu host_segments=%zu host_mb=%.1f",
        int(sync_mode),(unsigned long long)perf.frames,wall,f*1000.0/wall,double(perf.submit_cpu_ns)/1e6/f,double(perf.present_cpu_ns)/1e6/f,perf.gpu_samples?double(perf.gpu_ns)/1e6/f:-1.0,(unsigned long long)perf.gpu_samples,
        double(perf.submits)/f,double(perf.waits)/f,(unsigned long long)perf.sync_writeback,(unsigned long long)perf.sync_readback,(unsigned long long)perf.sync_query,double(perf.ops)/f,double(perf.folded_clears)/f,double(perf.passes)/f,double(perf.draws)/f,double(perf.dispatches)/f,
        (unsigned long long)perf.why_wait,(unsigned long long)perf.why_poll,(unsigned long long)perf.why_access,(unsigned long long)perf.access_by[1],(unsigned long long)perf.access_by[2],(unsigned long long)perf.access_by[3],(unsigned long long)perf.access_by[4],(unsigned long long)perf.access_by[5],(unsigned long long)perf.access_by[6],(unsigned long long)perf.access_by[8],(unsigned long long)perf.access_by[9],(unsigned long long)perf.access_by[10],(unsigned long long)perf.access_by[11],(unsigned long long)perf.why_cmdwait,(unsigned long long)perf.why_finish,(unsigned long long)perf.why_slot,(unsigned long long)perf.pending_views,(unsigned long long)perf.snapshot_hits,(unsigned long long)perf.snapshot_misses,double(perf.snapshot_bytes)/1048576.0,double(perf.arena_peak)/1048576.0,perf_counts().pipelines,perf_counts().layouts,perf_counts().samplers,perf_counts().passes,perf_counts().images,perf_counts().host_segments,double(perf_counts().host_bytes)/1048576.0);
    if(!timing_active){
        static const char *names[]={"other","draw","dispatch","descriptor","pass","buffer","pipeline","submit","fence","present","cleanup","hash","image","log","program","cpuwrite","snapshot"};
        char line[1024];int at=snprintf(line,sizeof(line),"NVN PERF parts_ms_per_frame timed=%llu",(unsigned long long)perf_timed_frames);
        for(unsigned i=0;i<XbgCpuTiming::Count && at<int(sizeof(line))-48;i++){const auto &c=timing.counters[i];if(c.calls<perf_timing_before[i].calls)perf_timing_before[i]=XbgCpuTiming::Counter{};uint64_t ns=c.exclusive_ns-perf_timing_before[i].exclusive_ns,calls=c.calls-perf_timing_before[i].calls;at+=snprintf(line+at,sizeof(line)-at," %s=%.2f/%.1f",names[i],double(ns)/1e6/tf,double(calls)/tf);perf_timing_before[i]=c;}
        xbg_logf("%s",line);
    }
    xbg_logf("NVN PERF storage_snapshots_per_frame=%.1f union_mb_per_frame=%.2f",storage_snap_count/f,storage_pages.size()*4096.0/1048576.0/f);storage_pages.clear();storage_snap_count=0;
    xbg_logf("NVN PERF wtrack copied_kb_per_frame=%.1f dirty_pages_per_frame=%.1f mismatches=%llu fallbacks=%llu",wtrack::copied_bytes/1024.0/f,wtrack::dirty_pages/f,(unsigned long long)wtrack::mismatches,(unsigned long long)wtrack::fallbacks);wtrack::copied_bytes=wtrack::dirty_pages=0;
    xbg_logf("NVN PERF uniform_trimmed_bindings=%llu snap_mb_per_frame other=%.2f vertex=%.2f index=%.2f storage=%.2f texel=%.2f uniform=%.2f",(unsigned long long)uniform_trimmed,snap_site_bytes[0]/1048576.0/f,snap_site_bytes[1]/1048576.0/f,snap_site_bytes[2]/1048576.0/f,snap_site_bytes[3]/1048576.0/f,snap_site_bytes[4]/1048576.0/f,snap_site_bytes[5]/1048576.0/f);for(auto &b:snap_site_bytes)b=0;
    {static uint64_t pm=0,pe=0;xbg_logf("NVN PERF pipelines_new=%llu evicted=%llu warm_hits=%llu created=%llu create_ms=%.1f create_max_ms=%.1f warm_built=%llu warm_failed=%llu warm_ms=%.0f",(unsigned long long)(pipeline_counts.misses-pm),(unsigned long long)(pipeline_counts.evictions-pe),(unsigned long long)warm::hits,(unsigned long long)warm::creates,warm::create_ns/1e6,warm::create_max_ns/1e6,(unsigned long long)warm::built.load(),(unsigned long long)warm::failed.load(),warm::build_ns.load()/1e6);warm::hits=warm::creates=warm::create_ns=warm::create_max_ns=0;pm=pipeline_counts.misses;pe=pipeline_counts.evictions;}
    xbg_logf("NVN PERF setcache ab=%d hits=%.1f misses=%.1f template=%.1f plain=%.1f memo_hits=%.1f memo_misses=%.1f why=texel%.1f/simg%.1f/sw%.1f/dynbuf%.1f/store%.1f",int(ab_frame_on),set_cache_hits/f,set_cache_misses/f,template_updates/f,plain_updates/f,desc_memo_hits/f,desc_memo_misses/f,desc_memo_why[0]/f,desc_memo_why[1]/f,desc_memo_why[2]/f,desc_memo_why[3]/f,desc_memo_why[4]/f);desc_memo_hits=desc_memo_misses=0;for(auto &w:desc_memo_why)w=0;xbg_logf("NVN PERF keydiff shader%.0f pool%.0f ubo%.0f uaddr%.0f size%.0f vp%.0f alpha%.0f drv%.0f tex%.0f samp%.0f saddr%.0f same%.0f new%.0f len%.0f",key_diff[0]/f,key_diff[1]/f,key_diff[2]/f,key_diff[3]/f,key_diff[4]/f,key_diff[5]/f,key_diff[6]/f,key_diff[7]/f,key_diff[8]/f,key_diff[9]/f,key_diff[10]/f,key_diff[11]/f,key_diff[12]/f,key_diff[13]/f);for(auto &k:key_diff)k=0;
    xbg_logf("NVN PERF binds_skipped_per_frame=%.1f xmemo_hits=%.1f drops=%llu bumps=%llu/%llu/%llu/%llu/%llu pinned=%zu/%zu",binds_skipped/f,xmemo_hits/f,(unsigned long long)xmemo_drops,(unsigned long long)memo_bump[0],(unsigned long long)memo_bump[1],(unsigned long long)memo_bump[2],(unsigned long long)memo_bump[3],(unsigned long long)memo_bump[4],memo_pinned[0].size(),memo_pinned[1].size());binds_skipped=0;xmemo_hits=0;xbg_logf("NVN PERF noload_passes_per_frame=%.2f img_cache_hits=%.1f misses=%.1f why=%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f/%.1f",noload_passes/f,img_cache_hits/f,img_cache_misses/f,noload_why[0]/f,noload_why[1]/f,noload_why[2]/f,noload_why[3]/f,noload_why[4]/f,noload_why[5]/f,noload_why[6]/f,noload_why[7]/f);for(auto &w:noload_why)w=0;noload_passes=0;img_cache_hits=img_cache_misses=0;xbg_logf("NVN PERF desc_ns_per_frame memo_key=%.2f tmpl_update=%.2f alloc=%.2f setkey=%.2f build=%.2f hit_total=%.2f",desc_ns[0]/1e6/tf,desc_ns[1]/1e6/tf,desc_ns[2]/1e6/tf,desc_ns[3]/1e6/tf,desc_ns[4]/1e6/tf,desc_ns[5]/1e6/tf);for(auto &x:desc_ns)x=0;template_updates=plain_updates=0;set_cache_hits=set_cache_misses=0;
    xbg_logf("NVN PERF async=%d drains=%.2f drain_ms=%.2f pace_ms=%.2f waits=%.2f",int(gfx_async),gfx_drain_calls.exchange(0)/f,gfx_drain_ns.exchange(0)/1e6/f,gfx_pace_ns.exchange(0)/1e6/f,gfx_wait_calls.exchange(0)/f);xbg_logf("NVN PERF events defer=%d deferred=%.2f cpu_reads_pending=%llu epochs=%zu reports=%zu",int(defer_events),events_deferred/f,(unsigned long long)event_reads,epochs.size(),pending_reports.size());events_deferred=0;
    {static const char *site_names[SITE_COUNT]={"other","ring","present","event","arena","gpu_sync","complete","poll","legacy_wait"};
     char line[768];int at=snprintf(line,sizeof(line),"NVN PERF block_ms");
     for(int i=0;i<SITE_COUNT && at<int(sizeof(line))-40;i++)if(block_calls[i])at+=snprintf(line+at,sizeof(line)-at," %s=%.2f/%.2f",site_names[i],block_ns[i]/1e6/f,block_calls[i]/f);
     at+=snprintf(line+at,sizeof(line)-at," legacy_wait_total=%.2f submit_by",legacy_wait_ns/1e6/f);
     for(int i=0;i<SITE_COUNT && at<int(sizeof(line))-40;i++)if(submit_by[i])at+=snprintf(line+at,sizeof(line)-at," %s=%.2f",site_names[i],submit_by[i]/f);
     xbg_logf("%s",line);for(int i=0;i<SITE_COUNT;i++)block_ns[i]=block_calls[i]=submit_by[i]=0;legacy_wait_ns=0;}
    xbg_logf("NVN PERF fb_hits=%.1f fb_creates=%.2f fb_live=%zu",fb_hits/f,fb_creates/f,persistent_framebuffers.size());fb_hits=fb_creates=0;
    xbg_logf("NVN PERF texel resident=%zu/%zu hits=%.1f refresh=%.1f fallback=%.2f evict=%.2f",texel_resident[0].size(),texel_resident[1].size(),texel_hits/f,texel_refreshes/f,texel_fallbacks/f,texel_evictions/f);texel_hits=texel_refreshes=texel_fallbacks=texel_evictions=0;
    xbg_logf("NVN PERF cleanup_ms_per_frame queries=%.2f writebacks=%.2f(%.1fMB) views=%.2f retire=%.2f arena_pools=%.2f",cleanup_ns[0]/1e6/f,cleanup_ns[1]/1e6/f,writeback_bytes/1048576.0/f,cleanup_ns[2]/1e6/f,cleanup_ns[3]/1e6/f,cleanup_ns[4]/1e6/f);for(auto &c:cleanup_ns)c=0;writeback_bytes=0;
    xbg_logf("NVN PERF occlusion batch=%d queries_per_frame=%.2f",int(occlusion_batch_enabled),double(occlusion_queries_started)/f);
    xbg_logf("NVN PERF early_submit enabled=%d submits_per_frame=%.2f blocked_writeback_per_frame=%.2f",int(early_submit_enabled),early_submit_window/f,early_blocked_window/f);
    early_submit_window=early_blocked_window=0;refresh_early_submit();
    occlusion_queries_started=0;refresh_occlusion_batch(); // 仅窗口边界读取诊断开关 / Read the A/B marker only at window boundaries.
    perf=Perf{};perf.window_start=now;perf_timed_frames=0;
}
void occlusion_query(VkQueryPool &pool,uint32_t &index){
    Slot &s=recording_slot();index=s.occlusion_used%256;
    if(s.occlusion_used/256==s.occlusion_pools.size()){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_OCCLUSION;q.queryCount=256;VkQueryPool p;VK(vkCreateQueryPool(cfg.device,&q,nullptr,&p));s.occlusion_pools.push_back(p);end_open_pass();vkCmdResetQueryPool(s.cmd,p,0,256);}
    pool=s.occlusion_pools[s.occlusion_used/256];{auto &e=epochs[counter_epoch];e.q.push_back({&s,pool,index,0,false});s.query_refs.push_back({counter_epoch,e.size()-1});}++s.occlusion_used;
}
void timestamp_query(VkQueryPool &pool,uint32_t &index,const void *&owner){
    Slot &s=recording_slot();index=s.timestamp_used%64;owner=&s;
    if(s.timestamp_used/64==s.timestamp_pools.size()){VkQueryPoolCreateInfo q{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};q.queryType=VK_QUERY_TYPE_TIMESTAMP;q.queryCount=64;VkQueryPool p;VK(vkCreateQueryPool(cfg.device,&q,nullptr,&p));s.timestamp_pools.push_back(p);end_open_pass();vkCmdResetQueryPool(s.cmd,p,0,64);}
    pool=s.timestamp_pools[s.timestamp_used/64];++s.timestamp_used;
}
// Synchronous point: everything recorded so far has completed on the GPU.
void gpu_sync(uint64_t &reason){++reason;SiteScope site(SITE_GPU_SYNC);submit_slots(true);}
void retire_handle(std::function<void()> f){recording_slot().retire.handles.push_back(std::move(f));}
template<class T>void retire_object(std::shared_ptr<T> p){recording_slot().retire.objects.push_back(std::move(p));}
Buffer stream_buffer(size_t n,const void *data=nullptr,size_t copy=0,bool zero=true){
    Buffer b=recording_slot().stream.alloc(n);
    if(data && copy){need(copy<=n,"Buffer copy overflow");memcpy(b.mapped,data,copy);}
    if(zero && copy<n)memset(static_cast<uint8_t*>(b.mapped)+copy,0,n-copy);
    b.flush();return b;
}
Buffer readback_buffer(size_t n,const void *data=nullptr,size_t copy=0){
    Buffer b=recording_slot().readback.alloc(n);
    if(data && copy){need(copy<=n,"Buffer copy overflow");memcpy(b.mapped,data,copy);}else copy=0;
    if(copy<n)memset(static_cast<uint8_t*>(b.mapped)+copy,0,n-copy);   // only the bytes the copy leaves
    b.flush();return b;
}
// Per-submission snapshot cache of guest memory ranges. Guest command streams
// cannot change memory between draws except through in-stream CPU operations
// (uniform updates, buffer copies, events, writebacks), which invalidate it.
std::map<std::pair<uint64_t,uint64_t>,Buffer> snapshots,uniform_snapshots;
uint64_t snapshot_maxlen=0;
void desc_memo_reset();
// Direct-mapped front for the two snapshot maps: most lookups repeat within a submission (~12k of 15k per D2R camp frame).
// Entries carry snapshot_epoch, which every reset and invalidation bumps, so a stale entry never matches.
struct SnapFront{uint64_t address=0,bytes=0,epoch=0;Buffer buffer;};SnapFront snap_front[2][2048];uint64_t snapshot_epoch=1;
inline SnapFront &snap_front_for(int which,uint64_t a,uint64_t n){uint64_t x=a*0x9E3779B97F4A7C15ull^n*0xC2B2AE3D27D4EB4Full;return snap_front[which][(x^(x>>31))&2047];}
void snapshot_reset(){snapshots.clear();uniform_snapshots.clear();snapshot_maxlen=0;desc_memo_reset();++snapshot_epoch;++texel_write_epoch;}
void snapshot_invalidate(uint64_t address,uint64_t bytes){
    ++snapshot_epoch;
    if(address<texel_hi && address+bytes>texel_lo)texel_touch(address,bytes);
    uint64_t lo=address>snapshot_maxlen?address-snapshot_maxlen:0;
    for(auto *m:{&snapshots,&uniform_snapshots})for(auto it=m->lower_bound({lo,0});it!=m->end() && it->first.first<address+bytes;){uint64_t b=it->first.first,e=b+it->first.second;if(address<e && address+bytes>b)it=m->erase(it);else ++it;}
}
// GPU-resident view of a guest range whose newest content is a pending
// (not yet written back) GPU result. The producing operation is followed by a
// full barrier, so later reads of the view observe it in queue order.
bool pending_view(uint64_t address,uint64_t bytes,VkDeviceSize align,Buffer &out){
    for(auto w=pending_writebacks.rbegin();w!=pending_writebacks.rend();++w){
        if(!(address<w->addr+w->bytes && address+bytes>w->addr))continue;
        if(address<w->addr || address+bytes>w->addr+w->bytes)return false;
        VkDeviceSize delta=address-w->addr;if(align>1 && (w->buffer.offset+delta)%align)return false;
        out=w->buffer;out.offset+=delta;out.mapped=static_cast<uint8_t*>(out.mapped)+delta;out.size=bytes;++perf.pending_views;return true;
    }
    return false;
}
// Write-tracked storage snapshots. Large read-only storage bindings (one 19.5 MB buffer per frame in the field,
// of which only ~7 pages change) used to be copied whole every frame. Guest pages are write-protected after a
// copy; a SIGSEGV on a tracked page marks it dirty and re-enables writes, so each frame copies only dirty pages
// into a persistent per-address buffer (two versions; a version is reused only once its frame slot retired).
// Every 240th use is verified against guest memory in full. files/gfx-no-wtrack disables it.
namespace wtrack {
constexpr size_t min_bytes=1u<<20;constexpr unsigned max_regions=8,versions=2;bool enabled=true;
struct Region{std::atomic<uintptr_t> begin{0},end{0};std::atomic<uint8_t> *dirty=nullptr;};
Region regions[max_regions];std::atomic<unsigned> nregions{0};struct sigaction previous{};bool installed=false;
void handler(int sig,siginfo_t *si,void *uc){
    uintptr_t a=uintptr_t(si->si_addr);unsigned n=nregions.load(std::memory_order_acquire);
    for(unsigned i=0;i<n;i++){uintptr_t b=regions[i].begin.load(std::memory_order_relaxed),e=regions[i].end.load(std::memory_order_relaxed);
        if(a>=b && a<e){size_t pg=(a-b)>>12;if(mprotect(reinterpret_cast<void*>(b+(pg<<12)),4096,PROT_READ|PROT_WRITE)==0){regions[i].dirty[pg].store(1,std::memory_order_release);return;}break;}}
    if(previous.sa_flags&SA_SIGINFO){if(previous.sa_sigaction){previous.sa_sigaction(sig,si,uc);return;}}
    else if(previous.sa_handler!=SIG_DFL && previous.sa_handler!=SIG_IGN && previous.sa_handler){previous.sa_handler(sig);return;}
    signal(sig,SIG_DFL); // return and re-fault into the default crash path
}
struct Version{VkBuffer buffer{};VkDeviceMemory memory{};uint8_t *mapped=nullptr;bool coherent=true;uint64_t last_use=0;std::vector<uint8_t> stale;};
struct Tracked{uint64_t address=0;size_t bytes=0;uintptr_t pbegin=0;size_t npages=0;unsigned region=0;Version v[versions];uint64_t uses=0;};
std::map<std::pair<uint64_t,size_t>,std::unique_ptr<Tracked>> tracked;bool exhausted=false;
uint64_t copied_bytes=0,dirty_pages=0,mismatches=0,fallbacks=0;
bool activate(Tracked &t){
    if(!installed){struct sigaction sa{};sa.sa_sigaction=handler;sa.sa_flags=SA_SIGINFO|SA_ONSTACK|SA_NODEFER;sigemptyset(&sa.sa_mask);if(sigaction(SIGSEGV,&sa,&previous)){enabled=false;return false;}installed=true;}
    unsigned n=nregions.load();if(n>=max_regions)return false;
    t.pbegin=uintptr_t(t.address)&~uintptr_t(4095);uintptr_t pend=(uintptr_t(t.address)+t.bytes+4095)&~uintptr_t(4095);
    for(unsigned i=0;i<n;i++){uintptr_t b=regions[i].begin.load(),e=regions[i].end.load();if(t.pbegin<e && pend>b)return false;} // overlapping regions unsupported
    size_t npages=(pend-t.pbegin)>>12;
    auto *dirty=new std::atomic<uint8_t>[npages];for(size_t i=0;i<npages;i++)dirty[i].store(0);
    for(auto &v:t.v){
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};ci.size=t.bytes;ci.usage=arena_usage;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VK(vkCreateBuffer(cfg.device,&ci,nullptr,&v.buffer));
        VkMemoryRequirements r;vkGetBufferMemoryRequirements(cfg.device,v.buffer,&r);uint32_t type=find_memory(r.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        v.coherent=(memories.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)!=0;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=r.size;ai.memoryTypeIndex=type;VK(vkAllocateMemory(cfg.device,&ai,nullptr,&v.memory));VK(vkBindBufferMemory(cfg.device,v.buffer,v.memory,0));
        void *p=nullptr;VK(vkMapMemory(cfg.device,v.memory,0,VK_WHOLE_SIZE,0,&p));v.mapped=static_cast<uint8_t*>(p);v.stale.assign(npages,1);
    }
    regions[n].dirty=dirty;regions[n].begin.store(t.pbegin);regions[n].end.store(pend);nregions.store(n+1,std::memory_order_release);t.region=n;
    if(mprotect(reinterpret_cast<void*>(t.pbegin),pend-t.pbegin,PROT_READ)){regions[n].end.store(t.pbegin);return false;}
    t.npages=npages;xbg_logf("NVN wtrack region=%u addr=%#llx bytes=%zu pages=%zu",n,(unsigned long long)t.address,t.bytes,t.npages);return true;
}
void copy_page(Tracked &t,Version &v,size_t pg){
    uint64_t lo=std::max<uint64_t>(t.address,t.pbegin+(pg<<12)),hi=std::min<uint64_t>(t.address+t.bytes,t.pbegin+((pg+1)<<12));
    if(hi>lo){memcpy(v.mapped+(lo-t.address),reinterpret_cast<const void*>(uintptr_t(lo)),size_t(hi-lo));copied_bytes+=hi-lo;}
}
bool snapshot(uint64_t address,size_t bytes,Buffer &out){
    auto &slot=tracked[{address,bytes}];
    if(!slot){if(exhausted)return false;slot=std::make_unique<Tracked>();slot->address=address;slot->bytes=bytes;if(!activate(*slot)){slot->npages=0;exhausted=nregions.load()>=max_regions;}}
    Tracked &t=*slot;if(!t.npages)return false;
    // Clear the flag, re-protect, then copy: a write after the protect faults and re-marks the page.
    Region &r=regions[t.region];
    for(size_t pg=0;pg<t.npages;){
        if(!r.dirty[pg].exchange(0,std::memory_order_acq_rel)){++pg;continue;}
        size_t end=pg+1;while(end<t.npages && r.dirty[end].exchange(0,std::memory_order_acq_rel))++end;
        if(mprotect(reinterpret_cast<void*>(t.pbegin+(pg<<12)),(end-pg)<<12,PROT_READ)){enabled=false;xbg_logf("NVN wtrack mprotect failed errno=%d; disabled",errno);return false;}
        for(size_t i=pg;i<end;i++)for(auto &v:t.v)v.stale[i]=1;dirty_pages+=end-pg;pg=end;
    }
    Version *pick=nullptr;
    for(auto &v:t.v)if(v.last_use==slot_serial){pick=&v;break;}
    if(pick && std::find(pick->stale.begin(),pick->stale.end(),uint8_t(1))!=pick->stale.end())pick=nullptr; // changed within this slot: take another version
    if(!pick)for(auto &v:t.v)if(v.last_use+2<=slot_serial && (!pick || v.last_use>pick->last_use))pick=&v;
    if(!pick){++fallbacks;return false;}
    for(size_t pg=0;pg<t.npages;pg++)if(pick->stale[pg]){copy_page(t,*pick,pg);pick->stale[pg]=0;}
    if(++t.uses%240==0){size_t bad=0;for(size_t pg=0;pg<t.npages;pg++){uint64_t lo=std::max<uint64_t>(t.address,t.pbegin+(pg<<12)),hi=std::min<uint64_t>(t.address+t.bytes,t.pbegin+((pg+1)<<12));
            if(hi>lo && memcmp(pick->mapped+(lo-t.address),reinterpret_cast<const void*>(uintptr_t(lo)),size_t(hi-lo))){++bad;copy_page(t,*pick,pg);}}
        if(bad){mismatches+=bad;xbg_logf("NVN wtrack MISMATCH addr=%#llx pages=%zu (untracked writes; resynced)",(unsigned long long)t.address,bad);}}
    if(!pick->coherent){VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};mr.memory=pick->memory;mr.size=VK_WHOLE_SIZE;VK(vkFlushMappedMemoryRanges(cfg.device,1,&mr));}
    pick->last_use=slot_serial;out=Buffer{};out.buffer=pick->buffer;out.memory=pick->memory;out.offset=0;out.size=bytes;out.mapped=pick->mapped;out.coherent=pick->coherent;return true;
}
}
Buffer snapshot_slow(uint64_t address,size_t bytes,VkDeviceSize align);
Buffer snapshot(uint64_t address,size_t bytes,VkDeviceSize align=4){CPU_SCOPE(Snapshot);
    SnapFront &f=snap_front_for(0,address,bytes);if(f.epoch==snapshot_epoch && f.address==address && f.bytes==bytes){++perf.snapshot_hits;return f.buffer;}
    Buffer b=snapshot_slow(address,bytes,align);f.address=address;f.bytes=bytes;f.epoch=snapshot_epoch;f.buffer=b;return b;
}
Buffer snapshot_slow(uint64_t address,size_t bytes,VkDeviceSize align){
    auto key=std::make_pair(address,uint64_t(bytes));auto it=snapshots.find(key);
    if(it!=snapshots.end()){++perf.snapshot_hits;return it->second;}
    {Buffer view;if(pending_view(address,bytes,align,view)){snapshots.emplace(key,view);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,bytes);return view;}}
    ++perf.snapshot_misses;perf.snapshot_bytes+=bytes;snap_site_bytes[snap_site<6?snap_site:0]+=bytes;if(snap_site==3){static unsigned logged=0;if(logged<40 && bytes>=65536){++logged;xbg_logf("NVN PERF storage_snapshot addr=%#llx bytes=%zu frame=%llu",(unsigned long long)address,bytes,(unsigned long long)stats.presents.load());}++storage_snap_count;for(uint64_t pg=address>>12;pg<=(address+bytes-1)>>12;pg++)storage_pages.insert(pg);}cpu_access(address,bytes,1);
    if(snap_site==3 && bytes>=wtrack::min_bytes && wtrack::enabled){Buffer t;if(wtrack::snapshot(address,bytes,t)){snapshots.emplace(key,t);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,bytes);return t;}}
    Buffer b=stream_buffer(bytes,reinterpret_cast<const void*>(address),bytes,false);snapshots.emplace(key,b);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,bytes);return b;
}
struct Commands {
    VkCommandBuffer command{};bool draw=false;
    // Non-draw operations keep the former submit-per-operation semantics: they
    // end any open pass and are fenced by full barriers before and after.
    explicit Commands(bool is_draw=false):draw(is_draw){Slot &s=recording_slot();command=s.cmd;++perf.ops;++s.ops;
        if(!draw){end_open_pass();full_barrier(command);}}
    void finish(){if(!draw){end_open_pass();full_barrier(command);}}
};
VkFormat format(int f) {
    static const VkFormat table[]={VK_FORMAT_UNDEFINED,
        VK_FORMAT_R8_UNORM,VK_FORMAT_R8_SNORM,VK_FORMAT_R8_UINT,VK_FORMAT_R8_SINT,
        VK_FORMAT_R16_SFLOAT,VK_FORMAT_R16_UNORM,VK_FORMAT_R16_SNORM,VK_FORMAT_R16_UINT,VK_FORMAT_R16_SINT,
        VK_FORMAT_R32_SFLOAT,VK_FORMAT_R32_UINT,VK_FORMAT_R32_SINT,
        VK_FORMAT_R8G8_UNORM,VK_FORMAT_R8G8_SNORM,VK_FORMAT_R8G8_UINT,VK_FORMAT_R8G8_SINT,
        VK_FORMAT_R16G16_SFLOAT,VK_FORMAT_R16G16_UNORM,VK_FORMAT_R16G16_SNORM,VK_FORMAT_R16G16_UINT,VK_FORMAT_R16G16_SINT,
        VK_FORMAT_R32G32_SFLOAT,VK_FORMAT_R32G32_UINT,VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R8G8B8_UNORM,VK_FORMAT_R8G8B8_SNORM,VK_FORMAT_R8G8B8_UINT,VK_FORMAT_R8G8B8_SINT,
        VK_FORMAT_R16G16B16_SFLOAT,VK_FORMAT_R16G16B16_UNORM,VK_FORMAT_R16G16B16_SNORM,VK_FORMAT_R16G16B16_UINT,VK_FORMAT_R16G16B16_SINT,
        VK_FORMAT_R32G32B32_SFLOAT,VK_FORMAT_R32G32B32_UINT,VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_R8G8B8A8_SNORM,VK_FORMAT_R8G8B8A8_UINT,VK_FORMAT_R8G8B8A8_SINT,
        VK_FORMAT_R16G16B16A16_SFLOAT,VK_FORMAT_R16G16B16A16_UNORM,VK_FORMAT_R16G16B16A16_SNORM,VK_FORMAT_R16G16B16A16_UINT,VK_FORMAT_R16G16B16A16_SINT,
        VK_FORMAT_R32G32B32A32_SFLOAT,VK_FORMAT_R32G32B32A32_UINT,VK_FORMAT_R32G32B32A32_SINT,
        VK_FORMAT_S8_UINT,VK_FORMAT_D16_UNORM,VK_FORMAT_X8_D24_UNORM_PACK32,VK_FORMAT_D32_SFLOAT,VK_FORMAT_D24_UNORM_S8_UINT,VK_FORMAT_D32_SFLOAT_S8_UINT};
    if(f>0 && f<int(sizeof(table)/sizeof(*table))) return table[f];
    // NVN ASTC 0x79-0x86 (LDR) / 0x87-0x94 (sRGB), order 4x4 5x4 5x5 6x5 6x6 8x5 8x6 8x8 10x5 10x6 10x8 10x10 12x10 12x12 = Vulkan order.
    if(f>=0x79 && f<=0x94){int k=(f-0x79)%14;return VkFormat(int(VK_FORMAT_ASTC_4x4_UNORM_BLOCK)+2*k+(f>=0x87?1:0));}
    switch(f){case 0x37:case 0x38:return VK_FORMAT_R8G8B8A8_SRGB;case 0x3d:return VK_FORMAT_A2B10G10R10_UNORM_PACK32;case 0x3e:return VK_FORMAT_A2B10G10R10_UINT_PACK32;case 0x3f:return VK_FORMAT_B10G11R11_UFLOAT_PACK32;case 0x40:return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;case 0x6d:return VK_FORMAT_R8G8B8A8_UNORM;case 0x9a:return VK_FORMAT_B8G8R8A8_UNORM;default:xbg_fatalf("Unsupported NVN format %#x",f);}
}
VkImageAspectFlags aspect(int f){if(f==0x31)return VK_IMAGE_ASPECT_STENCIL_BIT;if(f>=0x32 && f<=0x34)return VK_IMAGE_ASPECT_DEPTH_BIT;if(f==0x35 || f==0x36)return VK_IMAGE_ASPECT_DEPTH_BIT|VK_IMAGE_ASPECT_STENCIL_BIT;return VK_IMAGE_ASPECT_COLOR_BIT;}
VkFormat decoded_format(uint32_t pixels) {
    switch(pixels){
    case XBG_RGBA8_UNORM:return VK_FORMAT_R8G8B8A8_UNORM;
    case XBG_RGBA8_SRGB:return VK_FORMAT_R8G8B8A8_SRGB;
    case XBG_R16_UNORM:return VK_FORMAT_R16_UNORM;
    case XBG_RG16_UNORM:return VK_FORMAT_R16G16_UNORM;
    case XBG_R16_SNORM:return VK_FORMAT_R16_SNORM;
    case XBG_RG16_SNORM:return VK_FORMAT_R16G16_SNORM;
    case XBG_RGBA16_SFLOAT:return VK_FORMAT_R16G16B16A16_SFLOAT;
    default:xbg_fatalf("Invalid decoded BC pixel format %u",pixels);
    }
}
// Read-only capability report, logged once at initialization: what this device actually supports, so a missing capability can be
// named precisely (a Vulkan 1.1 driver lacks much of what 1.3 drivers report) instead of being read off the first fatal error.
void capability_report(){
    const VkPhysicalDeviceFeatures &f=cfg.enabled_core;const VkPhysicalDeviceLimits &l=properties.limits;
    xbg_logf("NVN caps device=\"%s\" vendor=%#x id=%#x api=%u.%u.%u driver=%#x",properties.deviceName,properties.vendorID,properties.deviceID,
        VK_API_VERSION_MAJOR(properties.apiVersion),VK_API_VERSION_MINOR(properties.apiVersion),VK_API_VERSION_PATCH(properties.apiVersion),properties.driverVersion);
    xbg_logf("NVN caps features drawParameters=%u storageReadWithoutFormat=%u storageWriteWithoutFormat=%u storageExtendedFormats=%u gatherExtended=%u "
        "tessellation=%u geometry=%u occlusionQueryPrecise=%u depthBounds=%u depthClamp=%u fillModeNonSolid=%u bc=%u astcLdr=%u etc2=%u "
        "anisotropy=%u logicOp=%u dualSrcBlend=%u independentBlend=%u clipDistance=%u cullDistance=%u float64=%u int64=%u int16=%u",
        unsigned(cfg.enabled_shader_draw_parameters),f.shaderStorageImageReadWithoutFormat,f.shaderStorageImageWriteWithoutFormat,f.shaderStorageImageExtendedFormats,
        f.shaderImageGatherExtended,f.tessellationShader,f.geometryShader,f.occlusionQueryPrecise,f.depthBounds,f.depthClamp,f.fillModeNonSolid,
        f.textureCompressionBC,f.textureCompressionASTC_LDR,f.textureCompressionETC2,f.samplerAnisotropy,f.logicOp,f.dualSrcBlend,f.independentBlend,
        f.shaderClipDistance,f.shaderCullDistance,f.shaderFloat64,f.shaderInt64,f.shaderInt16);
    xbg_logf("NVN caps subgroup size=%u stages=%#x ops=%#x quadAllStages=%u",subgroup.subgroupSize,subgroup.supportedStages,subgroup.supportedOperations,subgroup.quadOperationsInAllStages);
    xbg_logf("NVN caps limits ubRange=%u sbRange=%u ubDyn=%u sbDyn=%u ubAlign=%llu sbAlign=%llu texelAlign=%llu stageSamplers=%u stageSampled=%u "
        "stageStorageImages=%u stageUB=%u stageSB=%u boundSets=%u colorAttachments=%u sharedMemory=%u workgroupInvocations=%u viewports=%u maxAnisotropy=%g",
        l.maxUniformBufferRange,l.maxStorageBufferRange,l.maxDescriptorSetUniformBuffersDynamic,l.maxDescriptorSetStorageBuffersDynamic,
        (unsigned long long)l.minUniformBufferOffsetAlignment,(unsigned long long)l.minStorageBufferOffsetAlignment,(unsigned long long)l.minTexelBufferOffsetAlignment,
        l.maxPerStageDescriptorSamplers,l.maxPerStageDescriptorSampledImages,l.maxPerStageDescriptorStorageImages,l.maxPerStageDescriptorUniformBuffers,
        l.maxPerStageDescriptorStorageBuffers,l.maxBoundDescriptorSets,l.maxColorAttachments,l.maxComputeSharedMemorySize,l.maxComputeWorkGroupInvocations,
        l.maxViewports,double(l.maxSamplerAnisotropy));
    // Every core format up to ASTC 12x12 sRGB (BC, ETC2 and ASTC included) as vkformat:optimalTilingFeatures/bufferFeatures
    // (VkFormatFeatureFlagBits); formats with neither are left out.
    std::string line;unsigned n=0;
    for(int v=1;v<=int(VK_FORMAT_ASTC_12x12_SRGB_BLOCK);v++){const VkFormatProperties &p=format_properties(VkFormat(v));if(!p.optimalTilingFeatures && !p.bufferFeatures)continue;
        char item[40];snprintf(item,sizeof(item)," %d:%#x/%#x",v,p.optimalTilingFeatures,p.bufferFeatures);line+=item;
        if(++n%10==0){xbg_logf("NVN caps formats%s",line.c_str());line.clear();}}
    if(!line.empty())xbg_logf("NVN caps formats%s",line.c_str());
}
struct Image {
    tex_t source{}; uint64_t provenance_epoch=0,provenance_writer=0;unsigned provenance_write_site=0; VkImage image{};VkDeviceMemory memory{};VkFormat fmt{};VkImageLayout layout=VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspects{};VkImageUsageFlags usage{};uint32_t width{},height{},depth=1,levels{},layers{},bpp{};VkSampleCountFlagBits samples=VK_SAMPLE_COUNT_1_BIT;bool uploaded=false,gpu_written=false,decoded=false;
    uint64_t cpu_hash=0,checked_submission=UINT64_MAX,last_write=0;uint32_t stable_checks=0;
    std::map<std::array<uint32_t,8>,VkImageView> views;std::vector<std::vector<uint64_t>> fb_keys; // fb_keys: persistent framebuffers using these views
    ~Image(){forget_framebuffers(*this);for(auto &v:views)vkDestroyImageView(cfg.device,v.second,nullptr);if(image)vkDestroyImage(cfg.device,image,nullptr);if(memory)vkFreeMemory(cfg.device,memory,nullptr);}
    void transition(VkCommandBuffer cmd,VkImageLayout next){
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.oldLayout=layout;b.newLayout=next;b.image=image;b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange={aspects,0,levels,0,layers};b.srcAccessMask=layout==VK_IMAGE_LAYOUT_UNDEFINED?0:VK_ACCESS_MEMORY_WRITE_BIT|VK_ACCESS_MEMORY_READ_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,0,nullptr,0,nullptr,1,&b);layout=next;
    }
    VkFormat view_format(const view_t *v) const {
        if(!v || !v->set_format || v->format==source.d.format)return fmt;
        if(decoded){
            need(xbg_bc_view_compatible(source.d.format,v->format),"BC texture view must retain compression and signedness");
            xbg_bc_format info;need(xbg_bc_format_info(v->format,&info)==0,"Invalid BC view format");
            return decoded_format(info.pixels);
        }
        int bw,bh,bytes=nvn_fmt_bpb(v->format,&bw,&bh);
        need(aspects==VK_IMAGE_ASPECT_COLOR_BIT && aspect(v->format)==aspects && bw==1 && bh==1 && bytes==int(bpp),"Incompatible NVN texture view format class");
        return format(v->format);
    }
    VkImageView view(const view_t *v,bool attachment=false,bool storage=false){
        uint32_t base=v&&v->set_levels?v->base_level:0,n=v&&v->set_levels?v->nlevels:levels;
        uint32_t layer=v&&v->set_layers?v->min_layer:0,ls=v&&v->set_layers?v->nlayers:layers;
        need(base<levels && n && n<=levels-base && layer<layers && ls && ls<=layers-layer,"Invalid texture view range");
        if(attachment){need(ls==1,"Layered rendering not implemented");n=1;}
        VkFormat vf=view_format(v);
        int target=v&&v->set_target?v->target:source.d.target;
        VkImageViewType type=target==2?VK_IMAGE_VIEW_TYPE_3D:target==0?VK_IMAGE_VIEW_TYPE_1D:target==3?VK_IMAGE_VIEW_TYPE_1D_ARRAY:target==4||target==6?VK_IMAGE_VIEW_TYPE_2D_ARRAY:VK_IMAGE_VIEW_TYPE_2D;
        need(!attachment || source.d.target!=2,"Rendering directly to volume slices requires 2D compatible views");
        if(!attachment && (target==8 || target==9))type=target==8?VK_IMAGE_VIEW_TYPE_CUBE:VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        if(attachment){need(source.d.target!=0 && source.d.target!=3,"1D render targets require 2D storage adaptation");type=VK_IMAGE_VIEW_TYPE_2D;}
        VkImageAspectFlags view_aspects=!attachment && (aspects&VK_IMAGE_ASPECT_DEPTH_BIT)?VK_IMAGE_ASPECT_DEPTH_BIT:aspects;
        VkImageUsageFlags view_usage=attachment?(aspects==VK_IMAGE_ASPECT_COLOR_BIT?VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT):storage?VK_IMAGE_USAGE_STORAGE_BIT:VK_IMAGE_USAGE_SAMPLED_BIT;
        need((usage&view_usage)==view_usage,"Image was not allocated for requested view usage");
        const VkFormatProperties &supported=format_properties(vf);
        VkFormatFeatureFlags feature=attachment?(aspects==VK_IMAGE_ASPECT_COLOR_BIT?VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT:VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT):storage?VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT:VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        need(supported.optimalTilingFeatures&feature,"Actual view format does not support requested usage");
        std::array<uint32_t,8> key{base,n,layer,ls,uint32_t(vf),uint32_t(type),view_aspects,view_usage};auto it=views.find(key);if(it!=views.end())return it->second;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};vi.image=image;vi.viewType=type;vi.format=vf;vi.subresourceRange={aspects,base,n,layer,ls};
        vi.subresourceRange.aspectMask=view_aspects;
        VkImageViewUsageCreateInfo vu{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};vu.usage=view_usage;vi.pNext=&vu;
        unsigned view_code=v&&v->set_format?v->format:source.d.format;
        if(!attachment && !storage && (view_code==0x37 || view_code==0x6d))vi.components.a=VK_COMPONENT_SWIZZLE_ONE;
        VkImageView result;VK(vkCreateImageView(cfg.device,&vi,nullptr,&result));views[key]=result;return result;
    }
};
std::map<uint64_t,std::shared_ptr<Image>> images;
// Address-ordered index of image storage, rebuilt lazily when images change.
struct ImageRange {uint64_t begin,end;Image *image;};
std::vector<ImageRange> image_index;bool image_index_dirty=true;uint64_t image_index_maxlen=0;
void rebuild_image_index(){
    image_index.clear();image_index_maxlen=0;
    for(auto &e:images){uint64_t b=uintptr_t(e.second->source.mem);if(!b)continue;uint64_t n=e.second->source.L.size;image_index.push_back({b,b+n,e.second.get()});image_index_maxlen=std::max(image_index_maxlen,n);}
    std::sort(image_index.begin(),image_index.end(),[](const ImageRange &a,const ImageRange &b){return a.begin<b.begin;});image_index_dirty=false;
}
template<class F> void for_images_overlapping(uint64_t address,uint64_t bytes,F f){
    if(image_index_dirty)rebuild_image_index();
    uint64_t lo=address>image_index_maxlen?address-image_index_maxlen:0;
    auto it=std::lower_bound(image_index.begin(),image_index.end(),lo,[](const ImageRange &r,uint64_t v){return r.begin<v;});
    for(;it!=image_index.end() && it->begin<address+bytes;++it)if(it->end>address)f(*it->image);
}
// A sampled/storage read inside a draw: barrier only if the image may have been
// written since the last full barrier, is an attachment of the open pass, or
// needs a layout change.
void prepare_read(Image &im,VkCommandBuffer cmd){
    bool in_open=open_pass.cmd && std::find(open_pass.images.begin(),open_pass.images.end(),&im)!=open_pass.images.end();
    if(im.layout!=VK_IMAGE_LAYOUT_GENERAL || im.last_write>barrier_seq || in_open){
        end_open_pass();if(im.layout!=VK_IMAGE_LAYOUT_GENERAL)im.transition(cmd,VK_IMAGE_LAYOUT_GENERAL);full_barrier(cmd);
    }
}
void prepare_attachment(Image &im,VkCommandBuffer cmd){
    bool in_open=open_pass.cmd && std::find(open_pass.images.begin(),open_pass.images.end(),&im)!=open_pass.images.end();
    if(im.layout!=VK_IMAGE_LAYOUT_GENERAL && !in_open){end_open_pass();im.transition(cmd,VK_IMAGE_LAYOUT_GENERAL);}
    im.last_write=++write_seq;
}
void provenance_image(const char *event,const tex_t *t,const Image &im,const view_t *v=nullptr){
    if(!provenance_file || !provenance.window(stats.presents.load()))return;
    provenance_log("IMAGE event=%s requested=%llu source=%llu image=%p mem=%p pool=%p off=%lld bytes=%llu createEpoch=%llu currentEpoch=%llu writer=%llu writeSite=%u writerPhase=declared-or-complete-range-unspecified uploaded=%u gpu=%u fmt=%d target=%d extent=%dx%dx%d levels=%d samples=%d flags=%x stride=%d pkg=%d viewFmt=%d mip=%d/%d layer=%d/%d",
    event,(unsigned long long)t->serial,(unsigned long long)im.source.serial,(void*)im.image,t->mem,(void*)(t->mem && t->off>=0 && uint64_t(t->off)<=uint64_t(t->mem)?uint64_t(t->mem)-uint64_t(t->off):0),(long long)t->off,(unsigned long long)t->L.size,(unsigned long long)im.provenance_epoch,(unsigned long long)provenance.epoch((t->mem && t->off>=0 && uint64_t(t->off)<=uint64_t(t->mem)?uint64_t(t->mem)-uint64_t(t->off):0)),(unsigned long long)im.provenance_writer,im.provenance_write_site,im.uploaded,im.gpu_written,t->d.format,t->d.target,t->d.width,t->d.height,t->d.depth,t->d.levels,t->d.samples,t->d.flags,t->d.stride,t->d.pkg_block_h,v&&v->set_format?v->format:t->d.format,v&&v->set_levels?v->base_level:0,v&&v->set_levels?v->nlevels:int(im.levels),v&&v->set_layers?v->min_layer:0,v&&v->set_layers?v->nlayers:int(im.layers));
}
void provenance_write(Image &im,unsigned site){im.provenance_writer=++provenance.sequence;im.provenance_write_site=site;provenance_image("WRITE",&im.source,im);}

// Invoked only after a completed game draw/clear fence. Read the real Vulkan
// subresources and restore their original layout before guest execution resumes.
void capture_image(Image &im,const view_t *view,const char *role){
    if(!capture.active)return;
    uint32_t base=view&&view->set_levels?view->base_level:0,levels=view&&view->set_levels?view->nlevels:im.levels;
    uint32_t layer=view&&view->set_layers?view->min_layer:0,layers=view&&view->set_layers?view->nlayers:im.layers;
    capture_log("IMAGE role=%s serial=%llu image=%p source=%p sourceBytes=%llu format=%#x vk=%u extent=%ux%ux%u levels=%u baseMip=%u count=%u baseLayer=%u layers=%u layout=%u uploaded=%u gpuWritten=%u",role,(unsigned long long)im.source.serial,(void*)im.image,im.source.mem,(unsigned long long)im.source.L.size,im.source.d.format,unsigned(im.fmt),im.width,im.height,im.depth,im.levels,base,levels,layer,layers,unsigned(im.layout),im.uploaded,im.gpu_written);
    if(im.samples!=VK_SAMPLE_COUNT_1_BIT || im.layout==VK_IMAGE_LAYOUT_UNDEFINED){capture_log("SKIP role=%s reason=uninitialized-or-multisample",role);return;}
    if(base>=im.levels || levels>im.levels-base || layer>=im.layers || layers>im.layers-layer){capture_log("SKIP role=%s reason=invalid-view",role);return;}
    unsigned subresources=0;
    for(uint32_t l=layer;l<layer+layers;l++)for(uint32_t m=base;m<base+levels;m++){
        if(++subresources>16){capture_log("SKIP role=%s reason=16-subresource-limit",role);return;}
        uint32_t w=std::max(1u,im.width>>m),h=std::max(1u,im.height>>m),z=std::max(1u,im.depth>>m);
        VkImageAspectFlags plane=im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?VK_IMAGE_ASPECT_DEPTH_BIT:im.aspects;
        uint32_t bpp=im.bpp;if(plane==VK_IMAGE_ASPECT_DEPTH_BIT){xbg_depth_format f;if(xbg_depth_format_info(im.source.d.format,&f)){capture_log("SKIP reason=unknown-depth-format");return;}bpp=im.source.d.format==0x32 && depth16_as_d24?4:f.depth_bytes;}
        uint64_t bytes=uint64_t(w)*h*z*bpp;
        if(!bytes || bytes>capture_limit-capture.bytes){capture_log("SKIP role=%s mip=%u layer=%u bytes=%llu reason=64MiB-budget",role,m,l,(unsigned long long)bytes);continue;}
        Buffer buffer=readback_buffer(size_t(bytes));Commands command;VkImageLayout saved=im.layout;
        im.transition(command.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);VkBufferImageCopy region{};region.imageSubresource={plane,m,l,1};region.imageExtent={w,h,z};
        region.bufferOffset=buffer.offset;vkCmdCopyImageToBuffer(command.command,im.image,im.layout,buffer.buffer,1,&region);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;host.srcQueueFamilyIndex=host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;host.buffer=buffer.buffer;host.offset=buffer.offset;host.size=buffer.size;
        vkCmdPipelineBarrier(command.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&host,0,nullptr);
        im.transition(command.command,saved);command.finish();gpu_sync(perf.sync_readback);buffer.invalidate();
        char label[160];snprintf(label,sizeof(label),"%s-s%llu-m%u-l%u",role,(unsigned long long)im.source.serial,m,l);
        capture_log("PIXELS label=%s vk=%u aspect=%u width=%u height=%u depth=%u rowBytes=%u restoredLayout=%u",label,unsigned(im.fmt),plane,w,h,z,w*bpp,unsigned(im.layout));
        capture_data(label,buffer.mapped,size_t(bytes));
    }
}
std::map<uintptr_t,size_t> pools;
struct RegisteredTexture{tex_t texture{};view_t view{};bool has_view=false;uint64_t revision=0;};
// 只比较可影响图像/视图的语义字段，忽略 padding、snapshot 指针及未启用的 view 字段。
// Compare image/view semantics only, ignoring padding, snapshot pointers and inactive view fields.
static bool registered_texture_equal(const RegisteredTexture &a,const RegisteredTexture &b){
    const tex_t &x=a.texture,&y=b.texture;
    if(x.serial!=y.serial || x.mem!=y.mem || x.pool!=y.pool || x.off!=y.off ||
       x.d.target!=y.d.target || x.d.format!=y.d.format || x.d.flags!=y.d.flags || x.d.samples!=y.d.samples ||
       x.d.width!=y.d.width || x.d.height!=y.d.height || x.d.depth!=y.d.depth || x.d.levels!=y.d.levels || x.d.stride!=y.d.stride || x.d.pkg_block_h!=y.d.pkg_block_h ||
       x.L.size!=y.L.size || x.L.layer_size!=y.L.layer_size ||
       x.L.align!=y.L.align || x.L.layers!=y.L.layers || x.L.bpb!=y.L.bpb || x.L.tileH!=y.L.tileH || x.L.tileD!=y.L.tileD ||
       !!x.has_layout!=!!y.has_layout || (x.has_layout && memcmp(x.layout,y.layout,sizeof(x.layout))))return false;
    const view_t &u=a.view,&v=b.view;
    return a.has_view==b.has_view && (!a.has_view ||
        (!!u.set_levels==!!v.set_levels && (!u.set_levels || (u.base_level==v.base_level && u.nlevels==v.nlevels)) &&
         !!u.set_layers==!!v.set_layers && (!u.set_layers || (u.min_layer==v.min_layer && u.nlayers==v.nlayers)) &&
         !!u.set_format==!!v.set_format && (!u.set_format || u.format==v.format) &&
         !!u.set_target==!!v.set_target && (!u.set_target || u.target==v.target)));
}
static bool register_texture_record(RegisteredTexture &slot,const tex_t &texture,const view_t *view){
    RegisteredTexture next;next.texture=texture;
    if(view && (view->set_levels || view->set_layers || view->set_format || view->set_target)){next.view=*view;next.has_view=true;}
    bool changed=!slot.revision || !registered_texture_equal(slot,next);
    next.revision=slot.revision+uint64_t(changed);slot=next;return changed;
}
struct MemoImage{const tex_t *first;Image *second;const RegisteredTexture *record;uint64_t revision;};
using PoolKey=std::pair<uintptr_t,int64_t>;
PoolKey pool_key(const dpool_t *p){need(p && p->pool,"Missing descriptor pool");return {uintptr_t(p->pool->mem),p->off};}
std::map<PoolKey,std::map<int,RegisteredTexture>> textures;
std::map<int,RegisteredTexture> legacy_textures;std::map<int,samp_par> legacy_samplers;
std::map<int,RegisteredTexture> *texture_pool_cache=nullptr;PoolKey texture_pool_cache_key{}; // descriptors(): last texture pool (entries are never erased before shutdown)
std::map<PoolKey,std::map<int,samp_par>> samplers;
std::map<uint32_t,RegisteredTexture> null_textures;
struct HostSegment{cmd_t *owner;std::unique_ptr<uint8_t[]> memory;size_t bytes;};
std::vector<HostSegment> host_segments;
std::map<uintptr_t,size_t> host_ranges; // address-ordered index of host_segments
size_t host_bytes=0;
uint64_t hash_memory(const void *memory,size_t n){CPU_SCOPE(Hash);const uint8_t *p=static_cast<const uint8_t*>(memory);
    // Word-at-a-time 4-lane mix: change detection only, never persisted.
    uint64_t h[4]={1469598103934665603ull,0x9e3779b97f4a7c15ull,0xc2b2ae3d27d4eb4full,0x165667b19e3779f9ull};size_t i=0;
    for(;i+32<=n;i+=32){for(unsigned k=0;k<4;k++){uint64_t w;memcpy(&w,p+i+8*k,8);h[k]=(h[k]^w)*0x100000001b3ull;h[k]^=h[k]>>29;}}
    uint64_t hash=h[0]^(h[1]*3)^(h[2]*5)^(h[3]*7)^n;for(;i<n;i++){hash^=p[i];hash*=1099511628211ull;}return hash;}
void invalidate_cpu_range(uint64_t address,uint64_t bytes){CPU_SCOPE(CpuWrite);
    need(address+bytes>=address,"CPU write range overflow");cpu_access(address,bytes,4);snapshot_invalidate(address,bytes);
    for_images_overlapping(address,bytes,[&](Image &im){if(draw_detail)draw_timing.image();uint64_t begin=uintptr_t(im.source.mem),end=begin+im.source.L.size;
        // A full storage overwrite replaces the prior GPU contents. A partial
        // overwrite requires GPU-to-NVN readback/merge before applying CPU data.
        need(!im.gpu_written || (address<=begin && address+bytes>=end),"Partial CPU write aliases GPU-written texture; readback/merge required");
        im.gpu_written=false;provenance_write(im,__LINE__);im.uploaded=false;im.checked_submission=UINT64_MAX;
    });
}

bool resolving_pending=false;
// Writebacks of a slot whose fence has signalled can be applied without waiting.
void resolve_slot_writebacks(const void *slot){
    if(pending_writebacks.empty())return;resolving_pending=true;
    std::vector<PendingWriteback> keep;
    for(auto &w:pending_writebacks){if(w.slot!=slot){keep.push_back(w);continue;}writeback_bytes+=w.bytes;w.buffer.invalidate();invalidate_cpu_range(w.addr,w.bytes);memcpy(reinterpret_cast<void*>(w.addr),w.buffer.mapped,w.bytes);}
    pending_writebacks.swap(keep);resolving_pending=false;
}
void write_report(const PendingReport &r){
    uint64_t samples=0;auto e=epochs.find(r.epoch);if(e!=epochs.end()){e->second.advance_known();samples=e->second.prefix(r.count);}
    uint64_t value=r.ts_value;if(timestamp_bits<64)value&=(uint64_t(1)<<timestamp_bits)-1;
    NVNcounterData result{};result.counter=r.type==NVN_COUNTER_TYPE_SAMPLES_PASSED?samples:0;result.timestamp=uint64_t(double(value)*properties.limits.timestampPeriod*12.0/625.0);
    snapshot_invalidate(r.addr,sizeof(result));memcpy(reinterpret_cast<void*>(r.addr),&result,sizeof(result));
}
// Reports whose timestamp and samples are all known are written in order.
void resolve_known_reports(){
    size_t done=0;
    for(;done<pending_reports.size();done++){const auto &r=pending_reports[done];if(!r.ts_known)break;auto e=epochs.find(r.epoch);bool known=true;if(e!=epochs.end()){e->second.advance_known();known=e->second.known_upto>=std::min<uint64_t>(r.count,e->second.size());}if(!known)break;write_report(r);}
    if(done)pending_reports.erase(pending_reports.begin(),pending_reports.begin()+done);
    // Fold known queries into the running sum up to the earliest count still needed by a pending report, then drop them.
    for(auto &e:epochs){uint64_t limit=e.second.known_upto;for(auto &r:pending_reports)if(r.epoch==e.first)limit=std::min<uint64_t>(limit,r.count);e.second.prefix(limit);e.second.trim();}
    // A report only ever names the epoch current when it was recorded, so an older epoch that no pending report names is
    // dead (retire skips query refs to erased epochs). Waiting for no pending report at all never happens once frames
    // overlap, and the map then grew by every counter reset (~13 per D2R frame).
    for(auto it=epochs.begin();it!=epochs.end();){bool used=it->first==counter_epoch;for(auto &r:pending_reports)if(r.epoch==it->first){used=true;break;}
        if(used)++it;else it=epochs.erase(it);}
}
void resolve_pending(){
    if(resolving_pending || (pending_writebacks.empty() && pending_reports.empty()))return;
    resolving_pending=true;
    gpu_sync(pending_writebacks.empty()?perf.sync_query:perf.sync_writeback);
    auto writebacks=std::move(pending_writebacks);pending_writebacks.clear();
    for(auto &w:writebacks){w.buffer.invalidate();invalidate_cpu_range(w.addr,w.bytes);memcpy(reinterpret_cast<void*>(w.addr),w.buffer.mapped,w.bytes);}
    for(auto &r:pending_reports)write_report(r);
    pending_reports.clear();
    for(auto it=epochs.begin();it!=epochs.end();)if(it->first!=counter_epoch)it=epochs.erase(it);else ++it;
    apply_ready_events();
    resolving_pending=false;
}
void cpu_access(uint64_t address,uint64_t bytes,unsigned why){
    if(resolving_pending)return;
    for(auto &w:pending_writebacks)if(address<w.addr+w.bytes && address+bytes>w.addr){++perf.why_access;if(why<12)++perf.access_by[why];resolve_pending();return;}
    // GPU reads of counter memory (snapshots) see the last resolved report, i.e.
    // one frame of latency for occlusion-driven effects instead of a full stall.
    if(why==1 || why==2)return;
    for(auto &r:pending_reports)if(address<r.addr+sizeof(NVNcounterData) && address+bytes>r.addr){++perf.why_access;++perf.access_by[7];resolve_pending();return;}
}
// Guest fence completion. Work recorded for submitted guest command buffers is
// complete when every frame slot is retired; deferred results are then resolved.
void complete_all(){SiteScope site(SITE_COMPLETE);resolve_pending();submit_slots(true);completed=submitted;}
// Block until guest fence `fence` is complete, retiring only the slots needed.
void wait_fence(uint64_t fence){
    for(int i=0;i<2 && fence>completed;i++){Slot &older=slots[slot_index].pending?slots[slot_index]:slots[slot_index^1];if(!older.pending || older.covers<fence)break;retire_slot(older);}
    if(fence>completed)complete_all();
}
bool gpu_poll_complete(){
    SiteScope site(SITE_POLL);
    Slot &cur=slots[slot_index];if(cur.recording && cur.ops)submit_slots(false);
    for(auto &s:slots)if(s.pending){if(vkGetFenceStatus(cfg.device,s.fence)!=VK_SUCCESS)return false;}
    ++perf.why_poll;complete_all(); // everything signaled: retiring/resolving does not block
    return true;
}
// Vulkan buffer/image transfers name exactly one depth/stencil aspect. Keep
// both packed NVN channels, with independent tightly packed planes and copies.
void append_depth_planes(std::vector<uint8_t> &data,std::vector<VkBufferImageCopy> &copies,
    uint32_t format,uint32_t w,uint32_t h,uint32_t slices,const void *packed,size_t packed_bytes,
    size_t row,size_t image,const VkBufferImageCopy &region){
    xbg_depth_format f;need(xbg_depth_format_info(format,&f)==0,"Unknown NVN depth transfer format");
    bool widen=format==0x32 && depth16_as_d24;   // image texels are X8_D24 (4 bytes), guest texels D16 (2 bytes)
    uint64_t pixels=uint64_t(w)*h*slices;need(pixels<=256ull*1024*1024/5,"Depth staging extent exceeds bound");
    size_t begin=(data.size()+15)&~size_t(15),depth_bytes=size_t(pixels)*(widen?4:f.depth_bytes);
    size_t stencil_begin=(begin+depth_bytes+3)&~size_t(3),stencil_bytes=size_t(pixels)*f.stencil_bytes;
    need(stencil_begin+stencil_bytes<=256ull*1024*1024,"Depth staging allocation exceeds bound");data.resize(stencil_begin+stencil_bytes);
    std::vector<uint8_t> d16(packed && widen?size_t(pixels)*2:0);
    if(packed)need(xbg_depth_extract(format,w,h,slices,packed,packed_bytes,row,image,
        f.depth_bytes?(widen?d16.data():data.data()+begin):nullptr,widen?d16.size():depth_bytes,f.stencil_bytes?data.data()+stencil_begin:nullptr,stencil_bytes)==0,"Invalid packed NVN depth upload extent");
    for(size_t i=0;i<d16.size()/2;i++){uint16_t k;memcpy(&k,&d16[i*2],2);uint32_t v=d16_to_d24(k);memcpy(data.data()+begin+i*4,&v,4);}
    if(f.depth_bytes){auto r=region;r.bufferOffset=begin;r.imageSubresource.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;copies.push_back(r);}
    if(f.stencil_bytes){auto r=region;r.bufferOffset=stencil_begin;r.imageSubresource.aspectMask=VK_IMAGE_ASPECT_STENCIL_BIT;copies.push_back(r);}
}

void forget_framebuffers(Image &im,std::vector<VkFramebuffer> *deferred){
    for(auto &k:im.fb_keys){auto f=persistent_framebuffers.find(k);if(f==persistent_framebuffers.end())continue;   // already gone with another image
        if(deferred)deferred->push_back(f->second);else vkDestroyFramebuffer(cfg.device,f->second,nullptr);persistent_framebuffers.erase(f);}
    im.fb_keys.clear();
}
// Direct-mapped front for images.find (~12k get_image calls per D2R camp frame): erasing from or clearing images bumps
// images_epoch, so a front entry never outlives its Image.
struct ImgFront{uint64_t serial=0,epoch=0;Image *im=nullptr;};ImgFront img_front[1024];uint64_t images_epoch=1;
Image &get_image(const tex_t *t,bool upload){CPU_SCOPE(Image);
    need(t && t->serial && t->magic==MAGIC('T','E','X',0),"Invalid NVN texture");
    ImgFront &front=img_front[(t->serial*0x9E3779B97F4A7C15ull)>>54];Image *hit=front.epoch==images_epoch && front.serial==t->serial?front.im:nullptr;
    auto it=hit?images.end():images.find(t->serial);
    if(!hit){
    if(it==images.end() && t->mem){
        // Reinitialized or copied NVN objects with the same storage/layout share
        // GPU contents. Serial-only caches would incorrectly re-upload stale CPU
        // bytes when the new object aliases an already-rendered attachment.
        Image *alias=nullptr;for_images_overlapping(uint64_t(uintptr_t(t->mem)),1,[&](Image &c){if(!alias && c.source.mem==t->mem && !memcmp(&c.source.d,&t->d,sizeof(t->d)))alias=&c;});
        if(alias)for(auto &candidate:images)if(candidate.second.get()==alias){it=images.emplace(t->serial,candidate.second).first;image_index_dirty=true;++memo_bump[3];provenance_image("ALIAS",t,*it->second);break;}
    }
    if(it==images.end()){
        need(t->d.samples<=1 || t->d.samples==2 || t->d.samples==4 || t->d.samples==8,"Unsupported texture sample count");
        need(t->d.target==0 || t->d.target==1 || t->d.target==2 || t->d.target==3 || t->d.target==4 || t->d.target==5 || t->d.target==6 || t->d.target==7 || t->d.target==8 || t->d.target==9,"Unsupported texture target (1D/3D/buffer handled separately)");
        auto p=std::make_shared<Image>();p->source=*t;p->provenance_epoch=provenance.epoch((t->mem && t->off>=0 && uint64_t(t->off)<=uint64_t(t->mem)?uint64_t(t->mem)-uint64_t(t->off):0));p->width=t->d.width;p->height=(t->d.target==0 || t->d.target==3)?1:t->d.height;p->depth=t->d.target==2?t->d.depth:1;p->levels=t->d.levels;p->layers=t->L.layers;p->aspects=aspect(t->d.format);p->samples=VkSampleCountFlagBits(std::max(1,t->d.samples));
        need(p->width && p->height && p->levels && p->levels<=16 && p->layers,"Invalid texture dimensions");
        xbg_bc_format info;p->decoded=xbg_bc_format_info(t->d.format,&info)==0;
        if(p->decoded){p->bpp=info.bytes_per_pixel;p->fmt=decoded_format(info.pixels);need(p->samples==VK_SAMPLE_COUNT_1_BIT,"BC texture cannot be multisampled");}
        else {p->fmt=format(t->d.format);p->bpp=t->L.bpb;if(t->d.format==0x32 && depth16_as_d24)p->fmt=VK_FORMAT_X8_D24_UNORM_PACK32;}
        VkFormatProperties fp;vkGetPhysicalDeviceFormatProperties(cfg.physical,p->fmt,&fp);
        VkImageUsageFlags usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)usage|=VK_IMAGE_USAGE_SAMPLED_BIT;
        if((t->d.flags&0x4) && (fp.optimalTilingFeatures&VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && (p->samples==VK_SAMPLE_COUNT_1_BIT || cfg.enabled_core.shaderStorageImageMultisample))usage|=VK_IMAGE_USAGE_STORAGE_BIT;
        if(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)usage|=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)usage|=VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ci.imageType=t->d.target==2?VK_IMAGE_TYPE_3D:t->d.target==0||t->d.target==3?VK_IMAGE_TYPE_1D:VK_IMAGE_TYPE_2D;ci.format=p->fmt;ci.extent={p->width,p->height,p->depth};ci.mipLevels=p->levels;ci.arrayLayers=p->layers;ci.samples=p->samples;ci.tiling=VK_IMAGE_TILING_OPTIMAL;ci.usage=usage;ci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        p->usage=usage;if(p->aspects==VK_IMAGE_ASPECT_COLOR_BIT)ci.flags|=VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        if(t->d.target==8 || t->d.target==9)ci.flags|=VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        VkImageFormatProperties supported;VK(vkGetPhysicalDeviceImageFormatProperties(cfg.physical,ci.format,ci.imageType,ci.tiling,ci.usage,ci.flags,&supported));need(supported.sampleCounts&p->samples,"Texture sample count unsupported for actual format/usage");need(p->samples==VK_SAMPLE_COUNT_1_BIT || p->levels==1,"Multisample textures cannot have mipmaps");
        VK(vkCreateImage(cfg.device,&ci,nullptr,&p->image));VkMemoryRequirements r;vkGetImageMemoryRequirements(cfg.device,p->image,&r);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=r.size;ai.memoryTypeIndex=memory_type(r.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);VK(vkAllocateMemory(cfg.device,&ai,nullptr,&p->memory));VK(vkBindImageMemory(cfg.device,p->image,p->memory,0));
        it=images.emplace(t->serial,std::move(p)).first;image_index_dirty=true;++memo_bump[4];provenance_image("CREATE",t,*it->second);
    }
    hit=it->second.get();front.serial=t->serial;front.epoch=images_epoch;front.im=hit;}
    Image &im=*hit;
    // NVN CPU-coherent pools may be written without FlushMappedRange. Detect
    // such writes once per submit; command CPU writes also invalidate below.
    if(upload && im.uploaded && !im.gpu_written && im.checked_submission!=stats.submissions){
        // Textures unchanged for 4 checks are re-verified every 16th submission (staggered).
        if(im.stable_checks<4 || ((stats.submissions+t->serial)&15)==0){uint64_t hash=hash_memory(t->mem,t->L.size);if(hash!=im.cpu_hash){im.uploaded=false;im.stable_checks=0;}else ++im.stable_checks;}
        im.checked_submission=stats.submissions;}
    // Diagnostic: CPU writes into textures whose contents the GPU produced (copies) are invisible to us.
    if(upload && im.gpu_written && im.decoded && t->mem && ((stats.submissions+t->serial)&63)==0 && im.checked_submission!=stats.submissions){im.checked_submission=stats.submissions;uint64_t hash=hash_memory(t->mem,t->L.size);if(im.cpu_hash && hash!=im.cpu_hash){static unsigned n=0;if(n++<40)xbg_logf("NVN CPUGPU texture changed by CPU after GPU write serial=%llu fmt=%#x %ux%u layers=%u levels=%u target=%d",(unsigned long long)t->serial,t->d.format,im.width,im.height,im.layers,im.levels,t->d.target);}im.cpu_hash=hash;}
    if(upload && !im.uploaded && !im.gpu_written){
        need(im.samples==VK_SAMPLE_COUNT_1_BIT,"Cannot upload uninitialized multisample image from CPU storage");
        need(t->mem && t->pool && t->off>=0 && uint64_t(t->off)<=t->pool->size && t->L.size<=t->pool->size-uint64_t(t->off),"Texture storage outside memory pool");
        if(im.aspects!=VK_IMAGE_ASPECT_COLOR_BIT)xbg_logf("NVN depth initial upload serial=%llu format=%#x vkFormat=%u target=%d extent=%ux%ux%u levels=%u layers=%u source=%p bytes=%llu",(unsigned long long)t->serial,t->d.format,unsigned(im.fmt),t->d.target,im.width,im.height,im.depth,im.levels,im.layers,t->mem,(unsigned long long)t->L.size);
        std::vector<uint8_t> linear;std::vector<VkBufferImageCopy> copies;
        for(uint32_t layer=0;layer<im.layers;layer++)for(uint32_t level=0;level<im.levels;level++){
            uint32_t w=std::max(1u,im.width>>level),h=std::max(1u,im.height>>level),depth=std::max(1u,im.depth>>level);
            if(im.aspects!=VK_IMAGE_ASPECT_COLOR_BIT){
                size_t bytes=size_t(w)*h*depth*im.bpp;need(bytes<=256ull*1024*1024,"Packed depth image too large");std::vector<uint8_t> packed(bytes);
                if(t->d.flags&0x10){need(!layer && !level && depth==1,"Linear depth mip/layer/volume layout unsupported");size_t row=t->d.stride?t->d.stride:(size_t(w)*im.bpp+31)&~size_t(31);need(row>=size_t(w)*im.bpp && row*h<=t->L.size,"Linear depth storage extent invalid");for(uint32_t y=0;y<h;y++)memcpy(packed.data()+size_t(y)*w*im.bpp,t->mem+y*row,size_t(w)*im.bpp);}
                else need(nvn_deswizzle(&t->d,level,layer,t->mem,packed.data(),size_t(w)*im.bpp)==0,"Depth image deswizzle failed");
                VkBufferImageCopy region{};region.imageSubresource={0,level,layer,1};region.imageExtent={w,h,depth};
                append_depth_planes(linear,copies,t->d.format,w,h,depth,packed.data(),packed.size(),0,0,region);continue;
            }
            size_t begin=(linear.size()+15)&~size_t(15);
            int cbw=1,cbh=1;bool native_blocks=!im.decoded && nvn_fmt_bpb(t->d.format,&cbw,&cbh) && (cbw>1 || cbh>1);
            if(native_blocks){ // ASTC etc.: copy whole blocks, rows of blocks
                uint32_t bx=(w+cbw-1)/cbw,by=(h+cbh-1)/cbh;linear.resize(begin+size_t(bx)*by*depth*im.bpp);
                need(nvn_deswizzle(&t->d,level,layer,t->mem,linear.data()+begin,size_t(bx)*im.bpp)==0,"Compressed texture deswizzle failed");
                VkBufferImageCopy c{};c.bufferOffset=begin;c.imageSubresource={im.aspects,level,layer,1};c.imageExtent={w,h,depth};copies.push_back(c);continue;}
            linear.resize(begin+size_t(w)*h*depth*im.bpp);
            if(im.decoded){
                int bw,bh,block_bytes=nvn_fmt_bpb(t->d.format,&bw,&bh);uint32_t blocks_x=(w+bw-1)/bw,blocks_y=(h+bh-1)/bh;
                std::vector<uint8_t> blocks(size_t(blocks_x)*blocks_y*depth*block_bytes);
                need(nvn_deswizzle(&t->d,level,layer,t->mem,blocks.data(),size_t(blocks_x)*block_bytes)==0,"BC image deswizzle failed");
                need(xbg_decode_bc_linear(t->d.format,w,h,depth,blocks.data(),blocks.size(),0,0,linear.data()+begin,size_t(w)*h*depth*im.bpp)==0,"BC image decode failed");
            }else if(t->d.flags&0x10){need(!layer && !level && depth==1,"Linear mip/layer/volume texture not implemented");size_t pitch=t->d.stride?t->d.stride:(w*im.bpp+31)&~size_t(31);for(uint32_t y=0;y<h;y++)memcpy(linear.data()+begin+y*w*im.bpp,t->mem+y*pitch,w*im.bpp);}
            else need(nvn_deswizzle(&t->d,level,layer,t->mem,linear.data()+begin,size_t(w)*im.bpp)==0,"Texture deswizzle failed");
            VkBufferImageCopy c{};c.bufferOffset=begin;c.imageSubresource={im.aspects,level,layer,1};c.imageExtent={w,h,depth};copies.push_back(c);
        }
        Buffer staging=stream_buffer(linear.size(),linear.data(),linear.size(),false);for(auto &copy:copies)copy.bufferOffset+=staging.offset;Commands c;im.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);vkCmdCopyBufferToImage(c.command,staging.buffer,im.image,im.layout,copies.size(),copies.data());im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();im.uploaded=true;provenance_write(im,__LINE__);im.cpu_hash=hash_memory(t->mem,t->L.size);im.checked_submission=stats.submissions;
    }
    provenance_image(upload?"GET_UPLOAD":"GET",t,im);return im;
}
const RegisteredTexture &null_texture(const xbg_shader_binding &b){
    // Original host evidence: main+0x1746820 explicitly binds descriptor 0xFFFFF
    // for unused slots; descriptor 0 is the game's null handle. Only those
    // defined null cases receive zero resources. Missing in-pool IDs still fail.
    need(b.type!=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && b.type!=VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,"Null writable image descriptor needs robustness2 nullDescriptor support");
    uint32_t key=b.flags&0xffffff00u;auto it=null_textures.find(key);if(it!=null_textures.end())return it->second;
    unsigned dim=(b.flags&(1u<<31))?(b.flags>>8)&7:1;bool array=b.flags&(1u<<12);unsigned scalar=(b.flags>>16)&3;
    need(!(b.flags&(1u<<13)),"Null multisample texture descriptor unsupported");
    RegisteredTexture r;tex_t &t=r.texture;t.magic=MAGIC('T','E','X',0);t.serial=nvn_new_serial();t.d.width=t.d.height=t.d.depth=t.d.levels=1;
    t.d.format=b.flags&(1u<<14)?0x34:scalar==1?0x2f:scalar==2?0x30:dim==5?0x2e:0x25;
    t.d.target=dim==0?(array?3:0):dim==1?(array?4:1):dim==2?2:dim==3?(array?9:8):dim==5?10:-1;need(t.d.target>=0,"Unknown null texture dimension");
    nvn_tex_layout(&t.d,&t.L);
    if(dim==5){static uint8_t zero[65536]{};t.mem=zero;t.L.size=sizeof(zero);}
    else{Image &im=get_image(&t,false);Commands c;im.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageSubresourceRange range{im.aspects,0,1,0,im.layers};if(im.aspects==VK_IMAGE_ASPECT_COLOR_BIT){VkClearColorValue zero{};vkCmdClearColorImage(c.command,im.image,im.layout,&zero,1,&range);}else{VkClearDepthStencilValue zero{};vkCmdClearDepthStencilImage(c.command,im.image,im.layout,&zero,1,&range);}im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();im.gpu_written=true,provenance_write(im,__LINE__);}
    xbg_logf("NVN defined null sampled resource: dim=%u array=%d scalar=%u",dim,array,scalar);return null_textures.emplace(key,r).first->second;
}

/* SHA-256 over the original NVN 0x880-byte control segment. */
std::string control_key(const void *data){char key[17];need(xbg_shader_control_key(data,key)==0,"Missing shader control segment");return key;}
struct Shader {std::string key;uint32_t stage;VkShaderModule module{};std::vector<xbg_shader_binding> bindings;std::vector<uint32_t> inputs,input_classes;std::vector<uint8_t> c1;
    std::set<uint32_t> storage_writes; // Vulkan bindings of storage buffers the SPIR-V may write
    bool discards=true; // fragment shader may kill/demote (conservative default)
    Buffer c1_buffer{};bool c1_ready=false;
    ~Shader(){if(module)vkDestroyShaderModule(cfg.device,module,nullptr);}};
// Storage buffers written by a shader. Pointers rooted at a storage-buffer
// variable that reach OpStore/OpCopyMemory/atomics, function calls, phis or
// selects mark that binding written; anything unprovable marks all written.
// Bytes of each uniform block a shader can actually read (iOS c1/cbuf packing idea): Ryujinx declares
// constant buffers as struct{vec4 data[4096];}. When every access uses a constant element index, only
// (max index + 1) * 16 bytes need copying per draw; any other use keeps the full block.
std::map<uint32_t,uint32_t> spirv_uniform_extent(const std::vector<uint32_t> &w){
    std::map<uint32_t,uint32_t> binding,constant,stride,member_type,var_type,result;std::set<uint32_t> uniform_vars,full;std::map<uint32_t,uint32_t> max_index;
    std::map<uint32_t,uint32_t> pointer_pointee,struct_member0;
    for(size_t i=5;i<w.size();){uint32_t n=w[i]>>16,op=w[i]&65535;if(!n||i+n>w.size())return {};
        switch(op){
        case 71:if(n>=4 && w[i+2]==33)binding[w[i+1]]=w[i+3];if(n>=4 && w[i+2]==6)stride[w[i+1]]=w[i+3];break; // Binding / ArrayStride
        case 43:if(n>=4)constant[w[i+2]]=w[i+3];break; // OpConstant (32-bit)
        case 30:if(n>=3)struct_member0[w[i+1]]=w[i+2];break; // OpTypeStruct member 0
        case 32:if(n>=4)pointer_pointee[w[i+1]]=w[i+3];break;
        case 59:if(n>=4 && w[i+3]==2){uniform_vars.insert(w[i+2]);var_type[w[i+2]]=w[i+1];}break; // OpVariable Uniform
        default:break;
        }
        i+=n;
    }
    for(size_t i=5;i<w.size();){uint32_t n=w[i]>>16,op=w[i]&65535;
        if(op==59){i+=n;continue;}
        if(op==5||op==6||op==15||op==71||op==72||op==332||op==5632){i+=n;continue;} // names/entry-point interface/decorations reference without reading
        for(uint32_t k=1;k<n;k++){uint32_t id=w[i+k];if(!uniform_vars.count(id))continue;
            bool ok=(op==65||op==66) && k==3 && n>=6;
            if(ok){auto c=constant.find(w[i+5]);auto m=constant.find(w[i+4]);if(c==constant.end()||m==constant.end()||m->second!=0)ok=false;else max_index[id]=std::max(max_index[id],c->second);}
            if(!ok)full.insert(id);}
        i+=n;
    }
    for(uint32_t v:uniform_vars){auto b=binding.find(v);if(b==binding.end()||full.count(v))continue;
        auto pt=pointer_pointee.find(var_type[v]);if(pt==pointer_pointee.end())continue;auto m0=struct_member0.find(pt->second);if(m0==struct_member0.end())continue;
        auto st=stride.find(m0->second);if(st==stride.end()||st->second!=16)continue;
        auto mi=max_index.find(v);result[b->second]=mi==max_index.end()?16:(mi->second+1)*16;}
    return result;
}
std::set<uint32_t> spirv_storage_writes(const std::vector<uint32_t> &w,bool &unknown){
    std::map<uint32_t,uint32_t> binding,storage_class,root;std::set<uint32_t> written_vars;unknown=false;
    std::set<uint32_t> buffer_block_types;std::map<uint32_t,uint32_t> pointer_pointee;
    for(size_t i=5;i<w.size();){uint32_t n=w[i]>>16,op=w[i]&65535;if(!n||i+n>w.size()){unknown=true;break;}
        auto mark=[&](uint32_t ptr){auto r=root.find(ptr);if(r!=root.end())written_vars.insert(r->second);};
        switch(op){
        case 71:if(n>=4 && w[i+2]==33)binding[w[i+1]]=w[i+3];if(n>=3 && w[i+2]==3)buffer_block_types.insert(w[i+1]);break; // OpDecorate Binding/BufferBlock
        case 32:if(n>=4)pointer_pointee[w[i+1]]=w[i+3];break; // OpTypePointer
        case 59:if(n>=4){storage_class[w[i+2]]=w[i+3];root[w[i+2]]=w[i+2];}break; // OpVariable
        // Function parameters: call sites passing storage pointers are marked
        // at OpFunctionCall, so stores through parameters need no extra mark.
        case 55:if(n>=3)root[w[i+2]]=0xffffffffu;break;
        case 65:case 66:case 67:case 83:case 124:if(n>=4){auto r=root.find(w[i+3]);if(r!=root.end())root[w[i+2]]=r->second;}break;
        case 62:case 63:case 228:if(n>=2){if(root.count(w[i+1]))mark(w[i+1]);else unknown=true;}break; // OpStore/OpCopyMemory/OpAtomicStore
        case 57:for(uint32_t k=4;k<n;k++)mark(w[i+k]);break; // pointer arguments to calls
        case 169:case 245:for(uint32_t k=3;k<n;k++)mark(w[i+k]);break; // select/phi over pointers
        default:if(((op>=229 && op<=242) || op==6035 || op==5250) && n>=4){if(root.count(w[i+3]))mark(w[i+3]);else unknown=true;}break;
        }
        i+=n;
    }
    std::set<uint32_t> out;
    for(uint32_t v:written_vars){auto c=storage_class.find(v);if(c==storage_class.end())continue;if(c->second==12 || c->second==2){auto b=binding.find(v);if(b!=binding.end())out.insert(b->second);else unknown=true;}}
    return out;
}
std::map<std::string,std::unique_ptr<Shader>> shaders;
const uint32_t nvn_stage[6]={0,3,4,2,1,5};
struct State {
    p_set_rt rt{};Shader *shader[6]{};NVNbufferRange ub[6][16]{},sb[6][16]{},vb[16]{};
    uint64_t tex[6][128]{},samp[6][128]{},img[6][16]{};dpool_t texture_pool{},sampler_pool{};
    uint32_t attrib[16]{};vs_st stream[16]{};blend_st blend[8]{};color_st color{};chmask_st mask{0xffffffff};ms_st ms{};poly_st poly{0,1,2,0};ds_st ds{0,1,2,0,8,8,1,1,1,1,1,1};
    VkViewport viewport{0,0,1,1,0,1};VkRect2D scissor{{0,0},{UINT32_MAX,UINT32_MAX}};float blend_color[4]{},bias[3]{};uint32_t stencil_compare[2]{~0u,~0u},stencil_write[2]{~0u,~0u},stencil_ref[2]{};int64_t copy_stride=0,copy_image_stride=0;uint32_t patch_size=3;
    xbg_depth_bounds bounds{0,0,1};bool count_samples=false;uint64_t samples_passed=0;
} state;
void validate_capabilities(const std::vector<uint32_t> &w,uint32_t stage){
    need(w.size()>=5 && w[0]==0x07230203,"Invalid packaged SPIR-V");bool group_operations=false;VkSubgroupFeatureFlags groups=0;
    for(size_t i=5;i<w.size();){uint32_t count=w[i]>>16,op=w[i]&65535;need(count && i+count<=w.size(),"Invalid SPIR-V instruction bounds");if(op>=333 && op<=366)group_operations=true;
        if(op==17){uint32_t cap=w[i+1];bool ok=true;switch(cap){case 1:case 46:case 47:case 50:break;case 2:ok=cfg.enabled_core.geometryShader;break;case 3:ok=cfg.enabled_core.tessellationShader;break;case 10:ok=cfg.enabled_core.shaderFloat64;break;case 11:ok=cfg.enabled_core.shaderInt64;break;case 22:ok=cfg.enabled_core.shaderInt16;break;case 25:ok=cfg.enabled_core.shaderImageGatherExtended;break;case 32:ok=cfg.enabled_core.shaderClipDistance;break;case 33:ok=cfg.enabled_core.shaderCullDistance;break;case 35:ok=cfg.enabled_core.sampleRateShading;break;case 49:ok=cfg.enabled_core.shaderStorageImageExtendedFormats;break;case 55:ok=cfg.enabled_core.shaderStorageImageReadWithoutFormat;break;case 56:ok=cfg.enabled_core.shaderStorageImageWriteWithoutFormat;break;case 61:groups|=VK_SUBGROUP_FEATURE_BASIC_BIT;break;case 62:groups|=VK_SUBGROUP_FEATURE_VOTE_BIT;break;case 63:groups|=VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;break;case 64:groups|=VK_SUBGROUP_FEATURE_BALLOT_BIT;break;case 65:groups|=VK_SUBGROUP_FEATURE_SHUFFLE_BIT;break;case 66:groups|=VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;break;case 4427:ok=cfg.enabled_shader_draw_parameters;break;default:xbg_fatalf("Shader stage %u has unhandled capability %u",stage,cap);}if(!ok)xbg_fatalf("Shader stage %u requires disabled capability %u",stage,cap);}
        if(op==10)xbg_fatalf("Shader stage %u declares extension requiring explicit backend support",stage);i+=count;
    }
    need((subgroup.supportedOperations&groups)==groups,"Required subgroup operations unavailable");if(group_operations)need(subgroup.supportedStages&(1u<<stage),"Actual subgroup operation unsupported in shader stage");
}
Shader *load_shader(const void *control){
    std::string key=control_key(control);auto found=shaders.find(key);if(found!=shaders.end())return found->second.get();
    std::string path=std::string(cfg.shader_directory)+"/"+key+".xgs";FILE *f=fopen(path.c_str(),"rb");if(!f)xbg_fatalf("Missing genuine shader bundle %s",path.c_str());
    fseek(f,0,SEEK_END);long length=ftell(f);rewind(f);need(length>=32 && length<=16*1024*1024,"Invalid shader bundle length");std::vector<uint8_t> bytes(length);need(fread(bytes.data(),1,bytes.size(),f)==bytes.size(),"Shader bundle read failed");fclose(f);
    xbg_shader_header h;memcpy(&h,bytes.data(),sizeof(h));need(h.magic==0x31534758 && h.version==1 && h.stage<6 && h.binding_count<=256 && h.input_count<=16 && h.c1_bytes<=65536 && h.spirv_bytes%4==0,"Invalid shader bundle header");
    size_t offset=sizeof(h),expected=offset+size_t(h.binding_count)*sizeof(xbg_shader_binding)+size_t(h.input_count)*4+h.c1_bytes+h.spirv_bytes;need(expected==bytes.size(),"Shader bundle extent mismatch");
    auto p=std::make_unique<Shader>();p->key=key;p->stage=h.stage;p->bindings.resize(h.binding_count);memcpy(p->bindings.data(),bytes.data()+offset,h.binding_count*sizeof(xbg_shader_binding));offset+=h.binding_count*sizeof(xbg_shader_binding);p->inputs.resize(h.input_count);memcpy(p->inputs.data(),bytes.data()+offset,h.input_count*4);offset+=h.input_count*4;p->c1.assign(bytes.begin()+offset,bytes.begin()+offset+h.c1_bytes);offset+=h.c1_bytes;
    std::vector<uint32_t> words(h.spirv_bytes/4);memcpy(words.data(),bytes.data()+offset,h.spirv_bytes);validate_capabilities(words,h.stage);
    {auto extent=spirv_uniform_extent(words);uint32_t trimmed=0;for(auto &b:p->bindings)if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && b.source==XBG_SOURCE_UNIFORM){auto e=extent.find(b.binding);uint32_t declared=b.bytes?b.bytes:65536;if(e!=extent.end() && e->second<declared){b.bytes=e->second;++trimmed;}}uniform_trimmed+=trimmed;}
    {bool kill=false;for(size_t i=5;i<words.size();){uint32_t n=words[i]>>16,op=words[i]&0xffff;if(op==252||op==4416||op==5380){kill=true;break;}if(!n)break;i+=n;}p->discards=kill;}
    {bool unknown=false;p->storage_writes=spirv_storage_writes(words,unknown);if(unknown){for(auto &b:p->bindings)if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)p->storage_writes.insert(b.binding);}}
    if(h.stage==0){p->input_classes.resize(p->inputs.size());need(xbg_vertex_input_classes(words.data(),words.size(),p->inputs.data(),p->inputs.size(),p->input_classes.data())==0,"Unsupported vertex input scalar interface");}
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=h.spirv_bytes;ci.pCode=words.data();VK(vkCreateShaderModule(cfg.device,&ci,nullptr,&p->module));warm::register_module(p->module,key);Shader *out=p.get();shaders.emplace(key,std::move(p));++stats.shader_loads;xbg_logf("NVN shader %s stage=%u loaded",key.c_str(),h.stage);return out;
}
// nvnProgramSetShaders assigns a new serial, and NVN shader control/code memory
// must stay unchanged while a program uses it: cache resolved shaders by serial
// instead of re-hashing every control segment on each bind.
std::map<uint64_t,std::array<Shader*,6>> program_cache;
const std::array<Shader*,6> &program_shaders(const prog_t *prog){
    if(prog->serial){auto it=program_cache.find(prog->serial);if(it!=program_cache.end())return it->second;}
    std::array<Shader*,6> out{};for(unsigned i=0;i<prog->nstage && i<6;i++)out[i]=load_shader(prog->sd[i].control);
    if(!prog->serial){static std::array<Shader*,6> scratch;scratch=out;return scratch;}
    return program_cache.emplace(prog->serial,out).first->second;
}
} // namespace

extern "C" void xbg_logf(const char *fmt,...){CPU_SCOPE(Log);char text[1024];va_list a;va_start(a,fmt);vsnprintf(text,sizeof(text),fmt,a);va_end(a);if(cfg.log)cfg.log(cfg.userdata,text);else fprintf(stderr,"[xbg] %s\n",text);}
extern "C" __attribute__((noreturn)) void xbg_fatalf(const char *fmt,...){va_list a;va_start(a,fmt);vsnprintf(error_text,sizeof(error_text),fmt,a);va_end(a);failed=true;xbg_logf("FATAL: %s",error_text);if(cfg.fatal)cfg.fatal(cfg.userdata,error_text);abort();}
extern "C" int xbg_backend_failed(){return failed;}
extern "C" const char *xbg_game_last_error(){return error_text;}
extern "C" int xbg_device_integer(int p,int *out){
    need(initialized && out,"NVN device query before Vulkan initialization");const auto &l=properties.limits;
    switch(p){
    case NVN_DEVICE_INFO_MAX_UNIFORM_BUFFER_SIZE:*out=std::min(65536u,l.maxUniformBufferRange);break;
    case NVN_DEVICE_INFO_COLOR_BUFFER_BINDINGS:*out=std::min(8u,l.maxColorAttachments);break;
    case NVN_DEVICE_INFO_VERTEX_BUFFER_BINDINGS:*out=std::min(16u,l.maxVertexInputBindings);break;
    case NVN_DEVICE_INFO_VERTEX_ATTRIBUTES:*out=std::min(16u,l.maxVertexInputAttributes);break;
    case NVN_DEVICE_INFO_MAX_TEXTURE_ANISOTROPY:*out=cfg.enabled_core.samplerAnisotropy?int(l.maxSamplerAnisotropy):1;break;
    case NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_X:case NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Y:case NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_Z:*out=l.maxComputeWorkGroupSize[p-NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_X];break;
    case NVN_DEVICE_INFO_MAX_COMPUTE_WORK_GROUP_SIZE_THREADS:*out=l.maxComputeWorkGroupInvocations;break;
    case NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_X:case NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Y:case NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_Z:*out=l.maxComputeWorkGroupCount[p-NVN_DEVICE_INFO_MAX_COMPUTE_DISPATCH_WORK_GROUPS_X];break;
    case NVN_DEVICE_INFO_MAX_VIEWPORTS:*out=1;break;
    case NVN_DEVICE_INFO_MAX_TEXTURE_LAYERS:*out=l.maxImageArrayLayers;break;
    case NVN_DEVICE_INFO_MAX_TEXTURE_SIZE:case NVN_DEVICE_INFO_MAX_RECTANGLE_TEXTURE_SIZE:*out=l.maxImageDimension2D;break;
    case NVN_DEVICE_INFO_MAX_3D_TEXTURE_SIZE:*out=l.maxImageDimension3D;break;
    case NVN_DEVICE_INFO_MAX_CUBE_MAP_TEXTURE_SIZE:*out=l.maxImageDimensionCube;break;
    case NVN_DEVICE_INFO_MAX_BUFFER_TEXTURE_SIZE:*out=l.maxTexelBufferElements;break;
    case NVN_DEVICE_INFO_MAX_PATCH_SIZE:*out=cfg.enabled_core.tessellationShader?l.maxTessellationPatchSize:0;break;
    case NVN_DEVICE_INFO_SUBPIXEL_BITS:*out=l.subPixelPrecisionBits;break;
    case NVN_DEVICE_INFO_SUPPORTS_MIN_MAX_FILTERING:case NVN_DEVICE_INFO_SUPPORTS_ASTC_FORMATS:
    case NVN_DEVICE_INFO_SUPPORTS_CONSERVATIVE_RASTER:case NVN_DEVICE_INFO_SUPPORTS_ZERO_FROM_UNMAPPED_VIRTUAL_POOL_PAGES:
    case NVN_DEVICE_INFO_SUPPORTS_PASSTHROUGH_GEOMETRY_SHADERS:case NVN_DEVICE_INFO_SUPPORTS_VIEWPORT_SWIZZLE:
    case NVN_DEVICE_INFO_SUPPORTS_SPARSE_TILED_PACKAGED_TEXTURES:case NVN_DEVICE_INFO_SUPPORTS_ADVANCED_BLEND_MODES:
    case NVN_DEVICE_INFO_SUPPORTS_DRAW_TEXTURE:case NVN_DEVICE_INFO_SUPPORTS_TARGET_INDEPENDENT_RASTERIZATION:
    case NVN_DEVICE_INFO_SUPPORTS_FRAGMENT_COVERAGE_TO_COLOR:case NVN_DEVICE_INFO_SUPPORTS_POST_DEPTH_COVERAGE:
    case NVN_DEVICE_INFO_SUPPORTS_SAMPLE_LOCATIONS:case NVN_DEVICE_INFO_SUPPORTS_FRAGMENT_SHADER_INTERLOCK:*out=0;break;
    case NVN_DEVICE_INFO_SUPPORTS_STENCIL8_FORMAT:{VkFormatProperties f;vkGetPhysicalDeviceFormatProperties(cfg.physical,VK_FORMAT_S8_UINT,&f);*out=(f.optimalTilingFeatures&VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)!=0;break;}
    default:return 0; // NVN ABI sizes/alignments are host-emulated, not Vulkan addresses.
    }return 1;
}

namespace {
struct PipelineEntry {VkPipeline pipeline;uint64_t last_used;};
std::map<std::string,PipelineEntry> pipeline_objects;
uint64_t pipeline_tick=0;
constexpr size_t pipeline_object_limit=4096; // field scenes exceed 256 live pipelines; eviction+rebuild caused 100ms+ hitches
struct PipelineLease {
    VkPipeline pipeline{};bool transient=false;
    ~PipelineLease(){if(transient && pipeline){VkPipeline p=pipeline;retire_handle([p]{vkDestroyPipeline(cfg.device,p,nullptr);});}}
    PipelineLease()=default;
    PipelineLease(const PipelineLease&)=delete;
    PipelineLease& operator=(const PipelineLease&)=delete;
};
void report_pipeline_cache(){
    xbg_logf("NVN CPU_TIMING pipeline_cache hits=%llu misses=%llu evictions=%llu bypass=%llu entries=%zu limit=%zu",(unsigned long long)(pipeline_counts.hits-pipeline_before.hits),(unsigned long long)(pipeline_counts.misses-pipeline_before.misses),(unsigned long long)(pipeline_counts.evictions-pipeline_before.evictions),(unsigned long long)(pipeline_counts.bypass-pipeline_before.bypass),pipeline_objects.size(),pipeline_object_limit);
}
// Pipeline warm-up (the "precompiled shader cache" of console emulators). The Maleoon driver does not grow the
// VkPipelineCache, so every launch recompiled each graphics pipeline on first use, inside the frame (~100+ ms each).
// Every newly created graphics pipeline is appended to files/vk-pipeline-list.bin (shader bundle keys + the exact
// serialized create-info key, with shader modules identified by a stable hash of their bundle key). At start-up,
// background threads rebuild those pipelines from the list; a draw that needs one takes it ready-made.
// files/gfx-no-warm disables building (recording continues).
namespace warm {
std::mutex mu;std::unordered_map<VkShaderModule,uint64_t> ids;
uint64_t fnv(const std::string &s){uint64_t h=1469598103934665603ull;for(unsigned char c:s){h^=c;h*=1099511628211ull;}return h|(1ull<<63);}
uint64_t module_id(VkShaderModule m){std::lock_guard<std::mutex> l(mu);auto it=ids.find(m);return it==ids.end()?uint64_t(m):it->second;}
std::unordered_map<uint64_t,std::string> id_keys;
void register_module(VkShaderModule m,const std::string &key){std::lock_guard<std::mutex> l(mu);uint64_t id=fnv(key);ids[m]=id;id_keys[id]=key;}
struct Rec{std::vector<std::string> keys;std::vector<uint32_t> words;};
std::vector<Rec> recs;std::unordered_set<std::string> known;std::vector<uint8_t> pending;
std::unordered_map<std::string,VkPipeline> ready;std::vector<std::thread> threads;std::atomic<bool> stop{false};std::atomic<size_t> next{0};
std::atomic<uint64_t> built{0},failed{0},build_ns{0};uint64_t hits=0,creates=0,create_ns=0,create_max_ns=0;bool enabled=true;
std::unordered_map<std::string,VkShaderModule> modules;std::unordered_map<std::string,VkPipelineLayout> layouts;std::unordered_map<std::string,VkRenderPass> passes;
std::vector<VkDescriptorSetLayout> set_layouts;
std::string list_path(){std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');return (cut==std::string::npos?dir:dir.substr(0,cut))+"/vk-pipeline-list.bin";}
void record(const VkGraphicsPipelineCreateInfo &ci,const XbgPipelineKey &key){
    std::string bytes(reinterpret_cast<const char*>(key.words),size_t(key.count)*4);
    std::vector<std::string> keys;{std::lock_guard<std::mutex> l(mu);if(!known.insert(bytes).second)return;
        for(uint32_t i=0;i<ci.stageCount;i++){auto it=ids.find(ci.pStages[i].module);if(it==ids.end())return;keys.push_back(id_keys[it->second]);}}
    auto put=[](uint32_t v){for(int i=0;i<4;i++)pending.push_back(uint8_t(v>>(8*i)));};
    put(0x57504c58);put(uint32_t(keys.size()));for(auto &k:keys){put(uint32_t(k.size()));pending.insert(pending.end(),k.begin(),k.end());}
    put(key.count);pending.insert(pending.end(),bytes.begin(),bytes.end());
}
void flush(){if(pending.empty())return;FILE *f=fopen(list_path().c_str(),"ab");if(!f)return;bool ok=fwrite(pending.data(),1,pending.size(),f)==pending.size();ok=fclose(f)==0&&ok;if(ok)pending.clear();}
VkShaderModule load_module(const std::string &key){
    {std::lock_guard<std::mutex> l(mu);auto it=modules.find(key);if(it!=modules.end())return it->second;}
    FILE *f=fopen((std::string(cfg.shader_directory)+"/"+key+".xgs").c_str(),"rb");if(!f)return VK_NULL_HANDLE;
    fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);if(n<32 || n>16*1024*1024){fclose(f);return VK_NULL_HANDLE;}std::vector<uint8_t> b(static_cast<size_t>(n));bool ok=fread(b.data(),1,b.size(),f)==b.size();fclose(f);if(!ok)return VK_NULL_HANDLE;
    xbg_shader_header h;memcpy(&h,b.data(),sizeof(h));if(h.magic!=0x31534758 || h.version!=1 || h.spirv_bytes%4)return VK_NULL_HANDLE;
    size_t off=sizeof(h)+size_t(h.binding_count)*sizeof(xbg_shader_binding)+size_t(h.input_count)*4+h.c1_bytes;if(off+h.spirv_bytes!=b.size())return VK_NULL_HANDLE;
    std::vector<uint32_t> w(h.spirv_bytes/4);memcpy(w.data(),b.data()+off,h.spirv_bytes);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=h.spirv_bytes;ci.pCode=w.data();VkShaderModule m;if(vkCreateShaderModule(cfg.device,&ci,nullptr,&m)!=VK_SUCCESS)return VK_NULL_HANDLE;
    std::lock_guard<std::mutex> l(mu);auto ins=modules.emplace(key,m);if(!ins.second){vkDestroyShaderModule(cfg.device,m,nullptr);return ins.first->second;}return m;
}
struct Reader{const uint32_t *w;size_t n,i=0;bool bad=false;uint32_t u(){if(i>=n){bad=true;return 0;}return w[i++];}uint64_t q(){uint64_t lo=u();return lo|(uint64_t(u())<<32);}float f(){uint32_t b=u();float x;memcpy(&x,&b,4);return x;}};
VkPipelineLayout make_layout(const uint32_t *w,uint32_t n){
    std::string k(reinterpret_cast<const char*>(w),size_t(n)*4);{std::lock_guard<std::mutex> l(mu);auto it=layouts.find(k);if(it!=layouts.end())return it->second;}
    Reader r{w,n};VkDescriptorSetLayoutCreateInfo s{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};s.flags=r.u();uint32_t nb=r.u();if(nb>4096)return VK_NULL_HANDLE;std::vector<VkDescriptorSetLayoutBinding> bs(nb);
    for(auto &b:bs){b.binding=r.u();b.descriptorType=VkDescriptorType(r.u());b.descriptorCount=r.u();b.stageFlags=r.u();}s.bindingCount=nb;s.pBindings=bs.data();
    VkPipelineLayoutCreateInfo p{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};p.flags=r.u();uint32_t sets=r.u(),np=r.u();if(sets!=1 || np>16 || r.bad)return VK_NULL_HANDLE;std::vector<VkPushConstantRange> pr(np);for(auto &x:pr){x.stageFlags=r.u();x.offset=r.u();x.size=r.u();}
    if(r.bad || r.i!=n)return VK_NULL_HANDLE;VkDescriptorSetLayout sl;if(vkCreateDescriptorSetLayout(cfg.device,&s,nullptr,&sl)!=VK_SUCCESS)return VK_NULL_HANDLE;
    p.setLayoutCount=1;p.pSetLayouts=&sl;p.pushConstantRangeCount=np;p.pPushConstantRanges=pr.data();VkPipelineLayout pl;if(vkCreatePipelineLayout(cfg.device,&p,nullptr,&pl)!=VK_SUCCESS){vkDestroyDescriptorSetLayout(cfg.device,sl,nullptr);return VK_NULL_HANDLE;}
    std::lock_guard<std::mutex> l(mu);set_layouts.push_back(sl);layouts.emplace(k,pl);return pl;
}
VkRenderPass make_pass(const uint32_t *w,uint32_t n){
    std::string k(reinterpret_cast<const char*>(w),size_t(n)*4);{std::lock_guard<std::mutex> l(mu);auto it=passes.find(k);if(it!=passes.end())return it->second;}
    Reader r{w,n};VkRenderPassCreateInfo p{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};p.flags=r.u();uint32_t na=r.u();if(na>9)return VK_NULL_HANDLE;std::vector<VkAttachmentDescription> a(na);
    for(auto &x:a){x.flags=r.u();x.format=VkFormat(r.u());x.samples=VkSampleCountFlagBits(r.u());x.loadOp=VkAttachmentLoadOp(r.u());x.storeOp=VkAttachmentStoreOp(r.u());x.stencilLoadOp=VkAttachmentLoadOp(r.u());x.stencilStoreOp=VkAttachmentStoreOp(r.u());x.initialLayout=VkImageLayout(r.u());x.finalLayout=VkImageLayout(r.u());}
    VkSubpassDescription s{};s.flags=r.u();s.pipelineBindPoint=VkPipelineBindPoint(r.u());uint32_t nc=r.u();if(nc>8)return VK_NULL_HANDLE;std::vector<VkAttachmentReference> c(nc);for(auto &x:c){x.attachment=r.u();x.layout=VkImageLayout(r.u());}
    VkAttachmentReference d{};bool hd=r.u()!=0;if(hd){d.attachment=r.u();d.layout=VkImageLayout(r.u());}
    uint32_t nd=r.u();if(nd>8)return VK_NULL_HANDLE;std::vector<VkSubpassDependency> dep(nd);for(auto &x:dep){x.srcSubpass=r.u();x.dstSubpass=r.u();x.srcStageMask=r.u();x.dstStageMask=r.u();x.srcAccessMask=r.u();x.dstAccessMask=r.u();x.dependencyFlags=r.u();}
    if(r.bad || r.i!=n)return VK_NULL_HANDLE;
    s.colorAttachmentCount=nc;s.pColorAttachments=c.data();s.pDepthStencilAttachment=hd?&d:nullptr;p.attachmentCount=na;p.pAttachments=a.data();p.subpassCount=1;p.pSubpasses=&s;p.dependencyCount=nd;p.pDependencies=dep.data();
    VkRenderPass rp;if(vkCreateRenderPass(cfg.device,&p,nullptr,&rp)!=VK_SUCCESS)return VK_NULL_HANDLE;
    std::lock_guard<std::mutex> l(mu);auto ins=passes.emplace(k,rp);if(!ins.second){vkDestroyRenderPass(cfg.device,rp,nullptr);return ins.first->second;}return rp;
}
bool build(const Rec &rec){
    Reader r{rec.words.data(),rec.words.size()};if(r.u()!=1)return false;
    uint32_t ln=r.u();if(r.i+ln>r.n)return false;VkPipelineLayout layout=make_layout(r.w+r.i,ln);r.i+=ln;
    uint32_t pn=r.u();if(r.i+pn>r.n)return false;VkRenderPass pass=make_pass(r.w+r.i,pn);r.i+=pn;if(!layout || !pass)return false;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};ci.flags=r.u();ci.subpass=r.u();uint32_t ns=r.u();if(ns>5 || ns!=rec.keys.size())return false;
    VkPipelineShaderStageCreateInfo st[5]{};for(uint32_t i=0;i<ns;i++){st[i].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;st[i].flags=r.u();st[i].stage=VkShaderStageFlagBits(r.u());uint64_t id=r.q();if(id!=fnv(rec.keys[i]))return false;st[i].module=load_module(rec.keys[i]);if(!st[i].module)return false;st[i].pName="main";}
    VkPipelineVertexInputStateCreateInfo v{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};v.flags=r.u();uint32_t nvb=r.u();if(nvb>32)return false;std::vector<VkVertexInputBindingDescription> vb(nvb);for(auto &x:vb){x.binding=r.u();x.stride=r.u();x.inputRate=VkVertexInputRate(r.u());}
    uint32_t nva=r.u();if(nva>32)return false;std::vector<VkVertexInputAttributeDescription> va(nva);for(auto &x:va){x.location=r.u();x.binding=r.u();x.format=VkFormat(r.u());x.offset=r.u();}
    v.vertexBindingDescriptionCount=nvb;v.pVertexBindingDescriptions=vb.data();v.vertexAttributeDescriptionCount=nva;v.pVertexAttributeDescriptions=va.data();
    VkPipelineInputAssemblyStateCreateInfo a{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};a.flags=r.u();a.topology=VkPrimitiveTopology(r.u());a.primitiveRestartEnable=r.u();
    VkPipelineTessellationStateCreateInfo t{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};bool ht=r.u()!=0;if(ht){t.flags=r.u();t.patchControlPoints=r.u();}
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.flags=r.u();vp.viewportCount=r.u();vp.scissorCount=r.u();bool hv=r.u()!=0,hs=r.u()!=0;if(vp.viewportCount>16 || vp.scissorCount>16)return false;
    std::vector<VkViewport> vps(hv?vp.viewportCount:0);for(auto &x:vps){x.x=r.f();x.y=r.f();x.width=r.f();x.height=r.f();x.minDepth=r.f();x.maxDepth=r.f();}
    std::vector<VkRect2D> scs(hs?vp.scissorCount:0);for(auto &x:scs){x.offset.x=int32_t(r.u());x.offset.y=int32_t(r.u());x.extent.width=r.u();x.extent.height=r.u();}
    vp.pViewports=hv?vps.data():nullptr;vp.pScissors=hs?scs.data():nullptr;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.flags=r.u();rs.depthClampEnable=r.u();rs.rasterizerDiscardEnable=r.u();rs.polygonMode=VkPolygonMode(r.u());rs.cullMode=r.u();rs.frontFace=VkFrontFace(r.u());rs.depthBiasEnable=r.u();rs.depthBiasConstantFactor=r.f();rs.depthBiasClamp=r.f();rs.depthBiasSlopeFactor=r.f();rs.lineWidth=r.f();
    VkPipelineMultisampleStateCreateInfo m{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};m.flags=r.u();m.rasterizationSamples=VkSampleCountFlagBits(r.u());m.sampleShadingEnable=r.u();m.minSampleShading=r.f();m.alphaToCoverageEnable=r.u();m.alphaToOneEnable=r.u();bool hm=r.u()!=0;
    uint32_t mask[2]{};if(hm){if(m.rasterizationSamples>VK_SAMPLE_COUNT_64_BIT)return false;for(uint32_t i=0;i<(uint32_t(m.rasterizationSamples)+31)/32;i++)mask[i]=r.u();m.pSampleMask=mask;}
    auto stencil=[&](VkStencilOpState &s){s.failOp=VkStencilOp(r.u());s.passOp=VkStencilOp(r.u());s.depthFailOp=VkStencilOp(r.u());s.compareOp=VkCompareOp(r.u());s.compareMask=r.u();s.writeMask=r.u();s.reference=r.u();};
    VkPipelineDepthStencilStateCreateInfo d{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};d.flags=r.u();d.depthTestEnable=r.u();d.depthWriteEnable=r.u();d.depthCompareOp=VkCompareOp(r.u());d.depthBoundsTestEnable=r.u();d.stencilTestEnable=r.u();stencil(d.front);stencil(d.back);d.minDepthBounds=r.f();d.maxDepthBounds=r.f();
    VkPipelineColorBlendStateCreateInfo b{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};b.flags=r.u();b.logicOpEnable=r.u();b.logicOp=VkLogicOp(r.u());uint32_t nb=r.u();if(nb>8)return false;std::vector<VkPipelineColorBlendAttachmentState> ba(nb);
    for(auto &x:ba){x.blendEnable=r.u();x.srcColorBlendFactor=VkBlendFactor(r.u());x.dstColorBlendFactor=VkBlendFactor(r.u());x.colorBlendOp=VkBlendOp(r.u());x.srcAlphaBlendFactor=VkBlendFactor(r.u());x.dstAlphaBlendFactor=VkBlendFactor(r.u());x.alphaBlendOp=VkBlendOp(r.u());x.colorWriteMask=r.u();}
    b.attachmentCount=nb;b.pAttachments=ba.data();for(float &x:b.blendConstants)x=r.f();
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dy.flags=r.u();uint32_t nd=r.u();if(nd>64)return false;std::vector<VkDynamicState> ds(nd);for(auto &x:ds)x=VkDynamicState(r.u());dy.dynamicStateCount=nd;dy.pDynamicStates=ds.data();
    if(r.bad || r.i!=r.n)return false;
    ci.stageCount=ns;ci.pStages=st;ci.pVertexInputState=&v;ci.pInputAssemblyState=&a;ci.pTessellationState=ht?&t:nullptr;ci.pViewportState=&vp;ci.pRasterizationState=&rs;ci.pMultisampleState=&m;ci.pDepthStencilState=&d;ci.pColorBlendState=&b;ci.pDynamicState=&dy;ci.layout=layout;ci.renderPass=pass;
    VkPipeline pl;if(vkCreateGraphicsPipelines(cfg.device,pipeline_cache,1,&ci,nullptr,&pl)!=VK_SUCCESS)return false;
    std::string bytes(reinterpret_cast<const char*>(rec.words.data()),rec.words.size()*4);
    std::lock_guard<std::mutex> l(mu);if(!ready.emplace(bytes,pl).second)vkDestroyPipeline(cfg.device,pl,nullptr);return true;
}
void worker(){
    setpriority(PRIO_PROCESS,0,5); // below the game and graphics threads
    for(;;){if(stop.load())return;size_t i=next.fetch_add(1);if(i>=recs.size())return;uint64_t t0=timing_clock();if(build(recs[i]))++built;else ++failed;build_ns+=timing_clock()-t0;}
}
void start(){
    FILE *f=fopen(list_path().c_str(),"rb");if(f){fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);std::vector<uint8_t> b(n>0 && n<(64l<<20)?size_t(n):0);bool ok=b.empty() || fread(b.data(),1,b.size(),f)==b.size();fclose(f);
        size_t i=0;auto get=[&](uint32_t &v){if(i+4>b.size())return false;memcpy(&v,b.data()+i,4);i+=4;return true;};
        while(ok && i<b.size()){uint32_t magic,ns,count;if(!get(magic)||magic!=0x57504c58||!get(ns)||ns>5)break;Rec r;bool good=true;
            for(uint32_t s=0;s<ns && good;s++){uint32_t len;if(!get(len)||len>256||i+len>b.size()){good=false;break;}r.keys.emplace_back(reinterpret_cast<const char*>(b.data()+i),len);i+=len;}
            if(!good||!get(count)||count>XbgPipelineKey::Capacity||i+size_t(count)*4>b.size())break;r.words.resize(count);memcpy(r.words.data(),b.data()+i,size_t(count)*4);i+=size_t(count)*4;
            if(known.insert(std::string(reinterpret_cast<const char*>(r.words.data()),r.words.size()*4)).second)recs.push_back(std::move(r));}}
    xbg_logf("NVN warm list=%zu enabled=%d",recs.size(),int(enabled));
    if(!enabled || recs.empty())return;stop=false;next=0;unsigned n=std::min<unsigned>(3,std::max(1u,std::thread::hardware_concurrency()/3));
    for(unsigned i=0;i<n;i++)threads.emplace_back(worker);
}
void shutdown(){stop=true;for(auto &t:threads)if(t.joinable())t.join();threads.clear();std::lock_guard<std::mutex> l(mu);
    for(auto &p:ready)vkDestroyPipeline(cfg.device,p.second,nullptr);ready.clear();for(auto &p:layouts)vkDestroyPipelineLayout(cfg.device,p.second,nullptr);layouts.clear();for(auto s:set_layouts)vkDestroyDescriptorSetLayout(cfg.device,s,nullptr);set_layouts.clear();
    for(auto &p:passes)vkDestroyRenderPass(cfg.device,p.second,nullptr);passes.clear();for(auto &m:modules)vkDestroyShaderModule(cfg.device,m.second,nullptr);modules.clear();ids.clear();id_keys.clear();recs.clear();known.clear();}
bool take(const std::string &bytes,VkPipeline &out){std::lock_guard<std::mutex> l(mu);auto it=ready.find(bytes);if(it==ready.end())return false;out=it->second;ready.erase(it);return true;}
}
void pipeline_lookup(PipelineLease &out,const XbgPipelineKey &key,bool cacheable,const VkGraphicsPipelineCreateInfo *graphics,const VkComputePipelineCreateInfo *compute){
    static std::string bytes;bytes.clear();   // reused key buffer (graphics thread only)
    if(cacheable){
        bytes.assign(reinterpret_cast<const char*>(key.words),size_t(key.count)*4);
        auto found=pipeline_objects.find(bytes);
        if(found!=pipeline_objects.end()){++pipeline_counts.hits;found->second.last_used=++pipeline_tick;out.pipeline=found->second.pipeline;return;}
        ++pipeline_counts.misses;++pipelines_unsaved;
    }else ++pipeline_counts.bypass;
    if(graphics && cacheable && warm::take(bytes,out.pipeline)){++warm::hits;}
    else if(graphics){uint64_t t0=timing_clock();VK(vkCreateGraphicsPipelines(cfg.device,pipeline_cache,1,graphics,nullptr,&out.pipeline));uint64_t dt=timing_clock()-t0;++warm::creates;warm::create_ns+=dt;warm::create_max_ns=std::max(warm::create_max_ns,dt);if(cacheable)warm::record(*graphics,key);}
    else VK(vkCreateComputePipelines(cfg.device,pipeline_cache,1,compute,nullptr,&out.pipeline));
    if(!cacheable){out.transient=true;return;}
    if(pipeline_objects.size()==pipeline_object_limit){
        // All prior draws/dispatches waited their fence. This lookup occurs
        // before binding the current pipeline; no retained entry is in flight.
        auto oldest=std::min_element(pipeline_objects.begin(),pipeline_objects.end(),[](const auto &a,const auto &b){return a.second.last_used<b.second.last_used;});
        {VkPipeline old=oldest->second.pipeline;retire_handle([old]{vkDestroyPipeline(cfg.device,old,nullptr);});}pipeline_objects.erase(oldest);++pipeline_counts.evictions;
    }
    pipeline_objects.emplace(std::string(bytes),PipelineEntry{out.pipeline,++pipeline_tick});
}
void clear_pipeline_objects(){for(auto &p:pipeline_objects)vkDestroyPipeline(cfg.device,p.second.pipeline,nullptr);pipeline_objects.clear();}
struct PipelineDefinition {
    std::vector<uint32_t> words;bool valid=false;
    void assign(const XbgPipelineKey &k,bool ok){valid=ok && k.valid;if(valid)words.assign(k.words,k.words+k.count);}
    XbgPipelineKeyView view()const{return {words.data(),uint32_t(words.size()),valid};}
};
// Descriptor set layouts, pipeline layouts and samplers are immutable for a
// given binding list / sampler state, so they are created once and cached.
// Descriptor sets come from per-slot pools that are reset after the slot fence.
struct LayoutEntry {VkDescriptorSetLayout layout{};VkPipelineLayout pipeline_layout{};PipelineDefinition definition;VkDescriptorUpdateTemplate tmpl{};std::vector<uint64_t> tmpl_sig;};
// One descriptor payload slot for update templates (image/buffer info or texel view).
union DescEntry{VkDescriptorImageInfo image;VkDescriptorBufferInfo buffer;VkBufferView view;};

std::map<std::vector<uint32_t>,LayoutEntry> layout_cache;
std::map<std::array<const void*,7>,LayoutEntry*> shader_layouts; // shader set -> layout (bindings are immutable per shader)
std::map<std::array<uint32_t,16>,VkSampler> sampler_cache;
Arena static_arena; // never reset: immutable per-shader c1 constants
std::map<std::pair<float,float>,Buffer> support_static; // support block per viewport size (contents depend on it only)
struct SamplerFastKey{std::array<uint32_t,11> w;bool operator==(const SamplerFastKey &o)const{return w==o.w;}};
struct SamplerFastHash{size_t operator()(const SamplerFastKey &k)const{uint64_t h=1469598103934665603ull;for(uint32_t v:k.w){h^=v;h*=1099511628211ull;}return size_t(h);}};
std::unordered_map<SamplerFastKey,VkSampler,SamplerFastHash> sampler_fast;
void clear_descriptor_caches(){sampler_fast.clear();
    for(auto &l:layout_cache){if(l.second.tmpl)vkDestroyDescriptorUpdateTemplate(cfg.device,l.second.tmpl,nullptr);vkDestroyPipelineLayout(cfg.device,l.second.pipeline_layout,nullptr);vkDestroyDescriptorSetLayout(cfg.device,l.second.layout,nullptr);}layout_cache.clear();shader_layouts.clear();
    for(auto &x:sampler_cache)vkDestroySampler(cfg.device,x.second,nullptr);sampler_cache.clear();
    static_arena.release();support_static.clear();
}
VkDescriptorSet allocate_set(VkDescriptorSetLayout layout){
    Slot &s=recording_slot();
    for(;;){
        if(s.pool_next==s.pools.size()){
            std::vector<VkDescriptorPoolSize> sizes;for(uint32_t t=0;t<=VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;t++)sizes.push_back({VkDescriptorType(t),t==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER||t==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER||t==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC||t==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER?16384u:2048u});
            VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};ci.maxSets=2048;ci.poolSizeCount=sizes.size();ci.pPoolSizes=sizes.data();
            VkDescriptorPool pool;VK(vkCreateDescriptorPool(cfg.device,&ci,nullptr,&pool));s.pools.push_back(pool);
        }
        // Sets are allocated 16 at a time per layout and handed out from a per-slot spare list.
        auto &spare=s.spare_sets[layout];auto &used=s.used_sets[layout];if(!spare.empty()){VkDescriptorSet set=spare.back();spare.pop_back();used.push_back(set);return set;}
        VkDescriptorSetLayout layouts[16];for(auto &l:layouts)l=layout;VkDescriptorSet sets[16]{};
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=s.pools[s.pool_next];ai.descriptorSetCount=16;ai.pSetLayouts=layouts;
        VkResult r=vkAllocateDescriptorSets(cfg.device,&ai,sets);
        if(r!=VK_SUCCESS){ai.descriptorSetCount=1;r=vkAllocateDescriptorSets(cfg.device,&ai,sets);if(r==VK_SUCCESS){used.push_back(sets[0]);return sets[0];}}
        else{spare.insert(spare.end(),sets+1,sets+16);used.push_back(sets[0]);return sets[0];}
        if(r==VK_ERROR_OUT_OF_POOL_MEMORY || r==VK_ERROR_FRAGMENTED_POOL){++s.pool_next;continue;}
        check(r,"vkAllocateDescriptorSets");
    }
}
struct Descriptors {
    const LayoutEntry *entry=nullptr;VkDescriptorSet set{};VkPipelineLayout pipeline_layout{};std::vector<uint32_t> dynamic_offsets;
    std::vector<Buffer> buffers;std::vector<VkBufferView> buffer_views; // buffers: reserved up front, pointers stay valid
    std::vector<std::pair<std::pair<uint64_t,uint64_t>,Buffer*>> storage; // alias lookup for storage bindings
    std::vector<std::pair<std::pair<uint64_t,uint64_t>,Buffer*>> written; // ranges the GPU may write
    struct InputImage {Image *image;view_t view;bool has_view;std::string label;};
    std::vector<InputImage> capture_inputs;
    ~Descriptors(){for(auto x:buffer_views)vkDestroyBufferView(cfg.device,x,nullptr);}
    Buffer *keep(const Buffer &b){need(buffers.size()<buffers.capacity(),"Descriptor buffer reservation exceeded");buffers.push_back(b);return &buffers.back();}
    bool needs_writeback()const{return !written.empty();}
    void defer_writeback(){const void *slot=&slots[slot_index];for(auto &v:written){if(draw_detail)draw_timing.storage(v.first.second);snapshot_invalidate(v.first.first,v.first.second);pending_writebacks.push_back({v.first.first,v.first.second,*v.second,slot});}}
    void writeback(){for(auto &v:written){if(draw_detail)draw_timing.storage(v.first.second);v.second->invalidate();invalidate_cpu_range(v.first.first,v.first.second);memcpy(reinterpret_cast<void*>(v.first.first),v.second->mapped,v.first.second);}}
    void capture_read_images(){for(auto &i:capture_inputs)capture_image(*i.image,i.has_view?&i.view:nullptr,i.label.c_str());}
};
// Per-draw descriptor records were one make_shared + vector reservations each (≈1800/frame, slow musl malloc). They live
// until the slot's fence like before, then return to a free list with their capacity.
std::vector<std::unique_ptr<Descriptors>> desc_all;std::vector<Descriptors*> desc_free;
Descriptors &acquire_descriptors(){
    Descriptors *d;if(desc_free.empty()){desc_all.push_back(std::make_unique<Descriptors>());d=desc_all.back().get();}else{d=desc_free.back();desc_free.pop_back();}
    recording_slot().desc_used.push_back(d);return *d;
}
void release_descriptors(Slot &s){
    for(Descriptors *d:s.desc_used){for(auto x:d->buffer_views)vkDestroyBufferView(cfg.device,x,nullptr);
        d->entry=nullptr;d->set=VK_NULL_HANDLE;d->pipeline_layout=VK_NULL_HANDLE;d->dynamic_offsets.clear();d->buffers.clear();d->buffer_views.clear();d->storage.clear();d->written.clear();d->capture_inputs.clear();
        desc_free.push_back(d);}
    s.desc_used.clear();
}
// Fast path (iOS: samplers fetched directly instead of re-deriving per draw): raw NVN parameters +
// view format -> VkSampler, before the full conversion/validation below.
VkSampler sampler_slow(const samp_par &p,Image &im,VkFormat viewed,bool integer_border);
VkSampler sampler(const samp_par &p,Image &im,VkFormat viewed,bool integer_border){
    auto bits=[](float f){uint32_t u;memcpy(&u,&f,4);return u;};
    SamplerFastKey k{{uint32_t(p.minf)|uint32_t(p.magf)<<8|uint32_t(p.ws)<<16|uint32_t(p.wt)<<24,uint32_t(p.wr)|uint32_t(p.cmp_mode)<<8|uint32_t(p.cmp_func)<<16|uint32_t(integer_border)<<24,
        bits(p.lod_min),bits(p.lod_max),bits(p.lod_bias),bits(p.aniso),bits(p.border[0]),bits(p.border[1]),bits(p.border[2]),bits(p.border[3]),uint32_t(viewed)}};
    auto it=sampler_fast.find(k);if(it!=sampler_fast.end())return it->second;
    VkSampler out=sampler_slow(p,im,viewed,integer_border);sampler_fast.emplace(k,out);return out;
}
VkSampler sampler_slow(const samp_par &p,Image &im,VkFormat viewed,bool integer_border){
    xbg_nvn_sampler_desc input{p.minf,p.magf,p.ws,p.wt,p.wr,p.cmp_mode,p.cmp_func,p.lod_min,p.lod_max,p.lod_bias,p.aniso,{p.border[0],p.border[1],p.border[2],p.border[3]}};
    xbg_sampler_limits limits{cfg.enabled_core.samplerAnisotropy,sampler_mirror_clamp_enabled,VkBool32(integer_border),properties.limits.maxSamplerAnisotropy,properties.limits.maxSamplerLodBias};
    VkSamplerCreateInfo s{};int result=xbg_sampler_from_nvn(&input,&limits,&s);
    if(result)xbg_fatalf("NVN sampler %s result=%d min=%u mag=%u wrap=%u/%u/%u lod=%g/%g bias=%g aniso=%g compare=%u/%u border=%g,%g,%g,%g texture=%llu format=%#x vk=%u mirrorEnabled=%u",xbg_sampler_error(result),result,p.minf,p.magf,p.ws,p.wt,p.wr,p.lod_min,p.lod_max,p.lod_bias,p.aniso,p.cmp_mode,p.cmp_func,p.border[0],p.border[1],p.border[2],p.border[3],(unsigned long long)im.source.serial,im.source.d.format,unsigned(viewed),sampler_mirror_clamp_enabled);
    const VkFormatProperties &fp=format_properties(viewed);
    if((s.minFilter==VK_FILTER_LINEAR || s.magFilter==VK_FILTER_LINEAR) && !(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))xbg_fatalf("NVN sampler linear filtering unsupported min=%u mag=%u compare=%u/%u format=%#x vk=%u optimalFeatures=%#x texture=%llu",p.minf,p.magf,p.cmp_mode,p.cmp_func,im.source.d.format,unsigned(viewed),fp.optimalTilingFeatures,(unsigned long long)im.source.serial);
    need(s.pNext==nullptr,"Unexpected sampler extension chain");
    auto bits=[](float f){uint32_t u;memcpy(&u,&f,4);return u;};
    std::array<uint32_t,16> key{s.flags,uint32_t(s.magFilter),uint32_t(s.minFilter),uint32_t(s.mipmapMode),uint32_t(s.addressModeU),uint32_t(s.addressModeV),uint32_t(s.addressModeW),bits(s.mipLodBias),s.anisotropyEnable,bits(s.maxAnisotropy),s.compareEnable,uint32_t(s.compareOp),bits(s.minLod),bits(s.maxLod),uint32_t(s.borderColor),s.unnormalizedCoordinates};
    auto it=sampler_cache.find(key);if(it!=sampler_cache.end())return it->second;
    VkSampler out;VK(vkCreateSampler(cfg.device,&s,nullptr,&out));sampler_cache.emplace(key,out);return out;
}
void fill_driver(uint8_t *data,unsigned ns,unsigned bank){
    if(bank==0){unsigned base=ns==0?0x110:ns==5?0x310:ns==1?0x510:0;need(base,"Unverified storage-table base for tessellation/geometry stage");for(unsigned r=0;r<16;r++){memcpy(data+base+16*r,&state.sb[ns][r].address,8);uint32_t n=uint32_t(state.sb[ns][r].size);memcpy(data+base+16*r+8,&n,4);}if(ns==5)for(unsigned r=0;r<16;r++){memcpy(data+0x210+16*r,&state.ub[ns][r].address,8);uint32_t n=uint32_t(state.ub[ns][r].size);memcpy(data+0x218+16*r,&n,4);}}
    else{for(unsigned r=0;r<128;r++){memcpy(data+0x168+8*r,&state.tex[ns][r],8);memcpy(data+0x568+8*r,&state.samp[ns][r],8);}for(unsigned r=0;r<9;r++)memcpy(data+0x120+8*r,&state.img[ns][r],8);}
}
// Uniform data is bound with the guest-bound size (rounded to 16 bytes); reads
// past it rely on robustBufferAccess like the reference Ryujinx Vulkan backend.
size_t uniform_range(size_t copy){return std::max<size_t>(16,(copy+15)&~size_t(15));}
Buffer uniform_snapshot_slow(uint64_t address,size_t copy);
Buffer uniform_snapshot(uint64_t address,size_t copy){CPU_SCOPE(Snapshot);
    SnapFront &f=snap_front_for(1,address,copy);if(f.epoch==snapshot_epoch && f.address==address && f.bytes==copy){++perf.snapshot_hits;return f.buffer;}
    Buffer b=uniform_snapshot_slow(address,copy);f.address=address;f.bytes=copy;f.epoch=snapshot_epoch;f.buffer=b;return b;
}
Buffer uniform_snapshot_slow(uint64_t address,size_t copy){
    size_t n=uniform_range(copy);auto key=std::make_pair(address,uint64_t(copy));
    auto it=uniform_snapshots.find(key);if(it!=uniform_snapshots.end()){++perf.snapshot_hits;return it->second;}
    {Buffer view;if(pending_view(address,n,properties.limits.minUniformBufferOffsetAlignment,view)){uniform_snapshots.emplace(key,view);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,n);return view;}}
    ++perf.snapshot_misses;perf.snapshot_bytes+=copy;snap_site_bytes[5]+=copy;cpu_access(address,copy,2);
    Buffer b=stream_buffer(n,reinterpret_cast<const void*>(address),copy,true);uniform_snapshots.emplace(key,b);snapshot_maxlen=std::max<uint64_t>(snapshot_maxlen,copy);return b;
}
// Descriptor memo (iOS "same inputs -> reuse the parameter block"): within one guest submit, a draw whose shaders and
// bound inputs (buffer ranges, texture/sampler handles, pools, driver generations, viewport) equal an earlier draw's
// reuses that draw's set and dynamic offsets. Snapshots are per-submit too, so the bound data is the same. On a hit
// every input image still gets its pending CPU upload and read barrier. Not used for compute, written storage,
// texel buffers or storage images. files/gfx-no-descmemo disables it.
// Per-submit resolved sampled-image descriptors: (texture handle, sampler handle, pools, binding kind) -> image, view,
// sampler. A hit still performs the pending CPU upload and the read barrier for the image.
struct ImgKey{uint64_t h,sh,tp,to,sp,so,kind;bool operator==(const ImgKey &o)const{return h==o.h&&sh==o.sh&&tp==o.tp&&to==o.to&&sp==o.sp&&so==o.so&&kind==o.kind;}};
struct ImgKeyHash{size_t operator()(const ImgKey &k)const{uint64_t x=k.h*0x9E3779B97F4A7C15ull^k.sh*0xC2B2AE3D27D4EB4Full^k.tp^(k.to<<17)^(k.sp<<7)^k.so^(k.kind<<40);x^=x>>29;return size_t(x*0xBF58476D1CE4E5B9ull);}};
struct ImgEntry{const tex_t *tex;Image *im;VkImageView view;VkSampler sampler;const RegisteredTexture *record;uint64_t revision;};
std::unordered_map<ImgKey,ImgEntry,ImgKeyHash> img_cache;bool img_cache_enabled=true;uint64_t img_cache_hits=0,img_cache_misses=0;
struct DescMemo{const LayoutEntry *entry;VkDescriptorSet set;VkPipelineLayout pipeline_layout;std::vector<uint32_t> dynamic_offsets;std::vector<MemoImage> images;struct Dyn{uint64_t order;VkBuffer buffer;uint32_t offset;bool uniform;uint8_t src=0,ns=0,bank=0;};std::vector<Dyn> dyn;bool persistent=false;std::vector<std::pair<uint64_t,const void*>> tex_ids;struct Stor{uint64_t base,bytes;VkBuffer buffer;VkDeviceSize offset;};std::vector<Stor> stor;
    struct Texel{const RegisteredTexture *record;uint64_t revision,base,bytes,gen;uint32_t format;};std::vector<Texel> texel;};   // texel: resident uniform texel buffers (texel_acquire)
struct DynUbo{uint64_t order,address;size_t copy;};
// Cross-frame memo (Claude C147): the memo lives per frame slot and survives submits. An entry is persistent when its set
// holds no per-frame snapshot except dynamic uniform buffers (re-snapshotted on every hit, only the offset moves) and the
// support / driver parameter blocks (recomputed on every hit, buffer must be the same arena buffer). Entries with storage
// snapshots stay frame-local (cleared at submit, as before). Persistent sets are pinned out of the slot's set recycling;
// Texture registrations use per-record revisions (checked on every hit); sampler changes bump memo_epoch, which drops
// every slot's memo lazily and returns pinned sets to that slot (recycled only after its next fence).
// files/gfx-no-xmemo keeps the old per-submit behaviour.
std::unordered_map<std::vector<uint64_t>,DescMemo,SetKeyHash> desc_memo_slot[2];std::unordered_map<VkDescriptorSet,PinRef> memo_pinned[2];
std::atomic<uint64_t> memo_epoch{1};bool dyn_partial_enabled=true;uint64_t key_diff[14]={};uint64_t memo_bump[5]={};uint64_t memo_slot_epoch[2]={1,1};bool xmemo_enabled=true,memo_dyn_ok=false;uint64_t xmemo_hits=0,xmemo_drops=0;
#define desc_memo desc_memo_slot[slot_index]
uint64_t desc_ns[6]={};bool desc_memo_enabled=true;uint64_t desc_memo_hits=0,desc_memo_misses=0,desc_memo_why[6]={};
std::vector<MemoImage> desc_memo_images;std::vector<DescMemo::Dyn> desc_memo_dyn;bool desc_memo_recording=false;
std::vector<DescMemo::Texel> desc_memo_texel;bool desc_memo_texel_fail=false;
void descriptors_build(Descriptors &out,Commands &commands,bool compute);
// Dynamic uniform buffers (Claude C151): Maleoon allows 8 per pipeline layout. Mode 1 = every uniform buffer dynamic (fits);
// mode 2 = only the per-draw ones (guest uniform buffers, driver blocks) dynamic, the support block (depends on the viewport
// only, kept in the static arena) and c1 constants (static) bound at fixed offsets; mode 0 = none dynamic.
#define UBO_DYN(mode,b) ((b).type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && ((mode)==1 || ((mode)==2 && ((b).source==XBG_SOURCE_UNIFORM||(b).source==XBG_SOURCE_DRIVER0||(b).source==XBG_SOURCE_DRIVER2))))
unsigned dyn_mode_for(bool compute,bool *no_ubo=nullptr){unsigned all=0,vary=0;
    for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute)for(auto &b:state.shader[st]->bindings)if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER){all+=b.count;if(b.source==XBG_SOURCE_UNIFORM||b.source==XBG_SOURCE_DRIVER0||b.source==XBG_SOURCE_DRIVER2)vary+=b.count;}
    if(no_ubo)*no_ubo=all==0;unsigned lim=properties.limits.maxDescriptorSetUniformBuffersDynamic;
    return all && all<=lim?1:(vary && vary<=lim && dyn_partial_enabled?2:0);}
void memo_unpin_all(unsigned si){Slot &s=slots[si];for(auto &pn:memo_pinned[si])s.used_sets[pn.second.layout].push_back(pn.first);memo_pinned[si].clear();desc_memo_slot[si].clear();}
void memo_check_epoch(){unsigned si=slot_index;if(memo_slot_epoch[si]!=memo_epoch || memo_pinned[si].size()>8192){if(!desc_memo_slot[si].empty())++xmemo_drops;memo_unpin_all(si);memo_slot_epoch[si]=memo_epoch;}}
// End of a submit for the recording slot: frame-local entries go (their snapshots are per submit); persistent ones stay.
void desc_memo_reset(){img_cache.clear();
    if(!xmemo_enabled){memo_unpin_all(slot_index);return;}
    auto &m=desc_memo_slot[slot_index];for(auto it=m.begin();it!=m.end();){if(it->second.persistent)++it;else it=m.erase(it);}}
Buffer &support_block(){size_t n=1392;auto k=std::make_pair(state.viewport.width,state.viewport.height);
    auto it=support_static.find(k);if(it!=support_static.end())return it->second;
    Buffer s=static_arena.alloc(uniform_range(n));xbg_support_defaults(s.mapped,state.viewport.width,state.viewport.height);s.flush();return support_static.emplace(k,s).first->second;}
Buffer &driver_block(unsigned ns,unsigned bank){uint64_t gen=bank?tex_gen[ns]:buf_gen[ns];
    DriverCache &dc=driver_cache[ns][bank];if(dc.slot!=slot_serial || dc.gen!=gen || !dc.slot){Buffer s=stream_buffer(0x1000);fill_driver(static_cast<uint8_t*>(s.mapped),ns,bank?2:0);s.flush();dc.gen=gen;dc.slot=slot_serial;dc.buffer=s;}
    return dc.buffer;}
// Uniform texel buffers held in resident per-slot copies, so descriptor sets that bind them can be memoized across frames
// (D2R camp: 558 of 2188 draws per frame bind one and were rebuilt every time, ~21 us each against ~2.5 us for a memo hit).
// A copy belongs to one slot (refreshed only while that slot records, after its previous submission retired) and is
// refreshed from guest memory when its content hash changes, before its first read in a frame; content that changes again
// within the frame falls back to the per-frame snapshot. A memo entry checks the copy's generation, so an evicted and
// recreated copy is never mistaken for the one its set holds. Opt-in: gfx-texel-memo = '1'.
TexelResident *texel_acquire(uint64_t base,uint64_t bytes,VkFormat fmt){
    unsigned si=slot_index;auto &map=texel_resident[si];TexelKey key{base,bytes,uint32_t(fmt)};
    auto it=map.find(key);
    if(it!=map.end() && it->second.checked==slot_serial && it->second.check_epoch==texel_write_epoch){it->second.used=slot_serial;return &it->second;}
    cpu_access(base,bytes,8);   // pending GPU results land in guest memory first, as for snapshot()
    uint64_t h=hash_memory(reinterpret_cast<const void*>(uintptr_t(base)),size_t(bytes));
    if(it==map.end()){
        if(map.size()>=8192){++texel_fallbacks;return nullptr;}
        TexelResident r;r.buffer=texel_arena[si].alloc(size_t(bytes));
        VkBufferViewCreateInfo v{VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};v.buffer=r.buffer.buffer;v.offset=r.buffer.offset;v.format=fmt;v.range=bytes;VK(vkCreateBufferView(cfg.device,&v,nullptr,&r.view));
        memcpy(r.buffer.mapped,reinterpret_cast<const void*>(uintptr_t(base)),size_t(bytes));r.buffer.flush();r.hash=h;r.used=slot_serial;r.gen=++texel_gen;++texel_refreshes;
        r.checked=slot_serial;r.check_epoch=texel_write_epoch;texel_lo=std::min(texel_lo,base);texel_hi=std::max(texel_hi,base+bytes);
        return &map.emplace(key,r).first->second;}
    TexelResident &r=it->second;
    if(r.hash!=h){if(r.used==slot_serial){++texel_fallbacks;return nullptr;}   // this frame already reads the earlier content
        memcpy(r.buffer.mapped,reinterpret_cast<const void*>(uintptr_t(base)),size_t(bytes));r.buffer.flush();r.hash=h;++texel_refreshes;}
    r.used=slot_serial;r.checked=slot_serial;r.check_epoch=texel_write_epoch;return &r;
}
void texel_touch(uint64_t address,uint64_t bytes){
    auto &map=texel_resident[slot_index];if(map.size()>64){++texel_write_epoch;return;}   // many copies: invalidate coarsely
    for(auto &e:map)if(address<e.first.base+e.first.bytes && address+bytes>e.first.base){++texel_write_epoch;return;}
}
// While a slot starts recording (its previous submission retired): drop copies unused for ~128 submissions; when the holes they
// leave in the slot's arena pass 8 MB, start the slot's store over (memo entries then fail their generation check).
void texel_evict(unsigned si,bool all){
    auto &map=texel_resident[si];
    for(auto it=map.begin();it!=map.end();){if(all || it->second.used+128<slot_serial){vkDestroyBufferView(cfg.device,it->second.view,nullptr);texel_hole_bytes[si]+=it->first.bytes;it=map.erase(it);++texel_evictions;}else ++it;}
    if(all || texel_hole_bytes[si]>8ull*1024*1024){for(auto &e:map)vkDestroyBufferView(cfg.device,e.second.view,nullptr);map.clear();texel_arena[si].release();texel_hole_bytes[si]=0;}
}
void descriptors(Descriptors &out,Commands &commands,bool compute){
    uint64_t tm0=perf_clock();struct MemoClock{uint64_t t0;~MemoClock(){}} mc{tm0};
    static std::vector<uint64_t> key;key.clear();static std::vector<uint8_t> ktag;ktag.clear();
    bool ok=desc_memo_enabled && !compute && !capture.active && !provenance_file;
    if(ok){CPU_SCOPE(Descriptor);
        static std::vector<DynUbo> dyn_ubos;dyn_ubos.clear();
        bool no_ubo=false;const unsigned dmode=dyn_mode_for(false,&no_ubo);memo_dyn_ok=dmode!=0 || no_ubo;
        for(unsigned st=0;st<5;st++){key.push_back(uint64_t(uintptr_t(state.shader[st])));ktag.push_back(0);}
        key.push_back(state.texture_pool.pool?uint64_t(uintptr_t(state.texture_pool.pool->mem)):0);key.push_back(uint64_t(state.texture_pool.off));key.push_back(uint64_t(state.texture_pool.n));ktag.insert(ktag.end(),3,1);
        key.push_back(state.sampler_pool.pool?uint64_t(uintptr_t(state.sampler_pool.pool->mem)):0);key.push_back(uint64_t(state.sampler_pool.off));ktag.insert(ktag.end(),2,1);
        for(unsigned st=0;st<5 && ok;st++)if(state.shader[st]){Shader &shader=*state.shader[st];unsigned ns=nvn_stage[st];
            for(auto &b:shader.bindings){
                // Texel view 属于 slot，退休时销毁；memo 不追踪其快照或生命周期，不能保留旧 set。
                // Slot retirement destroys texel views; memo tracks neither their snapshots nor lifetime, so exclude both kinds.
                if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER || (b.type==VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER && !texel_memo_enabled)){ok=false;++desc_memo_why[0];break;}
                if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE){ok=false;++desc_memo_why[1];break;}
                if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && shader.storage_writes.count(b.binding)){ok=false;++desc_memo_why[2];break;}
                for(unsigned element=0;element<b.count;element++){unsigned slot=b.slot+element;
                    switch(b.source){
                    case XBG_SOURCE_UNIFORM:case XBG_SOURCE_STORAGE:{if(slot>=16){ok=false;break;}auto r=b.source==XBG_SOURCE_UNIFORM?state.ub[ns][slot]:state.sb[ns][slot];
                        if(UBO_DYN(dmode,b) && b.source==XBG_SOURCE_UNIFORM && r.address && r.size){size_t copy=std::min<size_t>(r.size,b.bytes?b.bytes:65536);key.push_back(copy);ktag.push_back(2);dyn_ubos.push_back({(uint64_t(b.binding)<<32)|element,r.address,copy});} // address only moves the dynamic offset
                        else{key.push_back(r.address);key.push_back(r.size);ktag.push_back(b.source==XBG_SOURCE_UNIFORM?3:10);ktag.push_back(4);}break;}
                    case XBG_SOURCE_SUPPORT:key.push_back((uint64_t(uint32_t(state.viewport.width))<<32)|uint32_t(state.viewport.height));key.push_back(state.color.alpha_func);ktag.push_back(5);ktag.push_back(6);break;
                    case XBG_SOURCE_C1:break;
                    case XBG_SOURCE_DRIVER0:case XBG_SOURCE_DRIVER2:key.push_back(UBO_DYN(dmode,b)?uint64_t(0xD0D0+b.source):(b.source==XBG_SOURCE_DRIVER0?tex_gen[ns]:buf_gen[ns]));ktag.push_back(7);break; // dynamic: block rebuilt from the current state on every hit (driver_block); otherwise its offset is baked into the set
                    case XBG_SOURCE_TEXTURE:case XBG_SOURCE_IMAGE:{if(slot>=(b.source==XBG_SOURCE_IMAGE?16u:128u)){ok=false;break;}key.push_back(b.source==XBG_SOURCE_IMAGE?state.img[ns][slot]:state.tex[ns][slot]);ktag.push_back(8);
                        if(b.sampler_slot!=UINT32_MAX){if(b.sampler_slot>=128){ok=false;break;}key.push_back(state.samp[ns][b.sampler_slot]);ktag.push_back(9);}break;}
                    default:ok=false;}
                    if(!ok)break;
                }
                if(!ok)break;
            }}
        if(ok){memo_check_epoch();auto hit=desc_memo.find(key);
            if(hit==desc_memo.end() && ktag.size()==key.size()){   // diagnostic: which key part differs from the last key of the same shaders
                static std::unordered_map<std::vector<uint64_t>,std::vector<uint64_t>,SetKeyHash> last;std::vector<uint64_t> pre(key.begin(),key.begin()+5);
                auto &prev=last[pre];if(prev.size()==key.size()){bool any=false;for(size_t i=0;i<key.size();i++)if(prev[i]!=key[i]){++key_diff[ktag[i]<12?ktag[i]:11];any=true;}if(!any)++key_diff[11];}else ++key_diff[prev.empty()?12:13];
                prev=key;}
            if(hit!=desc_memo.end()){DescMemo &m=hit->second;
                static std::vector<std::pair<uint64_t,Buffer>> fresh;fresh.clear();
                for(auto &u:dyn_ubos)fresh.push_back({u.order,uniform_snapshot(u.address,u.copy)});
                std::sort(fresh.begin(),fresh.end(),[](const std::pair<uint64_t,Buffer> &a,const std::pair<uint64_t,Buffer> &b){return a.first<b.first;});
                static std::vector<uint32_t> offsets;offsets.clear();bool same=true;size_t fi=0;
                {size_t elements=0;for(unsigned st=0;st<5;st++)if(state.shader[st])for(auto &b:state.shader[st]->bindings)elements+=b.count;out.buffers.reserve(elements*2+40);} // keep() below (support / driver blocks) and draw() vertex/index buffers
                for(auto &d:m.dyn){if(!d.uniform){
                        if(d.src==XBG_SOURCE_SUPPORT || d.src==XBG_SOURCE_DRIVER0 || d.src==XBG_SOURCE_DRIVER2){
                            Buffer &blk=d.src==XBG_SOURCE_SUPPORT?support_block():driver_block(d.ns,d.bank);if(blk.buffer!=d.buffer){same=false;break;}out.keep(blk);offsets.push_back(uint32_t(blk.offset));}
                        else offsets.push_back(d.offset);continue;}
                    if(fi>=fresh.size() || fresh[fi].first!=d.order || fresh[fi].second.buffer!=d.buffer){same=false;break;}offsets.push_back(uint32_t(fresh[fi].second.offset));++fi;}
                same=same && fi==fresh.size();
                // 每个缓存项均核注册版本（包括 view）和存活图像；仅重建失效项。
                // Every cached entry checks registration revision (including view) and live image identity.
                if(same)for(size_t k=0;k<m.images.size();k++){const auto &cached=m.images[k];const tex_t *t=cached.first;
                    if(cached.record->revision!=cached.revision){same=false;break;}
                    if(t->serial!=m.tex_ids[k].first || t->mem!=m.tex_ids[k].second){same=false;break;}
                    auto li=images.find(t->serial);if(li==images.end() || li->second.get()!=m.images[k].second){same=false;break;}}
                // Resident texel copies: same registration and range, the same copy the set holds (generation), content
                // refreshed if the guest changed it since this slot last recorded.
                if(same)for(auto &tr:m.texel){const tex_t &tt=tr.record->texture;
                    if(tr.record->revision!=tr.revision || uint64_t(uintptr_t(tt.mem))!=tr.base || uint64_t(tt.L.size)!=tr.bytes){same=false;break;}
                    TexelResident *r=texel_acquire(tr.base,tr.bytes,VkFormat(tr.format));if(!r || r->gen!=tr.gen){same=false;break;}++texel_hits;}
                // Read-only storage ranges: refresh the snapshot (write-tracked resident copies only copy dirty pages); the set
                // stays valid only if the range lands in the same buffer at the same offset as when the set was written.
                if(same && m.persistent)for(auto &st:m.stor){snap_site=3;Buffer b=snapshot(st.base,st.bytes,properties.limits.minStorageBufferOffsetAlignment);snap_site=0;
                    if(b.buffer!=st.buffer || b.offset!=st.offset){same=false;break;}}
                if(same){
                // 命中也走既有按 submit 节流的 CPU 内容复核，不强制重传。
                // Hits retain get_image's bounded per-submit CPU checks, without forcing uploads.
                for(auto &ti:m.images){Image &im=get_image(ti.first,true);prepare_read(im,commands.command);}
                {size_t elements=0;for(unsigned st=0;st<5;st++)if(state.shader[st])for(auto &b:state.shader[st]->bindings)elements+=b.count;out.buffers.reserve(elements*2+40);} // draw() keeps vertex/index buffers here
                out.entry=m.entry;out.set=m.set;out.pipeline_layout=m.pipeline_layout;out.dynamic_offsets=offsets;++desc_memo_hits;if(m.persistent)++xmemo_hits;desc_ns[5]+=perf_clock()-tm0;return;}
                ++desc_memo_why[3];
                if(m.persistent){auto pin=memo_pinned[slot_index].find(m.set);if(pin!=memo_pinned[slot_index].end() && --pin->second.refs==0){slots[slot_index].used_sets[pin->second.layout].push_back(m.set);memo_pinned[slot_index].erase(pin);}}
                desc_memo.erase(hit);}}
    }
    desc_ns[0]+=perf_clock()-tm0;
    desc_memo_images.clear();desc_memo_dyn.clear();desc_memo_texel.clear();desc_memo_texel_fail=false;desc_memo_recording=ok;
    {uint64_t tb=perf_clock();descriptors_build(out,commands,compute);desc_ns[4]+=perf_clock()-tb;}
    desc_memo_recording=false;
    if(ok && desc_memo_texel_fail){ok=false;++desc_memo_why[0];}
    if(ok && !(out.written.empty() && out.buffer_views.empty() && out.set && out.dynamic_offsets.size()==desc_memo_dyn.size()))++desc_memo_why[4];
    if(ok && out.written.empty() && out.buffer_views.empty() && out.set && out.dynamic_offsets.size()==desc_memo_dyn.size()){++desc_memo_misses;
        std::sort(desc_memo_dyn.begin(),desc_memo_dyn.end(),[](const DescMemo::Dyn &a,const DescMemo::Dyn &b){return a.order<b.order;});
        DescMemo dm{out.entry,out.set,out.pipeline_layout,out.dynamic_offsets,desc_memo_images,desc_memo_dyn};dm.texel=desc_memo_texel;
        dm.persistent=xmemo_enabled && memo_dyn_ok && out.written.empty();
        for(auto &st:out.storage)dm.stor.push_back({st.first.first,st.first.second,st.second->buffer,st.second->offset});
        for(auto &ti:dm.images)dm.tex_ids.push_back({ti.first->serial,ti.first->mem});
        for(auto &d:dm.dyn)if(!d.uniform && d.src!=XBG_SOURCE_SUPPORT && d.src!=XBG_SOURCE_DRIVER0 && d.src!=XBG_SOURCE_DRIVER2 && d.src!=XBG_SOURCE_C1)dm.persistent=false;
        if(dm.persistent){unsigned si=slot_index;VkDescriptorSetLayout lay=out.entry->layout;
            auto pin=memo_pinned[si].find(out.set);if(pin!=memo_pinned[si].end())++pin->second.refs;
            else{auto &u=slots[si].used_sets[lay];for(size_t i=0;i<u.size();i++)if(u[i]==out.set){u[i]=u.back();u.pop_back();break;}memo_pinned[si].emplace(out.set,PinRef{lay,1});}}
        desc_memo.emplace(key,std::move(dm));}
}
void descriptors_build(Descriptors &out,Commands &commands,bool compute){CPU_SCOPE(Descriptor);DescriptorRoot descriptor_measurement(&cpu_scope);
    // Uniform buffers use dynamic offsets (when the set fits the device limit) so per-draw constants do not change set contents.
    const unsigned dmode=dyn_mode_for(compute);
    // Aliased NVN storage ranges must refer to the same VkBuffer allocation.
    // Snapshotting each binding separately loses writes observed through aliases.
    {DESC_SCOPE(Setup);
    struct Range{uint64_t begin,end;bool written;};static std::vector<Range> ranges,merged;static std::vector<uint32_t> key;ranges.clear();merged.clear();key.clear();
    {size_t elements=0;for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute)for(auto &b:state.shader[st]->bindings)elements+=b.count;out.buffers.reserve(elements*2+40);}
    std::array<const void*,7> shader_key{};for(unsigned st=0;st<6;st++)if((st==5)==compute)shader_key[st]=state.shader[st];shader_key[6]=reinterpret_cast<const void*>(uintptr_t(compute));
    for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute)for(auto &b:state.shader[st]->bindings){
        need(b.count && b.count<=128 && b.type<=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,"Unsupported descriptor array/type");
        key.push_back(b.binding);key.push_back(UBO_DYN(dmode,b)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:b.type);key.push_back(b.count);key.push_back(1u<<st);
        if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && (b.source==XBG_SOURCE_STORAGE || b.source==XBG_SOURCE_UNIFORM)){bool written=state.shader[st]->storage_writes.count(b.binding)!=0;for(unsigned i=0;i<b.count;i++){
            need(b.slot+i<16,"Storage descriptor slot out of bounds");auto r=b.source==XBG_SOURCE_STORAGE?state.sb[nvn_stage[st]][b.slot+i]:state.ub[nvn_stage[st]][b.slot+i];
            need(r.address && r.size && r.size<=properties.limits.maxStorageBufferRange && r.address+r.size>=r.address,"Invalid storage descriptor address/range");ranges.push_back({r.address,r.address+r.size,written});
        }}
    }
    std::sort(ranges.begin(),ranges.end(),[](const Range &a,const Range &b){return a.begin<b.begin;});
    for(auto &r:ranges){if(!merged.empty() && r.begin<merged.back().end){merged.back().end=std::max(merged.back().end,r.end);merged.back().written=merged.back().written||r.written;}else merged.push_back(r);}
    for(auto &r:merged){uint64_t base=r.begin,bytes=r.end-base;
        if(r.written){Buffer view,*b;
            if(pending_view(base,bytes,1,view)){b=out.keep(readback_buffer(bytes));VkCommandBuffer cmd=recording_slot().cmd;end_open_pass();VkBufferCopy region{view.offset,b->offset,bytes};vkCmdCopyBuffer(cmd,view.buffer,b->buffer,1,&region);full_barrier(cmd);}
            else{cpu_access(base,bytes,3);b=out.keep(readback_buffer(bytes,reinterpret_cast<void*>(base),bytes));}out.storage.push_back({{base,bytes},b});out.written.push_back({{base,bytes},b});}
        else{snap_site=3;out.storage.push_back({{base,bytes},out.keep(snapshot(base,bytes,properties.limits.minStorageBufferOffsetAlignment))});snap_site=0;}
    }
    LayoutEntry *entry=nullptr;{auto quick=shader_layouts.find(shader_key);if(quick!=shader_layouts.end())entry=quick->second;}
    if(!entry){auto found=layout_cache.find(key);
    if(found==layout_cache.end()){
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for(size_t i=0;i<key.size();i+=4){VkDescriptorSetLayoutBinding l{};l.binding=key[i];l.descriptorType=VkDescriptorType(key[i+1]);l.descriptorCount=key[i+2];l.stageFlags=key[i+3];bindings.push_back(l);}
        LayoutEntry e;VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};li.bindingCount=bindings.size();li.pBindings=bindings.data();VK(vkCreateDescriptorSetLayout(cfg.device,&li,nullptr,&e.layout));
        VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pi.setLayoutCount=1;pi.pSetLayouts=&e.layout;VK(vkCreatePipelineLayout(cfg.device,&pi,nullptr,&e.pipeline_layout));
        {XbgPipelineKey k;e.definition.assign(k,xbg_pipeline_layout_key(k,li,pi));}
        found=layout_cache.emplace(key,e).first;
    }
    entry=&found->second;shader_layouts.emplace(shader_key,entry);}
    out.entry=entry;out.pipeline_layout=entry->pipeline_layout;
    }
    static std::vector<VkDescriptorBufferInfo> buffer_infos;static std::vector<VkDescriptorImageInfo> image_infos;static std::vector<VkBufferView> texel_views;static std::vector<VkWriteDescriptorSet> writes;static std::vector<std::pair<uint64_t,uint32_t>> dynamic_list;static std::vector<uint64_t> set_key;
    {size_t elements=0;for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute)for(auto &b:state.shader[st]->bindings)elements+=b.count;
     buffer_infos.clear();image_infos.clear();texel_views.clear();writes.clear();dynamic_list.clear();buffer_infos.reserve(elements);image_infos.reserve(elements);texel_views.reserve(elements);writes.reserve(elements);}
    for(unsigned st=0;st<6;st++)if(state.shader[st] && (st==5)==compute){Shader &shader=*state.shader[st];unsigned ns=nvn_stage[st];
        for(auto &b:shader.bindings)for(unsigned element=0;element<b.count;element++){DESC_SCOPE(ElementRest);
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.dstSet=VK_NULL_HANDLE;write.dstBinding=b.binding;write.dstArrayElement=element;write.descriptorCount=1;write.descriptorType=UBO_DYN(dmode,b)?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:VkDescriptorType(b.type);
            if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER){DESC_SCOPE(BufferBind);
                Buffer *buffer=nullptr;bool is_storage=b.type==VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;uint32_t slot=b.slot+element;size_t n=0,buffer_offset=0;const void *data=nullptr;size_t copy=0;
                if(b.source==XBG_SOURCE_UNIFORM || b.source==XBG_SOURCE_STORAGE){need(slot<16,"Buffer slot out of bounds");auto r=b.source==XBG_SOURCE_UNIFORM?state.ub[ns][slot]:state.sb[ns][slot];need(r.address && r.size,"Required NVN buffer slot is unbound");data=reinterpret_cast<void*>(r.address);copy=r.size;
                    if(is_storage){n=copy;for(auto &v:out.storage)if(r.address>=v.first.first && r.address+r.size<=v.first.first+v.first.second){buffer=v.second;buffer_offset=r.address-v.first.first;break;}need(buffer,"Storage alias allocation not found");need(buffer_offset%properties.limits.minStorageBufferOffsetAlignment==0,"NVN storage offset violates Vulkan alignment");}
                    else{copy=std::min<size_t>(copy,b.bytes?b.bytes:65536);n=uniform_range(copy);buffer=out.keep(uniform_snapshot(r.address,copy));}
                }else if(b.source==XBG_SOURCE_SUPPORT){n=1392;need(state.color.alpha_func==8,"Nontrivial NVN alpha test not translated");
                    Buffer &sb=support_block();data=sb.mapped;copy=n;buffer=out.keep(sb);}
                else if(b.source==XBG_SOURCE_C1){
                    if(!shader.c1_ready){size_t bytes=uniform_range(shader.c1.size());shader.c1_buffer=static_arena.alloc(bytes);memset(shader.c1_buffer.mapped,0,bytes);if(!shader.c1.empty())memcpy(shader.c1_buffer.mapped,shader.c1.data(),shader.c1.size());shader.c1_buffer.flush();shader.c1_ready=true;}
                    data=shader.c1.data();copy=shader.c1.size();n=shader.c1_buffer.size;buffer=&shader.c1_buffer;}
                else if(b.source==XBG_SOURCE_DRIVER0 || b.source==XBG_SOURCE_DRIVER2){n=0x1000;unsigned bank=b.source==XBG_SOURCE_DRIVER0?0:1;uint64_t gen=bank?tex_gen[ns]:buf_gen[ns]+(ns==5?0:0);
                    DriverCache &dc=driver_cache[ns][bank];if(dc.slot!=slot_serial || dc.gen!=gen || !dc.slot){Buffer s=stream_buffer(n);fill_driver(static_cast<uint8_t*>(s.mapped),ns,bank?2:0);s.flush();dc.gen=gen;dc.slot=slot_serial;dc.buffer=s;}
                    data=dc.buffer.mapped;copy=n;buffer=out.keep(dc.buffer);}
                else xbg_fatalf("Unsupported shader buffer source %u",b.source);
                need(n && n<=size_t(is_storage?properties.limits.maxStorageBufferRange:properties.limits.maxUniformBufferRange),"Descriptor range exceeds device limit");
                if(write.descriptorType==VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC){buffer_infos.push_back({buffer->buffer,0,n});if(desc_memo_recording)desc_memo_dyn.push_back({(uint64_t(b.binding)<<32)|element,buffer->buffer,uint32_t(buffer->offset+buffer_offset),b.source==XBG_SOURCE_UNIFORM,uint8_t(b.source),uint8_t(ns),uint8_t(b.source==XBG_SOURCE_DRIVER0?0:1)});dynamic_list.push_back({(uint64_t(b.binding)<<32)|element,uint32_t(buffer->offset+buffer_offset)});}
                else buffer_infos.push_back({buffer->buffer,buffer->offset+buffer_offset,n});write.pBufferInfo=&buffer_infos.back();
                if(capture.active){char label[100];snprintf(label,sizeof(label),"st%u-bind%u-el%u-buffer",st,b.binding,element);capture_log("BUFFER label=%s shader=%s nvnStage=%u sourceKind=%u slot=%u guest=%p copied=%zu descriptorBytes=%zu bufferOffset=%zu",label,shader.key.c_str(),ns,b.source,slot,data,copy,n,buffer_offset);capture_data(label,static_cast<const uint8_t*>(buffer->mapped)+buffer_offset,n);}
            }else{
                DESC_SCOPE(ImageLookup);
                bool img_fast=false,img_cacheable=false;ImgKey ikey{};
                if(img_cache_enabled && !capture.active && !provenance_file && b.source==XBG_SOURCE_TEXTURE && (b.type==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b.type==VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) && b.slot+element<128 && state.texture_pool.pool
                   && (b.sampler_slot==UINT32_MAX || b.sampler_slot<128)){
                    uint64_t handle=state.tex[ns][b.slot+element];int id=int(handle&0xfffff);
                    if(id!=0 && !(state.texture_pool.n>0 && id>=state.texture_pool.n)){
                        uint64_t sh=b.type==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER?(b.sampler_slot!=UINT32_MAX?state.samp[ns][b.sampler_slot]:handle):0;
                        ikey={handle,sh,uint64_t(uintptr_t(state.texture_pool.pool->mem)),uint64_t(state.texture_pool.off),state.sampler_pool.pool?uint64_t(uintptr_t(state.sampler_pool.pool->mem)):0,uint64_t(state.sampler_pool.off),uint64_t(b.type)|(uint64_t((b.flags>>16)&3)<<8)};img_cacheable=true;
                        auto it=img_cache.find(ikey);
                        auto live=it==img_cache.end()?images.end():images.find(it->second.tex->serial);
                        if(it!=img_cache.end() && it->second.record->revision==it->second.revision &&
                            live!=images.end() && live->second.get()==it->second.im){Image *im=&get_image(it->second.tex,true);
                            prepare_read(*im,commands.command);if(desc_memo_recording)desc_memo_images.push_back({it->second.tex,im,it->second.record,it->second.revision});
                            VkDescriptorImageInfo ii{};ii.imageView=it->second.view;ii.sampler=b.type==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER?it->second.sampler:VK_NULL_HANDLE;ii.imageLayout=VK_IMAGE_LAYOUT_GENERAL;
                            image_infos.push_back(ii);write.pImageInfo=&image_infos.back();img_fast=true;++img_cache_hits;}
                        else if(it!=img_cache.end())img_cache.erase(it);
                    }
                }
                if(!img_fast){
                need(b.source==XBG_SOURCE_TEXTURE || b.source==XBG_SOURCE_IMAGE,"Unsupported image source");unsigned slot=b.slot+element;need(slot<(b.source==XBG_SOURCE_IMAGE?16u:128u),"Image binding slot out of range");uint64_t handle=b.source==XBG_SOURCE_IMAGE?state.img[ns][slot]:state.tex[ns][slot];int id=int(handle&0xfffff);
                const RegisteredTexture *record=nullptr;
                if(id==0 || (state.texture_pool.n>0 && id>=state.texture_pool.n))record=&null_texture(b);
                else{PoolKey pk=pool_key(&state.texture_pool);if(!texture_pool_cache || !(pk==texture_pool_cache_key)){auto pool=textures.find(pk);texture_pool_cache=pool==textures.end()?&legacy_textures:&pool->second;texture_pool_cache_key=pk;}auto tex=texture_pool_cache->find(id);if(tex==texture_pool_cache->end())xbg_fatalf("Unregistered texture descriptor id=%d stage=%u slot=%u",id,ns,slot);record=&tex->second;}
                const auto &registered=*record;const tex_t &texture=registered.texture;
                if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || b.type==VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER){need(texture.d.target==10 && texture.mem && texture.L.size,"Invalid buffer texture");
                    TexelResident *res=nullptr;
                    if(b.type==VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER && texel_memo_enabled && !compute){Buffer pv;
                        if(!pending_view(uint64_t(texture.mem),texture.L.size,1,pv))res=texel_acquire(uint64_t(texture.mem),texture.L.size,format(texture.d.format));
                        if(res){if(desc_memo_recording)desc_memo_texel.push_back({record,record->revision,uint64_t(texture.mem),uint64_t(texture.L.size),res->gen,uint32_t(format(texture.d.format))});
                            texel_views.push_back(res->view);write.pTexelBufferView=&texel_views.back();}
                        else desc_memo_texel_fail=true;}   // the set then holds a per-frame view: not memoized
                    if(!res){
                    Buffer *buf=nullptr;uint64_t base=uint64_t(texture.mem),bytes=texture.L.size;
                    if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER){Buffer view;if(pending_view(base,bytes,1,view)){buf=out.keep(readback_buffer(bytes));VkCommandBuffer cmd=recording_slot().cmd;end_open_pass();VkBufferCopy region{view.offset,buf->offset,bytes};vkCmdCopyBuffer(cmd,view.buffer,buf->buffer,1,&region);full_barrier(cmd);}else{cpu_access(base,bytes,8);buf=out.keep(readback_buffer(bytes,texture.mem,bytes));}out.storage.push_back({{base,bytes},buf});out.written.push_back({{base,bytes},buf});}
                    else{snap_site=4;buf=out.keep(snapshot(base,bytes,properties.limits.minTexelBufferOffsetAlignment));snap_site=0;}
                    VkBufferViewCreateInfo v{VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};v.buffer=buf->buffer;v.offset=buf->offset;v.format=format(texture.d.format);v.range=bytes;VkBufferView bv;{auto &cache=recording_slot().texel_views;std::array<uint64_t,4> key{uint64_t(v.buffer),v.offset,v.range,uint64_t(v.format)};auto f=cache.find(key);if(f!=cache.end())bv=f->second;else{VK(vkCreateBufferView(cfg.device,&v,nullptr,&bv));cache.emplace(key,bv);}}texel_views.push_back(bv);write.pTexelBufferView=&texel_views.back();}}
                else{if(aspect(texture.d.format)!=VK_IMAGE_ASPECT_COLOR_BIT){auto prior=images.find(texture.serial);if(prior==images.end() || (!prior->second->uploaded && !prior->second->gpu_written))xbg_logf("NVN initial depth descriptor shader=%s stage=%u slot=%u id=%d serial=%llu format=%#x",shader.key.c_str(),ns,slot,id,(unsigned long long)texture.serial,texture.d.format);}
                    VkDescriptorImageInfo ii{};
                    Image &im=get_image(&texture,true);provenance_log("BIND shader=%s stage=%u slot=%u id=%d handle=%llu descriptorMem=%p descriptorOff=%lld type=%u",shader.key.c_str(),ns,slot,id,(unsigned long long)handle,state.texture_pool.pool?state.texture_pool.pool->mem:nullptr,(long long)state.texture_pool.off,unsigned(b.type));provenance_image("BIND",&texture,im,registered.has_view?&registered.view:nullptr);DESC_CALL(ImageTransition, prepare_read(im,commands.command));if(desc_memo_recording)desc_memo_images.push_back({&texture,&im,record,record->revision});ii.imageView=DESC_CALL(ImageView, im.view(registered.has_view?&registered.view:nullptr,false,b.type==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));ii.imageLayout=VK_IMAGE_LAYOUT_GENERAL;
                    if(capture.active){char label[100];snprintf(label,sizeof(label),"st%u-bind%u-el%u-input",st,b.binding,element);out.capture_inputs.push_back({&im,registered.view,registered.has_view,label});capture_log("TEXTURE label=%s shader=%s nvnStage=%u slot=%u handle=%#llx id=%d serial=%llu cachedSerial=%llu image=%p view=%p viewFormat=%u sourceKind=%u",label,shader.key.c_str(),ns,slot,(unsigned long long)handle,id,(unsigned long long)texture.serial,(unsigned long long)im.source.serial,(void*)im.image,(void*)ii.imageView,unsigned(im.view_format(registered.has_view?&registered.view:nullptr)),b.source);}
                    if(b.type==VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b.type==VK_DESCRIPTOR_TYPE_SAMPLER){DESC_SCOPE(SamplerBind);uint64_t sh=handle;if(b.sampler_slot!=UINT32_MAX){need(b.sampler_slot<128,"Sampler slot out of range");sh=state.samp[ns][b.sampler_slot];}int sid=int(sh>>20);samp_par parameters{};if(sid==0){parameters.ws=parameters.wt=parameters.wr=7;parameters.aniso=1;parameters.cmp_func=1;}
                        else{auto sp=samplers.find(pool_key(&state.sampler_pool));auto &table=sp==samplers.end()?legacy_samplers:sp->second;auto sa=table.find(sid);if(sa==table.end())xbg_fatalf("Unregistered sampler id=%d",sid);parameters=sa->second;}ii.sampler=sampler(parameters,im,im.view_format(registered.has_view?&registered.view:nullptr),((b.flags>>16)&3)!=0);if(capture.active)capture_log("SAMPLER st=%u binding=%u sid=%d samplerSlot=%u min=%u mag=%u wrap=%u/%u/%u lod=%g/%g bias=%g",st,b.binding,sid,b.sampler_slot,parameters.minf,parameters.magf,parameters.ws,parameters.wt,parameters.wr,parameters.lod_min,parameters.lod_max,parameters.lod_bias);}
                    if(b.type==VK_DESCRIPTOR_TYPE_STORAGE_IMAGE){VkFormat viewed=im.view_format(registered.has_view?&registered.view:nullptr);if(!(im.usage&VK_IMAGE_USAGE_STORAGE_BIT))xbg_fatalf("NVN storage image without IMAGE flag serial=%llu format=%#x flags=%#x",(unsigned long long)texture.serial,texture.d.format,texture.d.flags);const VkFormatProperties &fp=format_properties(viewed);need(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,"Format not usable as storage image");
                        if((b.flags&3) && format_feature_flags2){VkFormatProperties3 extra{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};VkFormatProperties2 all{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2};all.pNext=&extra;vkGetPhysicalDeviceFormatProperties2(cfg.physical,viewed,&all);if(b.flags&1)need(extra.optimalTilingFeatures&VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT,"Image format lacks formatless-write support");if(b.flags&2)need(extra.optimalTilingFeatures&VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT,"Image format lacks formatless-read support");}
                        else if(b.flags&3){VkFormatProperties fp{};vkGetPhysicalDeviceFormatProperties(cfg.physical,viewed,&fp);bool listed=storage_without_format_listed(viewed) && (fp.optimalTilingFeatures&VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
                            if((b.flags&1) && !(listed && cfg.enabled_core.shaderStorageImageWriteWithoutFormat))xbg_fatalf("Image format %d lacks formatless-write support without VK_KHR_format_feature_flags2 (shader=%s listedWithStorageImage=%d shaderStorageImageWriteWithoutFormat=%u)",int(viewed),shader.key.c_str(),int(listed),unsigned(cfg.enabled_core.shaderStorageImageWriteWithoutFormat));
                            if((b.flags&2) && !(listed && cfg.enabled_core.shaderStorageImageReadWithoutFormat))xbg_fatalf("Image format %d lacks formatless-read support without VK_KHR_format_feature_flags2 (shader=%s listedWithStorageImage=%d shaderStorageImageReadWithoutFormat=%u)",int(viewed),shader.key.c_str(),int(listed),unsigned(cfg.enabled_core.shaderStorageImageReadWithoutFormat));}im.gpu_written=true,provenance_write(im,__LINE__);im.last_write=++write_seq;}
                    image_infos.push_back(ii);write.pImageInfo=&image_infos.back();
                    if(img_cacheable){img_cache.emplace(ikey,ImgEntry{&texture,&im,ii.imageView,ii.sampler,record,record->revision});++img_cache_misses;}
                }
                }
            }
            writes.push_back(write);
        }
    }
    // Dynamic offsets are consumed in binding/array-element order.
    std::sort(dynamic_list.begin(),dynamic_list.end());out.dynamic_offsets.clear();for(auto &d:dynamic_list)out.dynamic_offsets.push_back(d.second);
    if(!set_cache_enabled){out.set=allocate_set(out.entry->layout);if(writes.empty())return;
        // Update templates: one driver call with packed payloads instead of parsing VkWriteDescriptorSet per element.
        LayoutEntry &e=*const_cast<LayoutEntry*>(out.entry);static std::vector<uint64_t> sig;sig.clear();
        for(auto &w:writes)sig.push_back((uint64_t(w.dstBinding)<<40)|(uint64_t(w.dstArrayElement)<<8)|uint64_t(w.descriptorType));
        if(template_enabled && !e.tmpl){std::vector<VkDescriptorUpdateTemplateEntry> entries;for(size_t i=0;i<writes.size();i++)entries.push_back({writes[i].dstBinding,writes[i].dstArrayElement,1,writes[i].descriptorType,i*sizeof(DescEntry),sizeof(DescEntry)});
            VkDescriptorUpdateTemplateCreateInfo ti{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};ti.descriptorUpdateEntryCount=uint32_t(entries.size());ti.pDescriptorUpdateEntries=entries.data();ti.templateType=VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;ti.descriptorSetLayout=e.layout;
            if(vkCreateDescriptorUpdateTemplate(cfg.device,&ti,nullptr,&e.tmpl)==VK_SUCCESS)e.tmpl_sig=sig;else e.tmpl=VK_NULL_HANDLE;}
        if(template_enabled && e.tmpl && sig==e.tmpl_sig){static std::vector<DescEntry> data;data.resize(writes.size());
            for(size_t i=0;i<writes.size();i++){auto &w=writes[i];if(w.pBufferInfo)data[i].buffer=*w.pBufferInfo;else if(w.pImageInfo)data[i].image=*w.pImageInfo;else data[i].view=*w.pTexelBufferView;}
            vkUpdateDescriptorSetWithTemplate(cfg.device,out.set,e.tmpl,data.data());++template_updates;return;}
        for(auto &w:writes)w.dstSet=out.set;vkUpdateDescriptorSets(cfg.device,uint32_t(writes.size()),writes.data(),0,nullptr);++plain_updates;return;}
    uint64_t tk0=perf_clock();
    set_key.clear();set_key.push_back(uint64_t(out.entry->layout));
    for(auto &w:writes){set_key.push_back((uint64_t(w.dstBinding)<<32)|w.dstArrayElement);set_key.push_back(w.descriptorType);
        if(w.pBufferInfo){set_key.push_back(uint64_t(w.pBufferInfo->buffer));set_key.push_back(w.pBufferInfo->offset);set_key.push_back(w.pBufferInfo->range);}
        else if(w.pImageInfo){set_key.push_back(uint64_t(w.pImageInfo->sampler));set_key.push_back(uint64_t(w.pImageInfo->imageView));set_key.push_back(w.pImageInfo->imageLayout);}
        else if(w.pTexelBufferView)set_key.push_back(uint64_t(*w.pTexelBufferView));}
    Slot &slot=recording_slot();bool use_cache=true;auto hit=use_cache?slot.set_cache.find(set_key):slot.set_cache.end();desc_ns[3]+=perf_clock()-tk0;
    if(hit!=slot.set_cache.end()){out.set=hit->second;++set_cache_hits;return;}
    {uint64_t t0=perf_clock();out.set=allocate_set(out.entry->layout);desc_ns[2]+=perf_clock()-t0;}++set_cache_misses;
    if(!writes.empty()){LayoutEntry &e=*const_cast<LayoutEntry*>(out.entry);static std::vector<uint64_t> sig2;sig2.clear();
        for(auto &w:writes)sig2.push_back((uint64_t(w.dstBinding)<<40)|(uint64_t(w.dstArrayElement)<<8)|uint64_t(w.descriptorType));
        if(!e.tmpl){std::vector<VkDescriptorUpdateTemplateEntry> entries;for(size_t i=0;i<writes.size();i++)entries.push_back({writes[i].dstBinding,writes[i].dstArrayElement,1,writes[i].descriptorType,i*sizeof(DescEntry),sizeof(DescEntry)});
            VkDescriptorUpdateTemplateCreateInfo ti{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};ti.descriptorUpdateEntryCount=uint32_t(entries.size());ti.pDescriptorUpdateEntries=entries.data();ti.templateType=VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;ti.descriptorSetLayout=e.layout;
            if(vkCreateDescriptorUpdateTemplate(cfg.device,&ti,nullptr,&e.tmpl)==VK_SUCCESS)e.tmpl_sig=sig2;else e.tmpl=VK_NULL_HANDLE;}
        if(e.tmpl && sig2==e.tmpl_sig){static std::vector<DescEntry> data;data.resize(writes.size());
            for(size_t i=0;i<writes.size();i++){auto &w=writes[i];if(w.pBufferInfo)data[i].buffer=*w.pBufferInfo;else if(w.pImageInfo)data[i].image=*w.pImageInfo;else data[i].view=*w.pTexelBufferView;}
            {uint64_t t0=perf_clock();vkUpdateDescriptorSetWithTemplate(cfg.device,out.set,e.tmpl,data.data());desc_ns[1]+=perf_clock()-t0;}++template_updates;}
        else{for(auto &w:writes)w.dstSet=out.set;vkUpdateDescriptorSets(cfg.device,uint32_t(writes.size()),writes.data(),0,nullptr);++plain_updates;}}
    if(use_cache)slot.set_cache.emplace(set_key,out.set);
}
// Render passes are cached by their full definition; framebuffers are cached
// per frame slot (they reference image views of possibly released images).
struct Pass {
    const PipelineDefinition *definition=nullptr;
    VkRenderPass pass{};VkFramebuffer framebuffer{};uint32_t width{},height{},color_slots{};VkSampleCountFlagBits samples=VK_SAMPLE_COUNT_1_BIT;ImageList images;
};
// 保留零值计数用于与旧版本日志比较；任意 guest draw 不再丢弃附件内容。
// Keep zero counters for baseline logs; arbitrary guest draws retain attachment contents.
uint64_t noload_passes=0;
struct RenderPassEntry {VkRenderPass pass{};PipelineDefinition definition;};
std::map<std::vector<uint32_t>,RenderPassEntry> render_pass_cache;
void clear_render_pass_cache(){for(auto &r:render_pass_cache)vkDestroyRenderPass(cfg.device,r.second.pass,nullptr);render_pass_cache.clear();}
bool create_pass(Pass &p,Commands &c,int clear_color=-1,bool clear_depth=false,bool soft=false){CPU_SCOPE(Pass);
    // 仅显式设置DM_VK_TRACE_ATTACHMENTS=1时输出成功诊断；首次调用读取，错误始终记录。
    // Read once: opt in with DM_VK_TRACE_ATTACHMENTS=1; errors are always logged.
    static const bool trace_attachments=[](){const char *v=std::getenv("DM_VK_TRACE_ATTACHMENTS");return v && std::strcmp(v,"1")==0;}();
    need(state.rt.n>=0 && state.rt.n<=8,"Invalid color target count");
    bool clear_only=clear_color>=0 || clear_depth;
    std::array<const tex_t*,9> targets{};std::array<const view_t*,9> selected{};std::array<xbg_framebuffer_attachment,9> extents{};
    for(int i=0;i<state.rt.n;i++)if(state.rt.color[i] && (!clear_only || clear_color==i)){targets[i]=state.rt.color[i];selected[i]=state.rt.has_cview[i]?&state.rt.cview[i]:nullptr;}
    if(state.rt.depth && (!clear_only || clear_depth)){targets[8]=state.rt.depth;selected[8]=state.rt.has_dview?&state.rt.dview:nullptr;}
    uint64_t trace=clear_only?1:0;
    for(unsigned i=0;i<9;i++)if(targets[i]){
        const tex_t &t=*targets[i];const view_t *v=selected[i];
        need(t.magic==MAGIC('T','E','X',0) && t.serial,"Invalid NVN render attachment");
        extents[i]={1,t.d.width,t.d.height,t.d.levels,t.L.layers,v&&v->set_levels?uint32_t(v->base_level):0,v&&v->set_levels?uint32_t(v->nlevels):t.d.levels,
            v&&v->set_layers?uint32_t(v->min_layer):0,v&&v->set_layers?uint32_t(v->nlayers):t.L.layers,uint32_t(std::max(1,t.d.samples))};
        trace=(trace^t.serial^uint64_t(i)) * 1099511628211ull;trace^=hash_memory(&extents[i],sizeof(extents[i]));
        trace^=uint64_t(v&&v->set_format?v->format:t.d.format)<<32;
    }
    xbg_framebuffer_extent plan{};uint32_t bad=UINT32_MAX;int result=xbg_framebuffer_plan(extents.data(),extents.size(),properties.limits.maxFramebufferWidth,properties.limits.maxFramebufferHeight,&plan,&bad);
    if(result || (trace_attachments && trace!=last_attachment_trace)){
        xbg_logf("NVN framebuffer mode=%s result=%d viewport=%g,%g:%gx%g depthRange=%g/%g scissor=%d,%d:%ux%u selected=%ux%u samples=%u",clear_color>=0?"clear-color":clear_depth?"clear-depth-stencil":"draw",result,state.viewport.x,state.viewport.y,state.viewport.width,state.viewport.height,state.viewport.minDepth,state.viewport.maxDepth,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,plan.width,plan.height,plan.samples);
        for(unsigned i=0;i<9;i++)if(targets[i]){const auto &e=extents[i];uint32_t w=e.base_level<32?std::max(1u,e.width>>e.base_level):0,h=e.base_level<32?std::max(1u,e.height>>e.base_level):0;
            xbg_logf("NVN attachment index=%d serial=%llu format=%#x viewFormat=%#x target=%d mip=%u/%u baseLayer=%u layerCount=%u totalLayers=%u base=%ux%u selected=%ux%u samples=%u",i==8?-1:int(i),(unsigned long long)targets[i]->serial,targets[i]->d.format,selected[i]&&selected[i]->set_format?selected[i]->format:targets[i]->d.format,targets[i]->d.target,e.base_level,e.level_count,e.base_layer,e.layer_count,e.layers,e.width,e.height,w,h,e.samples);
        }
        last_attachment_trace=trace;
    }
    if(result && soft)return false;
    if(result)xbg_fatalf("NVN framebuffer %s result=%d attachment=%d",xbg_framebuffer_error(result),result,bad==UINT32_MAX?-1:int(bad));
    p.width=plan.width;p.height=plan.height;p.samples=VkSampleCountFlagBits(plan.samples);
    static std::vector<VkAttachmentDescription> descriptions;descriptions.clear();static std::vector<VkImageView> views;views.clear();   // not re-entrant: graphics thread
    static std::vector<VkAttachmentReference> colors;colors.assign(size_t(state.rt.n),{VK_ATTACHMENT_UNUSED,VK_IMAGE_LAYOUT_GENERAL});VkAttachmentReference depth{VK_ATTACHMENT_UNUSED,VK_IMAGE_LAYOUT_GENERAL};
    for(unsigned slot=0;slot<9;slot++)if(targets[slot]){
        Image &im=get_image(targets[slot],false);const view_t *view=selected[slot];
        need((slot==8)==(im.aspects!=VK_IMAGE_ASPECT_COLOR_BIT),"Color/depth attachment role mismatches texture aspects");
        if(c.draw)prepare_attachment(im,c.command);else im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);VkAttachmentDescription a{};a.format=im.view_format(view);a.samples=im.samples;a.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;a.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a.stencilStoreOp=VK_ATTACHMENT_STORE_OP_STORE;a.initialLayout=a.finalLayout=VK_IMAGE_LAYOUT_GENERAL;
        uint32_t index=descriptions.size();descriptions.push_back(a);views.push_back(im.view(view,true));p.images.push_back(&im);im.gpu_written=true,provenance_write(im,__LINE__);
        if(slot==8)depth.attachment=index;else colors[slot].attachment=index;
    }
    p.color_slots=colors.size();VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.colorAttachmentCount=colors.size();sub.pColorAttachments=colors.data();sub.pDepthStencilAttachment=depth.attachment!=VK_ATTACHMENT_UNUSED?&depth:nullptr;
    VkSubpassDependency dep[2]{};dep[0]={VK_SUBPASS_EXTERNAL,0,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,0};dep[1]={0,VK_SUBPASS_EXTERNAL,VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT,0};
    VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=descriptions.size();rp.pAttachments=descriptions.data();rp.subpassCount=1;rp.pSubpasses=&sub;rp.dependencyCount=2;rp.pDependencies=dep;
    {XbgPipelineKey key;bool valid=xbg_renderpass_key(key,rp);
        if(valid && key.valid){static std::vector<uint32_t> words;words.assign(key.words,key.words+key.count);auto it=render_pass_cache.find(words);
            if(it==render_pass_cache.end()){RenderPassEntry e;VK(vkCreateRenderPass(cfg.device,&rp,nullptr,&e.pass));e.definition.assign(key,true);it=render_pass_cache.emplace(words,std::move(e)).first;}
            p.pass=it->second.pass;p.definition=&it->second.definition;
        }
        else{VK(vkCreateRenderPass(cfg.device,&rp,nullptr,&p.pass));VkRenderPass rpass=p.pass;retire_handle([rpass]{vkDestroyRenderPass(cfg.device,rpass,nullptr);});static PipelineDefinition invalid;p.definition=&invalid;}
    }
    Slot &slot=recording_slot();static std::vector<uint64_t> fb_key;fb_key.assign({uint64_t(p.pass),p.width,p.height});for(auto v:views)fb_key.push_back(uint64_t(v));
    // Cached render passes live until shutdown, so their framebuffers can outlive the frame; they go with the first of their
    // images (forget_framebuffers). A pass outside the cache is retired with this slot, and so are its framebuffers.
    bool persistent=persistent_fb_enabled && p.definition && p.definition->valid;
    auto &cache=persistent?persistent_framebuffers:slot.framebuffers;
    auto cached=cache.find(fb_key);
    if(cached!=cache.end()){p.framebuffer=cached->second;++fb_hits;}
    else{VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fb.renderPass=p.pass;fb.attachmentCount=views.size();fb.pAttachments=views.data();fb.width=p.width;fb.height=p.height;fb.layers=1;VK(vkCreateFramebuffer(cfg.device,&fb,nullptr,&p.framebuffer));cache.emplace(fb_key,p.framebuffer);++fb_creates;
        if(persistent)for(auto *im:p.images)im->fb_keys.push_back(fb_key);}
    return true;
}
VkRect2D scissor(const Pass &p){VkRect2D r{};need(xbg_clear_rectangle(p.width,p.height,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,&r)==0,"Invalid scissor extent");return r;}
int pass_prof_reserve(VkCommandBuffer cmd){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd)return -1;
    // 开始时预留首尾查询，预算耗尽时整段跳过，不能留下只有开头的假计时。
    // Reserve both queries up front; skip the whole interval when the frame budget is exhausted.
    if(s.prof_used+2>pass_prof_query_limit || pass_prof_queries+2>pass_prof_query_limit){++s.prof_skipped;return -1;}
    Slot::PassRec r;r.q0=s.prof_used++;r.q1=s.prof_used++;r.draws0=r.draws1=uint32_t(pass_prof_draws);pass_prof_queries+=2;
    s.prof_recs.push_back(std::move(r));int index=int(s.prof_recs.size())-1;
    vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,s.prof_pool,s.prof_recs[index].q0);return index;
}
void pass_prof_complete(VkCommandBuffer cmd,int index){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd || index<0 || size_t(index)>=s.prof_recs.size())return;
    auto &r=s.prof_recs[index];r.draws1=uint32_t(pass_prof_draws);vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,s.prof_pool,r.q1);r.complete=true;
}
void pass_prof_end(VkCommandBuffer cmd){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd)return;pass_prof_complete(cmd,s.prof_open);s.prof_open=-1;}
void pass_prof_begin(const Pass &p,VkCommandBuffer cmd){Slot &s=slots[slot_index];s.prof_open=pass_prof_reserve(cmd);if(s.prof_open<0)return;
    auto &r=s.prof_recs[s.prof_open];r.w=p.width;r.h=p.height;r.att=uint32_t(p.images.size());r.fmt=p.images.empty()?0u:uint32_t(p.images[0]->fmt);
}
void pass_prof_draw(VkCommandBuffer cmd){Slot &s=slots[slot_index];if(!s.prof || cmd!=s.cmd || s.prof_open<0)return;
    std::string keys;for(unsigned st=0;st<5;++st)if(state.shader[st]){if(!keys.empty())keys+=';';keys+=std::to_string(st)+":"+state.shader[st]->key;}
    auto &r=s.prof_recs[s.prof_open];if(r.first.empty())r.first=keys;else if(r.last!=keys)r.mixed=true;r.last=std::move(keys);
}
int pass_prof_compute_begin(VkCommandBuffer cmd,const p_dispatch &p){int index=pass_prof_reserve(cmd);if(index<0)return index;
    auto &r=slots[slot_index].prof_recs[index];r.compute=true;r.first=state.shader[5]->key;r.last=r.first;r.x=uint32_t(p.x);r.y=uint32_t(p.y);r.z=uint32_t(p.z);return index;
}
void begin_pass(Pass &p,Commands &c){pass_prof_begin(p,c.command);VkRenderPassBeginInfo b{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};b.renderPass=p.pass;b.framebuffer=p.framebuffer;b.renderArea.extent={p.width,p.height};vkCmdBeginRenderPass(c.command,&b,VK_SUBPASS_CONTENTS_INLINE);}
void capture_pass(Pass &pass){
    if(!capture.active)return;
    capture_log("PASS width=%u height=%u samples=%u viewport=%g,%g,%g,%g scissor=%d,%d,%u,%u mask=%#x blendEnable=%#x cull=%u front=%u depthTest=%u depthWrite=%u depthFunc=%u stencil=%u",pass.width,pass.height,unsigned(pass.samples),state.viewport.x,state.viewport.y,state.viewport.width,state.viewport.height,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,state.mask.mask,state.color.blend_en,state.poly.cull,state.poly.front,unsigned(state.ds.dt),unsigned(state.ds.dw),unsigned(state.ds.df),unsigned(state.ds.st));
    for(int index=0;index<9;index++){
        const tex_t *t=index==8?state.rt.depth:index<state.rt.n?state.rt.color[index]:nullptr;if(!t)continue;
        auto found=images.find(t->serial);if(found==images.end())continue;Image &im=*found->second;
        if(std::find(pass.images.begin(),pass.images.end(),&im)==pass.images.end())continue;
        bool has_view=index==8?state.rt.has_dview:state.rt.has_cview[index];view_t v=has_view?(index==8?state.rt.dview:state.rt.cview[index]):view_t{};
        if(!v.set_levels)v.base_level=0;if(!v.set_layers)v.min_layer=0;v.set_levels=v.set_layers=1;v.nlevels=v.nlayers=1;
        char label[40];snprintf(label,sizeof(label),"rt%d-output",index==8?-1:index);capture_image(im,&v,label);
    }
}
VkPrimitiveTopology topology(int prim){switch(prim){case 0:return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;case 1:return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;case 3:return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;case 4:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;case 5:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;case 6:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;case 10:return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;case 11:return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;case 12:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;case 13:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;case 14:return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;default:xbg_fatalf("Unsupported draw primitive %d",prim);}}
VkStencilOpState stencil(bool back){VkStencilOpState s{};s.failOp=VkStencilOp((back?state.ds.fail_b:state.ds.fail_f)-1);s.passOp=VkStencilOp((back?state.ds.zpass_b:state.ds.zpass_f)-1);s.depthFailOp=VkStencilOp((back?state.ds.zfail_b:state.ds.zfail_f)-1);s.compareOp=VkCompareOp((back?state.ds.sf_b:state.ds.sf_f)-1);s.compareMask=state.stencil_compare[back];s.writeMask=state.stencil_write[back];s.reference=state.stencil_ref[back];return s;}
// Redundant-bind filter for the draw path (~1800 draws/frame re-bound identical pipeline / set / viewport / scissor /
// vertex buffers). Valid only within one command buffer recording; any other bind site (clear pipeline) and every
// command-buffer begin reset it. files/gfx-no-bindcache disables.
struct BindCache{VkCommandBuffer cmd=VK_NULL_HANDLE;VkPipeline pipe=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkDescriptorSet set=VK_NULL_HANDLE;
    uint32_t noff=UINT32_MAX,off[32]{};bool has_vp=false,has_sc=false;VkViewport vp{};VkRect2D sc{};VkBuffer vb[16]{};VkDeviceSize vbo[16]{};
    VkBuffer ib=VK_NULL_HANDLE;VkDeviceSize ibo=0;VkIndexType it=VK_INDEX_TYPE_MAX_ENUM;};
BindCache bind_cache;bool bind_cache_enabled=true;uint64_t binds_skipped=0;
void bind_cache_reset(){bind_cache=BindCache{};}
// 顶点数量与全 viewport/scissor 不能证明几何覆盖所有像素（局部三角形也满足）。
// Vertex count and full viewport/scissor do not prove full pixel coverage: a partial triangle qualifies.
// Guest draws therefore preserve LOAD; explicit clears keep their separate clear semantics.
uint64_t noload_why[8]={};
void draw(int prim,int first,int count,const p_draw_idx *indexed,uint32_t instances=1,uint32_t first_instance=0){CPU_SCOPE(Draw);DrawRoot draw_timing_root(&cpu_scope);provenance_draw();
    need(state.rt.n>=0 && state.rt.n<=8,"Invalid draw color target count");
    VkShaderModule modules[5]{};for(unsigned i=0;i<5;i++)if(state.shader[i])modules[i]=state.shader[i]->module;
    uint32_t active_colors=0;for(int i=0;i<state.rt.n;i++)if(state.rt.color[i])active_colors|=1u<<i;
    static std::vector<VkPipelineShaderStageCreateInfo> stages;stages.assign(5,VkPipelineShaderStageCreateInfo{});uint32_t stage_count=0;   // draw(): graphics thread only, not re-entrant
    int stage_result=xbg_graphics_stages(modules,state.rt.depth!=nullptr,active_colors,state.mask.mask,state.ms.a2c,stages.data(),&stage_count);
    if(stage_result)xbg_fatalf("Invalid graphics stage/depth-only state: %d",stage_result);stages.resize(stage_count);need(count>=0 && first>=0,"Negative draw range");if(!count || (indexed && indexed->instances==0))return;provenance_draw();
    std::vector<uint32_t> quad_indices;p_draw_idx quad_indexed{};
    if(prim==7 && !indexed){
        need(instances<=INT32_MAX && first_instance<=INT32_MAX,"Quad instance range exceeds backend limits");
        if(!instances)return;
        uint32_t expanded=0;need(xbg_quad_index_count(count,&expanded)==0,"Quad index expansion exceeds 64MiB bound");
        if(!expanded)return;
        quad_indices.resize(expanded);xbg_draw_call source{7,-1,first,count,int32_t(instances),int32_t(first_instance),0,0},triangles{};
        need(xbg_quad_plan(&source,&triangles,quad_indices.data(),quad_indices.size())==0,"Quad expansion failed");
        quad_indexed={triangles.primitive,triangles.index_type,triangles.count,triangles.base_vertex,triangles.base_instance,triangles.instances,triangles.index_address};
        prim=4;first=0;count=triangles.count;indexed=&quad_indexed;
    }
    if(capture_begin("draw",stats.presents))capture_log("DRAW number=%llu vs=%s ps=%s prim=%d first=%d count=%d indexed=%u instances=%u baseInstance=%u baseVertex=%d",(unsigned long long)stats.draws.load(),state.shader[0]->key.c_str(),state.shader[4]?state.shader[4]->key.c_str():"none",prim,first,count,indexed!=nullptr,indexed?indexed->instances:instances,indexed?indexed->base_instance:first_instance,indexed?indexed->base_vertex:0);
    Commands c(true);Descriptors &d=acquire_descriptors();descriptors(d,c,false);Pass p;create_pass(p,c);
    static std::vector<VkVertexInputAttributeDescription> attributes;attributes.clear();VkVertexInputBindingDescription binding_slots[16];uint32_t binding_used=0;
    int zero_binding=-1;
    for(size_t input=0;input<state.shader[0]->inputs.size();input++){
        uint32_t loc=state.shader[0]->inputs[input];need(loc<16,"Vertex input location out of bounds");va_st a;memcpy(&a,&state.attrib[loc],4);
        if(!a.fmt){
            if(zero_binding<0)zero_binding=xbg_vertex_zero_binding(state.attrib,state.shader[0]->inputs.data(),state.shader[0]->inputs.size());
            need(zero_binding>=0,"No free binding for constant vertex input");
            uint32_t cls=state.shader[0]->input_classes[input];need(cls<=2,"Unknown vertex scalar type");
            VkFormat f=cls==0?VK_FORMAT_R32G32B32A32_SFLOAT:cls==1?VK_FORMAT_R32G32B32A32_SINT:VK_FORMAT_R32G32B32A32_UINT;
            const VkFormatProperties &fp=format_properties(f);need(fp.bufferFeatures&VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,"Constant vertex format unsupported");
            attributes.push_back({loc,uint32_t(zero_binding),f,0});need(zero_binding<16,"Zero binding out of range");binding_slots[zero_binding]={uint32_t(zero_binding),0,VK_VERTEX_INPUT_RATE_VERTEX};binding_used|=1u<<zero_binding;
            if(capture.active)capture_log("ATTRIBUTE location=%u constant=zero class=%u binding=%d",loc,cls,zero_binding);
            continue;
        }
        need(a.stream<16,"Vertex stream out of bounds");VkFormat fmt=format(a.fmt);const VkFormatProperties &fp=format_properties(fmt);need(fp.bufferFeatures&VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT,"Vertex format unsupported");need(state.stream[a.stream].divisor<=1,"Instance divisor >1 unsupported");attributes.push_back({loc,a.stream,fmt,a.offset});binding_slots[a.stream]={a.stream,state.stream[a.stream].stride,state.stream[a.stream].divisor?VK_VERTEX_INPUT_RATE_INSTANCE:VK_VERTEX_INPUT_RATE_VERTEX};binding_used|=1u<<a.stream;if(capture.active)capture_log("ATTRIBUTE location=%u stream=%u format=%u vk=%u offset=%u stride=%u divisor=%u",loc,a.stream,a.fmt,unsigned(fmt),a.offset,state.stream[a.stream].stride,state.stream[a.stream].divisor);
    }
    static std::vector<VkVertexInputBindingDescription> bindings;bindings.clear();for(uint32_t i=0;i<16;i++)if(binding_used&(1u<<i))bindings.push_back(binding_slots[i]);   // ascending binding order, as the former std::map
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};vi.vertexBindingDescriptionCount=bindings.size();vi.pVertexBindingDescriptions=bindings.data();vi.vertexAttributeDescriptionCount=attributes.size();vi.pVertexAttributeDescriptions=attributes.data();
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=topology(prim);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.viewportCount=vp.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.polygonMode=state.poly.mode==2?VK_POLYGON_MODE_FILL:state.poly.mode==1?VK_POLYGON_MODE_LINE:VK_POLYGON_MODE_POINT;need(rs.polygonMode==VK_POLYGON_MODE_FILL || cfg.enabled_core.fillModeNonSolid,"Non-solid polygon mode disabled");rs.cullMode=state.poly.cull;rs.frontFace=state.poly.front?VK_FRONT_FACE_COUNTER_CLOCKWISE:VK_FRONT_FACE_CLOCKWISE;rs.lineWidth=1;rs.depthBiasEnable=state.poly.offset_en!=0;rs.depthBiasConstantFactor=state.bias[1]*(state.rt.depth && state.rt.depth->d.format==0x32 && depth16_as_d24?256.0f:1.0f);   // bias units: 2^-16 for D16, 2^-24 for the X8_D24 stand-in
    rs.depthBiasSlopeFactor=state.bias[0];rs.depthBiasClamp=state.bias[2];need(!rs.depthBiasClamp || cfg.enabled_core.depthBiasClamp,"Depth bias clamp disabled");
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=p.samples;ms.alphaToCoverageEnable=state.ms.a2c;need(!state.ms.sl_en,"Custom sample locations unsupported");need(!state.ms.samples || state.ms.samples==uint32_t(p.samples),"NVN sample count differs from attachments");need(p.samples==VK_SAMPLE_COUNT_1_BIT || state.ms.enable,"Disabled multisampling on multisample target needs coverage emulation");
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};ds.depthTestEnable=state.ds.dt;ds.depthWriteEnable=state.ds.dw;ds.depthCompareOp=VkCompareOp(state.ds.df-1);ds.stencilTestEnable=state.ds.st;ds.front=stencil(false);ds.back=stencil(true);ds.depthBoundsTestEnable=state.bounds.enable;ds.minDepthBounds=state.bounds.minimum;ds.maxDepthBounds=state.bounds.maximum;need(!ds.depthBoundsTestEnable || cfg.enabled_core.depthBounds,"Depth bounds feature disabled");
    need(!(ds.depthTestEnable || ds.stencilTestEnable) || state.rt.depth,"Depth/stencil state requires attachment");
    static std::vector<VkPipelineColorBlendAttachmentState> attachments;attachments.assign(size_t(state.rt.n),VkPipelineColorBlendAttachmentState{});for(int i=0;i<state.rt.n;i++){const blend_st &b=state.blend[i];xbg_draw2d convert{};need(xbg_2d_blend_from_nvn(&convert,(state.color.blend_en>>i)&1,b.src_rgb,b.dst_rgb,b.src_a,b.dst_a,b.eq_rgb,b.eq_a,(state.mask.mask>>(4*i))&15)==0,"Unsupported blend state");attachments[i]=convert.blend;}
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};blend.attachmentCount=attachments.size();blend.pAttachments=attachments.data();memcpy(blend.blendConstants,state.blend_color,16);need(state.color.logic_op==3,"Non-copy logic operation unsupported");
    VkDynamicState dynamic[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dy.dynamicStateCount=2;dy.pDynamicStates=dynamic;
    VkPipelineTessellationStateCreateInfo tess{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};tess.patchControlPoints=state.patch_size;need((state.shader[1]!=nullptr)==(state.shader[2]!=nullptr),"Incomplete tessellation shader pair");
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};ci.stageCount=stages.size();ci.pStages=stages.data();ci.pVertexInputState=&vi;ci.pInputAssemblyState=&ia;ci.pViewportState=&vp;ci.pRasterizationState=&rs;ci.pMultisampleState=&ms;ci.pDepthStencilState=&ds;ci.pColorBlendState=&blend;ci.pDynamicState=&dy;ci.pTessellationState=state.shader[1]?&tess:nullptr;ci.layout=d.pipeline_layout;ci.renderPass=p.pass;
    PipelineLease pipeline;{CPU_SCOPE(Pipeline);XbgPipelineKey key;bool cacheable=xbg_graphics_pipeline_key(key,ci,d.entry->definition.view(),p.definition->view());pipeline_lookup(pipeline,key,cacheable,&ci,nullptr);}
    // Buffer copies precede the draw; original guest layout remains unchanged.
    static std::vector<std::pair<uint32_t,Buffer*>> vertex_buffers;vertex_buffers.clear();
    Buffer *ib=nullptr;VkIndexType index_type=VK_INDEX_TYPE_UINT16;
    {DRAW_SCOPE(VertexPrepare);for(auto &b:bindings){if(int(b.binding)==zero_binding){const uint32_t zero[4]={0};vertex_buffers.push_back({b.binding,d.keep(stream_buffer(sizeof(zero),zero,sizeof(zero)))});continue;}auto r=state.vb[b.binding];need(r.address && r.size,"Unbound vertex buffer");snap_site=1;vertex_buffers.push_back({b.binding,d.keep(snapshot(r.address,r.size))});snap_site=0;}
    if(indexed){need(indexed->itype>=0 && indexed->itype<=2 && indexed->instances>0 && indexed->base_instance>=0,"Invalid indexed draw");size_t bytes=size_t(count)*(1u<<indexed->itype);need(indexed->ibuf,"Unbound index buffer");if(indexed->itype==0){if(draw_detail)draw_timing.index(uint64_t(count));std::vector<uint16_t> expanded(count);const uint8_t *src=reinterpret_cast<const uint8_t*>(indexed->ibuf);for(int i=0;i<count;i++)expanded[i]=src[i];ib=d.keep(stream_buffer(expanded.size()*2,expanded.data(),expanded.size()*2));}else{index_type=indexed->itype==1?VK_INDEX_TYPE_UINT16:VK_INDEX_TYPE_UINT32;snap_site=2;ib=indexed==&quad_indexed?d.keep(stream_buffer(bytes,reinterpret_cast<void*>(indexed->ibuf),bytes)):d.keep(snapshot(uint64_t(indexed->ibuf),bytes));snap_site=0;}}
    } // VertexPrepare excludes capture, recording and storage writeback.
    if(capture.active){for(auto &v:vertex_buffers){char label[40];snprintf(label,sizeof(label),"vertex-stream%u",v.first);capture_log("VERTEX label=%s guest=%#llx bytes=%zu",label,(unsigned long long)state.vb[v.first].address,v.second->size);capture_data(label,v.second->mapped,v.second->size);}if(ib){capture_log("INDEX guest=%#llx vkType=%u bytes=%zu",(unsigned long long)indexed->ibuf,unsigned(index_type),ib->size);capture_data("index",ib->mapped,ib->size);}for(auto &input:d.capture_inputs)for(auto *output:p.images)if(input.image==output)capture_log("FEEDBACK input=%s serial=%llu",input.label.c_str(),(unsigned long long)output->source.serial);}
    // 在分配前关闭旧 pass 的计数；pool 增长可能结束 pass，故分配仍先于 begin_pass。
    // Close the previous pass before allocation; growing a pool may end the pass, so allocate before begin_pass.
    if(open_pass.cmd!=c.command || open_pass.pass!=p.pass || open_pass.framebuffer!=p.framebuffer)end_open_pass();
    VkQueryPool occlusion{};uint32_t occlusion_index=0;if(state.count_samples && !open_occlusion.cmd){need(cfg.enabled_core.occlusionQueryPrecise,"Precise samples-passed query disabled");occlusion_query(occlusion,occlusion_index);}
    if(open_pass.cmd!=c.command || open_pass.pass!=p.pass || open_pass.framebuffer!=p.framebuffer){end_open_pass();begin_pass(p,c);open_pass.cmd=c.command;open_pass.pass=p.pass;open_pass.framebuffer=p.framebuffer;open_pass.images=p.images;++perf.passes;}
    if(occlusion)begin_occlusion_query(c.command,occlusion,occlusion_index);
    BindCache &bc=bind_cache;if(!bind_cache_enabled || bc.cmd!=c.command || pipeline.transient){bc=BindCache{};bc.cmd=c.command;}
    if(bc.pipe!=pipeline.pipeline){vkCmdBindPipeline(c.command,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline.pipeline);bc.pipe=pipeline.pipeline;}else ++binds_skipped;
    {const uint32_t no=uint32_t(d.dynamic_offsets.size());
     if(bc.layout!=d.pipeline_layout || bc.set!=d.set || bc.noff!=no || no>32 || (no && memcmp(bc.off,d.dynamic_offsets.data(),no*4))){
        vkCmdBindDescriptorSets(c.command,VK_PIPELINE_BIND_POINT_GRAPHICS,d.pipeline_layout,0,1,&d.set,no,d.dynamic_offsets.data());
        if(no<=32){bc.layout=d.pipeline_layout;bc.set=d.set;bc.noff=no;if(no)memcpy(bc.off,d.dynamic_offsets.data(),no*4);}else bc.noff=UINT32_MAX;}
     else ++binds_skipped;}
    VkViewport viewport{};need(xbg_nvn_viewport(&viewport,state.viewport.x,state.viewport.y,state.viewport.width,state.viewport.height,state.viewport.minDepth,state.viewport.maxDepth)==0,"Invalid NVN viewport");
    need(viewport.width<=properties.limits.maxViewportDimensions[0] && -viewport.height<=properties.limits.maxViewportDimensions[1] && viewport.x>=properties.limits.viewportBoundsRange[0] && viewport.x+viewport.width<=properties.limits.viewportBoundsRange[1] && viewport.y>=properties.limits.viewportBoundsRange[0] && viewport.y<=properties.limits.viewportBoundsRange[1] && viewport.y+viewport.height>=properties.limits.viewportBoundsRange[0],"NVN viewport exceeds device bounds");
    if(!bc.has_vp || memcmp(&bc.vp,&viewport,sizeof viewport)){vkCmdSetViewport(c.command,0,1,&viewport);bc.vp=viewport;bc.has_vp=true;}else ++binds_skipped;
    VkRect2D sc=scissor(p);if(!bc.has_sc || memcmp(&bc.sc,&sc,sizeof sc)){vkCmdSetScissor(c.command,0,1,&sc);bc.sc=sc;bc.has_sc=true;}else ++binds_skipped;
    for(auto &b:vertex_buffers){VkDeviceSize off=b.second->offset;if(b.first<16 && bc.vb[b.first]==b.second->buffer && bc.vbo[b.first]==off){++binds_skipped;continue;}
        vkCmdBindVertexBuffers(c.command,b.first,1,&b.second->buffer,&off);if(b.first<16){bc.vb[b.first]=b.second->buffer;bc.vbo[b.first]=off;}}
    pass_prof_draw(c.command);
    if(indexed){if(bc.ib!=ib->buffer || bc.ibo!=ib->offset || bc.it!=index_type){vkCmdBindIndexBuffer(c.command,ib->buffer,ib->offset,index_type);bc.ib=ib->buffer;bc.ibo=ib->offset;bc.it=index_type;}else ++binds_skipped;vkCmdDrawIndexed(c.command,count,indexed->instances,0,indexed->base_vertex,indexed->base_instance);}else vkCmdDraw(c.command,count,instances,first,first_instance);
    if(!occlusion_batch_enabled)end_occlusion_query();c.finish();++perf.draws;++pass_prof_draws;
    if(d.needs_writeback() || capture.active){end_open_pass();full_barrier(c.command);VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);}
    {DRAW_SCOPE(Writeback);if(capture.active){if(d.needs_writeback())gpu_sync(perf.sync_writeback);d.writeback();}else d.defer_writeback();}++stats.draws;
    if(capture.active){capture_pass(p);d.capture_read_images();capture.active=false;}
}
void dispatch(const p_dispatch &p){CPU_SCOPE(Dispatch);provenance_log("DISPATCH shader=%s",state.shader[5]?state.shader[5]->key.c_str():"null");need(state.shader[5],"Dispatch has no genuine compute shader");need(p.x>=0 && p.y>=0 && p.z>=0,"Negative compute dispatch");if(!p.x||!p.y||!p.z)return;need(uint32_t(p.x)<=properties.limits.maxComputeWorkGroupCount[0] && uint32_t(p.y)<=properties.limits.maxComputeWorkGroupCount[1] && uint32_t(p.z)<=properties.limits.maxComputeWorkGroupCount[2],"Compute dispatch exceeds device limit");Commands c;Descriptors &d=acquire_descriptors();descriptors(d,c,true);VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=state.shader[5]->module;ci.stage.pName="main";ci.layout=d.pipeline_layout;PipelineLease pipeline;{CPU_SCOPE(Pipeline);XbgPipelineKey key;bool cacheable=xbg_compute_pipeline_key(key,ci,d.entry->definition.view());pipeline_lookup(pipeline,key,cacheable,nullptr,&ci);}vkCmdBindPipeline(c.command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline.pipeline);vkCmdBindDescriptorSets(c.command,VK_PIPELINE_BIND_POINT_COMPUTE,d.pipeline_layout,0,1,&d.set,uint32_t(d.dynamic_offsets.size()),d.dynamic_offsets.data());int prof_dispatch=pass_prof_compute_begin(c.command,p);vkCmdDispatch(c.command,p.x,p.y,p.z);pass_prof_complete(c.command,prof_dispatch);VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&b,0,nullptr,0,nullptr);c.finish();++perf.dispatches;d.defer_writeback();++stats.dispatches;}

struct MaskedClear {
    VkShaderModule vs{},fs{};VkPipelineLayout layout{};VkPipeline pipeline{};
    ~MaskedClear(){if(pipeline)vkDestroyPipeline(cfg.device,pipeline,nullptr);if(layout)vkDestroyPipelineLayout(cfg.device,layout,nullptr);if(fs)vkDestroyShaderModule(cfg.device,fs,nullptr);if(vs)vkDestroyShaderModule(cfg.device,vs,nullptr);}
    void prepare(const Pass &p,int attachment,int scalar,uint32_t cm,uint32_t sm,uint32_t reference){
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};si.codeSize=sizeof(xbg_clear_vs);si.pCode=xbg_clear_vs;VK(vkCreateShaderModule(cfg.device,&si,nullptr,&vs));
        if(cm){const uint32_t *words=scalar==1?xbg_clear_fs_sint:scalar==2?xbg_clear_fs_uint:xbg_clear_fs_float;size_t bytes=scalar==1?sizeof(xbg_clear_fs_sint):scalar==2?sizeof(xbg_clear_fs_uint):sizeof(xbg_clear_fs_float);std::vector<uint32_t> patched(words,words+bytes/4);patched[xbg_clear_fs_location_word]=uint32_t(attachment);si.codeSize=bytes;si.pCode=patched.data();VK(vkCreateShaderModule(cfg.device,&si,nullptr,&fs));}
        VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT,0,16};VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.pushConstantRangeCount=cm?1:0;li.pPushConstantRanges=cm?&push:nullptr;VK(vkCreatePipelineLayout(cfg.device,&li,nullptr,&layout));
        VkPipelineShaderStageCreateInfo stages[2]{};for(int i=0;i<(cm?2:1);i++){stages[i].sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;stages[i].stage=i?VK_SHADER_STAGE_FRAGMENT_BIT:VK_SHADER_STAGE_VERTEX_BIT;stages[i].module=i?fs:vs;stages[i].pName="main";}
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.viewportCount=vp.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.polygonMode=VK_POLYGON_MODE_FILL;rs.cullMode=VK_CULL_MODE_NONE;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;rs.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=p.samples;
        VkPipelineColorBlendAttachmentState color{};VkPipelineDepthStencilStateCreateInfo ds{};need(xbg_masked_clear_state(cm,sm,reference,&color,&ds)==0,"Invalid masked clear state");
        // Every other subpass color slot is UNUSED. Identical states across all
        // slots avoid any dependency on the optional independentBlend feature.
        std::vector<VkPipelineColorBlendAttachmentState> colors(p.color_slots,color);VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};blend.attachmentCount=colors.size();blend.pAttachments=colors.data();
        VkDynamicState dyn[2]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dynamic.dynamicStateCount=2;dynamic.pDynamicStates=dyn;
        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};ci.stageCount=cm?2:1;ci.pStages=stages;ci.pVertexInputState=&vi;ci.pInputAssemblyState=&ia;ci.pViewportState=&vp;ci.pRasterizationState=&rs;ci.pMultisampleState=&ms;ci.pDepthStencilState=&ds;ci.pColorBlendState=&blend;ci.pDynamicState=&dynamic;ci.layout=layout;ci.renderPass=p.pass;
        VK(vkCreateGraphicsPipelines(cfg.device,pipeline_cache,1,&ci,nullptr,&pipeline));
    }
    void encode(Commands &c,const Pass &p,const VkRect2D &rect,const void *color){
        bind_cache_reset();VkViewport viewport{0,0,float(p.width),float(p.height),0,1};vkCmdBindPipeline(c.command,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline);vkCmdSetViewport(c.command,0,1,&viewport);vkCmdSetScissor(c.command,0,1,&rect);
        if(color)vkCmdPushConstants(c.command,layout,VK_SHADER_STAGE_FRAGMENT_BIT,0,16,color);vkCmdDraw(c.command,3,1,0,0);
    }
};
void clear_attachment(const p_clear_color *color,const p_clear_ds *depth,int scalar=0){
    // 后端合成的 masked-clear draw 不属于游戏遮挡计数。
    // Backend-generated masked-clear draws do not contribute to the game's occlusion count.
    end_occlusion_query();
    VkClearAttachment a{};uint32_t cm=0,sm=0;bool partial=false;p_clear_color converted{};
    if(color){
        need(color->index>=0 && color->index<state.rt.n && state.rt.color[color->index],"Clear color attachment missing");need((uint32_t(color->mask)&~15u)==0,"Invalid color clear mask");cm=uint32_t(color->mask);
        Image &target=get_image(state.rt.color[color->index],false);const view_t *view=state.rt.has_cview[color->index]?&state.rt.cview[color->index]:nullptr;
        VkFormat viewed=target.view_format(view);static unsigned notes=0;if(notes++<8){uint32_t bits[4];memcpy(bits,color->rgba,16);xbg_logf("NVN clear-value attachment input=%d nvn=%#x view=%#x vk=%u mask=%#x bits=%08x/%08x/%08x/%08x",scalar,target.source.d.format,view&&view->set_format?view->format:target.source.d.format,unsigned(viewed),cm,bits[0],bits[1],bits[2],bits[3]);}VkClearColorValue value{};int converted_scalar=xbg_clear_value(viewed,scalar,color->rgba,&value);
        if(converted_scalar<0)xbg_fatalf("Color clear conversion failed nvn=%#x vk=%u input=%d rc=%d",target.source.d.format,unsigned(viewed),scalar,converted_scalar);
        converted=*color;memcpy(converted.rgba,&value,16);color=&converted;scalar=converted_scalar;
        if(cm)a.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;a.colorAttachment=color->index;memcpy(&a.clearValue.color,color->rgba,16);partial=cm && cm!=15;
    }else{
        need(state.rt.depth,"Clear depth attachment missing");sm=uint32_t(depth->stencil_mask)&255;
        VkImageAspectFlags requested=(depth->depth_mask?VK_IMAGE_ASPECT_DEPTH_BIT:0)|(sm?VK_IMAGE_ASPECT_STENCIL_BIT:0);need((aspect(state.rt.depth->d.format)&requested)==requested,"Clear requests absent depth/stencil aspect");
        if(depth->depth_mask){need(depth->depth>=0 && depth->depth<=1,"Depth clear value outside normalized range");a.aspectMask|=VK_IMAGE_ASPECT_DEPTH_BIT;}
        partial=sm && sm!=255;if(sm && !partial)a.aspectMask|=VK_IMAGE_ASPECT_STENCIL_BIT;a.clearValue.depthStencil={depth->depth,uint32_t(depth->stencil)&255};
    }
    if(!a.aspectMask && !partial)return;
    // Full (unmasked) clears are recorded inside the draw render pass of the
    // current targets, so a clear and the draws after it share one pass.
    if(!partial && a.aspectMask && !capture.trace && !provenance_file){
        const tex_t *t0=color?state.rt.color[color->index]:state.rt.depth;Image &ti=get_image(t0,false);
        const view_t *tv=color?(state.rt.has_cview[color->index]?&state.rt.cview[color->index]:nullptr):(state.rt.has_dview?&state.rt.dview:nullptr);
        uint32_t lvl=tv && tv->set_levels?tv->base_level:0;uint32_t cw=std::max(1u,ti.width>>lvl),ch=std::max(1u,ti.height>>lvl);VkRect2D keep{};
        bool needs_preserve=lvl<ti.levels && xbg_clear_rectangle(cw,ch,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,&keep)==0 && xbg_clear_requires_preserve(ti.levels,ti.layers,cw,ch,&keep,false,ti.aspects,a.aspectMask)!=0;
        if(!needs_preserve || ti.gpu_written || ti.uploaded){
            Commands c(true);Pass p;
            if(create_pass(p,c,-1,false,true)){
                if(open_pass.cmd!=c.command || open_pass.pass!=p.pass || open_pass.framebuffer!=p.framebuffer){end_open_pass();begin_pass(p,c);open_pass.cmd=c.command;open_pass.pass=p.pass;open_pass.framebuffer=p.framebuffer;open_pass.images=p.images;++perf.passes;}
                VkClearRect r{scissor(p),0,1};if(r.rect.extent.width && r.rect.extent.height)vkCmdClearAttachments(c.command,1,&a,1,&r);
                ++perf.folded_clears;return;
            }
        }
    }
    if(capture_begin(color?"clear-color":"clear-depth-stencil",stats.presents))capture_log("CLEAR target=%d scalar=%d colorMask=%u stencilMask=%u partial=%u",color?color->index:-1,scalar,cm,sm,unsigned(partial));
    const tex_t *target=color?state.rt.color[color->index]:state.rt.depth;Image &target_image=get_image(target,false);
    const view_t *target_view=color?(state.rt.has_cview[color->index]?&state.rt.cview[color->index]:nullptr):(state.rt.has_dview?&state.rt.dview:nullptr);
    uint32_t clear_level=target_view && target_view->set_levels?target_view->base_level:0;need(clear_level<target_image.levels,"Clear mip outside image");
    uint32_t clear_width=std::max(1u,target_image.width>>clear_level),clear_height=std::max(1u,target_image.height>>clear_level);VkRect2D preserve_rect{};
    need(xbg_clear_rectangle(clear_width,clear_height,state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,&preserve_rect)==0,"Invalid clear scissor");
    bool preserve=xbg_clear_requires_preserve(target_image.levels,target_image.layers,clear_width,clear_height,&preserve_rect,partial,target_image.aspects,a.aspectMask)!=0;
    // LOAD preserves actual previous contents only after importing CPU-backed
    // storage on first use; never mark gpu_written before this preservation.
    if(preserve){need(target_image.samples==VK_SAMPLE_COUNT_1_BIT || target_image.gpu_written,"Partial clear of uninitialized CPU multisample storage unsupported");get_image(target,true);}
    Commands c;Pass p;create_pass(p,c,color?color->index:-1,depth!=nullptr);VkClearRect r{scissor(p),0,1};auto masked_object=std::make_shared<MaskedClear>();MaskedClear &masked=*masked_object;retire_object(masked_object);
    bool area=r.rect.extent.width && r.rect.extent.height;
    if(partial && area)masked.prepare(p,color?color->index:-1,scalar,cm,color?0:sm,color?0:uint32_t(depth->stencil));
    begin_pass(p,c);
    if(area){
        if(!color && a.aspectMask)vkCmdClearAttachments(c.command,1,&a,1,&r);
        if(partial)masked.encode(c,p,r.rect,color?color->rgba:nullptr);
        else if(color)vkCmdClearAttachments(c.command,1,&a,1,&r);
    }
    vkCmdEndRenderPass(c.command);pass_prof_end(c.command);c.finish(); // transient pipeline/modules stay alive through this fence
    if(capture.active){capture_pass(p);capture.active=false;}
}
void region_check(const Image &im,uint32_t level,uint32_t layer,const NVNcopyRegion &r){
    need(level<im.levels && layer<im.layers && r.xoffset>=0 && r.yoffset>=0 && r.zoffset>=0 && r.width>0 && r.height>0 && r.depth>0,"Invalid texture copy region");
    bool volume=im.source.d.target==2;uint64_t zlimit=volume?std::max(1u,im.depth>>level):im.layers;
    need(uint64_t(r.xoffset)+r.width<=std::max(1u,im.width>>level) && uint64_t(r.yoffset)+r.height<=std::max(1u,im.height>>level) && uint64_t(volume?0:layer)+r.zoffset+r.depth<=zlimit,"Texture copy region exceeds image");
}
VkImageSubresourceLayers copy_layers(const Image &im,uint32_t level,uint32_t layer,const NVNcopyRegion &r){return {im.aspects,level,im.source.d.target==2?0:layer+uint32_t(r.zoffset),im.source.d.target==2?1:uint32_t(r.depth)};}
VkOffset3D copy_offset(const Image &im,const NVNcopyRegion &r){return {r.xoffset,r.yoffset,im.source.d.target==2?r.zoffset:0};}
VkExtent3D copy_extent(const Image &im,const NVNcopyRegion &r){return {uint32_t(r.width),uint32_t(r.height),im.source.d.target==2?uint32_t(r.depth):1};}
// Buffer->texture copies write the texture's pool storage on Switch. Mirror that
// into guest memory (block-linear) so the texture stays CPU-authoritative: later
// CPU streaming writes into the same storage keep being detected and uploaded.
bool guest_texture_write(const tex_t *t,unsigned level,unsigned layer,const NVNcopyRegion &r,const uint8_t *src,size_t row,size_t image_stride){
    uint32_t row_bytes,rows,slices,bpb;uint64_t base;unsigned th,td;
    if(!t->mem || r.width<=0 || r.height<=0 || r.depth<=0 || r.xoffset<0 || r.yoffset<0 || r.zoffset<0)return false;
    if(nvn_deswizzle_geom(&t->d,level,layer,&row_bytes,&rows,&slices,&base,&th,&td,&bpb))return false;
    int bw=1,bh=1;if(!nvn_fmt_bpb(t->d.format,&bw,&bh) || r.xoffset%bw || r.yoffset%bh)return false;
    uint32_t bx0=r.xoffset/bw,by0=r.yoffset/bh,nbx=(r.width+bw-1)/bw,nby=(r.height+bh-1)/bh;
    if(uint64_t(bx0+nbx)*bpb>row_bytes || by0+nby>rows || row<size_t(nbx)*bpb)return false;
    bool volume=t->d.target==2;if(volume?uint32_t(r.zoffset+r.depth)>slices:layer+uint32_t(r.zoffset+r.depth)>t->L.layers)return false;
    for(int k=0;k<r.depth;k++){
        uint64_t b=base;uint32_t z=0;
        if(volume)z=uint32_t(r.zoffset+k);
        else{uint32_t a,b2,c,d2;uint64_t lb;unsigned h1,h2;if(nvn_deswizzle_geom(&t->d,level,layer+uint32_t(r.zoffset+k),&a,&b2,&c,&lb,&h1,&h2,&d2))return false;b=lb;}
        for(uint32_t y=0;y<nby;y++)for(uint32_t x=0;x<nbx;x++){
            uint64_t at=b+nvn_bl_offset((bx0+x)*bpb,by0+y,z,row_bytes,rows,th,td);if(at+bpb>t->L.size)return false;
            memcpy(t->mem+at,src+size_t(k)*image_stride+size_t(y)*row+size_t(x)*bpb,bpb);
        }
    }
    return true;
}
void copy_texture_buffer(const p_copy_b2t &p,bool to_buffer){
    Image &im=get_image(p.tex,to_buffer);
    if(p.has_view && p.view.set_format)(void)im.view_format(&p.view);
    if(im.samples!=VK_SAMPLE_COUNT_1_BIT || (im.decoded&&to_buffer))
        xbg_fatalf("NVN %s format=%#x samples=%u region=%d,%d,%d:%dx%dx%d needs %s",to_buffer?"CopyTextureToBuffer":"CopyBufferToTexture",p.tex->d.format,uint32_t(im.samples),p.r.xoffset,p.r.yoffset,p.r.zoffset,p.r.width,p.r.height,p.r.depth,im.decoded?"compressed readback encoding":"depth/stencil or multisample packing");
    need((p.flags&~NVN_COPY_FLAGS_ENGINE_2D_BIT)==0,"Buffer/texture copy flags not implemented");
    uint32_t level=p.has_view&&p.view.set_levels?p.view.base_level:0,layer=p.has_view&&p.view.set_layers?p.view.min_layer:0;region_check(im,level,layer,p.r);need(p.src,"Null buffer image copy address");
    bool full_destination=xbg_copy_is_full({im.width,im.height,im.source.d.target==2?im.depth:im.layers,im.levels,level,layer,p.r.xoffset,p.r.yoffset,p.r.zoffset,p.r.width,p.r.height,p.r.depth})!=0;
    if(!to_buffer && !full_destination)get_image(p.tex,true);

    if(im.aspects!=VK_IMAGE_ASPECT_COLOR_BIT){
        xbg_depth_format f;need(xbg_depth_format_info(p.tex->d.format,&f)==0,"Unknown NVN depth copy format");
        size_t row,image,bytes;
        need(nvn_copy_pitch(p.r.width,p.r.height,p.r.depth,1,1,f.packed_bytes,state.copy_stride,state.copy_image_stride,&row,&image,&bytes) && row%f.packed_bytes==0 && bytes<=256ull*1024*1024,"Invalid packed depth copy stride/extent");
        need(p.src<=UINT64_MAX-bytes,"Depth copy address overflow");
        std::vector<uint8_t> planes;std::vector<VkBufferImageCopy> copies;VkBufferImageCopy region{};
        region.imageSubresource=copy_layers(im,level,layer,p.r);region.imageOffset=copy_offset(im,p.r);region.imageExtent=copy_extent(im,p.r);
        append_depth_planes(planes,copies,p.tex->d.format,p.r.width,p.r.height,p.r.depth,
            to_buffer?nullptr:reinterpret_cast<const void*>(p.src),bytes,row,image,region);
        Buffer staging=to_buffer?readback_buffer(planes.size()):stream_buffer(planes.size(),planes.data(),planes.size(),false);for(auto &copy:copies)copy.bufferOffset+=staging.offset;
        Commands command;im.transition(command.command,to_buffer?VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        if(to_buffer)vkCmdCopyImageToBuffer(command.command,im.image,im.layout,staging.buffer,copies.size(),copies.data());
        else vkCmdCopyBufferToImage(command.command,staging.buffer,im.image,im.layout,copies.size(),copies.data());
        im.transition(command.command,VK_IMAGE_LAYOUT_GENERAL);command.finish();
        if(to_buffer){
            gpu_sync(perf.sync_readback);staging.invalidate();invalidate_cpu_range(p.src,bytes);const uint8_t *dp=nullptr,*sp=nullptr;
            for(auto &r:copies){if(r.imageSubresource.aspectMask==VK_IMAGE_ASPECT_DEPTH_BIT)dp=static_cast<const uint8_t*>(staging.mapped)+(r.bufferOffset-staging.offset);else sp=static_cast<const uint8_t*>(staging.mapped)+(r.bufferOffset-staging.offset);}
            size_t pixels=size_t(p.r.width)*p.r.height*p.r.depth;
            std::vector<uint8_t> d16;if(dp && p.tex->d.format==0x32 && depth16_as_d24){d16.resize(pixels*2);for(size_t i=0;i<pixels;i++){uint32_t v;memcpy(&v,dp+i*4,4);uint16_t k=uint16_t(d24_to_d16(v));memcpy(&d16[i*2],&k,2);}dp=d16.data();}
            need(xbg_depth_merge(p.tex->d.format,p.r.width,p.r.height,p.r.depth,reinterpret_cast<void*>(p.src),bytes,row,image,dp,pixels*f.depth_bytes,sp,pixels*f.stencil_bytes)==0,"Packed NVN depth readback merge failed");
        }else im.gpu_written=true,provenance_write(im,__LINE__);
        xbg_logf("NVN depth %s serial=%llu format=%#x level=%u layer=%u region=%dx%dx%d planes=%zu",to_buffer?"readback":"upload",(unsigned long long)p.tex->serial,p.tex->d.format,level,layer,p.r.width,p.r.height,p.r.depth,copies.size());return;
    }
    if(im.decoded){
        int bw,bh,bpb=nvn_fmt_bpb(p.tex->d.format,&bw,&bh);need(bw==4 && bh==4 && !(p.r.xoffset%4) && !(p.r.yoffset%4),"BC upload origin must be block aligned");
        need(p.r.width%4==0 || uint32_t(p.r.xoffset+p.r.width)==std::max(1u,im.width>>level),"BC upload width must reach a block or mip edge");
        need(p.r.height%4==0 || uint32_t(p.r.yoffset+p.r.height)==std::max(1u,im.height>>level),"BC upload height must reach a block or mip edge");
        size_t row,image,source_bytes;
        need(nvn_copy_pitch(p.r.width,p.r.height,p.r.depth,bw,bh,bpb,state.copy_stride,state.copy_image_stride,&row,&image,&source_bytes) && row%bpb==0 && source_bytes<=256ull*1024*1024,"Invalid BC copy row stride/extent");
        need(p.src<=UINT64_MAX-source_bytes,"BC copy address overflow");
        size_t bytes=size_t(p.r.width)*p.r.height*p.r.depth*im.bpp;
        if(!im.gpu_written && !(p.tex->d.flags&0x10)){cpu_access(p.src,source_bytes,9);invalidate_cpu_range(uint64_t(uintptr_t(p.tex->mem)),p.tex->L.size);
            if(guest_texture_write(p.tex,level,layer,p.r,reinterpret_cast<const uint8_t*>(p.src),row,image)){im.uploaded=false;im.checked_submission=UINT64_MAX;return;}}
        need(bytes<=256ull*1024*1024,"Decoded BC staging exceeds memory bound");std::vector<uint8_t> decoded(bytes);
        int result=xbg_decode_bc_linear(p.tex->d.format,p.r.width,p.r.height,p.r.depth,reinterpret_cast<const void*>(p.src),source_bytes,row,image,decoded.data(),decoded.size());
        if(result)xbg_fatalf("NVN BC staging decode failed format=%#x level=%u result=%d",p.tex->d.format,level,result);
        Buffer buffer=stream_buffer(bytes,decoded.data(),bytes,false);Commands command;im.transition(command.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy region{};region.bufferOffset=buffer.offset;region.imageSubresource=copy_layers(im,level,layer,p.r);region.imageOffset=copy_offset(im,p.r);region.imageExtent=copy_extent(im,p.r);
        vkCmdCopyBufferToImage(command.command,buffer.buffer,im.image,im.layout,1,&region);im.transition(command.command,VK_IMAGE_LAYOUT_GENERAL);command.finish();im.gpu_written=true,provenance_write(im,__LINE__);
        xbg_logf("NVN BC upload format=%#x level=%u layer=%u region=%dx%dx%d src=%zu decoded=%zu",p.tex->d.format,level,layer,p.r.width,p.r.height,p.r.depth,source_bytes,bytes);return;
    }
    size_t row,image,n;
    need(nvn_copy_pitch(p.r.width,p.r.height,p.r.depth,1,1,im.bpp,state.copy_stride,state.copy_image_stride,&row,&image,&n) && n<=256ull*1024*1024,"Invalid or oversized color copy stride/extent");
    need(p.src<=UINT64_MAX-n,"Color copy address overflow");
    if(!to_buffer && !im.gpu_written && !(p.tex->d.flags&0x10)){cpu_access(p.src,n,9);invalidate_cpu_range(uint64_t(uintptr_t(p.tex->mem)),p.tex->L.size);
        if(guest_texture_write(p.tex,level,layer,p.r,reinterpret_cast<const uint8_t*>(p.src),row,image)){im.uploaded=false;im.checked_submission=UINT64_MAX;return;}}
    size_t slice_alignment=4;while(slice_alignment%im.bpp)slice_alignment+=4;
    size_t staging_row=size_t(p.r.width)*im.bpp;
    size_t staging_image=((staging_row*size_t(p.r.height)+slice_alignment-1)/slice_alignment)*slice_alignment,staging_bytes=staging_image*size_t(p.r.depth);
    need(staging_bytes<=256ull*1024*1024,"Color staging exceeds memory bound");
    std::vector<uint8_t> upload;
    if(!to_buffer){cpu_access(p.src,n,9);upload.resize(staging_bytes);
        for(int k=0;k<p.r.depth;k++)for(int y=0;y<p.r.height;y++)
            memcpy(upload.data()+size_t(k)*staging_image+size_t(y)*staging_row,reinterpret_cast<const uint8_t*>(p.src)+size_t(k)*image+size_t(y)*row,size_t(p.r.width)*im.bpp);
    }
    Buffer b=to_buffer?readback_buffer(staging_bytes):stream_buffer(staging_bytes,upload.data(),staging_bytes,false);
    Commands c;im.transition(c.command,to_buffer?VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    // 每层单独记录，允许字节图距不整除行距。
    // Record each slice separately to allow byte image strides indivisible by row strides.
    std::vector<VkBufferImageCopy> copies;
    for(int k=0;k<p.r.depth;k++){
        VkBufferImageCopy r{};r.bufferOffset=b.offset+size_t(k)*staging_image;r.bufferRowLength=0;
        r.imageSubresource=copy_layers(im,level,layer,p.r);r.imageSubresource.layerCount=1;
        r.imageOffset=copy_offset(im,p.r);r.imageExtent={uint32_t(p.r.width),uint32_t(p.r.height),1};
        if(im.source.d.target==2)r.imageOffset.z+=k;else r.imageSubresource.baseArrayLayer+=k;
        copies.push_back(r);
    }
    if(to_buffer)vkCmdCopyImageToBuffer(c.command,im.image,im.layout,b.buffer,uint32_t(copies.size()),copies.data());else vkCmdCopyBufferToImage(c.command,b.buffer,im.image,im.layout,uint32_t(copies.size()),copies.data());im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();
    if(to_buffer){gpu_sync(perf.sync_readback);b.invalidate();invalidate_cpu_range(p.src,n);
        for(int k=0;k<p.r.depth;k++)for(int y=0;y<p.r.height;y++){
            size_t off=size_t(k)*image+size_t(y)*row;
            memcpy(reinterpret_cast<uint8_t*>(p.src)+off,static_cast<const uint8_t*>(b.mapped)+size_t(k)*staging_image+size_t(y)*staging_row,size_t(p.r.width)*im.bpp);
        }
    }else im.gpu_written=true,provenance_write(im,__LINE__);
}
void copy_texture(const p_copy_t2t &p){Image &s=get_image(p.src,true),&d=get_image(p.dst,false);{static unsigned n=0;if(n++<60)xbg_logf("NVN COPYT2T src=%llu fmt=%#x %ux%ux%u L%u lv%u -> dst=%llu fmt=%#x %ux%u L%u lv%u decoded=%d/%d region=%d,%d,%d %dx%dx%d -> %d,%d,%d sv=%d/%u/%u dv=%d/%u/%u",(unsigned long long)p.src->serial,p.src->d.format,s.width,s.height,s.depth,s.layers,s.levels,(unsigned long long)p.dst->serial,p.dst->d.format,d.width,d.height,d.layers,d.levels,int(s.decoded),int(d.decoded),p.sr.xoffset,p.sr.yoffset,p.sr.zoffset,p.sr.width,p.sr.height,p.sr.depth,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,int(p.has_sv),p.has_sv?p.sv.base_level:0,p.has_sv?p.sv.min_layer:0,int(p.has_dv),p.has_dv?p.dv.base_level:0,p.has_dv?p.dv.min_layer:0);}
    if(p.has_sv && p.sv.set_format)(void)s.view_format(&p.sv);
    if(p.has_dv && p.dv.set_format)(void)d.view_format(&p.dv);
    // No-BC + ASTC pack (iOS copy_t2t_astc): the game copies file textures (now ASTC 4x4) block-for-block into its own BC3 arrays,
    // which we decode to RGBA8. Same-size NEAREST blit decodes ASTC exactly; sRGB-ness matches on both sides (BC3 sRGB <-> ASTC sRGB).
    if(!s.decoded && d.decoded && s.fmt>=VK_FORMAT_ASTC_4x4_UNORM_BLOCK && s.fmt<=VK_FORMAT_ASTC_4x4_SRGB_BLOCK && (d.fmt==VK_FORMAT_R8G8B8A8_UNORM || d.fmt==VK_FORMAT_R8G8B8A8_SRGB)
       && p.sr.width==p.dr.width && p.sr.height==p.dr.height && p.sr.depth==p.dr.depth && s.source.d.target!=2 && d.source.d.target!=2){
        VkFormatProperties sf{},df{};vkGetPhysicalDeviceFormatProperties(cfg.physical,s.fmt,&sf);vkGetPhysicalDeviceFormatProperties(cfg.physical,d.fmt,&df);
        need((sf.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_SRC_BIT) && (df.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_DST_BIT),"ASTC to decoded-BC copy needs blit support");
        uint32_t sl=p.has_sv&&p.sv.set_levels?p.sv.base_level:0,dl=p.has_dv&&p.dv.set_levels?p.dv.base_level:0,sa=p.has_sv&&p.sv.set_layers?p.sv.min_layer:0,da=p.has_dv&&p.dv.set_layers?p.dv.min_layer:0;
        region_check(s,sl,sa,p.sr);region_check(d,dl,da,p.dr);need(&s!=&d,"Same-image copy needs overlap validation");
        bool full_destination=xbg_copy_is_full({d.width,d.height,d.layers,d.levels,dl,da,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,p.dr.width,p.dr.height,p.dr.depth})!=0;
        if(!full_destination)get_image(p.dst,true);
        VkImageBlit blit{};blit.srcSubresource=copy_layers(s,sl,sa,p.sr);blit.dstSubresource=copy_layers(d,dl,da,p.dr);
        VkOffset3D so=copy_offset(s,p.sr),dof=copy_offset(d,p.dr);VkExtent3D e=copy_extent(s,p.sr);
        blit.srcOffsets[0]=so;blit.srcOffsets[1]={so.x+int32_t(e.width),so.y+int32_t(e.height),so.z+1};blit.dstOffsets[0]=dof;blit.dstOffsets[1]={dof.x+int32_t(e.width),dof.y+int32_t(e.height),dof.z+1};
        Commands c;s.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);d.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);vkCmdBlitImage(c.command,s.image,s.layout,d.image,d.layout,1,&blit,VK_FILTER_NEAREST);
        s.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);d.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();d.gpu_written=true,provenance_write(d,__LINE__);
        {static unsigned n=0;if(n++<8)xbg_logf("NVN ASTC2BC blit src=%llu fmt=%#x -> dst=%llu fmt=%#x %dx%d lv%u/%u layer%u/%u",(unsigned long long)p.src->serial,p.src->d.format,(unsigned long long)p.dst->serial,p.dst->d.format,p.sr.width,p.sr.height,sl,dl,sa,da);}
        return;
    }
    if(s.decoded || d.decoded)need(s.decoded && d.decoded && xbg_bc_view_compatible(s.source.d.format,d.source.d.format),"BC texture copy must retain compression and signedness");
    need((uint32_t(p.flags)&~3u)==0 && s.fmt==d.fmt && s.aspects==d.aspects && s.samples==d.samples,"Texture copy requires matching formats/aspects and supported flags");
    uint32_t sl=p.has_sv&&p.sv.set_levels?p.sv.base_level:0,dl=p.has_dv&&p.dv.set_levels?p.dv.base_level:0,sa=p.has_sv&&p.sv.set_layers?p.sv.min_layer:0,da=p.has_dv&&p.dv.set_layers?p.dv.min_layer:0;
    region_check(s,sl,sa,p.sr);region_check(d,dl,da,p.dr);need(&s!=&d,"Same-image copy needs overlap validation");need((s.source.d.target==2)==(d.source.d.target==2),"Mixed volume/array copy requires slice mapping");
    bool scaled=p.sr.width!=p.dr.width || p.sr.height!=p.dr.height || p.sr.depth!=p.dr.depth;
    VkImageBlit blit{};VkFilter filter{};
    if(scaled){
        need(!s.decoded && !d.decoded && s.source.d.target!=0 && s.source.d.target!=2 && s.source.d.target!=3 && d.source.d.target!=0 && d.source.d.target!=2 && d.source.d.target!=3 && s.aspects==VK_IMAGE_ASPECT_COLOR_BIT && s.samples==VK_SAMPLE_COUNT_1_BIT,"Scaled blit requires native single-sample color 2D/array images");
        need(s.view_format(p.has_sv?&p.sv:nullptr)==s.fmt && d.view_format(p.has_dv?&p.dv:nullptr)==d.fmt,"Scaled blit cannot reinterpret image views");
        VkFormatProperties sf{},df{};vkGetPhysicalDeviceFormatProperties(cfg.physical,s.fmt,&sf);vkGetPhysicalDeviceFormatProperties(cfg.physical,d.fmt,&df);
        XbgBlitRegion src{s.width,s.height,s.layers,s.levels,sl,sa,p.sr.xoffset,p.sr.yoffset,p.sr.zoffset,p.sr.width,p.sr.height,p.sr.depth};
        XbgBlitRegion dst{d.width,d.height,d.layers,d.levels,dl,da,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,p.dr.width,p.dr.height,p.dr.depth};
        need(xbg_blit_plan(src,dst,uint32_t(p.flags),xbg_clear_color_class(s.source.d.format),sf.optimalTilingFeatures,df.optimalTilingFeatures,&blit,&filter)==0,"Scaled blit region/type/filter/device capability unsupported");
    }
    // Import initial CPU contents before a partial destination overwrite; this
    // also preserves untouched mip levels and array layers for ordinary copies.
    bool full_destination=xbg_copy_is_full({d.width,d.height,d.source.d.target==2?d.depth:d.layers,d.levels,dl,da,p.dr.xoffset,p.dr.yoffset,p.dr.zoffset,p.dr.width,p.dr.height,p.dr.depth})!=0;
    if(!full_destination){need(d.samples==VK_SAMPLE_COUNT_1_BIT || d.gpu_written,"Partial copy into initial CPU multisample storage unsupported");get_image(p.dst,true);}
    if(scaled){Commands c;s.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);d.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);vkCmdBlitImage(c.command,s.image,s.layout,d.image,d.layout,1,&blit,filter);s.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);d.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();d.gpu_written=true,provenance_write(d,__LINE__);return;}
    Commands c;s.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);d.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageCopy r{};r.srcSubresource=copy_layers(s,sl,sa,p.sr);r.dstSubresource=copy_layers(d,dl,da,p.dr);r.srcOffset=copy_offset(s,p.sr);r.dstOffset=copy_offset(d,p.dr);r.extent=copy_extent(s,p.sr);std::vector<VkImageCopy> regions;
    for(VkImageAspectFlags flag: {VkImageAspectFlags(VK_IMAGE_ASPECT_COLOR_BIT),VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT),VkImageAspectFlags(VK_IMAGE_ASPECT_STENCIL_BIT)})if(s.aspects&flag){auto one=r;one.srcSubresource.aspectMask=one.dstSubresource.aspectMask=flag;regions.push_back(one);}
    vkCmdCopyImage(c.command,s.image,s.layout,d.image,d.layout,regions.size(),regions.data());s.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);d.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();d.gpu_written=true,provenance_write(d,__LINE__);}
void partial_clear_texture(Image &im,const p_clear_tex &request,uint32_t level,uint32_t base_layer,int scalar){
    need(im.source.d.target!=2,"Partial 3D texture clear requires 2D slice-view support");
    bool color=im.aspects==VK_IMAGE_ASPECT_COLOR_BIT;need(color || request.mask==15,"Depth/stencil texture channel-mask semantics not implemented");
    for(uint32_t slice=0;slice<uint32_t(request.r.depth);slice++){
        Commands c;Pass p;p.width=std::max(1u,im.width>>level);p.height=std::max(1u,im.height>>level);p.samples=im.samples;p.color_slots=color?1:0;
        uint32_t selected_layer=0;VkRect2D rect{};
        need(xbg_texture_clear_slice(p.width,p.height,im.layers,base_layer,request.r.xoffset,request.r.yoffset,request.r.zoffset,request.r.width,request.r.height,request.r.depth,slice,&rect,&selected_layer)==0,"Invalid texture clear slice");
        view_t view=request.view;view.set_levels=view.set_layers=1;view.base_level=level;view.nlevels=1;view.min_layer=selected_layer;view.nlayers=1;
        VkImageView image_view=im.view(&view,true);im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);
        VkAttachmentDescription attachment{};attachment.format=im.view_format(&view);attachment.samples=p.samples;attachment.loadOp=attachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_LOAD;attachment.storeOp=attachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_STORE;attachment.initialLayout=attachment.finalLayout=VK_IMAGE_LAYOUT_GENERAL;
        VkAttachmentReference ref{0,VK_IMAGE_LAYOUT_GENERAL};VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.colorAttachmentCount=color?1:0;sub.pColorAttachments=color?&ref:nullptr;sub.pDepthStencilAttachment=color?nullptr:&ref;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=1;rp.pAttachments=&attachment;rp.subpassCount=1;rp.pSubpasses=&sub;VK(vkCreateRenderPass(cfg.device,&rp,nullptr,&p.pass));
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fb.renderPass=p.pass;fb.attachmentCount=1;fb.pAttachments=&image_view;fb.width=p.width;fb.height=p.height;fb.layers=1;VK(vkCreateFramebuffer(cfg.device,&fb,nullptr,&p.framebuffer));
        {VkFramebuffer f=p.framebuffer;VkRenderPass rp_=p.pass;retire_handle([f,rp_]{vkDestroyFramebuffer(cfg.device,f,nullptr);vkDestroyRenderPass(cfg.device,rp_,nullptr);});}
        auto masked_object=std::make_shared<MaskedClear>();MaskedClear &masked=*masked_object;retire_object(masked_object);bool partial=color && request.mask!=15;if(partial)masked.prepare(p,0,scalar,uint32_t(request.mask),0,0);
        begin_pass(p,c);
        if(partial)masked.encode(c,p,rect,&request.c);
        else{VkClearAttachment clear{};clear.aspectMask=im.aspects;if(color)memcpy(&clear.clearValue.color,&request.c,16);else clear.clearValue.depthStencil={im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?request.c.f[0]:0.0f,request.c.u[im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?1:0]};VkClearRect region{rect,0,1};vkCmdClearAttachments(c.command,1,&clear,1,&region);}
        vkCmdEndRenderPass(c.command);pass_prof_end(c.command);c.finish();
    }
    im.gpu_written=true,provenance_write(im,__LINE__);
}
void clear_texture(const p_clear_tex &input){
    p_clear_tex p=input;
    Image &im=get_image(p.tex,false);need((uint32_t(p.mask)&~15u)==0,"Invalid texture clear mask");if(!p.mask)return;
    uint32_t level=p.has_view&&p.view.set_levels?p.view.base_level:0,layer=p.has_view&&p.view.set_layers?p.view.min_layer:0;
    region_check(im,level,layer,p.r);uint32_t w=std::max(1u,im.width>>level),h=std::max(1u,im.height>>level);
    need(p.is_uint>=0 && p.is_uint<=2,"Invalid texture clear scalar type");int scalar=p.is_uint==2?1:p.is_uint==1?2:0;
    if(im.aspects==VK_IMAGE_ASPECT_COLOR_BIT){VkFormat viewed=im.view_format(p.has_view?&p.view:nullptr);static unsigned notes=0;if(notes++<8)xbg_logf("NVN clear-value texture input=%d nvn=%#x view=%#x vk=%u mip=%u mask=%#x bits=%08x/%08x/%08x/%08x",scalar,im.source.d.format,p.has_view&&p.view.set_format?p.view.format:im.source.d.format,unsigned(viewed),level,unsigned(p.mask),p.c.u[0],p.c.u[1],p.c.u[2],p.c.u[3]);VkClearColorValue value{};int converted=xbg_clear_value(viewed,scalar,&p.c,&value);
        if(converted<0)xbg_fatalf("Texture clear conversion failed nvn=%#x view=%#x vk=%u input=%d rc=%d",im.source.d.format,p.has_view&&p.view.set_format?p.view.format:im.source.d.format,unsigned(viewed),scalar,converted);
        memcpy(&p.c,&value,16);scalar=converted;}

    bool partial=p.mask!=15 || p.r.xoffset || p.r.yoffset || uint32_t(p.r.width)!=w || uint32_t(p.r.height)!=h;
    if(partial)need(im.source.d.target!=2,"Partial 3D texture clear requires 2D slice-view support");
    if(im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT)need(p.c.f[0]>=0 && p.c.f[0]<=1,"Texture depth clear outside normalized range");
    if(partial || im.levels>1 || im.layers>1){need(im.samples==VK_SAMPLE_COUNT_1_BIT || im.gpu_written,"Partial clear of uninitialized CPU multisample storage unsupported");get_image(p.tex,true);}
    if(partial){partial_clear_texture(im,p,level,layer,scalar);return;}
    bool volume=im.source.d.target==2;need(!volume || (p.r.zoffset==0 && uint32_t(p.r.depth)==std::max(1u,im.depth>>level)),"Partial volume clear requires shader clear");
    VkFormat viewed=im.view_format(p.has_view?&p.view:nullptr);Commands c;VkClearValue value{};memcpy(&value.color,&p.c,16);
    if(viewed!=im.fmt){
        // VkClearColorImage uses the image's creation format. The original game
        // clears through integer views, so use an actual view/render-pass clear
        // to preserve raw bit patterns instead of numerically converting them.
        need(!volume && im.aspects==VK_IMAGE_ASPECT_COLOR_BIT,"Reinterpreted volume/depth clear unsupported");
        im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);
        std::vector<VkFramebuffer> framebuffers;VkRenderPass pass{};
        VkAttachmentDescription attachment{};attachment.format=viewed;attachment.samples=im.samples;attachment.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;attachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;attachment.initialLayout=attachment.finalLayout=VK_IMAGE_LAYOUT_GENERAL;
        VkAttachmentReference ref{0,VK_IMAGE_LAYOUT_GENERAL};VkSubpassDescription sub{};sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sub.colorAttachmentCount=1;sub.pColorAttachments=&ref;
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=1;rp.pAttachments=&attachment;rp.subpassCount=1;rp.pSubpasses=&sub;VK(vkCreateRenderPass(cfg.device,&rp,nullptr,&pass));
        for(uint32_t z=0;z<uint32_t(p.r.depth);z++){
            view_t view=p.view;view.set_levels=view.set_layers=1;view.base_level=level;view.nlevels=1;view.min_layer=layer+p.r.zoffset+z;view.nlayers=1;
            VkImageView image_view=im.view(&view,true);VkFramebuffer fb;VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fi.renderPass=pass;fi.attachmentCount=1;fi.pAttachments=&image_view;fi.width=w;fi.height=h;fi.layers=1;VK(vkCreateFramebuffer(cfg.device,&fi,nullptr,&fb));framebuffers.push_back(fb);
            VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};begin.renderPass=pass;begin.framebuffer=fb;begin.renderArea.extent={w,h};begin.clearValueCount=1;begin.pClearValues=&value;
            if(slots[slot_index].prof){Pass profile;profile.width=w;profile.height=h;profile.images.push_back(&im);pass_prof_begin(profile,c.command);}
            vkCmdBeginRenderPass(c.command,&begin,VK_SUBPASS_CONTENTS_INLINE);vkCmdEndRenderPass(c.command);pass_prof_end(c.command);
        }
        im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();retire_handle([framebuffers,pass]{for(auto f:framebuffers)vkDestroyFramebuffer(cfg.device,f,nullptr);vkDestroyRenderPass(cfg.device,pass,nullptr);});
    }else{
        im.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageSubresourceRange range{im.aspects,level,1,volume?0:layer+uint32_t(p.r.zoffset),volume?1:uint32_t(p.r.depth)};
        if(im.aspects==VK_IMAGE_ASPECT_COLOR_BIT)vkCmdClearColorImage(c.command,im.image,im.layout,&value.color,1,&range);
        else{VkClearDepthStencilValue ds{im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?p.c.f[0]:0.0f,p.c.u[im.aspects&VK_IMAGE_ASPECT_DEPTH_BIT?1:0]};vkCmdClearDepthStencilImage(c.command,im.image,im.layout,&ds,1,&range);}
        im.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();
    }
    im.gpu_written=true,provenance_write(im,__LINE__);
}
// Buffer copy whose source is a pending GPU result: copy on the GPU and keep
// the destination pending too, instead of stalling for the CPU memmove.
bool gpu_buffer_copy(uint64_t src,uint64_t dst,uint64_t n){
    if(!n || (src<dst+n && dst<src+n))return false;
    for(auto &w:pending_writebacks)if(dst<w.addr+w.bytes && dst+n>w.addr)return false;
    Buffer view;if(!pending_view(src,n,1,view))return false;
    invalidate_cpu_range(dst,n);
    Slot &s=recording_slot();end_open_pass();Buffer b=readback_buffer(n);
    VkBufferCopy region{view.offset,b.offset,n};vkCmdCopyBuffer(s.cmd,view.buffer,b.buffer,1,&region);full_barrier(s.cmd);
    pending_writebacks.push_back({dst,n,b,&s});return true;
}
void resolve_texture(const p_copy_t2t &p){
    Image &s=get_image(p.src,true),&d=get_image(p.dst,false);
    need(s.samples!=VK_SAMPLE_COUNT_1_BIT && d.samples==VK_SAMPLE_COUNT_1_BIT && s.fmt==d.fmt,"Resolve requires matching MS source / single-sample destination");
    need(s.aspects==VK_IMAGE_ASPECT_COLOR_BIT && d.aspects==VK_IMAGE_ASPECT_COLOR_BIT,"Depth/stencil resolve mode is not yet verified against original NVN");
    uint32_t sl=p.has_sv&&p.sv.set_levels?p.sv.base_level:0,dl=p.has_dv&&p.dv.set_levels?p.dv.base_level:0,sa=p.has_sv&&p.sv.set_layers?p.sv.min_layer:0,da=p.has_dv&&p.dv.set_layers?p.dv.min_layer:0;
    uint32_t sn=p.has_sv&&p.sv.set_layers?p.sv.nlayers:s.layers,dn=p.has_dv&&p.dv.set_layers?p.dv.nlayers:d.layers;
    need(sl<s.levels && dl<d.levels && sa+sn<=s.layers && da+dn<=d.layers && sn==dn,"Resolve view out of range");
    uint32_t w=std::max(1u,s.width>>sl),h=std::max(1u,s.height>>sl);need(w==std::max(1u,d.width>>dl)&&h==std::max(1u,d.height>>dl),"Resolve extents differ");
    Commands c;s.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);d.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);VkImageResolve r{};r.srcSubresource={s.aspects,sl,sa,sn};r.dstSubresource={d.aspects,dl,da,dn};r.extent={w,h,1};vkCmdResolveImage(c.command,s.image,s.layout,d.image,d.layout,1,&r);s.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);d.transition(c.command,VK_IMAGE_LAYOUT_GENERAL);c.finish();d.gpu_written=true,provenance_write(d,__LINE__);
}

// CPU-visible writes occur only after all preceding synchronous GPU work finished.
// Every byte count and opcode is checked; unknown commands terminate with context.
// Consecutive commands nearly always sit in one range: remember the last one (pools change only under the mutex,
// and clear it; host segments are freed only at shutdown).
uintptr_t cmd_range_lo=0;size_t cmd_range_bytes=0;
size_t command_available(const uint8_t *p){uintptr_t address=uintptr_t(p);if(address-cmd_range_lo<cmd_range_bytes)return cmd_range_bytes-(address-cmd_range_lo);auto it=pools.upper_bound(address);if(it!=pools.begin()){--it;if(address>=it->first && address-it->first<it->second){cmd_range_lo=it->first;cmd_range_bytes=it->second;return it->second-(address-it->first);}}{auto h=host_ranges.upper_bound(address);if(h!=host_ranges.begin()){--h;if(address>=h->first && address-h->first<h->second){cmd_range_lo=h->first;cmd_range_bytes=h->second;return h->second-(address-h->first);}}}xbg_fatalf("NVN command address %#llx is outside registered command pools",(unsigned long long)address);}
void execute_draw_call(const xbg_draw_call &p){
    need(p.index_type>=-1 && p.index_type<=2 && p.first>=0 && p.count>=0 && p.instances>=0 && p.base_instance>=0,"Invalid recorded draw plan");
    if(!p.count || !p.instances)return;
    if(p.index_type<0)draw(p.primitive,p.first,p.count,nullptr,uint32_t(p.instances),uint32_t(p.base_instance));
    else{
        uint64_t bytes=uint64_t(p.count)<<p.index_type;
        need(bytes<=command_available(reinterpret_cast<const uint8_t*>(p.index_address)),"Indirect index read exceeds registered storage");
        p_draw_idx indexed{p.primitive,p.index_type,p.count,p.base_vertex,p.base_instance,p.instances,p.index_address};
        draw(p.primitive,0,p.count,&indexed);
    }
}
void execute_indirect(const xbg_indirect &p,bool single){
    need(p.arguments && p.max_count>=0 && p.max_count<=1000000,"Invalid indirect draw request");
    uint32_t count=1;
    if(!single){
        need(p.count_address && !(p.count_address&3),"Invalid indirect count address");
        need(command_available(reinterpret_cast<const uint8_t*>(p.count_address))>=4,"Truncated indirect count storage");
        cpu_access(p.count_address,4,5);memcpy(&count,reinterpret_cast<const void*>(p.count_address),4);count=std::min(count,uint32_t(p.max_count));
    }
    for(uint32_t i=0;i<count;i++){
        uint64_t address=0;need(xbg_draw_indirect_address(p.arguments,i,p.stride,p.index_type,&address)==0,"Invalid indirect argument address/stride");
        size_t bytes=p.index_type<0?16:20;
        need(bytes<=command_available(reinterpret_cast<const uint8_t*>(address)),"Indirect arguments exceed registered storage");cpu_access(address,bytes,5);
        // Read at execution, after preceding synchronous GPU storage writeback,
        // never at command recording. Both single and multi use this path.
        xbg_draw_call plan{};int result=xbg_draw_indirect_plan(&plan,p.primitive,p.index_type,p.index_address,reinterpret_cast<const void*>(address),bytes);
        if(result)xbg_fatalf("Invalid indirect draw parameters: %d",result);
        execute_draw_call(plan);
    }
}
template<class T>T payload(const uint8_t *p,size_t n){need(n>=sizeof(T),"Truncated NVN command payload");T out;memcpy(&out,p,sizeof(out));return out;}
template<class T>T variable_payload(const uint8_t *p,size_t n){T out{};memcpy(&out,p,std::min(n,sizeof(T)));return out;}
// Graphics thread: retire submitted slots whose fence already signalled (non-blocking) so a game
// thread blocked on an older fence is released while the next frame is still being translated.
void poll_retire(){
    bool any=false;for(auto &s:slots)if(s.pending && !s.recording && vkGetFenceStatus(cfg.device,s.fence)==VK_SUCCESS){retire_slot(s);any=true;}
    if(any){{std::lock_guard<std::mutex> l(gfx_job_mutex);}gfx_idle_cv.notify_all();}
}
void execute(const uint8_t *pc,unsigned depth,uint64_t &budget){
    need(depth<64,"NVN command CALL nesting limit exceeded");
    for(;;){need(budget++<10000000,"NVN command count/cycle limit exceeded");
        if(gfx_async && (budget&63)==0)poll_retire();
        // Bound host arena use per frame slot; only at command boundaries.
        {Slot &cur=slots[slot_index];if(cur.recording && cur.stream.used_total+cur.readback.used_total>arena_flush_bytes){SiteScope site(SITE_ARENA);submit_slots(false);}}
        if(early_submit_enabled){
            const uint64_t frame=stats.presents.load();if(early_submit_frame!=frame){early_submit_frame=frame;early_submit_count=0;}
            Slot &cur=slots[slot_index];
            if(early_submit_count<2 && cur.recording && cur.ops>=256 && !open_pass.cmd){
                if(pending_writebacks.empty()){
                    need(!open_occlusion.cmd,"Early submit with an open occlusion query");
                    submit_slots(false);++early_submit_count;++early_submit_window;
                }else ++early_blocked_window;
            }
        }
        size_t available=command_available(pc);need(available>=8,"Truncated command header");op_hdr h;memcpy(&h,pc,8);need(h.size>=8 && !(h.size&7) && h.size<=available,"NVN command size outside pool");const uint8_t *p=pc+8;size_t n=h.size-8;
        switch(h.op){
        case OP_END:return;
        case OP_BEGIN:break;
        case OP_JUMP:pc=reinterpret_cast<const uint8_t*>(payload<uint64_t>(p,n));continue;
        case OP_CALL:execute(reinterpret_cast<const uint8_t*>(payload<uint64_t>(p,n)),depth+1,budget);break;
        case OP_SIGNAL_EVENT:{need(n>=16,"Truncated event");uint64_t addr;uint32_t mode,v;memcpy(&addr,p,8);memcpy(&mode,p+8,4);memcpy(&v,p+12,4);need(addr,"Null event address");SiteScope site(SITE_EVENT);
            if(defer_events){   // hand the work so far to the GPU without waiting; publish when that submission retires
                Slot &cur=slots[slot_index];
                if(defer_events==2 && cur.recording && cur.ops){recording_events.push_back({addr,mode,v,&cur});++events_deferred;break;}
                if(cur.recording && cur.ops)submit_slots(false);
                Slot &last=slots[slot_index^1];   // most recent submission (just made, or the previous one)
                if(last.pending){pending_events.push_back({addr,mode,v,&last});++events_deferred;}else apply_event(addr,mode,v);
                break;}
            complete_all();cpu_access(addr,4,10);snapshot_invalidate(addr,4);auto ptr=reinterpret_cast<uint32_t*>(addr);if(mode==NVN_EVENT_SIGNAL_MODE_ADD)__atomic_fetch_add(ptr,v,__ATOMIC_SEQ_CST);else{need(mode==NVN_EVENT_SIGNAL_MODE_WRITE,"Unknown event mode");__atomic_store_n(ptr,v,__ATOMIC_SEQ_CST);}break;}
        case OP_COPY_B2B:{need(n>=24,"Truncated buffer copy");uint64_t a[3];memcpy(a,p,24);need(a[0]&&a[1]&&a[2]<=256ull*1024*1024,"Invalid buffer copy");if(gpu_buffer_copy(a[0],a[1],a[2]))break;cpu_access(a[0],a[2],11);invalidate_cpu_range(a[1],a[2]);memmove(reinterpret_cast<void*>(a[1]),reinterpret_cast<void*>(a[0]),a[2]);break;}
        case OP_UPDATE_UB:{need(n>=16,"Truncated uniform update");uint64_t a[2];memcpy(a,p,16);need(a[0] && a[1]<=n-16,"Invalid uniform update");invalidate_cpu_range(a[0],a[1]);memcpy(reinterpret_cast<void*>(a[0]),p+16,a[1]);break;}
        case OP_SET_RT:state.rt=payload<p_set_rt>(p,n);break;
        case OP_CLEAR_COLOR:{auto a=payload<p_clear_color>(p,n);clear_attachment(&a,nullptr);break;}
        case OP_COUNT_+9:{auto a=payload<xbg_typed_clear>(p,n);clear_attachment(&a.color,nullptr,a.scalar);break;}
        case OP_CLEAR_DS:{auto a=payload<p_clear_ds>(p,n);clear_attachment(nullptr,&a);break;}
        case OP_VIEWPORTS:{auto a=payload<p_viewports>(p,n);need(a.first==0 && a.n==1,"Multiple viewports unsupported");state.viewport.x=a.v[0][0];state.viewport.y=a.v[0][1];state.viewport.width=a.v[0][2];state.viewport.height=a.v[0][3];break;}
        case OP_SCISSOR:{auto a=payload<p_scissor>(p,n);need(a.w>=0 && a.h>=0,"Invalid scissor dimensions");state.scissor={{a.x,a.y},{uint32_t(a.w),uint32_t(a.h)}};break;}
        case OP_DEPTH_RANGE:{auto a=payload<p_depth_range>(p,n);state.viewport.minDepth=a.n;state.viewport.maxDepth=a.f;break;}
        case OP_BIND_PROGRAM:{CPU_SCOPE(Program);auto a=payload<p_bind_program>(p,n);need(!a.prog || a.prog->nstage<=6,"Invalid bound program");for(unsigned st=0;st<6;st++)if(a.stages&(1u<<nvn_stage[st]))state.shader[st]=nullptr;if(!a.prog)break;const std::array<Shader*,6> &loaded=program_shaders(a.prog);for(unsigned i=0;i<a.prog->nstage;i++){Shader *s=loaded[i];if(a.stages&(1u<<nvn_stage[s->stage]))state.shader[s->stage]=s;}break;}
        case OP_BIND_STATE:{auto a=payload<p_bind_state>(p,n);switch(a.kind){case 0:{blend_st b;memcpy(&b,a.bytes,sizeof(b));state.blend[b.target]=b;break;}case 1:memcpy(&state.color,a.bytes,sizeof(state.color));break;case 2:memcpy(&state.mask,a.bytes,sizeof(state.mask));break;case 3:memcpy(&state.ms,a.bytes,sizeof(state.ms));break;case 4:memcpy(&state.poly,a.bytes,sizeof(state.poly));break;case 5:memcpy(&state.ds,a.bytes,sizeof(state.ds));break;default:xbg_fatalf("Unknown NVN state object %d",a.kind);}break;}
        case OP_BIND_VATTR:{auto a=variable_payload<p_bind_vattr>(p,n);need(a.first==0 && a.n>=0 && a.n<=16 && n>=offsetof(p_bind_vattr,st)+size_t(a.n)*4,"Invalid vertex attribute record");need(xbg_vertex_attributes_replace(state.attrib,reinterpret_cast<const uint32_t*>(a.st),a.n)==0,"Invalid vertex attribute replacement");break;}
        case OP_BIND_VSTREAM:{auto a=variable_payload<p_bind_vstream>(p,n);need(a.first>=0 && a.n>=0 && a.first+a.n<=16 && n>=offsetof(p_bind_vstream,st)+size_t(a.n)*sizeof(vs_st),"Invalid vertex stream record");memcpy(state.stream+a.first,a.st,a.n*sizeof(vs_st));break;}
        case OP_BIND_VBUF:{auto a=variable_payload<p_bind_vbuf>(p,n);need(a.first>=0 && a.n>=0 && a.first+a.n<=16 && n>=offsetof(p_bind_vbuf,r)+size_t(a.n)*sizeof(NVNbufferRange),"Invalid vertex buffer record");memcpy(state.vb+a.first,a.r,a.n*sizeof(NVNbufferRange));break;}
        case OP_BIND_UBO:case OP_BIND_SSBO:{auto a=payload<p_bind_buf>(p,n);need(a.stage>=0 && a.stage<6 && a.index>=0 && a.index<16,"Invalid shader buffer slot");(h.op==OP_BIND_UBO?state.ub:state.sb)[a.stage][a.index]={a.addr,a.size};++buf_gen[a.stage];break;}
        case OP_BIND_TEX:case OP_BIND_SAMP:case OP_BIND_IMG:{auto a=variable_payload<p_bind_handles>(p,n);need(a.stage>=0 && a.stage<6 && a.first>=0 && a.n>=0 && a.n<=32 && a.first+a.n<=(h.op==OP_BIND_IMG?16:128) && n>=offsetof(p_bind_handles,h)+size_t(a.n)*8,"Invalid descriptor binding record");uint64_t *dst=h.op==OP_BIND_IMG?state.img[a.stage]:h.op==OP_BIND_SAMP?state.samp[a.stage]:state.tex[a.stage];memcpy(dst+a.first,a.h,a.n*8);++tex_gen[a.stage];break;}
        case OP_STENCIL:{auto a=payload<p_stencil>(p,n);need(a.face>=0 && a.face<=3 && a.which>=0 && a.which<=2,"Invalid stencil state");for(unsigned i=0;i<2;i++)if(a.face&(1<<i)){if(a.which==0)state.stencil_compare[i]=a.value_mask;else if(a.which==1)state.stencil_write[i]=a.mask;else state.stencil_ref[i]=a.ref;}break;}
        case OP_POLY_OFFSET:{auto a=payload<p_poly_offset>(p,n);state.bias[0]=a.factor;state.bias[1]=a.units;state.bias[2]=a.clamp;break;}
        case OP_DRAW:{auto a=payload<p_draw>(p,n);draw(a.prim,a.first,a.count,nullptr);break;}
        case OP_DRAW_IDX:{auto a=payload<p_draw_idx>(p,n);draw(a.prim,0,a.count,&a);break;}
        case OP_DISPATCH:dispatch(payload<p_dispatch>(p,n));break;
        case OP_BARRIER:{Commands c;VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};b.srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;b.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&b,0,nullptr,0,nullptr);c.finish();break;}
        case OP_SET_TEX_POOL:{auto a=payload<p_set_pool>(p,n);need(a.pool,"Null texture pool");state.texture_pool=*a.pool;break;}
        case OP_SET_SAMP_POOL:{auto a=payload<p_set_pool>(p,n);need(a.pool,"Null sampler pool");state.sampler_pool=*a.pool;break;}
        case OP_COPY_B2T:copy_texture_buffer(payload<p_copy_b2t>(p,n),false);break;
        case OP_COPY_T2B:copy_texture_buffer(payload<p_copy_b2t>(p,n),true);break;
        case OP_COPY_T2T:copy_texture(payload<p_copy_t2t>(p,n));break;
        case OP_CLEAR_TEX:clear_texture(payload<p_clear_tex>(p,n));break;
        case OP_COPY_STRIDE:state.copy_stride=payload<p_copy_stride>(p,n).stride;need(state.copy_stride>=0,"Negative copy stride");break;
        case OP_COPY_IMAGE_STRIDE:state.copy_image_stride=payload<p_copy_stride>(p,n).stride;need(state.copy_image_stride>=0,"Negative copy image stride");break;
        case OP_BLEND_COLOR:{auto a=payload<p_blend_color>(p,n);memcpy(state.blend_color,a.rgba,16);break;}
        case OP_COUNT_:{uint32_t size=payload<uint32_t>(p,n);need(size && size<=properties.limits.maxTessellationPatchSize,"Invalid tessellation patch size");state.patch_size=size;break;}
        case OP_COUNT_+1:{auto s=payload<sync_t>(p,n);need(s.fenced && s.fence<=submitted+1,"Command wait on unresolved external sync");
            // GPU-side wait on an earlier fence of this same in-order queue: already
            // satisfied by submission order (every operation starts with a barrier).
            break;}
        case OP_COUNT_+2:{uint64_t addr=payload<uint64_t>(p,n);need(addr,"Null indirect dispatch address");cpu_access(addr,12,5);uint32_t xyz[3];memcpy(xyz,reinterpret_cast<void*>(addr),12);p_dispatch a{int32_t(xyz[0]),int32_t(xyz[1]),int32_t(xyz[2]),0};dispatch(a);break;}
        case OP_COUNT_+3:execute_indirect(payload<xbg_indirect>(p,n),false);break;
        case OP_COUNT_+7:execute_draw_call(payload<xbg_draw_call>(p,n));break;
        case OP_COUNT_+8:execute_indirect(payload<xbg_indirect>(p,n),true);break;
        case OP_COUNT_+4:{auto a=payload<xbg_wait_event>(p,n);need(a.address && a.mode<=1,"Invalid event wait");
            for(auto *list:{&pending_events,&recording_events})if(std::any_of(list->begin(),list->end(),[&](const PendingEvent &e){return e.addr==a.address;})){complete_all();break;}
            auto deadline=std::chrono::steady_clock::now()+std::chrono::nanoseconds(cfg.fence_timeout_ns);for(;;){uint32_t value=__atomic_load_n(reinterpret_cast<uint32_t*>(a.address),__ATOMIC_ACQUIRE);if(a.mode==0?value==a.value:int32_t(value-a.value)>=0)break;need(std::chrono::steady_clock::now()<deadline,"NVN event wait timeout");std::this_thread::yield();}break;}
        case OP_COUNT_+5:state.bounds=payload<xbg_depth_bounds>(p,n);break;
        case OP_COUNT_+6:resolve_texture(payload<p_copy_t2t>(p,n));break;
        case OP_RESET_COUNTER:{uint64_t type=payload<uint64_t>(p,n);need(type==NVN_COUNTER_TYPE_SAMPLES_PASSED,"Unsupported pipeline counter reset");end_occlusion_query();state.samples_passed=0;++counter_epoch;state.count_samples=true;break;}
        case OP_REPORT_COUNTER:{need(n>=16,"Truncated counter report");uint64_t type,addr;memcpy(&type,p,8);memcpy(&addr,p+8,8);need(addr,"Null counter output");need(type==NVN_COUNTER_TYPE_TIMESTAMP || type==NVN_COUNTER_TYPE_SAMPLES_PASSED,"Unsupported pipeline statistics query");need(timestamp_bits,"Queue family does not support GPU timestamps");
            Commands c;PendingReport r{addr,type,counter_epoch,epochs[counter_epoch].size(),nullptr,VK_NULL_HANDLE,0,0,false};timestamp_query(r.ts,r.ts_index,r.slot);vkCmdWriteTimestamp(c.command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,r.ts,r.ts_index);c.finish();
            pending_reports.push_back(r);break;}
        default:xbg_fatalf("NVN opcode %u at %#llx not implemented",h.op,(unsigned long long)uintptr_t(pc));
        }
        pc+=h.size;
    }
}
PerfCounts perf_counts(){return {pipeline_objects.size(),layout_cache.size(),sampler_cache.size(),render_pass_cache.size(),images.size(),host_segments.size(),host_bytes};}
void destroy_slots(){
    texel_evict(0,true);texel_evict(1,true);
    for(auto &s:slots){s.retire.run();release_descriptors(s);for(auto &v:s.texel_views)vkDestroyBufferView(cfg.device,v.second,nullptr);s.texel_views.clear();for(auto &f:s.framebuffers)vkDestroyFramebuffer(cfg.device,f.second,nullptr);s.framebuffers.clear();for(auto p:s.pools)vkDestroyDescriptorPool(cfg.device,p,nullptr);s.pools.clear();s.pool_next=0;s.stream.release();s.readback.release();if(s.fence)vkDestroyFence(cfg.device,s.fence,nullptr);if(s.timestamps)vkDestroyQueryPool(cfg.device,s.timestamps,nullptr);if(s.prof_pool)vkDestroyQueryPool(cfg.device,s.prof_pool,nullptr);for(auto q:s.occlusion_pools)vkDestroyQueryPool(cfg.device,q,nullptr);for(auto q:s.timestamp_pools)vkDestroyQueryPool(cfg.device,q,nullptr);if(s.pool)vkDestroyCommandPool(cfg.device,s.pool,nullptr);s=Slot{};}
    slot_index=0;snapshot_reset();
}
} // namespace

// Zero-copy probe (C097): can exportable host-visible Vulkan memory be mmap-ed at an address we choose?
// If yes, guest NVN memory pools can be re-backed by GPU-visible memory in place (iOS: Metal buffers on pool memory).
void zero_copy_probe(){
    for(uint32_t i=0;i<memories.memoryTypeCount;i++)xbg_logf("NVN caps memtype %u flags=%#x heap=%u heapsize=%lluMB",i,memories.memoryTypes[i].propertyFlags,memories.memoryTypes[i].heapIndex,(unsigned long long)(memories.memoryHeaps[memories.memoryTypes[i].heapIndex].size>>20));
    auto getfd=reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(cfg.device,"vkGetMemoryFdKHR"));
    if(!getfd){xbg_logf("NVN caps zerocopy getfd=missing");return;}
    const VkDeviceSize n=4<<20;
    for(uint32_t want:{uint32_t(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT),uint32_t(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}){
        uint32_t type=UINT32_MAX;for(uint32_t i=0;i<memories.memoryTypeCount;i++)if((memories.memoryTypes[i].propertyFlags&want)==want){type=i;break;}
        if(type==UINT32_MAX){xbg_logf("NVN caps zerocopy flags=%#x type=none",want);continue;}
        for(VkExternalMemoryHandleTypeFlagBits ht:{VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT}){
            VkExportMemoryAllocateInfo ex{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};ex.handleTypes=ht;
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&ex;ai.allocationSize=n;ai.memoryTypeIndex=type;VkDeviceMemory mem{};
            VkResult r=vkAllocateMemory(cfg.device,&ai,nullptr,&mem);if(r!=VK_SUCCESS){xbg_logf("NVN caps zerocopy flags=%#x handle=%#x alloc=%d",want,unsigned(ht),r);continue;}
            VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};gi.memory=mem;gi.handleType=ht;int fd=-1;r=getfd(cfg.device,&gi,&fd);
            int ok=0;void *vk=nullptr;
            if(r==VK_SUCCESS && fd>=0){
                void *at=mmap(nullptr,n,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
                void *m=at==MAP_FAILED?MAP_FAILED:mmap(at,n,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,fd,0);
                if(m!=MAP_FAILED && vkMapMemory(cfg.device,mem,0,n,0,&vk)==VK_SUCCESS){
                    memset(m,0x5a,n);static_cast<uint8_t*>(m)[12345]=0x77;
                    if(!(want&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){VkMappedMemoryRange rr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};rr.memory=mem;rr.size=VK_WHOLE_SIZE;vkInvalidateMappedMemoryRanges(cfg.device,1,&rr);}
                    ok=static_cast<uint8_t*>(vk)[12345]==0x77 && static_cast<uint8_t*>(vk)[0]==0x5a ? 1 : 2;
                    vkUnmapMemory(cfg.device,mem);
                }
                if(at!=MAP_FAILED)munmap(at,n);close(fd);
            }
            xbg_logf("NVN caps zerocopy flags=%#x type=%u handle=%#x getfd=%d mmap_fixed_shared=%s",want,type,unsigned(ht),r,ok==1?"OK":ok==2?"MISMATCH":"FAIL");
            vkFreeMemory(cfg.device,mem,nullptr);
        }
    }
}
extern "C" int xbg_game_initialize(const xbg_game_config *c,char *error,size_t bytes){
    if(!c || c->size!=sizeof(*c) || !c->physical || !c->device || !c->queue || !c->shader_directory || !c->fence_timeout_ns){if(error&&bytes)snprintf(error,bytes,"Invalid xbg_game_config");return -1;}
    std::lock_guard<std::recursive_mutex> lock(mutex);if(initialized){if(error&&bytes)snprintf(error,bytes,"Game graphics already initialized");return -1;}
    cfg=*c;vkGetPhysicalDeviceProperties(cfg.physical,&properties);vkGetPhysicalDeviceMemoryProperties(cfg.physical,&memories);
    VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};pp.pNext=&subgroup;vkGetPhysicalDeviceProperties2(cfg.physical,&pp);
    VkPhysicalDeviceFeatures available;vkGetPhysicalDeviceFeatures(cfg.physical,&available);const VkBool32 *want=reinterpret_cast<const VkBool32*>(&cfg.enabled_core),*have=reinterpret_cast<const VkBool32*>(&available);for(size_t i=0;i<sizeof(available)/sizeof(VkBool32);i++)if(want[i]&&!have[i]){if(error&&bytes)snprintf(error,bytes,"Requested enabled feature %zu is unsupported",i);return -1;}
    uint32_t count;vkGetPhysicalDeviceQueueFamilyProperties(cfg.physical,&count,nullptr);std::vector<VkQueueFamilyProperties> queues(count);vkGetPhysicalDeviceQueueFamilyProperties(cfg.physical,&count,queues.data());if(cfg.queue_family>=count || (queues[cfg.queue_family].queueFlags&(VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT))!=(VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT)){if(error&&bytes)snprintf(error,bytes,"Graphics+compute queue family required");return -1;}
    timestamp_bits=queues[cfg.queue_family].timestampValidBits;
    {const auto &l=properties.limits;VkDeviceSize a=256;for(VkDeviceSize x:{l.minUniformBufferOffsetAlignment,l.minStorageBufferOffsetAlignment,l.minTexelBufferOffsetAlignment,l.optimalBufferCopyOffsetAlignment,l.nonCoherentAtomSize,VkDeviceSize(16)})while(a<x)a<<=1;arena_alignment=a;}
    {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');std::string marker=(cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-sync-mode";FILE *m=fopen(marker.c_str(),"rb");sync_mode=false;if(m){sync_mode=fgetc(m)=='1';fclose(m);}}
    open_occlusion=OpenOcclusion{};occlusion_queries_started=0;
    {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');occlusion_batch_marker=(cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-query-batch";refresh_occlusion_batch();}
    {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');early_submit_marker=(cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-early-submit";refresh_early_submit();}
    xbg_logf("NVN occlusion batch enabled=%d",int(occlusion_batch_enabled));
     {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-defer-events").c_str(),"rb");defer_events=0;if(m){int c=fgetc(m);defer_events=c=='1'?1:c=='2'?2:0;fclose(m);}xbg_logf("NVN defer events=%d (marker gfx-defer-events)",int(defer_events));{FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-perf-sample").c_str(),"rb");perf_sample=8;if(m){if(fscanf(m,"%u",&perf_sample)!=1)perf_sample=8;fclose(m);}perf_frame_timed=perf_sample!=0;perf_timed_frames=0;xbg_logf("NVN perf timing every %u frames (marker gfx-perf-sample; 0 = off)",perf_sample);}}
    for(auto &sl:slots)sl=Slot{};slot_index=0;pass_prof_enabled=false;pass_prof_next=300;pass_prof_frame=UINT64_MAX;pass_prof_draws=0;pass_prof_queries=pass_prof_slots=pass_prof_skipped_slots=0;perf=Perf{};for(auto &c:perf_timing_before)c=XbgCpuTiming::Counter{};static_arena.cached=false;support_static.clear();format_property_cache.clear();for(auto &k:format_flat_known)k=false;
    VkPhysicalDeviceShaderDrawParametersFeatures draw_parameters{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES};VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};features.pNext=&draw_parameters;vkGetPhysicalDeviceFeatures2(cfg.physical,&features);if(cfg.enabled_shader_draw_parameters && !draw_parameters.shaderDrawParameters){if(error&&bytes)snprintf(error,bytes,"shaderDrawParameters unsupported");return -1;}
    VkCommandPoolCreateInfo cp{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cp.queueFamilyIndex=cfg.queue_family;cp.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;VkResult r=vkCreateCommandPool(cfg.device,&cp,nullptr,&command_pool);if(r!=VK_SUCCESS){if(error&&bytes)snprintf(error,bytes,"vkCreateCommandPool=%d",r);return r;}
    VkPipelineCacheCreateInfo cache{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};std::vector<uint8_t> cache_data=load_pipeline_cache();cache.initialDataSize=cache_data.size();cache.pInitialData=cache_data.empty()?nullptr:cache_data.data();r=vkCreatePipelineCache(cfg.device,&cache,nullptr,&pipeline_cache);if(r!=VK_SUCCESS){vkDestroyCommandPool(cfg.device,command_pool,nullptr);command_pool=VK_NULL_HANDLE;if(error&&bytes)snprintf(error,bytes,"vkCreatePipelineCache=%d",r);return r;}
    xbg_module_id_hook=warm::module_id;{std::string lp=warm::list_path();FILE *nw=fopen((lp.substr(0,lp.find_last_of('/'))+"/gfx-no-warm").c_str(),"rb");warm::enabled=!nw;if(nw)fclose(nw);}warm::start();
    wtrack::enabled=false;state=State{};state.color.logic_op=3;state.color.alpha_func=8;for(unsigned i=0;i<8;i++)state.blend[i]={i,2,1,2,1,1,1};stats.reset();timing=XbgCpuTiming(timing_clock);pipeline_counts={};pipeline_before={};pipeline_tick=0;timing_active=false;timing_first=timing_last=0;live_profile_used=false;descriptor_detail=false;descriptor_timing=XbgDescriptorTiming{};descriptor_root=nullptr;draw_detail=false;draw_timing=XbgDrawTiming{};draw_root=nullptr;completed=0;last_attachment_trace=UINT64_MAX;sampler_mirror_clamp_enabled=VK_FALSE;failed=false;error_text[0]=0;initialized=true;
    {std::string dir=cfg.shader_directory;size_t cut=dir.find_last_of('/');FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-sync-submit").c_str(),"rb");bool sync_submit=false;if(m){sync_submit=fgetc(m)=='1';fclose(m);}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-bindcache").c_str(),"rb");if(w){bind_cache_enabled=false;fclose(w);}}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-imgcache").c_str(),"rb");if(w){img_cache_enabled=false;fclose(w);}}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-pass-profile").c_str(),"rb");if(w){pass_prof_enabled=true;fclose(w);}}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-descmemo").c_str(),"rb");if(w){desc_memo_enabled=false;fclose(w);}xbg_logf("NVN descmemo enabled=%d",int(desc_memo_enabled));}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-dynpartial").c_str(),"rb");if(w){dyn_partial_enabled=false;fclose(w);}}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-xmemo").c_str(),"rb");if(w){xmemo_enabled=false;fclose(w);}xbg_logf("NVN cross-frame memo enabled=%d",int(xmemo_enabled));}{FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-no-wtrack").c_str(),"rb");if(w){wtrack::enabled=false;fclose(w);}xbg_logf("NVN wtrack enabled=%d",int(wtrack::enabled));{auto marker=[&](const char *name){FILE *w=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+name).c_str(),"rb");int c=-1;if(w){c=fgetc(w);fclose(w);}return c;};persistent_fb_enabled=marker("/gfx-no-persistent-fb")<0;slot_pools_enabled=marker("/gfx-no-slot-pools")<0;readback_cached=marker("/gfx-readback-cached")=='1';texel_memo_enabled=marker("/gfx-texel-memo")=='1';xbg_logf("NVN persistent framebuffers=%d slot command pools=%d cached readback=%d",int(persistent_fb_enabled),int(slot_pools_enabled),int(readback_cached));}}
     /* Android: no guest memory remapping probe. */
     {uint32_t n=0;vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&n,nullptr);std::vector<VkExtensionProperties> ext(n);vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&n,ext.data());
      std::string list;for(auto &e:ext){list+=e.extensionName;list+=' ';}
      format_feature_flags2=properties.apiVersion>=VK_API_VERSION_1_3;for(auto &e:ext)if(!strcmp(e.extensionName,VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME))format_feature_flags2=true;
      xbg_logf("NVN format-feature-flags2=%d (formatless storage checks %s)",int(format_feature_flags2),format_feature_flags2?"per format":"by the pre-1.3 format list");
      capability_report();
      {const VkFormatProperties &d16=format_properties(VK_FORMAT_D16_UNORM),&d24=format_properties(VK_FORMAT_X8_D24_UNORM_PACK32);
       const VkFormatFeatureFlags d24_needs=VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT|VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_FORMAT_FEATURE_TRANSFER_SRC_BIT|VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
       bool d16_linear=(d16.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)!=0;depth16_as_d24=!d16_linear && (d24.optimalTilingFeatures&d24_needs)==d24_needs;
       xbg_logf("NVN depth16 stored as %s (D16 linear filtering=%u, X8_D24 features=%#x)",depth16_as_d24?"X8_D24":"D16",unsigned(d16_linear),d24.optimalTilingFeatures);
       xbg_logf("NVN compatibility format_feature_flags2=%d depth16_as_d24=%d",int(format_feature_flags2),int(depth16_as_d24));}
      for(size_t at=0;at<list.size();at+=900)xbg_logf("NVN caps extensions[%zu] %s",at/900,list.substr(at,900).c_str());
      xbg_logf("NVN caps api=%u.%u.%u driver=%u maxUBdyn=%u maxPerStageUB=%u minUBAlign=%llu maxBoundSets=%u",VK_API_VERSION_MAJOR(properties.apiVersion),VK_API_VERSION_MINOR(properties.apiVersion),VK_API_VERSION_PATCH(properties.apiVersion),properties.driverVersion,properties.limits.maxDescriptorSetUniformBuffersDynamic,properties.limits.maxPerStageDescriptorUniformBuffers,(unsigned long long)properties.limits.minUniformBufferOffsetAlignment,properties.limits.maxBoundDescriptorSets);}
     {FILE *m=fopen(((cut==std::string::npos?dir:dir.substr(0,cut))+"/gfx-ab-toggle").c_str(),"rb");ab_period=0;if(m){if(fscanf(m,"%d",&ab_period)!=1)ab_period=0;fclose(m);}xbg_logf("NVN PERF ab_period=%d",ab_period);}
     ++slot_serial;gfx_enqueued=0;gfx_executed=0;gfx_presents_queued=0;gfx_async=false; /* Android game command memory remains guest-owned. */if(gfx_async)gfx_worker=std::thread(gfx_worker_loop);xbg_logf("NVN async translation=%d",int(gfx_async));}
    return 0;
}
// nvnEventGetValue on a deferred event: finish the GPU work the signal covers, then the value is published (apply_ready_events).
extern "C" void xbg_game_event_read(const void *addr){
    gfx_drain();
    std::lock_guard<std::recursive_mutex> lock(mutex);if(!initialized || failed)return;
    for(auto *list:{&pending_events,&recording_events})for(auto &e:*list)if(e.addr==uint64_t(reinterpret_cast<uintptr_t>(addr))){++event_reads;complete_all();return;}
}
extern "C" int xbg_game_set_sampler_mirror_clamp_to_edge(VkBool32 enabled){
    if(enabled!=VK_FALSE && enabled!=VK_TRUE)return -1;
    std::lock_guard<std::recursive_mutex> lock(mutex);if(!initialized || failed || stats.submissions || stats.draws || stats.dispatches)return -1;
    if(enabled){
        uint32_t count=0;VkResult result=vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&count,nullptr);if(result!=VK_SUCCESS)return result;
        std::vector<VkExtensionProperties> extensions(count);result=vkEnumerateDeviceExtensionProperties(cfg.physical,nullptr,&count,extensions.data());if(result!=VK_SUCCESS)return result;
        bool supported=false;for(const auto &e:extensions)if(!strcmp(e.extensionName,VK_KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_EXTENSION_NAME))supported=true;
        if(!supported && properties.apiVersion>=VK_API_VERSION_1_2){VkPhysicalDeviceVulkan12Features core{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};f.pNext=&core;vkGetPhysicalDeviceFeatures2(cfg.physical,&f);supported=core.samplerMirrorClampToEdge;}
        if(!supported)return -2;
    }
    sampler_mirror_clamp_enabled=enabled;xbg_logf("NVN sampler mirror-clamp-to-edge actualEnabled=%u",enabled);return 0;
}

extern "C" int xbg_game_provenance(const char *path){
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!path){provenance_close("CLOSE");return 0;}
    if(!initialized || provenance_file || capture.trace || timing_active || timing_first || failed || stats.presents.load()>UINT64_MAX-2)return -1;
    FILE *f=fopen(path,"wx");if(!f)return -2;
    provenance.arm(stats.presents.load());provenance_file=f;
    char begin[128];int n=snprintf(begin,sizeof(begin),"BEGIN target=%llu maxDraws=64 maxEvents=2048 maxText=262144\n",(unsigned long long)provenance.target);
    if(n<0 || size_t(n)>=sizeof(begin) || fwrite(begin,1,size_t(n),f)!=size_t(n) || fflush(f)){provenance_close("OPEN_IO_FAIL");return -2;}
    provenance.bytes=uint64_t(n);return 0;
}

extern "C" int xbg_game_capture_frame(const char *directory,uint64_t frame){
    if(!directory || !directory[0] || !frame || strlen(directory)>800)return -1;
    std::lock_guard<std::recursive_mutex> lock(mutex);if(!initialized || failed || stats.submissions || stats.draws || stats.dispatches || capture.trace || timing_first)return -1;
    std::string path=std::string(directory)+"/capture.txt";FILE *trace=fopen(path.c_str(),"wb");if(!trace)return -2;
    capture.directory=directory;capture.trace=trace;capture.frame=frame;capture.bytes=0;capture.events=0;capture.active=false;
    capture_log("CAPTURE version=1 frame=%llu maxEvents=16 maxBytes=%llu semantics=post-fence-raw-no-state-replacement",(unsigned long long)frame,(unsigned long long)capture_limit);
    xbg_logf("NVN one-frame capture frame=%llu directory=%s maxEvents=16 maxBytes=%llu",(unsigned long long)frame,directory,(unsigned long long)capture_limit);return 0;
}
extern "C" int xbg_game_profile_cpu_live(uint32_t frame_count){gfx_drain();
    if(!frame_count || frame_count>60)return -1;
    std::lock_guard<std::recursive_mutex> lock(mutex);
    const uint64_t current=stats.presents.load();
    if(!initialized || failed || live_profile_used || timing_active || timing_first || capture.trace || provenance_file || current>UINT64_MAX-uint64_t(frame_count)-1)return -1;
    // Requested from inside the present callback (a timing scope is open): arm at the next submit instead.
    if(timing.top){deferred_live_profile=frame_count;live_profile_used=true;return 0;}
    timing=XbgCpuTiming(timing_clock);timing_first=current+2;timing_last=timing_first+frame_count-1;live_profile_used=true;
    xbg_logf("NVN CPU_TIMING armed-live first=%llu count=%u maxSeconds=60 clock=steady elapsed-not-thread-cpu once=1",(unsigned long long)timing_first,frame_count);
    return 0;
}
extern "C" int xbg_game_profile_descriptor_live(uint32_t frame_count){gfx_drain();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // Reuse exact once/pending/active/nonempty-scope/provenance/capture checks.
    int result=xbg_game_profile_cpu_live(frame_count);if(result)return result;
    descriptor_timing=XbgDescriptorTiming{};descriptor_detail=true;
    xbg_logf("NVN DESC_TIMING armed v=2 first=%llu count=%u once=1 maxSeconds=60",(unsigned long long)timing_first,frame_count);
    return 0;
}
extern "C" int xbg_game_profile_draw_live(uint32_t frame_count){gfx_drain();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // Reuse the original live once/pending/active/provenance/capture gates.
    int result=xbg_game_profile_cpu_live(frame_count);if(result)return result;
    draw_timing=XbgDrawTiming{};draw_detail=true;
    xbg_logf("NVN DRAW_TIMING armed v=1 first=%llu count=%u once=1 maxSeconds=60",(unsigned long long)timing_first,frame_count);
    return 0;
}
extern "C" int xbg_game_profile_cpu(uint64_t first_frame,uint32_t frame_count){
    if(!first_frame || !frame_count || frame_count>120 || first_frame>UINT64_MAX-frame_count)return -1;
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(!initialized || failed || stats.submissions || stats.draws || stats.dispatches || stats.presents || timing_first || capture.trace)return -1;
    timing=XbgCpuTiming(timing_clock);timing_first=first_frame;timing_last=first_frame+frame_count-1;
    xbg_logf("NVN CPU_TIMING armed first=%llu count=%u maxSeconds=60 clock=steady elapsed-not-thread-cpu",(unsigned long long)first_frame,frame_count);
    if(first_frame==1)timing_start();return 0;
}
extern "C" int xbg_game_shutdown(){gfx_drain();gfx_async_stop();std::lock_guard<std::recursive_mutex> lock(mutex);if(!initialized)return 0;timing_report("shutdown-partial");provenance_close("SHUTDOWN");resolve_pending();submit_slots(true);save_pipeline_cache();pending_writebacks.clear();pending_reports.clear();pending_events.clear();recording_events.clear();epochs.clear();VkResult r=vkQueueWaitIdle(cfg.queue);if(r!=VK_SUCCESS)return r;destroy_slots();clear_pipeline_objects();warm::shutdown();clear_descriptor_caches();clear_render_pass_cache();images.clear();++images_epoch;for(auto &f:persistent_framebuffers)vkDestroyFramebuffer(cfg.device,f.second,nullptr);persistent_framebuffers.clear();image_index.clear();image_index_dirty=true;program_cache.clear();shaders.clear();textures.clear();texture_pool_cache=nullptr;samplers.clear();null_textures.clear();texture_pool_cache=nullptr;pools.clear();host_segments.clear();host_ranges.clear();host_bytes=0;cmd_range_lo=0;cmd_range_bytes=0;vkDestroyPipelineCache(cfg.device,pipeline_cache,nullptr);pipeline_cache=VK_NULL_HANDLE;vkDestroyCommandPool(cfg.device,command_pool,nullptr);command_pool=VK_NULL_HANDLE;initialized=false;return 0;}
extern "C" void xbg_game_get_stats(xbg_game_stats *out){if(out)*out={stats.submissions.load(),stats.draws.load(),stats.dispatches.load(),stats.presents.load(),stats.shader_loads.load()};}
static void xbg_legacy_pool_add_now(void *p,size_t n){std::lock_guard<std::recursive_mutex> lock(mutex);need(p&&n,"Invalid NVN pool");pools[uintptr_t(p)]=n;cmd_range_lo=0;cmd_range_bytes=0;auto epoch=provenance.pool(uint64_t(p),true);provenance_log("POOL_ADD mem=%p bytes=%zu epoch=%llu",p,n,epoch);}
static void xbg_legacy_pool_del_now(void *p){std::lock_guard<std::recursive_mutex> lock(mutex);provenance_log("POOL_DEL mem=%p epoch=%llu",p,provenance.epoch(uint64_t(p)));provenance.pool(uint64_t(p),false);pools.erase(uintptr_t(p));cmd_range_lo=0;cmd_range_bytes=0;}
static void xbg_register_texture_now(const dpool_t *p,int id,const tex_t *t,const view_t *v){std::lock_guard<std::recursive_mutex> lock(mutex);need(p&&id>=0&&id<p->n&&t,"Invalid texture registration");auto &slot=textures[pool_key(p)][id];if(register_texture_record(slot,*t,v))++memo_bump[0];}
static void xbg_register_sampler_now(const dpool_t *p,int id,const samp_par *s){std::lock_guard<std::recursive_mutex> lock(mutex);need(p&&id>=0&&id<p->n&&s,"Invalid sampler registration");auto &slot=samplers[pool_key(p)][id];if(memcmp(&slot,s,sizeof(*s))){++memo_epoch;++memo_bump[1];}slot=*s;}
static void xbg_legacy_tex_release_now(tex_t *t){std::lock_guard<std::recursive_mutex> lock(mutex);if(t){auto i=images.find(t->serial);if(i!=images.end()){provenance_image("RELEASE",t,*i->second);retire_object(i->second);++memo_bump[2];}images.erase(t->serial);image_index_dirty=true;++images_epoch;}}
static void xbg_legacy_prog_release_now(prog_t *p){/* Immutable shader modules are keyed by original control digest, shared safely. */std::lock_guard<std::recursive_mutex> lock(mutex);if(p && p->serial)program_cache.erase(p->serial);}
static void xbg_texture_invalidate_now(const tex_t *t){std::lock_guard<std::recursive_mutex> lock(mutex);if(t){auto i=images.find(t->serial);if(i!=images.end()){need(!i->second->gpu_written,"CPU invalidation of GPU-written texture needs explicit range merge");i->second->uploaded=false;}}}
extern "C" void *xbg_host_command_memory(cmd_t *owner,size_t n){gfx_drain();std::lock_guard<std::recursive_mutex> lock(mutex);need(n<=64*1024*1024 && host_bytes+n<=256ull*1024*1024,"Host command memory cap reached; finalize handles before reusing");auto p=std::make_unique<uint8_t[]>(n);void *out=p.get();host_ranges[uintptr_t(out)]=n;host_segments.push_back({owner,std::move(p),n});host_bytes+=n;return out;}
extern "C" void xbg_release_command_handle(uint64_t handle){
    // A handle can CALL another handle and game-supplied segments can point to
    // host segments. Retain until shutdown until reference ownership is proven.
    (void)handle;
}
extern "C" void xbg_copy_commands(cmd_t *command,int count,const uint64_t *handles){gfx_drain();
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto available=[](void *,const void *p)->size_t{return command_available(static_cast<const uint8_t*>(p));};
    size_t bytes=0;int result=xbg_command_clone(count,handles,nullptr,256ull*1024*1024,available,nullptr,&bytes);
    if(result)xbg_fatalf("CopyCommands invalid source chain: %d",result);
    std::vector<uint8_t> snapshot(bytes);size_t actual=0;
    result=xbg_command_clone(count,handles,snapshot.data(),snapshot.size(),available,nullptr,&actual);
    need(result==0 && actual==bytes,"CopyCommands source changed while recording");
    for(size_t offset=0;offset<bytes;){op_hdr h;memcpy(&h,snapshot.data()+offset,8);xbg_record_command(command,snapshot.data()+offset,h.size);offset+=h.size;}
}
extern "C" void xbg_invoke_command_callback(cmd_t *c,size_t n){uint64_t args[]={uint64_t(c),NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY,n,uint64_t(c->cb_data)};if(cfg.invoke_guest)cfg.invoke_guest(cfg.userdata,reinterpret_cast<void*>(c->cb),args,4);else c->cb(reinterpret_cast<NVNcommandBuffer*>(c),NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY,n,c->cb_data);}
static void xbg_submit_now(int n,const uint64_t *handles){std::lock_guard<std::recursive_mutex> lock(mutex);
    if(deferred_live_profile && !timing.top){uint64_t current=stats.presents.load();timing=XbgCpuTiming(timing_clock);timing_first=current+2;timing_last=timing_first+deferred_live_profile-1;deferred_live_profile=0;}
   TimingRoot timing_root;CPU_SCOPE(Other);need(initialized && !failed,"Game Vulkan backend not initialized/healthy");uint64_t begin=timing_clock();snapshot_reset();uint64_t budget=0;for(int i=0;i<n;i++)execute(reinterpret_cast<const uint8_t*>(handles[i]),0,budget);++stats.submissions;++submitted;if(sync_mode)complete_all();perf.submit_cpu_ns+=timing_clock()-begin;}
extern "C" uint64_t xbg_legacy_last_fence(){return gfx_async?gfx_enqueued.load():[]{std::lock_guard<std::recursive_mutex> lock(mutex);return submitted;}();}
extern "C" int xbg_legacy_wait(uint64_t fence,uint64_t timeout){
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if(failed || fence>submitted)return 0;if(fence<=completed)return 1;
    SiteScope site(SITE_LEGACY_WAIT);uint64_t w0=timing_clock();struct Spent{uint64_t w0;~Spent(){legacy_wait_ns+=timing_clock()-w0;}}spent{w0};
    submit_slots(false);auto begin=std::chrono::steady_clock::now();
    for(auto &slot:slots)if(slot.pending){uint64_t remaining=timeout;
        if(timeout!=UINT64_MAX){auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();remaining=uint64_t(elapsed)>=timeout?0:timeout-uint64_t(elapsed);}
        bool busy=vkGetFenceStatus(cfg.device,slot.fence)==VK_NOT_READY;uint64_t b0=busy?timing_clock():0;
        VkResult r=vkWaitForFences(cfg.device,1,&slot.fence,VK_TRUE,remaining);if(busy){block_ns[SITE_LEGACY_WAIT]+=timing_clock()-b0;++block_calls[SITE_LEGACY_WAIT];}
        if(r==VK_TIMEOUT)return 0;VK(r);retire_slot(slot);
    }
    resolve_known_reports();if(pending_writebacks.empty()&&pending_reports.empty())completed=submitted;
    return fence<=completed;
}
extern "C" void xbg_legacy_finish(){gfx_drain();std::lock_guard<std::recursive_mutex> lock(mutex);need(initialized,"Queue finish before Vulkan initialization");++perf.why_finish;complete_all();VK(vkQueueWaitIdle(cfg.queue));}
static void xbg_present_texture_now(const win_t *window,const tex_t *texture){std::lock_guard<std::recursive_mutex> lock(mutex);{SiteScope site(SITE_PRESENT);TimingRoot timing_root;CPU_SCOPE(Other);need(initialized&&cfg.present,"Game presentation callback not installed");uint64_t present_begin=timing_clock();Image &im=get_image(texture,false);need(im.gpu_written,"Game tried to present a texture never rendered");Commands c;im.transition(c.command,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);c.finish();submit_slots(false);int r;{CPU_SCOPE(Present);r=cfg.present(cfg.userdata,window->native,im.image,im.fmt,im.width,im.height,window->crop,window->interval);}if(r)xbg_fatalf("Game presentation failed: %d",r);++stats.presents;if(pass_prof_enabled && stats.presents.load()==pass_prof_frame)xbg_logf("NVN GPASS boundary frame=%llu slots=%u skipped_slots=%u reserved_queries=%u max_queries=%u max_slots=%u",(unsigned long long)pass_prof_frame,pass_prof_slots,pass_prof_skipped_slots,pass_prof_queries,pass_prof_query_limit,pass_prof_slot_limit);perf.present_cpu_ns+=timing_clock()-present_begin;perf_frame();if(provenance_file && stats.presents.load()>=provenance.target)provenance_close("END");if(capture.trace && stats.presents==capture.frame){capture_log("END frame=%llu events=%u bytes=%llu presentedSerial=%llu image=%p",(unsigned long long)capture.frame,capture.events,(unsigned long long)capture.bytes,(unsigned long long)texture->serial,(void*)im.image);fclose(capture.trace);capture.trace=nullptr;capture.active=false;xbg_logf("NVN one-frame capture complete frame=%llu events=%u bytes=%llu",(unsigned long long)capture.frame,capture.events,(unsigned long long)capture.bytes);}}timing_boundary();}
extern "C" void xbg_legacy_read_texels(const tex_t *t,unsigned level,unsigned layer,const NVNcopyRegion *r,uint8_t *dst,size_t row,size_t image){gfx_drain();std::lock_guard<std::recursive_mutex> lock(mutex);need(t&&r&&dst,"Invalid ReadTexels arguments");auto found=images.find(t->serial);if(found==images.end() || !found->second->gpu_written){int bw,bh,bpp=nvn_fmt_bpb(t->d.format,&bw,&bh);need(bpp && r->xoffset==0 && r->yoffset==0 && r->zoffset==0 && r->depth==1 && uint32_t(r->width)==std::max(1u,t->d.width>>level) && uint32_t(r->height)==std::max(1u,t->d.height>>level),"CPU ReadTexels partial region unsupported");need(!(t->d.flags&0x10),"Linear CPU ReadTexels not implemented");need(nvn_deswizzle(&t->d,level,layer,t->mem,dst,row)==0,"ReadTexels deswizzle failed");}
    else{p_copy_b2t p{};p.src=uint64_t(dst);p.tex=t;p.has_view=1;p.view.set_levels=p.view.set_layers=1;p.view.base_level=level;p.view.nlevels=1;p.view.min_layer=layer;p.view.nlayers=1;p.r=*r;int64_t old=state.copy_stride,old_image=state.copy_image_stride;state.copy_stride=row;state.copy_image_stride=image;copy_texture_buffer(p,true);state.copy_stride=old;state.copy_image_stride=old_image;}}

// ---- public entry points (queued in order on the graphics thread) ----
extern "C" void xbg_legacy_pool_add(void *p,size_t n){gfx_enqueue([p,n]{xbg_legacy_pool_add_now(p,n);});}
extern "C" void xbg_legacy_pool_del(void *p){gfx_enqueue([p]{xbg_legacy_pool_del_now(p);});}
extern "C" void xbg_register_texture(const dpool_t *p,int id,const tex_t *t,const view_t *v){
    if(!gfx_async){xbg_register_texture_now(p,id,t,v);return;}
    need(p&&t,"Invalid texture registration");dpool_t pool=*p;tex_t tex=*t;view_t view{};bool has=v!=nullptr;if(v)view=*v;
    gfx_enqueue([pool,id,tex,view,has]{xbg_register_texture_now(&pool,id,&tex,has?&view:nullptr);});}
extern "C" void xbg_register_sampler(const dpool_t *p,int id,const samp_par *s){
    if(!gfx_async){xbg_register_sampler_now(p,id,s);return;}
    need(p&&s,"Invalid sampler registration");dpool_t pool=*p;samp_par sp=*s;gfx_enqueue([pool,id,sp]{xbg_register_sampler_now(&pool,id,&sp);});}
extern "C" void xbg_legacy_tex_release(tex_t *t){if(!t)return;if(!gfx_async){xbg_legacy_tex_release_now(t);return;}tex_t copy=*t;gfx_enqueue([copy]()mutable{xbg_legacy_tex_release_now(&copy);});}
extern "C" void xbg_legacy_prog_release(prog_t *p){if(!p)return;if(!gfx_async){xbg_legacy_prog_release_now(p);return;}prog_t copy=*p;gfx_enqueue([copy]()mutable{xbg_legacy_prog_release_now(&copy);});}
extern "C" void xbg_texture_invalidate(const tex_t *t){if(!t)return;if(!gfx_async){xbg_texture_invalidate_now(t);return;}tex_t copy=*t;gfx_enqueue([copy]{xbg_texture_invalidate_now(&copy);});}
extern "C" void xbg_submit(int n,const uint64_t *handles){
    if(!gfx_async){xbg_submit_now(n,handles);return;}
    std::vector<uint64_t> list(handles,handles+n);++gfx_enqueued;
    gfx_enqueue([list]{xbg_submit_now(int(list.size()),list.data());gfx_executed.store(submitted);});}
extern "C" void xbg_present_texture(const win_t *window,const tex_t *texture){
    if(!gfx_async){xbg_present_texture_now(window,texture);return;}
    // At most two frames queued ahead of the graphics thread.
    {uint64_t t0=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());std::unique_lock<std::mutex> l(gfx_job_mutex);gfx_idle_cv.wait(l,[]{return gfx_presents_queued.load()<2;});gfx_pace_ns+=uint64_t(std::chrono::steady_clock::now().time_since_epoch().count())-t0;}
    win_t w=*window;tex_t t=*texture;++gfx_presents_queued;
    gfx_enqueue([w,t]{xbg_present_texture_now(&w,&t);{std::lock_guard<std::mutex> l(gfx_job_mutex);--gfx_presents_queued;}gfx_idle_cv.notify_all();});}

extern "C" void xbg_legacy_register_texture(int id,const tex_t *t,const view_t *v){std::lock_guard<std::recursive_mutex> l(mutex);need(id>=0&&t,"Invalid legacy texture registration");if(register_texture_record(legacy_textures[id],*t,v))++memo_bump[0];}
extern "C" void xbg_legacy_register_sampler(int id,const samp_par *s){std::lock_guard<std::recursive_mutex> l(mutex);need(id>=0&&s,"Invalid legacy sampler registration");legacy_samplers[id]=*s;++memo_epoch;++memo_bump[1];}
