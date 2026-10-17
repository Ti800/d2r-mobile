// 真C++改写器回归与仓外XGS审计入口：合成SPIR-V，不链接Vulkan、不启动游戏。
// Real C++ rewriter regression and external XGS audit entry: synthetic SPIR-V, no Vulkan linkage or game launch.
#include "../xbg_spirv_quad.h"
#include <algorithm>
#include <array>
#include <cassert>
#ifdef NDEBUG
#error "Quad regression tests require active assertions; compile with -UNDEBUG."
#endif
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace fs=std::filesystem;
using Words=std::vector<uint32_t>;
using Quad=std::array<float,4>;
enum class Mode { numeric,source_bits,source_bits_copy,result_bits,result_bits_copy,integer_use,memory_use,branch,scope,xor4,index,existing_coord };
static void add(Words &v,uint32_t op,std::initializer_list<uint32_t> args) {
    v.push_back((uint32_t(args.size()+1)<<16)|op);v.insert(v.end(),args.begin(),args.end());
}
static Words fixture(unsigned mask,Mode mode=Mode::numeric) {
    Words w={0x07230203,0x00010300,0,400,0};
    add(w,17,{1});add(w,17,{61});add(w,17,{65});add(w,14,{0,1});
    if(mode==Mode::existing_coord)add(w,15,{4,100,0x6e69616d,0,10,11,12,19});
    else if(mode==Mode::source_bits || mode==Mode::source_bits_copy)add(w,15,{4,100,0x6e69616d,0,10,11,12,19});
    else add(w,15,{4,100,0x6e69616d,0,10,11,12});
    add(w,16,{100,7});add(w,71,{10,11,41});add(w,71,{10,14});add(w,71,{11,30,0});add(w,71,{12,30,0});
    if(mode==Mode::existing_coord)add(w,71,{19,11,15});
    if(mode==Mode::source_bits || mode==Mode::source_bits_copy) {add(w,71,{19,30,1});add(w,71,{19,14});}
    add(w,19,{1});add(w,21,{2,32,0});add(w,22,{3,32});add(w,23,{4,3,4});add(w,20,{5});
    add(w,32,{6,1,2});add(w,32,{7,1,3});add(w,32,{8,3,3});add(w,33,{9,1});
    if(mode==Mode::existing_coord)add(w,32,{16,1,4});
    if(mode==Mode::memory_use)add(w,32,{16,7,3});
    add(w,59,{6,10,1});add(w,59,{7,11,1});add(w,59,{8,12,3});
    if(mode==Mode::existing_coord)add(w,59,{16,19,1});
    if(mode==Mode::source_bits || mode==Mode::source_bits_copy)add(w,59,{6,19,1});
    add(w,43,{2,13,mode==Mode::scope?2u:3u});add(w,43,{2,14,mode==Mode::xor4?4u:mask});add(w,41,{5,20});
    add(w,54,{1,100,0,9});add(w,248,{101});
    if(mode==Mode::memory_use)add(w,59,{16,19,7});
    add(w,61,{2,102,10});add(w,198,{2,103,102,14});
    if(mode==Mode::source_bits || mode==Mode::source_bits_copy) {
        add(w,61,{2,116,19});add(w,124,{3,mode==Mode::source_bits?104u:117u,116});
        if(mode==Mode::source_bits_copy)add(w,83,{3,104,117});
    } else add(w,61,{3,104,11});
    if(mode==Mode::branch) {add(w,247,{111,0});add(w,250,{20,112,113});add(w,248,{112});}
    add(w,345,{3,105,13,104,mode==Mode::index?102u:103u});
    add(w,62,{12,105});
    if(mode==Mode::result_bits)add(w,124,{2,118,105});
    if(mode==Mode::result_bits_copy) {add(w,83,{3,117,105});add(w,124,{2,118,117});}
    if(mode==Mode::integer_use)add(w,109,{2,118,105});
    if(mode==Mode::memory_use)add(w,62,{19,105});
    if(mode==Mode::branch) {add(w,249,{111});add(w,248,{113});add(w,249,{111});add(w,248,{111});}
    add(w,253,{});add(w,56,{});return w;
}

// 在四个活跃lane上执行真实输出指令，检查对角项和非有限回退，不模拟GPU辅助调用。
// Execute the actual emitted instructions over four active lanes to check diagonals and nonfinite fallback, without pretending to simulate GPU helpers.
static Quad evaluate(const Words &w,const Quad &input) {
    std::map<uint32_t,Quad> f;
    std::map<uint32_t,std::array<uint32_t,4>> u;
    std::map<uint32_t,std::array<bool,4>> b;
    std::map<uint32_t,std::array<Quad,4>> vectors;
    Quad result={};
    for(size_t at=5;at<w.size();) {
        auto *p=w.data()+at;unsigned wc=p[0]>>16,op=p[0]&65535;
        if(op==43 && p[1]==2)u[p[2]].fill(p[3]);
        if(op==61) {
            if(p[1]==2)u[p[2]]={0,1,2,3};
            else if(p[1]==3)f[p[2]]=input;
            else for(unsigned lane=0;lane<4;++lane)vectors[p[2]][lane]={float(lane&1)+.5f,float(lane>>1)+.5f,0,1};
        }
        if(op==198 || op==199)for(unsigned lane=0;lane<4;++lane)u[p[2]][lane]=op==198?(u[p[3]][lane]^u[p[4]][lane]):(u[p[3]][lane]&u[p[4]][lane]);
        if(op==171)for(unsigned lane=0;lane<4;++lane)b[p[2]][lane]=u[p[3]][lane]!=u[p[4]][lane];
        if(op==345)for(unsigned lane=0;lane<4;++lane)f[p[2]][lane]=f[p[4]][u[p[5]][lane]&3];
        if(op==81)for(unsigned lane=0;lane<4;++lane)f[p[2]][lane]=vectors[p[3]][lane][p[4]];
        if(op==109)for(unsigned lane=0;lane<4;++lane)u[p[2]][lane]=uint32_t(f[p[3]][lane]);
        if(op==210) {auto q=f[p[3]];f[p[2]]={q[1]-q[0],q[1]-q[0],q[3]-q[2],q[3]-q[2]};}
        if(op==211) {auto q=f[p[3]];f[p[2]]={q[2]-q[0],q[3]-q[1],q[2]-q[0],q[3]-q[1]};}
        if(op==127)for(unsigned lane=0;lane<4;++lane)f[p[2]][lane]=-f[p[3]][lane];
        if(op==129)for(unsigned lane=0;lane<4;++lane)f[p[2]][lane]=f[p[3]][lane]+f[p[4]][lane];
        if(op==156 || op==157)for(unsigned lane=0;lane<4;++lane)b[p[2]][lane]=op==156?std::isnan(f[p[3]][lane]):std::isinf(f[p[3]][lane]);
        if(op==166)for(unsigned lane=0;lane<4;++lane)b[p[2]][lane]=b[p[3]][lane]||b[p[4]][lane];
        if(op==169)for(unsigned lane=0;lane<4;++lane)f[p[2]][lane]=b[p[3]][lane]?f[p[4]][lane]:f[p[5]][lane];
        if(op==62 && p[1]==12)result=f[p[2]];
        at+=wc;
    }
    return result;
}
static void save(const fs::path &path,const Words &words) {
    std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char *>(words.data()),std::streamsize(words.size()*4));
    if(!file)throw std::runtime_error("cannot write synthetic SPIR-V");
}
static void run_tests(const fs::path &output={}) {
    unsigned cases=0;
    if(!output.empty())fs::create_directories(output);
    for(unsigned mask=1;mask<=3;++mask) {
        auto original=fixture(mask);Words rewritten;xbg_spirv_quad_stats stats;
        assert(xbg_spirv_quad_derivatives(original.data(),original.size(),rewritten,&stats)==1);
        assert(stats.candidates==1 && stats.rewritten==1 && stats.guarded_nonfinite==1);
        Quad values={1,4,7,20};auto actual=evaluate(rewritten,values);
        for(unsigned lane=0;lane<4;++lane)assert(actual[lane]==values[lane^mask]);
        if(mask==3) {
            size_t horizontal=0,vertical=0;uint32_t dy_source=0;
            for(size_t at=5;at<rewritten.size();at+=rewritten[at]>>16) {
                unsigned op=rewritten[at]&65535;
                if(op==210)horizontal=at;if(op==211) {vertical=at;dy_source=rewritten[at+3];}
            }
            assert(horizontal && vertical>horizontal && dy_source!=104);
        }
        Quad exceptional={INFINITY,2,NAN,-INFINITY};auto fallback=evaluate(rewritten,exceptional);
        for(unsigned lane=0;lane<4;++lane)assert(std::isnan(exceptional[lane^mask])?std::isnan(fallback[lane]):fallback[lane]==exceptional[lane^mask]);
        if(!output.empty()) {save(output/(("xor"+std::to_string(mask))+"-before.spv"),original);save(output/(("xor"+std::to_string(mask))+"-after.spv"),rewritten);}
        ++cases;
    }
    struct Rejection {Mode mode;xbg_quad_skip reason;const char *name;};
    for(auto test:{Rejection{Mode::source_bits,xbg_quad_skip::bit_pattern,"source_bits"},
        {Mode::source_bits_copy,xbg_quad_skip::bit_pattern,"transitive_source_bits"},
        {Mode::result_bits,xbg_quad_skip::bit_pattern,"result_bits"},
        {Mode::result_bits_copy,xbg_quad_skip::bit_pattern,"transitive_result_bits"},
        {Mode::integer_use,xbg_quad_skip::integer_flow,"integer_use"},
        {Mode::memory_use,xbg_quad_skip::memory_flow,"private_memory"},
        {Mode::branch,xbg_quad_skip::control_flow,"conditional_branch"},
        {Mode::scope,xbg_quad_skip::lane_index,"wrong_scope"},
        {Mode::xor4,xbg_quad_skip::lane_index,"cross_quad"},
        {Mode::index,xbg_quad_skip::lane_index,"unproven_index"}}) {
        auto original=fixture(3,test.mode);Words rewritten={1,2,3};xbg_spirv_quad_stats stats;
        assert(!xbg_spirv_quad_derivatives(original.data(),original.size(),rewritten,&stats));
        assert(rewritten.empty() && stats.candidates==1 && stats.skipped[size_t(test.reason)]==1);
        if(!output.empty())save(output/(std::string(test.name)+"-unchanged.spv"),original);
        ++cases;
    }
    auto existing=fixture(3,Mode::existing_coord);Words out;xbg_spirv_quad_stats stats;
    assert(xbg_spirv_quad_derivatives(existing.data(),existing.size(),out,&stats)==1);++cases;
    auto invalid=fixture(3);invalid[5]=0;assert(!xbg_spirv_quad_derivatives(invalid.data(),invalid.size(),out,&stats));assert(out.empty());++cases;
    assert(!xbg_spirv_quad_derivatives(nullptr,0,out,&stats));assert(stats.skipped[size_t(xbg_quad_skip::malformed)]==1);++cases;
    auto ordinary=fixture(1);assert(xbg_spirv_quad_derivatives(ordinary.data(),ordinary.size(),out)==1);++cases;
    auto rounded=evaluate(out,{1e20f,1,1e20f,1});
    assert(rounded[0]!=1);++cases;
    std::cout<<"PASS quad_rewriter cases="<<cases<<"; active-lane arithmetic only; finite_rounding_limit_confirmed=1; device_helpers=NOT_TESTED\n";
}
static void audit(const fs::path &directory,const fs::path &output={}) {
    size_t modules=0,total=0;std::vector<fs::path> files;
    for(const auto &entry:fs::directory_iterator(directory))if(entry.path().extension()==".xgs")files.push_back(entry.path());
    std::sort(files.begin(),files.end());if(!output.empty())fs::create_directories(output);
    for(const auto &path:files) {
        std::ifstream file(path,std::ios::binary);std::vector<char> raw((std::istreambuf_iterator<char>(file)),{});
        if(raw.size()<32)throw std::runtime_error("short XGS");
        uint32_t h[8];std::memcpy(h,raw.data(),32);size_t offset=32+size_t(h[3])*32+size_t(h[4])*4+h[5];
        if(h[0]!=0x31534758 || h[1]!=1 || h[6]%4 || offset>raw.size() || h[6]!=raw.size()-offset)throw std::runtime_error("invalid XGS extent");
        Words before(h[6]/4),after;std::memcpy(before.data(),raw.data()+offset,h[6]);xbg_spirv_quad_stats stats;
        total+=xbg_spirv_quad_derivatives(before.data(),before.size(),after,&stats);++modules;
        std::cout<<"{\"shader\":\""<<path.filename().string()<<"\",\"stage\":"<<h[2]<<",\"candidates\":"<<stats.candidates<<",\"rewritten\":"<<stats.rewritten<<",\"skipped\":{";
        bool first=true;
        for(size_t i=0;i<size_t(xbg_quad_skip::count);++i)if(stats.skipped[i]) {if(!first)std::cout<<',';first=false;std::cout<<'"'<<xbg_spirv_quad_skip_name(xbg_quad_skip(i))<<"\":"<<stats.skipped[i];}
        std::cout<<"}}\n";
        if(!output.empty() && stats.candidates)save(output/(path.stem().string()+".spv"),after.empty()?before:after);
    }
    std::cerr<<"AUDIT modules="<<modules<<" rewritten="<<total<<"\n";
}
int main(int argc,char **argv) {
    try {
        if(argc>=3 && std::string(argv[1])=="--audit")audit(argv[2],argc>=4?fs::path(argv[3]):fs::path{});
        else if(argc==3 && std::string(argv[1])=="--emit-fixtures")run_tests(argv[2]);
        else if(argc==1)run_tests();
        else {std::cerr<<"usage: quad_test [--emit-fixtures dir | --audit xgs-dir [spv-dir]]\n";return 2;}
        return 0;
    } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
