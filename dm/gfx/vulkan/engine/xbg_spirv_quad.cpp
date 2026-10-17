// PR14 a90826a的有界修订：只改已证明lane XOR 1..3、直线主函数、无整数/位用途的32位浮点。
// XOR3先求水平邻居n1，再取dFdyFine(n1)，避免原补丁遗漏混合二阶项。
// 非有限值保留原shuffle回退；有限数值仍受IEEE舍入影响，须另验截图，不宣称位精确。
// Bounded correction of PR14 a90826a: require proven lane XOR 1..3, a straight-line entry function, and numeric float32 data without bit/integer uses.
// XOR3 first takes horizontal neighbour n1, then dFdyFine(n1), preserving the mixed second-order term missing from the original patch.
// Keep the original shuffle as a nonfinite fallback; IEEE rounding remains for finite values and requires separate screenshot validation, not a bit-exact claim.
// 规范依据 | Specification: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
#include "xbg_spirv_quad.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace {
enum : uint32_t {
    ExtInst=12, EntryPoint=15, ExecutionMode=16, Capability=17, TypeVoid=19, TypeBool=20, TypeInt=21,
    TypeFloat=22, TypeVector=23, TypeMatrix=24, TypePointer=32, Constant=43, ConstantComposite=44,
    Function=54, FunctionEnd=56, FunctionCall=57, Variable=59, Load=61, Store=62, AccessChain=65,
    InBoundsAccessChain=66, Decorate=71, VectorShuffle=79, CompositeConstruct=80, CompositeExtract=81,
    CompositeInsert=82, CopyObject=83, ConvertFToU=109, Bitcast=124, FNegate=127, FAdd=129,
    IsNan=156, IsInf=157, LogicalOr=166, Select=169, INotEqual=171, BitwiseXor=198,
    BitwiseAnd=199, DPdxFine=210, DPdyFine=211, Label=248, Branch=249, BranchConditional=250,
    Switch=251, Kill=252, Return=253, ReturnValue=254, Unreachable=255, Shuffle=345,
    Input=1, Output=3, Uniform=2, PushConstant=9, BuiltIn=11, FragCoord=15, Lane=41, Fragment=4
};
struct Instruction { size_t offset; uint32_t op,wc,type=0,id=0,fn=0; };
struct Type { uint32_t op=0,a=0,b=0; };
uint32_t word(uint32_t op,uint32_t wc) { return op|(wc<<16); }
struct Module {
    const uint32_t *w; size_t n; uint32_t bound,entry_fn=0,entry_count=0,lane_var=0,coord_var=0;
    uint32_t t_float=0,t_uint=0,t_bool=0,t_vec4=0,t_ptr4=0;
    size_t entry_offset=0,first_type=0,first_func=0,first_cap=0;
    bool fragment=true,derivatives=false,origin_upper=false,valid=true;
    std::vector<Instruction> instructions;
    std::vector<Type> types;
    std::vector<size_t> definitions;
    std::vector<std::vector<size_t>> uses;
    std::vector<uint32_t> unsafe_functions;
    std::vector<uint32_t> glsl_sets;
    explicit Module(const uint32_t *p,size_t count):w(p),n(count),bound(p[3]),types(bound),definitions(bound,SIZE_MAX),uses(bound) {}
    const Instruction *def(uint32_t id) const { return id<bound && definitions[id]!=SIZE_MAX?&instructions[definitions[id]]:nullptr; }
    bool float_type(uint32_t id,unsigned depth=0) const {
        if(!id || id>=bound || depth>32)return false;
        const Type &t=types[id];
        return t.op==TypeFloat || ((t.op==TypeVector || t.op==TypeMatrix) && float_type(t.a,depth+1));
    }
    bool int_type(uint32_t id,unsigned depth=0) const {
        if(!id || id>=bound || depth>32)return false;
        const Type &t=types[id];return t.op==TypeInt || (t.op==TypeVector && int_type(t.a,depth+1));
    }
    bool constant(uint32_t id,uint32_t &value) const {
        const auto *i=def(id);
        if(!i || i->op!=Constant || i->wc!=4 || !int_type(i->type))return false;
        value=w[i->offset+3];return true;
    }
    uint32_t pointer_root(uint32_t id,unsigned depth=0) const {
        if(depth>32)return 0;
        const auto *i=def(id);if(!i)return 0;
        if(i->op==Variable)return id;
        if((i->op==AccessChain || i->op==InBoundsAccessChain || i->op==CopyObject) && i->wc>=4)
            return pointer_root(w[i->offset+3],depth+1);
        return 0;
    }
    uint32_t storage(uint32_t ptr) const {
        const auto *i=def(pointer_root(ptr));return i && i->wc>=4?w[i->offset+3]:UINT32_MAX;
    }
    bool lane_value(uint32_t id,unsigned depth=0) const {
        if(depth>16)return false;
        const auto *i=def(id);if(!i)return false;
        const auto *p=w+i->offset;
        if(i->op==Load && i->wc==4 && p[3]==lane_var)return true;
        return (i->op==Bitcast || i->op==CopyObject) && i->wc==4 && int_type(i->type) && lane_value(p[3],depth+1);
    }
};

// 只列实际ID操作数；未知指令宁可把字面量当ID多跳过，也不漏过整数消费者。
// Enumerate actual ID operands where known; unknown instructions may over-classify literals as IDs to avoid missing integer consumers.
std::vector<uint32_t> operands(const Module &m,const Instruction &i) {
    const auto *p=m.w+i.offset;std::vector<uint32_t> ids;
    auto tail=[&](unsigned first){for(unsigned j=first;j<i.wc;++j)ids.push_back(p[j]);};
    if(i.op==Constant || i.op==Variable || i.op==Label || i.op==Function || i.op==FunctionEnd)return ids;
    if(i.op==VectorShuffle) { if(i.wc>=5)ids={p[3],p[4]}; }
    else if(i.op==CompositeExtract || i.op==Load || i.op==CopyObject || i.op==Bitcast) { if(i.wc>=4)ids={p[3]}; }
    else if(i.op==CompositeInsert) { if(i.wc>=5)ids={p[3],p[4]}; }
    else if(i.op==ExtInst)tail(5);
    else if(i.op==Store) { if(i.wc>=3)ids={p[1],p[2]}; }
    else if(i.op==BranchConditional) { if(i.wc>=2)ids={p[1]}; }
    else if(i.op==Switch) { if(i.wc>=2)ids={p[1]}; }
    else if(i.op==ReturnValue) { if(i.wc>=2)ids={p[1]}; }
    else if(i.op==Branch || i.op==Return || i.op==Kill || i.op==Unreachable || i.op==246 || i.op==247)return ids;
    else if(i.op==245) { for(unsigned j=3;j+1<i.wc;j+=2)ids.push_back(p[j]); }
    else if(i.type)tail(3);
    else if(i.fn)tail(1);
    return ids;
}
bool parse(Module &m) {
    uint32_t fn=0;
    for(size_t off=5;off<m.n;) {
        uint32_t wc=m.w[off]>>16,op=m.w[off]&65535;
        if(!wc || wc>m.n-off)return false;
        const auto *p=m.w+off;
        if(op>=19 && op<=39) {
            if(wc<2 || !p[1] || p[1]>=m.bound)return false;
            if(!m.first_type)m.first_type=off;
            auto &t=m.types[p[1]];t.op=op;if(wc>=3)t.a=p[2];if(wc>=4)t.b=p[3];
            if(op==TypeFloat && wc==3 && p[2]==32)m.t_float=p[1];
            if(op==TypeInt && wc==4 && p[2]==32 && !p[3])m.t_uint=p[1];
            if(op==TypeBool && wc==2)m.t_bool=p[1];
        }
        if(op==Function) { if(wc!=5)return false;fn=p[2];if(!m.first_func)m.first_func=off; }
        m.instructions.push_back({off,op,wc,0,0,fn});
        if(op==FunctionEnd)fn=0;
        if(op==Capability) {if(wc!=2)return false;if(!m.first_cap)m.first_cap=off;if(p[1]==51)m.derivatives=true;}
        if(op==EntryPoint) {if(wc<4)return false;++m.entry_count;m.fragment&=p[1]==Fragment;m.entry_fn=p[2];m.entry_offset=off;}
        if(op==ExecutionMode && wc>=3 && p[2]==7)m.origin_upper=true;
        if(op==Decorate && wc>=4 && p[2]==BuiltIn) {
            if(p[3]==Lane) {if(m.lane_var)return false;m.lane_var=p[1];}
            if(p[3]==FragCoord) {if(m.coord_var)return false;m.coord_var=p[1];}
        }
        if(op==11 && wc>=5 && std::memcmp(p+2,"GLSL.std.450",12)==0)m.glsl_sets.push_back(p[1]);
        if(fn && (op==FunctionCall || op==BranchConditional || op==Switch || op==Kill || op==Unreachable || op==246 || op==247 || op==5380 || op==4416))
            m.unsafe_functions.push_back(fn);
        off+=wc;
    }
    for(size_t index=0;index<m.instructions.size();++index) {
        auto &i=m.instructions[index];const auto *p=m.w+i.offset;
        if((i.fn || (i.op>=41 && i.op<=52) || i.op==Variable || i.op==1) && i.wc>=3 && p[1]<m.bound && m.types[p[1]].op && !(i.op>=19 && i.op<=39)) {
            if(!p[2] || p[2]>=m.bound || m.definitions[p[2]]!=SIZE_MAX)return false;
            i.type=p[1];i.id=p[2];m.definitions[i.id]=index;
        }
    }
    for(size_t index=0;index<m.instructions.size();++index)
        for(uint32_t id:operands(m,m.instructions[index]))if(id<m.bound)m.uses[id].push_back(index);
    for(uint32_t id=1;id<m.bound;++id) {
        const auto &t=m.types[id];
        if(t.op==TypeVector && t.a==m.t_float && t.b==4)m.t_vec4=id;
    }
    for(uint32_t id=1;id<m.bound;++id)if(m.types[id].op==TypePointer && m.types[id].a==Input && m.types[id].b==m.t_vec4)m.t_ptr4=id;
    return true;
}
bool numeric_op(uint32_t op) {
    return op==FNegate || op==129 || op==131 || op==133 || op==136 || op==140 || op==141 ||
        (op>=142 && op<=148) || op==115 || op==116 || op==VectorShuffle || op==CompositeConstruct ||
        op==CompositeExtract || op==CompositeInsert || op==CopyObject || op==Select ||
        (op>=207 && op<=215) || op==Shuffle;
}
bool numeric_ext(const Module &m,const Instruction &i) {
    if(i.op!=ExtInst || i.wc<6)return false;
    const auto *p=m.w+i.offset;uint32_t op=p[4];
    bool set=std::find(m.glsl_sets.begin(),m.glsl_sets.end(),p[3])!=m.glsl_sets.end();
    return set && ((op>=1 && op<=34 && op!=5 && op!=7) || (op>=37 && op<=50) || op==53 || (op>=66 && op<=72) || (op>=79 && op<=81));
}
bool source_safe(const Module &m,uint32_t id,std::vector<uint8_t> &seen,xbg_quad_skip &reason,unsigned depth=0) {
    if(depth>256 || !id || id>=m.bound) {reason=xbg_quad_skip::unknown_source;return false;}
    if(seen[id]==2)return true;
    if(seen[id]==1) {reason=xbg_quad_skip::unknown_source;return false;}
    const auto *i=m.def(id);if(!i) {reason=xbg_quad_skip::unknown_source;return false;}
    if(i->op==Bitcast) {reason=xbg_quad_skip::bit_pattern;return false;}
    if(m.int_type(i->type)) {reason=xbg_quad_skip::integer_flow;return false;}
    if(!m.float_type(i->type)) {reason=xbg_quad_skip::unknown_source;return false;}
    if(i->op==Constant)return true;
    const auto *p=m.w+i->offset;
    if(i->op==Load) {
        uint32_t storage=m.storage(p[3]);
        if(storage==Input || storage==Uniform || storage==PushConstant)return true;
        reason=xbg_quad_skip::memory_flow;return false;
    }
    if(i->op>=87 && i->op<=98)return true;
    if(!numeric_op(i->op) && !numeric_ext(m,*i) && i->op!=ConstantComposite) {reason=xbg_quad_skip::unknown_source;return false;}
    seen[id]=1;
    std::vector<uint32_t> deps;
    if(i->op==Select && i->wc==6)deps={p[4],p[5]};
    else if(i->op==Shuffle && i->wc==6)deps={p[4]};
    else deps=operands(m,*i);
    for(uint32_t dep:deps)if(!source_safe(m,dep,seen,reason,depth+1))return false;
    seen[id]=2;return true;
}
bool uses_safe(const Module &m,uint32_t id,std::vector<uint8_t> &seen,xbg_quad_skip &reason,unsigned depth=0) {
    if(depth>256) {reason=xbg_quad_skip::unknown_use;return false;}
    if(seen[id])return true;seen[id]=1;
    for(size_t index:m.uses[id]) {
        const auto &i=m.instructions[index];const auto *p=m.w+i.offset;
        if(i.op==Bitcast) {reason=xbg_quad_skip::bit_pattern;return false;}
        if(m.int_type(i.type)) {reason=xbg_quad_skip::integer_flow;return false;}
        if(i.op==Store) {
            bool readback=false;
            uint32_t root=m.pointer_root(p[1]);
            for(const auto &other:m.instructions)if(other.op==Load && other.wc>=4 && m.pointer_root(m.w[other.offset+3])==root)readback=true;
            if(i.wc==3 && p[2]==id && m.storage(p[1])==Output && !readback)continue;
            reason=xbg_quad_skip::memory_flow;return false;
        }
        if(i.op==IsNan || i.op==IsInf || (i.op>=180 && i.op<=191))continue;
        if(!i.id || !m.float_type(i.type) || (!numeric_op(i.op) && !numeric_ext(m,i))) {reason=xbg_quad_skip::unknown_use;return false;}
        if(!uses_safe(m,i.id,seen,reason,depth+1))return false;
    }
    return true;
}
}

const char *xbg_spirv_quad_skip_name(xbg_quad_skip reason) {
    static const char *const names[]={"malformed","module_contract","non_float","lane_index","control_flow","bit_pattern","integer_flow","unknown_source","unknown_use","memory_flow"};
    auto index=static_cast<size_t>(reason);return index<static_cast<size_t>(xbg_quad_skip::count)?names[index]:"invalid_reason";
}

size_t xbg_spirv_quad_derivatives(const uint32_t *in,size_t n,std::vector<uint32_t> &out,xbg_spirv_quad_stats *stats) {
    out.clear();xbg_spirv_quad_stats local;auto &s=stats?*stats:local;s={};
    auto skip=[&](xbg_quad_skip reason){++s.skipped[static_cast<size_t>(reason)];};
    if(!in || n<5 || in[0]!=0x07230203 || !in[3] || in[3]>(1u<<22) || n>(1u<<24)) {skip(xbg_quad_skip::malformed);return 0;}
    Module m(in,n);if(!parse(m)) {skip(xbg_quad_skip::malformed);return 0;}
    struct Plan {size_t index;uint32_t mask;};std::vector<Plan> plans;
    bool contract=m.entry_count==1 && m.fragment && m.origin_upper && m.first_type && m.first_func && m.first_cap && m.t_float && m.t_uint && m.lane_var;
    const auto *lane=m.def(m.lane_var);
    contract=contract && lane && lane->op==Variable && lane->type<m.bound && m.types[lane->type].op==TypePointer &&
        m.types[lane->type].a==Input && m.int_type(m.types[lane->type].b) && m.types[m.types[lane->type].b].a==32;
    if(m.coord_var) {
        const auto *coord=m.def(m.coord_var);
        contract=contract && coord && coord->op==Variable && coord->type<m.bound && m.types[coord->type].op==TypePointer &&
            m.types[coord->type].a==Input && m.types[coord->type].b==m.t_vec4;
    }
    for(size_t index=0;index<m.instructions.size();++index) {
        const auto &i=m.instructions[index];if(i.op!=Shuffle)continue;++s.candidates;
        const auto *p=in+i.offset;
        if(i.wc!=6) {skip(xbg_quad_skip::malformed);continue;}
        if(i.type!=m.t_float) {skip(xbg_quad_skip::non_float);continue;}
        if(!contract) {skip(xbg_quad_skip::module_contract);continue;}
        if(i.fn!=m.entry_fn || std::find(m.unsafe_functions.begin(),m.unsafe_functions.end(),i.fn)!=m.unsafe_functions.end()) {skip(xbg_quad_skip::control_flow);continue;}
        xbg_quad_skip reason=xbg_quad_skip::unknown_source;std::vector<uint8_t> visited(m.bound);
        if(!source_safe(m,p[4],visited,reason)) {skip(reason);continue;}
        std::fill(visited.begin(),visited.end(),0);
        if(!uses_safe(m,p[2],visited,reason)) {skip(reason);continue;}
        uint32_t scope=0,mask=0;const auto *idx=m.def(p[5]);bool index_ok=false;
        if(m.constant(p[3],scope) && scope==3 && idx && idx->op==BitwiseXor && idx->wc==5) {
            const auto *q=in+idx->offset;
            index_ok=(m.lane_value(q[3]) && m.constant(q[4],mask)) || (m.lane_value(q[4]) && m.constant(q[3],mask));
        }
        if(!index_ok || mask<1 || mask>3) {skip(xbg_quad_skip::lane_index);continue;}
        plans.push_back({index,mask});
    }
    if(plans.empty())return 0;
    uint32_t bound=m.bound;auto fresh=[&](){return bound++;};
    std::vector<uint32_t> declarations,decorations;
    if(!m.t_bool) {m.t_bool=fresh();declarations.insert(declarations.end(),{word(TypeBool,2),m.t_bool});}
    uint32_t one=fresh(),zero=fresh();
    declarations.insert(declarations.end(),{word(Constant,4),m.t_uint,one,1,word(Constant,4),m.t_uint,zero,0});
    bool new_coord=!m.coord_var;
    if(new_coord) {
        if(!m.t_vec4) {m.t_vec4=fresh();declarations.insert(declarations.end(),{word(TypeVector,4),m.t_vec4,m.t_float,4});}
        if(!m.t_ptr4) {m.t_ptr4=fresh();declarations.insert(declarations.end(),{word(TypePointer,4),m.t_ptr4,Input,m.t_vec4});}
        m.coord_var=fresh();declarations.insert(declarations.end(),{word(Variable,4),m.t_ptr4,m.coord_var,Input});
        decorations.insert(decorations.end(),{word(Decorate,4),m.coord_var,BuiltIn,FragCoord});
    }
    if(bound+plans.size()*64>0x3fffff) {skip(xbg_quad_skip::malformed);return 0;}
    out.insert(out.end(),in,in+5);size_t next=0;
    auto emit=[&](std::initializer_list<uint32_t> words){out.insert(out.end(),words.begin(),words.end());};
    auto nonfinite=[&](uint32_t value) {
        uint32_t nan=fresh(),inf=fresh(),bad=fresh();
        emit({word(IsNan,4),m.t_bool,nan,value});emit({word(IsInf,4),m.t_bool,inf,value});emit({word(LogicalOr,5),m.t_bool,bad,nan,inf});return bad;
    };
    for(size_t index=0;index<m.instructions.size();++index) {
        const auto &i=m.instructions[index];const auto *p=in+i.offset;
        if(i.offset==m.first_cap && !m.derivatives)emit({word(Capability,2),51});
        if(i.offset==m.first_type)out.insert(out.end(),decorations.begin(),decorations.end());
        if(i.offset==m.first_func)out.insert(out.end(),declarations.begin(),declarations.end());
        if(i.offset==m.entry_offset && new_coord) {out.push_back(word(EntryPoint,i.wc+1));out.insert(out.end(),p+1,p+i.wc);out.push_back(m.coord_var);continue;}
        if(next<plans.size() && plans[next].index==index) {
            uint32_t mask=plans[next++].mask,original=fresh(),coord=fresh();
            emit({word(Shuffle,6),p[1],original,p[3],p[4],p[5]});
            emit({word(Load,4),m.t_vec4,coord,m.coord_var});
            uint32_t value=p[4],bad=nonfinite(value);
            auto merge_bad=[&](uint32_t v){uint32_t other=nonfinite(v),combined=fresh();emit({word(LogicalOr,5),m.t_bool,combined,bad,other});bad=combined;};
            auto neighbour=[&](uint32_t source,uint32_t axis) {
                uint32_t derivative=fresh(),component=fresh(),integer=fresh(),parity=fresh(),odd=fresh(),negative=fresh(),signed_delta=fresh(),result=fresh();
                emit({word(axis?DPdyFine:DPdxFine,4),m.t_float,derivative,source});
                emit({word(CompositeExtract,5),m.t_float,component,coord,axis});
                emit({word(ConvertFToU,4),m.t_uint,integer,component});
                emit({word(BitwiseAnd,5),m.t_uint,parity,integer,one});
                emit({word(INotEqual,5),m.t_bool,odd,parity,zero});
                emit({word(FNegate,4),m.t_float,negative,derivative});
                emit({word(Select,6),m.t_float,signed_delta,odd,negative,derivative});
                emit({word(FAdd,5),m.t_float,result,source,signed_delta});
                merge_bad(derivative);merge_bad(result);return result;
            };
            if(mask&1)value=neighbour(value,0);
            if(mask&2)value=neighbour(value,1);
            emit({word(Select,6),m.t_float,p[2],bad,original,value});
            ++s.rewritten;++s.guarded_nonfinite;
        } else out.insert(out.end(),p,p+i.wc);
    }
    out[3]=bound;return s.rewritten;
}
size_t xbg_spirv_quad_derivatives(const uint32_t *in,size_t n,std::vector<uint32_t> &out) {
    return xbg_spirv_quad_derivatives(in,n,out,nullptr);
}
