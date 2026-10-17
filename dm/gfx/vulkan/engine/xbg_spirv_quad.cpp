// 片元着色器的四像素邻居交换：着色器库把 Maxwell 的 SHFL.BFLY（同一 2x2 四元组里取左右 / 上下邻居，lane ^ 1 / ^ 2）
// 译成 OpGroupNonUniformShuffle。Vulkan 不保证辅助调用（三角形外、只为求导存在的像素）参与子组操作；Adreno 上它们取到的
// 是垃圾值，D2R 罪恶洞穴打斗时由此生出的 NaN 经模糊链扩成单位周围的黑方块。这里把浮点的这种交换改成由精细导数求邻居：
// 同一四元组里 dFdxFine(v) = 右 - 左、dFdyFine(v) = 下 - 上，辅助调用一定参与求导，左（上）像素的邻居是 v + d，右（下）
// 像素的是 v - d；对角（异或 3）取两者之和（差一个二阶项）。像素在四元组里的位置由 gl_FragCoord 的奇偶得出。
// The quad-neighbour exchange of fragment shaders: the shader library translates Maxwell's SHFL.BFLY (the left / right or
// upper / lower neighbour in the same 2x2 quad, lane ^ 1 / ^ 2) into OpGroupNonUniformShuffle. Vulkan does not promise that
// helper invocations (pixels outside the triangle that exist only for derivatives) take part in subgroup operations; on
// Adreno they read garbage, and in the D2R Den of Evil fight the resulting NaNs spread through the blur chain into black
// squares around units. This rewrites such float exchanges to take the neighbour from fine derivatives: within a quad
// dFdxFine(v) = right - left and dFdyFine(v) = lower - upper, helper invocations always take part in derivatives, the left
// (upper) pixel's neighbour is v + d and the right (lower) pixel's is v - d; the diagonal (xor 3) adds both (off by a
// second-order term). The pixel's place in its quad comes from the parity of gl_FragCoord.
#include "xbg_spirv_quad.h"
#include <cstring>

namespace {
enum : uint32_t {
    OpCapability = 17, OpEntryPoint = 15, OpDecorate = 71, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
    OpTypePointer = 32, OpConstant = 43, OpVariable = 59, OpFunction = 54, OpLoad = 61, OpBitcast = 124, OpBitwiseXor = 198,
    OpBitwiseAnd = 199, OpINotEqual = 171, OpDPdxFine = 210, OpDPdyFine = 211, OpCompositeExtract = 81, OpConvertFToU = 109,
    OpFNegate = 127, OpSelect = 169, OpFAdd = 129, OpGroupNonUniformShuffle = 345,
    CapDerivativeControl = 51, DecBuiltIn = 11, BuiltInFragCoord = 15, BuiltInSubgroupLocalInvocationId = 41, StorageInput = 1,
    ModelFragment = 4,
};
inline uint32_t inst(uint32_t op, uint32_t words) { return (words << 16) | op; }
bool is_global_decl(uint32_t op) { return (op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == OpVariable || op == 1 /*OpUndef*/ || op == 322 /*OpTypeForwardPointer*/; }
}

size_t xbg_spirv_quad_derivatives(const uint32_t *in, size_t n, std::vector<uint32_t> &out) {
    out.clear();
    if (n < 5 || in[0] != 0x07230203u) return 0;
    uint32_t bound = in[3];
    // Pass 1: what the module already has.
    bool has_deriv = false; size_t first_cap = 0, entry = 0, first_type = 0, first_func = 0;
    uint32_t fragcoord = 0, lane_var = 0, t_bool = 0, t_uint = 0, t_int = 0, t_float = 0, t_v4f = 0, t_ptr_in_v4f = 0;
    uint32_t c_u0 = 0, c_u1 = 0, c_u2 = 0, c_f0 = 0;
    std::vector<size_t> shuffles;
    struct Type { uint32_t op, a, b; };
    std::vector<Type> types(bound, Type{0, 0, 0});   // by id: OpTypeInt (width, signedness) / OpTypeFloat (width) / OpTypePointer (class, pointee)
    std::vector<uint32_t> var_type(bound, 0);        // global OpVariable: pointer type
    for (size_t i = 5; i < n;) {
        uint32_t wc = in[i] >> 16, op = in[i] & 0xffff;
        if (!wc || i + wc > n) return 0;
        if (op == OpCapability) { if (!first_cap) first_cap = i; if (in[i + 1] == CapDerivativeControl) has_deriv = true; }
        else if (op == OpEntryPoint) { if (in[i + 1] != ModelFragment) return 0; if (!entry) entry = i; }
        else if (op == OpDecorate && wc >= 4 && in[i + 2] == DecBuiltIn) {
            if (in[i + 3] == BuiltInFragCoord) fragcoord = in[i + 1];
            else if (in[i + 3] == BuiltInSubgroupLocalInvocationId) lane_var = in[i + 1];
        }
        if (!first_func) {
            if (is_global_decl(op) && !first_type) first_type = i;
            if (op == OpTypeBool) t_bool = in[i + 1];
            else if (op == OpTypeInt && wc == 4 && in[i + 1] < bound) { types[in[i + 1]] = {op, in[i + 2], in[i + 3]}; if (in[i + 2] == 32) (in[i + 3] ? t_int : t_uint) = in[i + 1]; }
            else if (op == OpTypeFloat && wc == 3 && in[i + 1] < bound) { types[in[i + 1]] = {op, in[i + 2], 0}; if (in[i + 2] == 32) t_float = in[i + 1]; }
            else if (op == OpTypeVector && wc == 4 && in[i + 2] && in[i + 2] == t_float && in[i + 3] == 4) t_v4f = in[i + 1];
            else if (op == OpTypePointer && wc == 4 && in[i + 1] < bound) { types[in[i + 1]] = {op, in[i + 2], in[i + 3]}; if (in[i + 2] == StorageInput && t_v4f && in[i + 3] == t_v4f) t_ptr_in_v4f = in[i + 1]; }
            else if (op == OpConstant && wc == 4) {
                if (t_uint && in[i + 1] == t_uint) { if (in[i + 3] == 0 && !c_u0) c_u0 = in[i + 2]; if (in[i + 3] == 1 && !c_u1) c_u1 = in[i + 2]; if (in[i + 3] == 2 && !c_u2) c_u2 = in[i + 2]; }
                if (t_float && in[i + 1] == t_float && in[i + 3] == 0 && !c_f0) c_f0 = in[i + 2];
            }
            else if (op == OpVariable && wc >= 4 && in[i + 2] < bound) var_type[in[i + 2]] = in[i + 1];
        }
        if (op == OpFunction && !first_func) first_func = i;
        if (op == OpGroupNonUniformShuffle && wc == 6 && t_float && in[i + 1] == t_float) shuffles.push_back(i);
        i += wc;
    }
    if (shuffles.empty() || !entry || !first_cap || !first_type || !first_func || !lane_var || !t_float || !t_uint) return 0;
    // The lane variable's pointee (uint, or int: then bitcast); an index of int type is bitcast too.
    uint32_t lane_ptr = var_type[lane_var] < bound ? var_type[lane_var] : 0, lane_t = lane_ptr ? types[lane_ptr].b : 0;
    if (!lane_t || types[lane_t].op != OpTypeInt || types[lane_t].a != 32) return 0;
    // Pass 2: new ids for whatever is missing.
    std::vector<uint32_t> decl, deco;   // global declarations to add before the first function, decorations before the first type
    auto fresh = [&] { return bound++; };
    if (!t_uint) { t_uint = fresh(); decl.insert(decl.end(), {inst(OpTypeInt, 4), t_uint, 32, 0}); }
    if (!t_bool) { t_bool = fresh(); decl.insert(decl.end(), {inst(OpTypeBool, 2), t_bool}); }
    if (!c_u0) { c_u0 = fresh(); decl.insert(decl.end(), {inst(OpConstant, 4), t_uint, c_u0, 0}); }
    if (!c_u1) { c_u1 = fresh(); decl.insert(decl.end(), {inst(OpConstant, 4), t_uint, c_u1, 1}); }
    if (!c_u2) { c_u2 = fresh(); decl.insert(decl.end(), {inst(OpConstant, 4), t_uint, c_u2, 2}); }
    if (!c_f0) { c_f0 = fresh(); decl.insert(decl.end(), {inst(OpConstant, 4), t_float, c_f0, 0}); }
    bool new_fragcoord = !fragcoord;
    if (new_fragcoord) {
        if (!t_v4f) { t_v4f = fresh(); decl.insert(decl.end(), {inst(OpTypeVector, 4), t_v4f, t_float, 4}); }
        if (!t_ptr_in_v4f) { t_ptr_in_v4f = fresh(); decl.insert(decl.end(), {inst(OpTypePointer, 4), t_ptr_in_v4f, StorageInput, t_v4f}); }
        fragcoord = fresh(); decl.insert(decl.end(), {inst(OpVariable, 4), t_ptr_in_v4f, fragcoord, StorageInput});
        deco.insert(deco.end(), {inst(OpDecorate, 4), fragcoord, DecBuiltIn, BuiltInFragCoord});
    } else if (!t_v4f) return 0;   // an existing FragCoord needs its vec4 type known
    // Pass 3: write the module.
    out.reserve(n + 64 + shuffles.size() * 96);
    out.insert(out.end(), in, in + 5);
    size_t next_shuffle = 0;
    for (size_t i = 5; i < n;) {
        uint32_t wc = in[i] >> 16;
        if (i == first_cap && !has_deriv) out.insert(out.end(), {inst(OpCapability, 2), CapDerivativeControl});
        if (i == first_type) out.insert(out.end(), deco.begin(), deco.end());
        if (i == first_func) out.insert(out.end(), decl.begin(), decl.end());
        if (i == entry && new_fragcoord) { out.push_back(inst(OpEntryPoint, wc + 1)); out.insert(out.end(), in + i + 1, in + i + wc); out.push_back(fragcoord); i += wc; continue; }
        if (next_shuffle < shuffles.size() && i == shuffles[next_shuffle]) {
            ++next_shuffle;
            uint32_t res = in[i + 2], val = in[i + 4], idxu = in[i + 5];   // the shuffle's index is an unsigned integer by rule
            uint32_t lane0 = fresh(), lane = lane0;
            out.insert(out.end(), {inst(OpLoad, 4), lane_t, lane0, lane_var});
            if (lane_t != t_uint) { lane = fresh(); out.insert(out.end(), {inst(OpBitcast, 4), t_uint, lane, lane0}); }
            uint32_t mask = fresh(), mx = fresh(), my = fresh(), hx = fresh(), hy = fresh();
            out.insert(out.end(), {inst(OpBitwiseXor, 5), t_uint, mask, idxu, lane});
            out.insert(out.end(), {inst(OpBitwiseAnd, 5), t_uint, mx, mask, c_u1});
            out.insert(out.end(), {inst(OpBitwiseAnd, 5), t_uint, my, mask, c_u2});
            out.insert(out.end(), {inst(OpINotEqual, 5), t_bool, hx, mx, c_u0});
            out.insert(out.end(), {inst(OpINotEqual, 5), t_bool, hy, my, c_u0});
            uint32_t dx = fresh(), dy = fresh(), fc = fresh(), fx = fresh(), fy = fresh(), ux = fresh(), uy = fresh(), px = fresh(), py = fresh(), rx = fresh(), ry = fresh();
            out.insert(out.end(), {inst(OpDPdxFine, 4), t_float, dx, val});
            out.insert(out.end(), {inst(OpDPdyFine, 4), t_float, dy, val});
            out.insert(out.end(), {inst(OpLoad, 4), t_v4f, fc, fragcoord});
            out.insert(out.end(), {inst(OpCompositeExtract, 5), t_float, fx, fc, 0});
            out.insert(out.end(), {inst(OpCompositeExtract, 5), t_float, fy, fc, 1});
            out.insert(out.end(), {inst(OpConvertFToU, 4), t_uint, ux, fx});
            out.insert(out.end(), {inst(OpConvertFToU, 4), t_uint, uy, fy});
            out.insert(out.end(), {inst(OpBitwiseAnd, 5), t_uint, px, ux, c_u1});
            out.insert(out.end(), {inst(OpBitwiseAnd, 5), t_uint, py, uy, c_u1});
            out.insert(out.end(), {inst(OpINotEqual, 5), t_bool, rx, px, c_u0});
            out.insert(out.end(), {inst(OpINotEqual, 5), t_bool, ry, py, c_u0});
            uint32_t ndx = fresh(), ndy = fresh(), sdx = fresh(), sdy = fresh(), tx = fresh(), ty = fresh(), a = fresh();
            out.insert(out.end(), {inst(OpFNegate, 4), t_float, ndx, dx});
            out.insert(out.end(), {inst(OpFNegate, 4), t_float, ndy, dy});
            out.insert(out.end(), {inst(OpSelect, 6), t_float, sdx, rx, ndx, dx});   // right column: v - dx; left: v + dx
            out.insert(out.end(), {inst(OpSelect, 6), t_float, sdy, ry, ndy, dy});   // lower row: v - dy; upper: v + dy
            out.insert(out.end(), {inst(OpSelect, 6), t_float, tx, hx, sdx, c_f0});
            out.insert(out.end(), {inst(OpSelect, 6), t_float, ty, hy, sdy, c_f0});
            out.insert(out.end(), {inst(OpFAdd, 5), t_float, a, val, tx});
            out.insert(out.end(), {inst(OpFAdd, 5), t_float, res, a, ty});
            i += wc; continue;
        }
        out.insert(out.end(), in + i, in + i + wc);
        i += wc;
    }
    out[3] = bound;
    return shuffles.size();
}
