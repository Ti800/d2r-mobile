// 片元着色器四元组邻居交换改用精细导数（见 xbg_spirv_quad.cpp）
// Fragment-shader quad-neighbour exchanges rewritten to fine derivatives (see xbg_spirv_quad.cpp)
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Rewrites every float OpGroupNonUniformShuffle of a fragment-shader module into the quad neighbour taken from
// OpDPdxFine / OpDPdyFine, adding DerivativeControl and a FragCoord input where missing. Returns the number of
// shuffles rewritten (0: out is empty and the module is to be used unchanged).
size_t xbg_spirv_quad_derivatives(const uint32_t *words, size_t count, std::vector<uint32_t> &out);
