// 按 Maxwell 32 道线程束写的计算着色器在更宽的子组上运行（见 xbg_spirv_warp.cpp）
// Compute shaders written for Maxwell's 32-lane warps, run on wider subgroups (see xbg_spirv_warp.cpp)
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Rewrites a compute module translated for 32-lane subgroups so that each 32-lane half (or quarter) of a 64- or 128-lane
// subgroup acts as its own warp: the lane id becomes id & 31, word 0 of a ballot or subgroup mask becomes word id >> 5, a
// vote is answered from the ballot word of the lane's warp, and a shuffle reads from the same warp ((id & 0x60) | index).
// Returns the number of instructions rewritten (0: out is empty and the module is to be used unchanged, including when it
// uses a subgroup feature this does not know).
size_t xbg_spirv_warp32(const uint32_t *words, size_t count, std::vector<uint32_t> &out);
