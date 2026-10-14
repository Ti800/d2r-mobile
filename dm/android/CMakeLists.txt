# 安卓宿主构建（NDK，arm64-v8a）。源文件清单不手抄：从 dm/host/Makefile 的 SRC / GFX / LIBDEFLATE 取（与 iOS 的 ios/tools/gen_build.py 同一思路），
# 排除 Mac 专用（*_mac.*）、Objective-C（*.m，安卓另有实现）与 main.c（单独编：命令行 dmhost 用原名 main，App 的 libdm.so 以 -Dmain=dm_main 编）。
# 汇编（Mach-O 写法）经 tools/asm_elf.py 转 ELF 后编；compat/ 只在安卓构建的头文件路径里，苹果构建看不到。
#
# 必需：-DDM_GEN_DIR=<目录>，内含由游戏文件生成的 dm_vault_tab.h / dm_hud_exp.h / dm_bright_levels.inc（生成方法同 dm/host/Makefile 对应规则，不入库）。
# 产物：dmhost（命令行，adb shell 跑，开发验证用）；libdm.so（App 用，待接 JNI）。
#
# Android host build (NDK, arm64-v8a). The source list is not copied by hand: it is taken from SRC / GFX / LIBDEFLATE in dm/host/Makefile (the same idea as iOS's ios/tools/gen_build.py),
# excluding Mac-only files (*_mac.*), Objective-C (*.m, Android has its own implementations) and main.c (built separately: the command-line dmhost keeps the name main, the App's libdm.so is built with -Dmain=dm_main).
# Assembly (Mach-O syntax) is converted to ELF by tools/asm_elf.py before assembling; compat/ is only on the Android build's header path, invisible to Apple builds.
#
# Required: -DDM_GEN_DIR=<directory> containing dm_vault_tab.h / dm_hud_exp.h / dm_bright_levels.inc generated from the game files (generated as by the matching rules in dm/host/Makefile, not committed).
# Outputs: dmhost (command line, run from adb shell, for development checks); libdm.so (for the App, JNI to be wired up).
cmake_minimum_required(VERSION 3.22)
project(dm_android C ASM)

set(DM ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(HOST ${DM}/host)
if(NOT DM_GEN_DIR OR NOT EXISTS ${DM_GEN_DIR}/dm_vault_tab.h)
  message(FATAL_ERROR "要 -DDM_GEN_DIR=<由游戏文件生成的宿主表头目录>（dm_vault_tab.h / dm_hud_exp.h / dm_bright_levels.inc，见 dm/host/Makefile）")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---- 从 Makefile 取源文件 | Take the source files from the Makefile ----
# 逐行读（CMake 的正则是回溯实现，整文件多行匹配会爆栈）：SRC=\ 起、到第一行不以 \ 结尾止
# Read line by line (CMake's regex is a backtracking implementation, and a multi-line match over the whole file overflows the stack): from SRC=\ to the first line not ending in \
file(STRINGS ${HOST}/Makefile MK_LINES)
set(HOST_SRCS)
set(in_src 0)
set(LD_LINE "")
foreach(l IN LISTS MK_LINES)
  if(l MATCHES "^SRC=")
    set(in_src 1)
  elseif(in_src)
    string(STRIP "${l}" t)
    string(REGEX REPLACE "[ \t]*\\\\$" "" t "${t}")
    if(t MATCHES "^[A-Za-z0-9_./-]+\\.(c|m|S)$")
      get_filename_component(n ${t} NAME_WE)
      if(NOT (t MATCHES "\\.(m|S)$" OR n MATCHES "_mac$" OR t STREQUAL "main.c"))
        list(APPEND HOST_SRCS ${HOST}/${t})
      endif()
    endif()
    if(NOT l MATCHES "\\\\$")
      set(in_src 0)
    endif()
  endif()
  if(l MATCHES "^LIBDEFLATE=")
    set(LD_LINE "${l}")
  endif()
endforeach()
file(GLOB HLE_SRCS ${HOST}/hle_*.c)                      # Makefile：$(sort $(wildcard hle_*.c))
file(GLOB GFX_SRCS ${DM}/gfx/*.c ${DM}/gfx/nvn/*.c ${DM}/gfx/tex/*.c)   # Makefile GFX 里的 .c（.m / .S 另处理） | the .c files in the Makefile's GFX (.m / .S handled separately)
string(REGEX MATCHALL "[A-Za-z0-9_./-]+\\.c" LD_LIST "${LD_LINE}")
set(LIBDEFLATE_SRCS)
foreach(s ${LD_LIST})
  list(APPEND LIBDEFLATE_SRCS ${HOST}/${s})
endforeach()
list(LENGTH HOST_SRCS nh)
if(nh LESS 40)
  message(FATAL_ERROR "从 dm/host/Makefile 只解析出 ${nh} 个宿主源文件：Makefile 的 SRC 写法变了，改本文件的解析")
endif()

# ---- 汇编：Mach-O → ELF | Assembly: Mach-O → ELF ----
set(ASM_SRCS)
foreach(s ${HOST}/entry.S ${DM}/gfx/nvn/nvn_entry.S ${DM}/gfx/nvn/nvn_thunks.S)
  get_filename_component(n ${s} NAME)
  set(out ${CMAKE_CURRENT_BINARY_DIR}/asm/${n})
  add_custom_command(OUTPUT ${out} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/asm_elf.py ${s} ${out}
                     DEPENDS ${s} ${CMAKE_CURRENT_SOURCE_DIR}/tools/asm_elf.py VERBATIM)
  list(APPEND ASM_SRCS ${out})
endforeach()

set(ANDROID_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/front_android.c
  ${CMAKE_CURRENT_SOURCE_DIR}/compat/clonefile.c
  ${CMAKE_CURRENT_SOURCE_DIR}/compat/digest.c
  ${CMAKE_CURRENT_SOURCE_DIR}/compat/mach_thread.c
  ${CMAKE_CURRENT_SOURCE_DIR}/compat/mach_symbols.c
  ${DM}/gfx/vulkan/nvn_vk.c)

add_library(dmcore STATIC ${HOST_SRCS} ${HLE_SRCS} ${GFX_SRCS} ${LIBDEFLATE_SRCS} ${ASM_SRCS} ${ANDROID_SRCS})
list(REMOVE_DUPLICATES HOST_SRCS)
target_include_directories(dmcore PUBLIC ${HOST} ${DM}/gfx/nvn/include ${CMAKE_CURRENT_SOURCE_DIR}/compat/include ${DM_GEN_DIR})
# 与 Makefile 同档优化（-O1 -g）；dm_android_compat.h 强制包含；-fno-strict-aliasing 同苹果 clang 缺省行为下宿主对客体内存的读写习惯
# Same optimization level as the Makefile (-O1 -g); dm_android_compat.h force-included; -fno-strict-aliasing matches the host's habit of reading and writing guest memory under Apple clang's default behaviour
set(DM_CFLAGS -O1 -g -Wall -Wno-unused-function -fno-strict-aliasing -include ${CMAKE_CURRENT_SOURCE_DIR}/compat/include/dm_android_compat.h)
target_compile_options(dmcore PUBLIC $<$<COMPILE_LANGUAGE:C>:${DM_CFLAGS} -std=gnu17>)
target_link_libraries(dmcore PUBLIC aaudio log m dl)

# 命令行宿主：main.c 原样 | Command-line host: main.c as is
add_executable(dmhost ${HOST}/main.c)
target_link_libraries(dmhost PRIVATE -Wl,--whole-archive dmcore -Wl,--no-whole-archive)
target_link_options(dmhost PRIVATE -Wl,--no-undefined -Wl,-z,nostart-stop-gc -Wl,--export-dynamic)   # 导出符号：崩溃回溯里 dladdr 能认出宿主函数 | export symbols: dladdr in crash backtraces can name host functions
