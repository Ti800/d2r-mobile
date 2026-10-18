# 安卓移植

English: [README.md](README.md)

这是 ARM64 安卓开发版。APK 已接入公用宿主、Vulkan 回放引擎、AAudio、透明触摸覆盖层及系统输入框。编译通过不代表可玩：本版本仍须真机验证出图、游戏交互、声音、生命周期和保存重进，尚不是完成的安卓发行版。

## 源码结构

- `app/`：Java Activity、全屏 SurfaceView、触摸覆盖层及 Gradle 应用工程。
- `jni_android.c`、`front_android.c`：启动、资源路径、生命周期及平台前端接口。
- `touch_android*`：接入公用 `dm/ios/Sources/dm_touch_controls.c` 状态机；`text_input_jni.c` 将客体同步输入请求接到安卓系统对话框。
- `vulkan_present.*`：安卓窗口引用、交换链及呈现。
- `../gfx/vulkan/`：NVN 回放适配器和 Vulkan 引擎。
- `compat/`：宿主 Apple 平台接口的安卓实现。
- `tools/`：源码清单解析、ELF 汇编转换及本地化资源生成。

## 本地构建

当前配置使用 JDK 17 或更新版本、Gradle 9.8.1、Android SDK 36、NDK 29.0.14206865、CMake 3.31.6 和 Python 3。ARM64 设备要求 Android 12/API 31 或更新系统及 Vulkan 1.3。这些是构建和清单要求，不代表符合条件的机型都已验证。首台目标是 vivo X200 Ultra（Android 16/API 36、Adreno 830）。

按根目录准备说明提供自己的游戏文件。参考 `dm/host/Makefile` 的规则，在仓外生成 `dm_vault_tab.h`、`dm_hud_exp.h`、`dm_bright_levels.inc`。`ANDROID_HOME` 与 `ANDROID_SDK_ROOT` 必须指向同一个 SDK，`JAVA_HOME` 指向 JDK。在仓库根目录执行：

```sh
gradle -p dm/android :app:assembleDebug :app:lintDebug \
  -PdmGenDir=/absolute/path/to/private/generated/headers \
  -PdmPython=/absolute/path/to/python
```

APK 位于 `dm/android/app/build/outputs/apk/debug/app-debug.apk`，原生链接启用 `--no-undefined`。`-PdmPrebuiltJni=...` 只供使用已有库打包，不是当前原生源码编译通过的证据。

## 仓外运行材料

APK 和 Git 仓库均不含游戏程序、着色器、游戏美术或存档。当前开发启动器需要独立导入材料，尚无通用导入界面。

| 位置 | 内容 |
| --- | --- |
| 应用内部 `files/dm/out/` | 使用 `DM_TARGET=android` 生成的 `main`、`sdk` 改写产物，含 patched 模块、fixups 和 gates |
| 应用内部 `files/dm/shaders/` | 与游戏及 Vulkan 后端匹配的仓外 `.xgs` 包 |
| 应用内部 `files/dm/overlay/` | 必需 CASC 触摸覆盖层，含 `overlay.manifest`，由匹配源码及用户资源生成 |
| 应用内部 `files/dm/overlay/touchart/` | 可选本地生成的 `dm_touch_joy.png`；没有时绘制备用摇杆 |
| 共享 `Download/data/` | 用户资源档案及 `.build.config`，需要安卓“所有文件访问”授权 |
| 应用内部 `files/save/` | 存档，升级时必须保留 |
| 应用内部 `files/temp/`、`files/dm/boot.log` | 临时文件及追加写入的启动日志 |

内部没有主程序时，启动器回退到应用外部私有目录下的 `dm/`。启动前会检查必要路径，但不能据此证明着色器身份及资源兼容性。调试包可使用 `run-as` 导入内部文件，升级使用相同签名覆盖安装，不以卸载或清数据作为升级步骤。真机验证前备份已有存档，不向 PR 上传 APK 或上述私人文件。

覆盖层生成器 `dm/ios/tools/gen_romfs_overlay.py` 支持用 `DM_DESWIZZLE_LIB` 指定本地编译的纹理处理库。生成的精简 CASC 只含变更并引用原资源，不得替换原始资源目录。

## 验证与剩余工作

执行根目录贡献检查及 `:app:lintDebug`。纯 Java 输入测试不需要手机：

```sh
javac -encoding UTF-8 -d /tmp/dm-input-tests \
  dm/android/app/src/main/java/com/xiaoboluo/d2r/TextInputRules.java \
  dm/android/tests/TextInputRulesTest.java
java -cp /tmp/dm-input-tests com.xiaoboluo.d2r.TextInputRulesTest
```

真机验收须分别记录安装、真实 GPU 出图/帧率、建角、进局、移动停止、NPC 交互、背包/技能、声音、后台恢复及保存重进。无窗口宿主帧率不等于渲染帧率。本后端尚未接入 Metal GPU 计时及实际扫描显示时间，缺失读数不能当作零耗时。

Vulkan 当前每次全画，并使用同步宿主翻译（`DM_NVN_SYNC=1`）；上游新增异步路径需独立审计对象生命周期。呈现层仍逐帧等待队列。iOS 技能图、空技能槽菜单、专用数量/属性加点浮层及商店拖动跟随图尚未完整实现，通用控件和系统输入框不代表完整 iOS 功能对齐。提交上游集成前，还须在 Apple 构建机上验证 macOS/iOS 回归。

覆盖层生成器将源码指纹和逐文件哈希写入 `overlay.manifest`，APK 编入同一生成器输入指纹。缺失、不完整或不匹配时，启动界面提示并在宿主线程启动前退出启动流程。须部署完整生成目录，不可把新清单贴到旧资源上，也不可在手机启动时补清单；生成器或布局修改后应重新生成。这项校验不负责认证第三方下载来源。
