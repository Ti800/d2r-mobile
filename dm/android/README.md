# Android port

中文：[README.zh-CN.md](README.zh-CN.md)

This is an ARM64 Android development port. The APK now links the shared host,
the Vulkan replay engine, AAudio, a transparent touch overlay, and system text
input. Compilation is not a playability result: this revision still needs device
validation of rendering, gameplay, audio, lifecycle, and save/reload. It is not a
finished Android release.

## Source layout

- `app/`: Java Activity, fullscreen SurfaceView, touch overlay, and Gradle app.
- `jni_android.c`, `front_android.c`: startup, resource locations, lifecycle, and
  platform front-end interfaces.
- `touch_android*`: bridge to the shared `dm/ios/Sources/dm_touch_controls.c` state
  machine; `text_input_jni.c` bridges synchronous guest prompts to Android dialogs.
- `vulkan_present.*`: Android window ownership, swapchain, and presentation.
- `../gfx/vulkan/`: NVN replay adapter and Vulkan engine.
- `compat/`: Android implementations of the host's Apple platform interfaces.
- `tools/`: source-list parsing, ELF assembly conversion, and localized resources.

## Build locally

The current configuration uses JDK 17 or newer, Gradle 9.8.1, Android SDK 36,
NDK 29.0.14206865, CMake 3.31.6, and Python 3. ARM64 devices must support Android
12/API 31 or newer and Vulkan 1.3. These are build/manifest requirements, not a
claim that all matching devices have been tested. The first target is vivo X200
Ultra (Android 16/API 36, Adreno 830).

Prepare your own game files according to the root setup documentation. Generate
`dm_vault_tab.h`, `dm_hud_exp.h`, and `dm_bright_levels.inc` outside the repository
using the rules in `dm/host/Makefile`. Set `ANDROID_HOME` and `ANDROID_SDK_ROOT` to
the same SDK directory, and `JAVA_HOME` to the JDK. From the repository root:

```sh
gradle -p dm/android :app:assembleDebug :app:lintDebug \
  -PdmGenDir=/absolute/path/to/private/generated/headers \
  -PdmPython=/absolute/path/to/python
```

The APK is `dm/android/app/build/outputs/apk/debug/app-debug.apk`. Native linking
uses `--no-undefined`. `-PdmPrebuiltJni=...` is a packaging-only fallback; it is not
evidence that current native sources compile.

## Private runtime files

The APK and Git repository contain no game program, shaders, game art, or saves.
The development launcher currently needs separate provisioning; a general import
screen is not implemented.

| Location | Contents |
| --- | --- |
| App internal `files/dm/out/` | `main` and `sdk` rewriter products generated with `DM_TARGET=android`, including patched modules, fixups, and gates |
| App internal `files/dm/shaders/` | External `.xgs` bundles matching the game and Vulkan backend |
| App internal `files/dm/overlay/` | Required CASC touch overlay, including `overlay.manifest`, generated from the matching source and the user's resources |
| App internal `files/dm/overlay/touchart/` | Optional locally generated `dm_touch_joy.png`; otherwise the UI draws a fallback joystick |
| Shared `Download/data/` | User's resource archives and `.build.config`; requires Android's all-files-access setting |
| App internal `files/save/` | Saves; preserve during upgrades |
| App internal `files/temp/`, `files/dm/boot.log` | Temporary files and appended startup logs |

The launcher uses external app-private `dm/` as a fallback if no internal main
module exists. It checks required paths before startup; this check does not prove
shader identity or resource compatibility. Use the debug application's `run-as`
access to provision internal files, and overlay-install with the same signing
key. Never uninstall or clear data as an upgrade step. Back up existing saves
before device validation. Do not upload APKs or any of these private files to a PR.

The overlay generator `dm/ios/tools/gen_romfs_overlay.py` supports
`DM_DESWIZZLE_LIB` for a locally compiled native texture helper. Its thin CASC
output contains only changes and references the original archives; it must not
replace the original resource directory.

## Validation and remaining work

Run the root contribution checks, plus `:app:lintDebug`. The pure Java input
checks can run without a device:

```sh
javac -encoding UTF-8 -d /tmp/dm-input-tests \
  dm/android/app/src/main/java/com/xiaoboluo/d2r/TextInputRules.java \
  dm/android/tests/TextInputRulesTest.java
java -cp /tmp/dm-input-tests com.xiaoboluo.d2r.TextInputRulesTest
```

Device acceptance must separately record installation, genuine GPU output/FPS,
character creation, entering a world, moving and stopping, NPC interaction,
inventory/skills, sound, background/resume, and saving/reloading. Headless host
FPS is not a rendering result. Metal GPU timing and scanout timestamps are not
implemented by this backend; unavailable metrics must not be treated as zero cost.

The Vulkan bridge currently replays all draws and uses synchronous host
translation (`DM_NVN_SYNC=1`); the upstream asynchronous path needs an independent
object-lifetime audit. The presenter still waits for its queue each frame.
Advanced iOS skill artwork, empty-slot skill menus, specialized quantity/stat
popovers, and shop drag-follow visuals are incomplete. Generic controls and text
input do not establish complete iOS parity. macOS/iOS regressions must be run on
an Apple build host before upstream integration.

The overlay generator writes a source fingerprint and exact file hashes to `overlay.manifest`. The APK embeds the same generator-input fingerprint. Missing, incomplete, or mismatched overlays stop startup with a localized message before the host thread starts. Deploy the complete generated directory; do not copy a manifest onto old resources or generate one on the phone. Regenerate after generator/layout changes. This contract does not authenticate third-party resource downloads.
