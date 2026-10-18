#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable result values. JNI may map MISSING to -9 and INVALID to -10.
 * Missing directory/manifest is distinct from an installed but invalid overlay.
 * Validation is read-only: no repairs, process exit, save or resource writes.
 */
typedef enum {
    DM_OVERLAY_OK = 0,
    DM_OVERLAY_MISSING = 1,
    DM_OVERLAY_INVALID = 2
} dm_overlay_result;

dm_overlay_result dm_overlay_validate(const char *directory,
                                      const char *expected_source_sha256,
                                      char *reason, size_t reason_capacity);

#ifdef __cplusplus
}
#endif
