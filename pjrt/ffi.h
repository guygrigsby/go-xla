#ifndef GOMLX_GOPJRT_FFI
#define GOMLX_GOPJRT_FFI

#include "pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// find_ffi_extension walks the PJRT_Api extension chain and returns the
// PJRT_FFI_Extension pointer, or NULL if the plugin does not expose one.
const PJRT_FFI_Extension* find_ffi_extension(const PJRT_Api* api);

// call_PJRT_FFI_Register_Handler registers handler under target_name for
// platform_name via the FFI extension.  Returns a PJRT_Error* (NULL = OK).
PJRT_Error* call_PJRT_FFI_Register_Handler(const PJRT_FFI_Extension* ext,
                                            const char* target_name,
                                            size_t target_name_size,
                                            void* handler,
                                            const char* platform_name,
                                            size_t platform_name_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GOMLX_GOPJRT_FFI
