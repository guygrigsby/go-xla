#include "ffi.h"

const PJRT_FFI_Extension* find_ffi_extension(const PJRT_Api* api) {
    const PJRT_Extension_Base* ext = api->extension_start;
    while (ext != NULL) {
        if (ext->type == PJRT_Extension_Type_FFI) {
            return (const PJRT_FFI_Extension*)ext;
        }
        ext = ext->next;
    }
    return NULL;
}

PJRT_Error* call_PJRT_FFI_Register_Handler(const PJRT_FFI_Extension* ext,
                                            const char* target_name,
                                            size_t target_name_size,
                                            void* handler,
                                            const char* platform_name,
                                            size_t platform_name_size) {
    PJRT_FFI_Register_Handler_Args args;
    args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
    args.target_name = target_name;
    args.target_name_size = target_name_size;
    args.handler = handler;
    args.platform_name = platform_name;
    args.platform_name_size = platform_name_size;
    args.traits = 0;
    return ext->register_handler(&args);
}
