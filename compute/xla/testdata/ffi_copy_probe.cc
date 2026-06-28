// ffi_copy_probe.cc: minimal ROCm XLA FFI handler that copies input to output
// via hipMemcpyAsync on the HIP stream.  Used by TestRegisterFFIHandlerDispatches
// to verify Go-side FFI handler registration and dispatch end-to-end.
//
// Build on the GPU machine:
//   JAXROCM=$HOME/venvs/jaxrocm/lib/python3.12/site-packages
//   ROCM_SDK=$JAXROCM/_rocm_sdk_core
//   $ROCM_SDK/lib/llvm/bin/amdclang++ -std=c++17 -fPIC -shared \
//     -I $JAXROCM/jaxlib/include \
//     -I $ROCM_SDK/include \
//     -o /tmp/ffi_copy_probe.so ffi_copy_probe.cc \
//     -L $ROCM_SDK/lib -lamdhip64
//
// The .so exports the symbol "CopyProbe" as an extern-C function with the
// XLA_FFI_Handler signature (XLA_FFI_Error* fn(XLA_FFI_CallFrame*)).
// The Go test resolves it via dlsym and passes it to Plugin.RegisterFFIHandler.

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

static ffi::Error CopyImpl(hipStream_t stream,
                            ffi::AnyBuffer in,
                            ffi::Result<ffi::AnyBuffer> out) {
  hipMemcpyAsync(out->untyped_data(), in.untyped_data(), in.size_bytes(),
                 hipMemcpyDeviceToDevice, stream);
  return ffi::Error::Success();
}

// CopyProbe is the XLA FFI handler symbol: extern "C" XLA_FFI_Error*
// CopyProbe(XLA_FFI_CallFrame*).  The Go test dlsym-resolves it and passes
// the pointer directly to Plugin.RegisterFFIHandler.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CopyProbe, CopyImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>());
