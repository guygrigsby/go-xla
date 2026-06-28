package pjrt

/*
#include "pjrt_c_api.h"
#include "ffi.h"
*/
import "C"
import (
	"unsafe"

	"github.com/pkg/errors"
)

// RegisterFFIHandler registers handler (an XLA_FFI_Handler*) under targetName
// for this plugin's platform via the PJRT FFI extension.
//
// The plugin must expose PJRT_Extension_Type_FFI in its extension chain; if it
// does not, an error is returned and the caller may try the process-global
// XLA_FFI_GetApi path instead.
func (p *Plugin) RegisterFFIHandler(targetName string, handler unsafe.Pointer) error {
	ext := C.find_ffi_extension(p.api)
	if ext == nil {
		return errors.Errorf("plugin %q does not expose PJRT_Extension_Type_FFI", p.name)
	}

	cName := C.CString(targetName)
	defer cFree(cName)

	platform := p.platformName()
	cPlatform := C.CString(platform)
	defer cFree(cPlatform)

	pErr := C.call_PJRT_FFI_Register_Handler(
		ext,
		cName, C.size_t(len(targetName)),
		handler,
		cPlatform, C.size_t(len(platform)),
	)
	return toError(p, pErr)
}

// platformName returns the canonical XLA platform name for this plugin, used
// when registering FFI handlers.  The PJRT ROCm plugin reports "ROCM".
func (p *Plugin) platformName() string {
	if p.IsROCm() {
		return "ROCM"
	}
	if p.IsCUDA() {
		return "CUDA"
	}
	return "Host"
}
