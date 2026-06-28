//go:build linux

package pjrt

// TestRegisterFFIHandlerDispatches proves that RegisterFFIHandler can register
// a typed XLA FFI handler (XLA_FFI_Handler*) with the ROCm PJRT plugin and
// have it dispatch via a custom_call op.
//
// Prerequisites — build the probe .so on the GPU machine before running:
//
//	JAXROCM=$HOME/venvs/jaxrocm/lib/python3.12/site-packages
//	ROCM_SDK=$JAXROCM/_rocm_sdk_core
//	$ROCM_SDK/lib/llvm/bin/amdclang++ -std=c++17 -fPIC -shared \
//	  -I $JAXROCM/jaxlib/include \
//	  -I $ROCM_SDK/include \
//	  -o /tmp/ffi_copy_probe.so \
//	  <repo>/compute/xla/testdata/ffi_copy_probe.cc \
//	  -L $ROCM_SDK/lib -lamdhip64
//
// Run on the GPU machine (wrap in gputex for the GPU lock):
//
//	GOMLX_BACKEND=xla:rocm \
//	  PJRT_PLUGIN_LIBRARY_PATH=$HOME/.local/lib/go-xla/rocm \
//	  LD_LIBRARY_PATH=$ROCM_SDK/lib:$JAXROCM/_rocm_sdk_libraries_gfx120X_all/lib \
//	  FFI_PROBE_SO=/tmp/ffi_copy_probe.so \
//	  go test -v -run TestRegisterFFIHandlerDispatches -plugin rocm ./pjrt/

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>
*/
import "C"
import (
	"os"
	"testing"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-xla/stablehlo"
	"github.com/gomlx/go-xla/types/shapes"
)

// probeSOPath returns the path to the pre-built ffi_copy_probe.so.
// Override via FFI_PROBE_SO; default /tmp/ffi_copy_probe.so.
func probeSOPath() string {
	if p := os.Getenv("FFI_PROBE_SO"); p != "" {
		return p
	}
	return "/tmp/ffi_copy_probe.so"
}

// loadProbeHandler dlopen-s the probe .so, resolves the "CopyProbe" symbol
// (an XLA_FFI_Handler*), and returns it as unsafe.Pointer.
// Skips the test with t.Skip if the .so is missing.
// The caller must call C.dlclose(handle) when done.
func loadProbeHandler(t *testing.T) (handle unsafe.Pointer, handler unsafe.Pointer) {
	t.Helper()
	soPath := probeSOPath()
	cPath := C.CString(soPath)
	defer cFree(cPath)

	handle = C.dlopen(cPath, C.RTLD_LAZY|C.RTLD_GLOBAL)
	if handle == nil {
		t.Skipf("ffi_copy_probe.so not found at %s (build it first — see test doc comment): %s",
			soPath, C.GoString(C.dlerror()))
	}

	symC := C.CString("CopyProbe")
	defer cFree(symC)
	C.dlerror()
	handler = C.dlsym(handle, symC)
	if e := C.dlerror(); e != nil {
		C.dlclose(handle)
		t.Fatalf("dlsym CopyProbe in %s: %s", soPath, C.GoString(e))
	}
	if handler == nil {
		C.dlclose(handle)
		t.Fatalf("dlsym CopyProbe returned nil in %s", soPath)
	}
	return handle, handler
}

func TestRegisterFFIHandlerDispatches(t *testing.T) {
	if *FlagPluginName != "rocm" {
		t.Skipf("FFI handler dispatch test requires -plugin rocm (have %q)", *FlagPluginName)
	}

	handle, handlerPtr := loadProbeHandler(t)
	defer C.dlclose(handle)

	plugin, err := GetPlugin(*FlagPluginName)
	requireNoError(t, err, "GetPlugin rocm")
	t.Logf("plugin: %s", plugin)

	err = plugin.RegisterFFIHandler("copy_probe", handlerPtr)
	requireNoError(t, err, "RegisterFFIHandler")
	t.Logf("registered copy_probe handler at %p", handlerPtr)

	client, err := plugin.NewClient(nil)
	requireNoError(t, err, "NewClient")

	// Build a one-op StableHLO program: custom_call("copy_probe", input) -> output.
	// api_version=4 is XLA_CUSTOM_CALL_API_VERSION_TYPED_FFI.
	const n = 16
	shape := shapes.Make(dtypes.Float32, n)
	b := stablehlo.New("ffi_copy_probe_test")
	fn := b.Main()
	input := must1(fn.NamedInput("x", shape))
	results, err := stablehlo.CustomCall(
		"copy_probe",
		stablehlo.CustomCallAPIVersionTypedFFI,
		"", // no backendConfig
		"", // no operandLayouts
		"", // no resultLayouts
		[]shapes.Shape{shape},
		input,
	)
	requireNoError(t, err, "CustomCall build")
	must(fn.Return(results[0]))

	exec, err := client.Compile().WithStableHLO(must1(b.Build())).Done()
	requireNoError(t, err, "Compile")
	defer exec.Destroy()

	// Upload input: 0, 1, 2, ..., n-1 as float32.
	inputData := make([]float32, n)
	for i := range inputData {
		inputData[i] = float32(i)
	}
	inBuf, err := client.BufferFromHost().FromFlatDataWithDimensions(inputData, []int{n}).Done()
	requireNoError(t, err, "BufferFromHost")
	defer destroyAll(inBuf)

	outBufs, err := exec.Execute(inBuf).DonateNone().Done()
	requireNoError(t, err, "Execute")
	defer destroyAll(outBufs...)
	assertLen(t, outBufs, 1)

	flat, _, err := outBufs[0].ToFlatDataAndDimensions()
	requireNoError(t, err, "ToFlatDataAndDimensions")
	out := flat.([]float32)
	assertLen(t, out, n)

	for i, v := range out {
		if v != inputData[i] {
			t.Errorf("out[%d] = %v, want %v", i, v, inputData[i])
		}
	}
	t.Logf("PASS: copy_probe output matches input (%d float32 elements)", n)
}
