package pjrt

import "testing"

func TestIsRocm(t *testing.T) {
	for name, want := range map[string]bool{
		"rocm": true, "ROCM": true, "xla_rocm_plugin": true,
		"cuda": false, "cpu": false,
	} {
		if got := isRocm(name); got != want {
			t.Errorf("isRocm(%q)=%v want %v", name, got, want)
		}
	}
}
