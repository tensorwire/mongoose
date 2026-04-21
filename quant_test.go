package mongoose

import (
	"testing"
)

func TestHasQ8Matvec(t *testing.T) {
	// On non-Linux or without kernels, should return false
	result := HasQ8Matvec()
	t.Logf("HasQ8Matvec: %v", result)
}

func TestHasQ4Matvec(t *testing.T) {
	result := HasQ4Matvec()
	t.Logf("HasQ4Matvec: %v", result)
}

func TestAsTensorEngine_CPU(t *testing.T) {
	cpu := &CPU{}
	te := AsTensorEngine(cpu)
	if te != nil {
		t.Error("CPU should not be a TensorEngine")
	}
}

func TestAsResidentWeightEngine_CPU(t *testing.T) {
	cpu := &CPU{}
	rwe := AsResidentWeightEngine(cpu)
	if rwe != nil {
		t.Error("CPU should not be a ResidentWeightEngine")
	}
}

func TestIsTensorEngine_CPU(t *testing.T) {
	cpu := &CPU{}
	if IsTensorEngine(cpu) {
		t.Error("CPU should not implement TensorEngine")
	}
}
