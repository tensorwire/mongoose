//go:build linux && cgo

package mongoose

import "unsafe"

func HasFusedDispatch() bool   { return false }
func HasSignalMailbox() bool   { return false }

func KSignalMailbox(_ unsafe.Pointer) {}

func KSignalMailboxOnStream(_, _ unsafe.Pointer) {}

func KPartialStepQ4Fused(
	_, _, _, _, _ unsafe.Pointer,
	_, _, _, _ unsafe.Pointer,
	_, _ unsafe.Pointer,
	_, _, _, _, _ unsafe.Pointer,
	_, _, _ unsafe.Pointer,
	_, _, _ unsafe.Pointer,
	_ unsafe.Pointer,
	_, _ unsafe.Pointer,
	_, _, _, _, _, _, _ int,
	_, _, _, _ int,
) {
}

func KPartialStepQ4FusedOnStream(
	_, _, _, _, _ unsafe.Pointer,
	_, _, _, _ unsafe.Pointer,
	_, _ unsafe.Pointer,
	_, _, _, _, _ unsafe.Pointer,
	_, _, _ unsafe.Pointer,
	_, _, _ unsafe.Pointer,
	_ unsafe.Pointer,
	_, _ unsafe.Pointer,
	_, _, _, _, _, _, _ int,
	_, _, _, _ int,
	_ unsafe.Pointer,
) {
}
