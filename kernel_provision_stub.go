//go:build !linux || !cgo

package mongoose

func ProvisionKernels() string { return "" }
