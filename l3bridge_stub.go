//go:build !linux || !cgo

package mongoose

func (c *CUDA) AllocL3Bridge(bytes int) *L3Bridge      { return nil }
func (c *CUDA) RegisterL3Bridge(bridge *L3Bridge) error { return nil }
