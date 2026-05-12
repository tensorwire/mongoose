//go:build !darwin || !cgo

package mongoose

type SQ4Metal struct{}

func SQ4MetalInit() bool                { return false }
func HasSQ4MetalKernels() bool          { return false }
func NewSQ4Metal(eng *Metal) *SQ4Metal  { return nil }
func (s *SQ4Metal) Matvec(act, packed, bands, out *Tensor, rows, cols int) {}
func (s *SQ4Metal) OutlierCorrect(outlierIdx, outlierVal, packed, bands, act, out *Tensor, outlierCount, cols int) {}
