package mongoose

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// decode_attn exists twice: in kernels/infer.metal, which is compiled ahead of
// time into infer.metallib, and as an inline C string in metal_impl_darwin.m
// used as a runtime-compile fallback when the metallib is missing.
//
// The metallib wins when present, so a divergence between the two does not
// fail loudly — it produces two different attention implementations selected by
// whether a file happens to be on disk. This project has already lost hours to
// exactly that class of bug (three copies of infer.metallib, two in sync and
// one stale), so the invariant is asserted rather than remembered.
func TestDecodeAttnCopiesAreIdentical(t *testing.T) {
	metalSrc, err := os.ReadFile("kernels/infer.metal")
	if err != nil {
		t.Fatalf("read kernels/infer.metal: %v", err)
	}
	objcSrc, err := os.ReadFile("metal_impl_darwin.m")
	if err != nil {
		t.Fatalf("read metal_impl_darwin.m: %v", err)
	}

	fromMetal := extractMetalFunc(t, string(metalSrc), "decode_attn")
	fromObjC := extractInlineFunc(t, string(objcSrc), "decode_attn")

	if fromMetal != fromObjC {
		t.Errorf("decode_attn has diverged between kernels/infer.metal and the "+
			"inline fallback in metal_impl_darwin.m.\n\n"+
			"--- kernels/infer.metal ---\n%s\n\n--- inline fallback ---\n%s",
			fromMetal, fromObjC)
	}
}

// extractMetalFunc pulls a kernel body out of a .metal source file.
func extractMetalFunc(t *testing.T, src, name string) string {
	t.Helper()
	start := strings.Index(src, "kernel void "+name+"(")
	if start < 0 {
		t.Fatalf("kernel %q not found in .metal source", name)
	}
	end := strings.Index(src[start:], "\n}\n")
	if end < 0 {
		t.Fatalf("unterminated kernel %q in .metal source", name)
	}
	return strings.TrimSpace(src[start : start+end+2])
}

var inlineLine = regexp.MustCompile(`(?m)^"(.*)\\n"`)

// extractInlineFunc reconstructs a kernel body from the C string literal form
// that metal_impl_darwin.m stores it in, undoing the escaping.
func extractInlineFunc(t *testing.T, src, name string) string {
	t.Helper()
	start := strings.Index(src, `"kernel void `+name+`(\n"`)
	if start < 0 {
		t.Fatalf("inline kernel %q not found", name)
	}
	end := strings.Index(src[start:], `"}\n";`)
	if end < 0 {
		t.Fatalf("unterminated inline kernel %q", name)
	}
	block := src[start : start+end+len(`"}\n";`)]

	var b strings.Builder
	for _, m := range inlineLine.FindAllStringSubmatch(block, -1) {
		line := strings.ReplaceAll(m[1], `\"`, `"`)
		line = strings.ReplaceAll(line, `\\`, `\`)
		b.WriteString(line)
		b.WriteByte('\n')
	}
	return strings.TrimSpace(b.String())
}

// The 4096-element threadgroup array that used to stage attention scores capped
// usable context at 4096 tokens — silently, by truncation rather than error.
// The tiled kernel stages only one tile, so nothing in decode_attn may depend
// on seqLen for its threadgroup allocation.
func TestDecodeAttnHasNoSeqLenSizedThreadgroupArray(t *testing.T) {
	src, err := os.ReadFile("kernels/infer.metal")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	body := extractMetalFunc(t, string(src), "decode_attn")
	if strings.Contains(body, "threadgroup float scores[4096]") {
		t.Error("decode_attn still stages all scores in a 4096-element threadgroup " +
			"array; that caps context at 4096 tokens by silent truncation")
	}
}
