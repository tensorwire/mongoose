# mongoose — build rules
#
# The metallib rule exists because its absence has cost this project hours more
# than once. kernels/*.metal is compiled ahead of time into infer.metallib, and
# at runtime metal_impl_darwin.m PREFERS that file over the inline kernel source
# it carries as a fallback. So editing a .metal file and rebuilding only the Go
# binary changes nothing observable — the stale metallib keeps winning, and the
# resulting "my fix did nothing" is indistinguishable from a wrong fix.
#
# Always `make kernels` after touching kernels/.

METAL      := xcrun -sdk macosx metal
METALLIB   := xcrun -sdk macosx metallib
METAL_SRC  := kernels/infer.metal
METAL_LIB  := kernels/infer.metallib
BUILD_DIR  := .build

.PHONY: all kernels test clean verify-kernels

all: kernels
	go build ./...

kernels: $(METAL_LIB)

$(METAL_LIB): $(METAL_SRC)
	@mkdir -p $(BUILD_DIR)
	$(METAL) -O3 -c $(METAL_SRC) -o $(BUILD_DIR)/infer.air
	$(METALLIB) $(BUILD_DIR)/infer.air -o $@
	@echo "built $@ ($$(md5 -q $@))"

# verify-kernels fails if the metallib is older than its source. Use it in CI:
# a stale metallib is a silent correctness bug, not a build error.
verify-kernels:
	@if [ $(METAL_SRC) -nt $(METAL_LIB) ]; then \
		echo "STALE: $(METAL_LIB) is older than $(METAL_SRC) — run 'make kernels'"; \
		exit 1; \
	fi
	@echo "kernels up to date"

test: kernels
	go test ./...

clean:
	rm -rf $(BUILD_DIR)
