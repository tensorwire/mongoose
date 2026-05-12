// sq4_metal_darwin.m — Self-contained SQ4 matvec dispatch.
// Loads sq4_matvec.metallib. Does NOT reference any globals from metal_impl_darwin.m.
// GPU-side: packed nibbles [sign:1|band:3], 2 per byte. 16-entry LUT.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;

static id<MTLLibrary> g_sq4_lib = nil;
id<MTLComputePipelineState> g_ps_sq4_matvec = nil;
id<MTLComputePipelineState> g_ps_sq4_outlier = nil;

static id<MTLBuffer> g_sq4_zero_buf = nil;
static id<MTLBuffer> g_sq4_cb_oc = nil;

static id<MTLBuffer> sq4_const_u(uint32_t v) {
    return [g_device newBufferWithBytes:&v length:4 options:MTLResourceStorageModeShared];
}

int mtl_sq4_init(const char* path) {
    if (g_sq4_lib) return 1;
    if (!g_device || !path) return 0;

    NSString* p = [NSString stringWithUTF8String:path];
    if (![[NSFileManager defaultManager] fileExistsAtPath:p]) return 0;

    NSError* e = nil;
    g_sq4_lib = [g_device newLibraryWithURL:[NSURL fileURLWithPath:p] error:&e];
    if (!g_sq4_lib) {
        NSLog(@"[SQ4] failed to load %@: %@", p, e);
        return 0;
    }

    id<MTLFunction> fn_mv = [g_sq4_lib newFunctionWithName:@"sq4_matvec"];
    id<MTLFunction> fn_oc = [g_sq4_lib newFunctionWithName:@"sq4_outlier_correct"];
    if (fn_mv) g_ps_sq4_matvec = [g_device newComputePipelineStateWithFunction:fn_mv error:&e];
    if (fn_oc) g_ps_sq4_outlier = [g_device newComputePipelineStateWithFunction:fn_oc error:&e];
    if (!g_ps_sq4_matvec) {
        NSLog(@"[SQ4] pipeline creation failed");
        return 0;
    }

    uint32_t zero = 0;
    g_sq4_zero_buf = [g_device newBufferWithBytes:&zero length:4 options:MTLResourceStorageModeShared];
    g_sq4_cb_oc = [g_device newBufferWithBytes:&zero length:4 options:MTLResourceStorageModeShared];

    NSLog(@"[SQ4] loaded %@", p);
    return 1;
}

int mtl_sq4_ready(void) {
    return g_ps_sq4_matvec != nil ? 1 : 0;
}

void mtl_sq4_matvec(void* actRef, void* packedRef, void* bandsRef, void* outRef,
    int rows, int cols) {
    if (!g_ps_sq4_matvec) return;
    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_sq4_matvec];
    [enc setBuffer:(__bridge id<MTLBuffer>)actRef    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)packedRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bandsRef  offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)outRef    offset:0 atIndex:3];
    [enc setBuffer:sq4_const_u(cols) offset:0 atIndex:4];
    [enc setBuffer:sq4_const_u(rows) offset:0 atIndex:5];
    [enc setBuffer:g_sq4_zero_buf offset:0 atIndex:6];
    [enc setBuffer:g_sq4_zero_buf offset:0 atIndex:7];
    [enc setBuffer:g_sq4_zero_buf offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    }
}

void mtl_sq4_matvec_fused(void* actRef, void* packedRef, void* bandsRef, void* outRef,
    int rows, int cols,
    void* outlierIdxRef, void* outlierValRef, int outlierCount) {
    if (!g_ps_sq4_matvec) return;
    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_sq4_matvec];
    [enc setBuffer:(__bridge id<MTLBuffer>)actRef    offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)packedRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)bandsRef  offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)outRef    offset:0 atIndex:3];
    [enc setBuffer:sq4_const_u(cols) offset:0 atIndex:4];
    [enc setBuffer:sq4_const_u(rows) offset:0 atIndex:5];
    if (outlierCount > 0 && outlierIdxRef && outlierValRef) {
        [enc setBuffer:(__bridge id<MTLBuffer>)outlierIdxRef offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)outlierValRef offset:0 atIndex:7];
    } else {
        [enc setBuffer:g_sq4_zero_buf offset:0 atIndex:6];
        [enc setBuffer:g_sq4_zero_buf offset:0 atIndex:7];
    }
    ((uint32_t*)g_sq4_cb_oc.contents)[0] = (uint32_t)outlierCount;
    [enc setBuffer:g_sq4_cb_oc offset:0 atIndex:8];
    [enc dispatchThreadgroups:MTLSizeMake((rows + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    }
}

void mtl_sq4_outlier_correct(void* outlierIdxRef, void* outlierValRef,
    void* packedRef, void* bandsRef, void* actRef, void* outRef,
    int outlierCount, int cols) {
    if (!g_ps_sq4_outlier || outlierCount <= 0) return;
    @autoreleasepool {
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ps_sq4_outlier];
    [enc setBuffer:(__bridge id<MTLBuffer>)outlierIdxRef offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)outlierValRef offset:0 atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)packedRef     offset:0 atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)bandsRef      offset:0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)actRef        offset:0 atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)outRef        offset:0 atIndex:5];
    [enc setBuffer:sq4_const_u(cols) offset:0 atIndex:6];
    [enc setBuffer:sq4_const_u(outlierCount) offset:0 atIndex:7];
    int groups = (outlierCount + 255) / 256;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted];
    }
}
