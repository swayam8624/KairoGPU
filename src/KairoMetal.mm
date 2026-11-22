// TEACHING NATIVE BRIDGE: C++ Kairo.GPU's `extern "C"` functions
// delegate selected buffer and Float32 compute operations to Objective-C++
// Metal APIs. MTLDevice and MTLBuffer objects are returned as opaque void*
// handles with explicit bridging retain/release ownership.
//
// Each native kernel path must check shader library/function, pipeline
// creation, buffer lengths, dispatch dimensions and command completion.
// The public device capabilities report a bounded feature set; a
// platform's generic Metal availability is not a promise of every
// tensor or arbitrary kernel operation.
//
// The Device wrapper owns the Metal device and buffers. This bridge
// should never be called with a stale/foreign device or destroyed buffer.
// Explicit upload/download is required by the higher-level TensorGPU API.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <cstring>

extern "C" {

// Retain the Objective-C MTLDevice for the C++ wrapper. The matching
// kairo_metal_destroy_device releases this ownership exactly once.
// C ABI ownership boundary: MTLCreateSystemDefaultDevice returns an ARC-
// managed Objective-C object; CFBridgingRetain gives C++ Device a +1 retain
// released by kairo_metal_destroy_device during Device teardown.
void* kairo_metal_create_device()
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device ? const_cast<void*>(CFBridgingRetain(device)) : nullptr;
}

void kairo_metal_destroy_device(void* handle)
{
    if (handle) CFBridgingRelease(handle);
}

unsigned long long kairo_metal_max_buffer_bytes(void* handle)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)handle;
    return device ? device.maxBufferLength : 0;
}

unsigned int kairo_metal_max_threads_per_group(void* handle)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)handle;
    return device ? 1024 : 0;
}

int kairo_metal_supports_float16(void* handle)
{
    return handle != nullptr;
}

int kairo_metal_supports_unified_memory(void* handle)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)handle;
    return device && device.hasUnifiedMemory;
}

// Allocate a shared-storage Metal buffer. Shared means CPU-visible contents
// for this native implementation; it is NOT a portable GPU synchronization
// policy. The C++ wrapper registers the retained native buffer in its table.
void* kairo_metal_create_buffer(void* deviceHandle, unsigned long long byteSize)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
    if (!device || byteSize == 0) return nullptr;
    id<MTLBuffer> buffer = [device newBufferWithLength:byteSize options:MTLResourceStorageModeShared];
    return buffer ? const_cast<void*>(CFBridgingRetain(buffer)) : nullptr;
}

void kairo_metal_destroy_buffer(void* handle)
{
    if (handle) CFBridgingRelease(handle);
}

// Raw host-byte transfer into MTLBuffer.contents. This helper checks the
// native buffer length but knows nothing about Tensor strides or Float32.
int kairo_metal_write_buffer(void* handle, const void* source, unsigned long long byteSize)
{
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    if (!buffer || !source || byteSize > buffer.length) return 0;
    std::memcpy(buffer.contents, source, static_cast<size_t>(byteSize));
    return 1;
}

// Reverse memcpy into caller-provided CPU bytes; callers must ensure the
// relevant GPU command completed first (current dispatch waits synchronously).
int kairo_metal_read_buffer(void* handle, void* destination, unsigned long long byteSize)
{
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    if (!buffer || !destination || byteSize > buffer.length) return 0;
    std::memcpy(destination, buffer.contents, static_cast<size_t>(byteSize));
    return 1;
}

// Compute Float32 elementwise addition with a Metal command.
 // Caller must supply three valid MTLBuffers of sufficient size.
// Vector Float32 path: on first use compile embedded MSL to a pipeline and
// queue, then encode buffers at bindings 0..2 and count at binding 3.
// dispatch_once keeps these objects process-scoped, NOT per-device. A future
// multi-device runtime must review pipeline/queue device compatibility.
// WaitUntilCompleted makes this call synchronous; the C++ timer includes
// compile/encode/submit/wait overhead, not GPU hardware kernel time alone.
int kairo_metal_vector_add(void* deviceHandle, void* lhsHandle, void* rhsHandle, void* outputHandle, unsigned long long count)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
    id<MTLBuffer> lhs = (__bridge id<MTLBuffer>)lhsHandle;
    id<MTLBuffer> rhs = (__bridge id<MTLBuffer>)rhsHandle;
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
    if (!device || !lhs || !rhs || !output || count == 0) return 0;
    static dispatch_once_t initialization;
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTLCommandQueue> queue = nil;
    dispatch_once(&initialization, ^{
        static NSString* source = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void kairo_vector_add(device const float* lhs [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* output [[buffer(2)]], constant uint& count [[buffer(3)]], uint index [[thread_position_in_grid]]) { if (index < count) output[index] = lhs[index] + rhs[index]; }";
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = library && !error ? [library newFunctionWithName:@"kairo_vector_add"] : nil;
        pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        queue = pipeline && !error ? [device newCommandQueue] : nil;
    });
    if (!pipeline || !queue) return 0;
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    const uint32_t elementCount = static_cast<uint32_t>(count);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:lhs offset:0 atIndex:0];
    [encoder setBuffer:rhs offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&elementCount length:sizeof(elementCount) atIndex:3];
    const NSUInteger width = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted && !command.error;
}

// Multiplication mirrors Add's binding scheme and wait semantics. Each GPU
// thread writes at most one Float32 output, guarded by its grid index.
int kairo_metal_vector_multiply(
    void* deviceHandle,
    void* lhsHandle,
    void* rhsHandle,
    void* outputHandle,
    unsigned long long count)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
    id<MTLBuffer> lhs = (__bridge id<MTLBuffer>)lhsHandle;
    id<MTLBuffer> rhs = (__bridge id<MTLBuffer>)rhsHandle;
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
    if (!device || !lhs || !rhs || !output || count == 0) return 0;

    static dispatch_once_t initialization;
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTLCommandQueue> queue = nil;
    dispatch_once(&initialization, ^{
        static NSString* source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "kernel void kairo_vector_multiply(device const float* lhs [[buffer(0)]], "
             "device const float* rhs [[buffer(1)]], device float* output [[buffer(2)]], "
             "constant uint& count [[buffer(3)]], uint index [[thread_position_in_grid]]) "
             "{ if (index < count) output[index] = lhs[index] * rhs[index]; }";
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function =
            library && !error ? [library newFunctionWithName:@"kairo_vector_multiply"] : nil;
        pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        queue = pipeline && !error ? [device newCommandQueue] : nil;
    });
    if (!pipeline || !queue) return 0;

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    const uint32_t elementCount = static_cast<uint32_t>(count);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:lhs offset:0 atIndex:0];
    [encoder setBuffer:rhs offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&elementCount length:sizeof(elementCount) atIndex:3];
    const NSUInteger width = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted && !command.error;
}

// Tiled Float32 GEMM, row-major [rows,inner] * [inner,columns]. Each 16x16
// threadgroup loads two threadgroup-memory tiles per K step, synchronizes
// before consumption and before overwriting tile storage, then stores one
// output per valid (row,column). Padding lanes load zeros; 17x17 exercises
// this edge case in tests/smoke.cpp. The pipeline/queue are first-device
// process-wide statics, and the returned result is synchronous.
int kairo_metal_matmul(
    void* deviceHandle,
    void* lhsHandle,
    void* rhsHandle,
    void* outputHandle,
    unsigned int rows,
    unsigned int inner,
    unsigned int columns)
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)deviceHandle;
    id<MTLBuffer> lhs = (__bridge id<MTLBuffer>)lhsHandle;
    id<MTLBuffer> rhs = (__bridge id<MTLBuffer>)rhsHandle;
    id<MTLBuffer> output = (__bridge id<MTLBuffer>)outputHandle;
    if (!device || !lhs || !rhs || !output || rows == 0 || inner == 0 || columns == 0) return 0;

    static dispatch_once_t initialization;
    static id<MTLComputePipelineState> pipeline = nil;
    static id<MTLCommandQueue> queue = nil;
    dispatch_once(&initialization, ^{
        static NSString* source =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "constant uint tile_width = 16;\n"
             "kernel void kairo_matmul(device const float* lhs [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* output [[buffer(2)]], constant uint3& shape [[buffer(3)]], uint2 local [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {\n"
             "  threadgroup float lhs_tile[16][16];\n"
             "  threadgroup float rhs_tile[16][16];\n"
             "  const uint row = group.y * tile_width + local.y;\n"
             "  const uint column = group.x * tile_width + local.x;\n"
             "  float sum = 0.0f;\n"
             "  for (uint base = 0; base < shape.y; base += tile_width) {\n"
             "    lhs_tile[local.y][local.x] = (row < shape.x && base + local.x < shape.y) ? lhs[row * shape.y + base + local.x] : 0.0f;\n"
             "    rhs_tile[local.y][local.x] = (base + local.y < shape.y && column < shape.z) ? rhs[(base + local.y) * shape.z + column] : 0.0f;\n"
             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
             "    for (uint offset = 0; offset < tile_width; ++offset) sum += lhs_tile[local.y][offset] * rhs_tile[offset][local.x];\n"
             "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
             "  }\n"
             "  if (row < shape.x && column < shape.z) output[row * shape.z + column] = sum;\n"
             "}";
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = library && !error ? [library newFunctionWithName:@"kairo_matmul"] : nil;
        pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        queue = pipeline && !error ? [device newCommandQueue] : nil;
    });
    if (!pipeline || !queue || pipeline.maxTotalThreadsPerThreadgroup < 256) return 0;

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    const uint32_t shape[3] = { rows, inner, columns };
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:lhs offset:0 atIndex:0];
    [encoder setBuffer:rhs offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:shape length:sizeof(shape) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((columns + 15u) / 16u, (rows + 15u) / 16u, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted && !command.error;
}

}
