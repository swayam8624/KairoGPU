#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

#include <cstring>

extern "C" {

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

int kairo_metal_write_buffer(void* handle, const void* source, unsigned long long byteSize)
{
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    if (!buffer || !source || byteSize > buffer.length) return 0;
    std::memcpy(buffer.contents, source, static_cast<size_t>(byteSize));
    return 1;
}

int kairo_metal_read_buffer(void* handle, void* destination, unsigned long long byteSize)
{
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)handle;
    if (!buffer || !destination || byteSize > buffer.length) return 0;
    std::memcpy(destination, buffer.contents, static_cast<size_t>(byteSize));
    return 1;
}

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

}
