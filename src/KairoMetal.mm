#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

}
