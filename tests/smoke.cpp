import Kairo.GPU;

#include <cassert>

int main()
{
    kairo::gpu::Device device;
    assert(!device.IsAvailable());
    assert(kairo::gpu::CompiledBackendCount() ==
#if defined(KAIRO_GPU_METAL)
        2
#else
        1
#endif
    );
    assert(kairo::gpu::IsBackendCompiled(kairo::gpu::Backend::None));
    kairo::gpu::CommandList commands;
    commands.Barrier();
    assert(!commands.Empty());
#if defined(KAIRO_GPU_METAL)
    assert(kairo::gpu::IsBackendCompiled(kairo::gpu::Backend::Metal));
    kairo::gpu::Device metal({ .backend = kairo::gpu::Backend::Metal, .debugName = "smoke" });
    assert(metal.IsAvailable());
    assert(metal.Capabilities().maxBufferBytes > 0);
    const auto buffer = metal.CreateBuffer({ .byteSize = 4096, .usage = kairo::gpu::BufferUsage::Storage });
    assert(buffer.Valid());
#endif
    return 0;
}
