import Kairo.GPU;

#include <cassert>

int main()
{
    kairo::gpu::Device device;
    assert(!device.IsAvailable());
    assert(kairo::gpu::CompiledBackendCount() == 1);
    assert(kairo::gpu::IsBackendCompiled(kairo::gpu::Backend::None));
    return 0;
}
