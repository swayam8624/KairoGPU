#include <array>
#include <bit>
#include <cassert>
#include <span>

import Kairo.GPU;

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
    const std::array<std::byte, 4> upload{ std::byte{0x10}, std::byte{0x20}, std::byte{0x30}, std::byte{0x40} };
    std::array<std::byte, 4> download{};
    metal.Upload(buffer, upload);
    metal.Download(buffer, download);
    assert(upload == download);

    const std::array<float, 4> lhs{ 1.0f, 2.0f, 3.0f, 4.0f };
    const std::array<float, 4> rhs{ 10.0f, 20.0f, 30.0f, 40.0f };
    std::array<float, 4> sum{};
    const auto lhsBuffer = metal.CreateBuffer({ .byteSize = sizeof(lhs), .usage = kairo::gpu::BufferUsage::Storage });
    const auto rhsBuffer = metal.CreateBuffer({ .byteSize = sizeof(rhs), .usage = kairo::gpu::BufferUsage::Storage });
    const auto sumBuffer = metal.CreateBuffer({ .byteSize = sizeof(sum), .usage = kairo::gpu::BufferUsage::Storage });
    metal.Upload(lhsBuffer, std::as_bytes(std::span(lhs)));
    metal.Upload(rhsBuffer, std::as_bytes(std::span(rhs)));
    metal.VectorAddFloat(lhsBuffer, rhsBuffer, sumBuffer, sum.size());
    metal.Download(sumBuffer, std::as_writable_bytes(std::span(sum)));
    assert(sum[0] == 11.0f && sum[3] == 44.0f);
    metal.VectorAddFloat(sumBuffer, lhsBuffer, rhsBuffer, sum.size());
    metal.Download(rhsBuffer, std::as_writable_bytes(std::span(sum)));
    assert(sum[0] == 12.0f && sum[3] == 48.0f);
#endif
    return 0;
}
