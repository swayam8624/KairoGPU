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
    const auto buffer = metal.CreateBuffer({ .byteSize = 4096, .usage = kairo::gpu::BufferUsage::Storage, .debugName = "round-trip" });
    assert(buffer.Valid());
    const std::array<std::byte, 4> upload{ std::byte{0x10}, std::byte{0x20}, std::byte{0x30}, std::byte{0x40} };
    std::array<std::byte, 4> download{};
    metal.Upload(buffer, upload);
    metal.Download(buffer, download);
    assert(upload == download);

    const std::array<float, 4> lhs{ 1.0f, 2.0f, 3.0f, 4.0f };
    const std::array<float, 4> rhs{ 10.0f, 20.0f, 30.0f, 40.0f };
    std::array<float, 4> sum{};
    const auto lhsBuffer = metal.CreateBuffer({ .byteSize = sizeof(lhs), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "vector-lhs" });
    const auto rhsBuffer = metal.CreateBuffer({ .byteSize = sizeof(rhs), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "vector-rhs" });
    const auto sumBuffer = metal.CreateBuffer({ .byteSize = sizeof(sum), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "vector-output" });
    metal.Upload(lhsBuffer, std::as_bytes(std::span(lhs)));
    metal.Upload(rhsBuffer, std::as_bytes(std::span(rhs)));
    metal.VectorAddFloat(lhsBuffer, rhsBuffer, sumBuffer, sum.size());
    metal.Download(sumBuffer, std::as_writable_bytes(std::span(sum)));
    assert(sum[0] == 11.0f && sum[3] == 44.0f);
    metal.VectorAddFloat(sumBuffer, lhsBuffer, rhsBuffer, sum.size());
    metal.Download(rhsBuffer, std::as_writable_bytes(std::span(sum)));
    assert(sum[0] == 12.0f && sum[3] == 48.0f);

    const std::array<float, 6> matrixLhs{ 1, 2, 3, 4, 5, 6 };
    const std::array<float, 6> matrixRhs{ 7, 8, 9, 10, 11, 12 };
    std::array<float, 4> matrixOutput{};
    const auto matrixLhsBuffer = metal.CreateBuffer({ .byteSize = sizeof(matrixLhs), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "matmul-lhs" });
    const auto matrixRhsBuffer = metal.CreateBuffer({ .byteSize = sizeof(matrixRhs), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "matmul-rhs" });
    const auto matrixOutputBuffer = metal.CreateBuffer({ .byteSize = sizeof(matrixOutput), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "matmul-output" });
    metal.Upload(matrixLhsBuffer, std::as_bytes(std::span(matrixLhs)));
    metal.Upload(matrixRhsBuffer, std::as_bytes(std::span(matrixRhs)));
    metal.MatMulFloat(matrixLhsBuffer, matrixRhsBuffer, matrixOutputBuffer, 2, 3, 2);
    metal.Download(matrixOutputBuffer, std::as_writable_bytes(std::span(matrixOutput)));
    assert(matrixOutput[0] == 58.0f && matrixOutput[1] == 64.0f);
    assert(matrixOutput[2] == 139.0f && matrixOutput[3] == 154.0f);

    std::array<float, 17 * 17> boundaryLhs{};
    std::array<float, 17 * 17> boundaryIdentity{};
    std::array<float, 17 * 17> boundaryOutput{};
    for (std::size_t row = 0; row < 17; ++row)
    {
        for (std::size_t column = 0; column < 17; ++column)
        {
            boundaryLhs[row * 17 + column] = static_cast<float>(row * 17 + column);
            boundaryIdentity[row * 17 + column] = row == column ? 1.0f : 0.0f;
        }
    }
    const auto boundaryLhsBuffer = metal.CreateBuffer({ .byteSize = sizeof(boundaryLhs), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "boundary-lhs" });
    const auto boundaryIdentityBuffer = metal.CreateBuffer({ .byteSize = sizeof(boundaryIdentity), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "boundary-identity" });
    const auto boundaryOutputBuffer = metal.CreateBuffer({ .byteSize = sizeof(boundaryOutput), .usage = kairo::gpu::BufferUsage::Storage, .debugName = "boundary-output" });
    metal.Upload(boundaryLhsBuffer, std::as_bytes(std::span(boundaryLhs)));
    metal.Upload(boundaryIdentityBuffer, std::as_bytes(std::span(boundaryIdentity)));
    metal.MatMulFloat(boundaryLhsBuffer, boundaryIdentityBuffer, boundaryOutputBuffer, 17, 17, 17);
    metal.Download(boundaryOutputBuffer, std::as_writable_bytes(std::span(boundaryOutput)));
    assert(boundaryOutput == boundaryLhs);
#endif
    return 0;
}
