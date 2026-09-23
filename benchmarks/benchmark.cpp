#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <span>
#include <vector>

import Kairo.GPU;

int main()
{
#if !defined(KAIRO_GPU_METAL)
    std::cout << "{\"schema\":\"kairo.gpu.benchmark.v1\",\"backend\":\"unavailable\"}\n";
    return 0;
#else
    using namespace kairo::gpu;

    constexpr std::size_t count = 1u << 20u;
    std::vector<float> lhs(count, 1.5f);
    std::vector<float> rhs(count, 2.0f);
    std::vector<float> output(count, 0.0f);

    Device device({ .backend = Backend::Metal, .debugName = "benchmark" });
    const auto a = device.CreateBuffer(
        { .byteSize = lhs.size() * sizeof(float), .usage = BufferUsage::Storage, .debugName = "lhs" });
    const auto b = device.CreateBuffer(
        { .byteSize = rhs.size() * sizeof(float), .usage = BufferUsage::Storage, .debugName = "rhs" });
    const auto out = device.CreateBuffer(
        { .byteSize = output.size() * sizeof(float), .usage = BufferUsage::Storage, .debugName = "out" });

    device.Upload(a, std::as_bytes(std::span(lhs)));
    device.Upload(b, std::as_bytes(std::span(rhs)));

    constexpr std::size_t iterations = 64u;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration)
        device.VectorMultiplyFloat(a, b, out, count);
    const auto stop = std::chrono::steady_clock::now();

    device.Download(out, std::as_writable_bytes(std::span(output)));
    const auto stats = device.Stats();
    const auto elapsedNs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    const bool correct =
        output.front() == 3.0f && output[count / 2u] == 3.0f && output.back() == 3.0f;

    std::cout
        << "{\"schema\":\"kairo.gpu.benchmark.v1\","
        << "\"backend\":\"metal\","
        << "\"elements\":" << count << ","
        << "\"iterations\":" << iterations << ","
        << "\"elapsed_ns\":" << elapsedNs << ","
        << "\"dispatch_count\":" << stats.dispatchCount << ","
        << "\"total_dispatch_ns\":" << stats.totalDispatchNanoseconds << ","
        << "\"max_dispatch_ns\":" << stats.maxDispatchNanoseconds << ","
        << "\"allocated_bytes\":" << stats.allocatedBytes << ","
        << "\"correct\":" << (correct ? "true" : "false")
        << "}\n";
    return correct ? 0 : 2;
#endif
}
