module;

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

export module Kairo.GPU;

export namespace kairo::gpu
{
    enum class Backend
    {
        None,
        Metal,
        Vulkan,
        CUDA,
        WebGPU
    };

    enum class BufferUsage : std::uint32_t
    {
        None = 0,
        Storage = 1u << 0u,
        Uniform = 1u << 1u,
        Upload = 1u << 2u,
        Readback = 1u << 3u
    };

    [[nodiscard]]
    constexpr BufferUsage operator|(BufferUsage lhs, BufferUsage rhs) noexcept
    {
        return static_cast<BufferUsage>(
            static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
    }

    struct DeviceDesc final
    {
        Backend backend = Backend::None;
        std::string debugName;
        bool enableValidation = true;
    };

    struct BufferDesc final
    {
        std::size_t byteSize = 0;
        BufferUsage usage = BufferUsage::Storage;
        std::string debugName;
    };

    struct DispatchSize final
    {
        std::uint32_t x = 1;
        std::uint32_t y = 1;
        std::uint32_t z = 1;
    };

    struct KernelDesc final
    {
        std::string name;
        DispatchSize threadsPerGroup{};
    };

    class UnsupportedBackend final : public std::runtime_error
    {
    public:
        explicit UnsupportedBackend(const std::string& message)
            : std::runtime_error(message)
        {
        }
    };

    class Device final
    {
    public:
        explicit Device(DeviceDesc desc = {})
            : m_desc(std::move(desc))
        {
            if (m_desc.backend != Backend::None)
            {
                throw UnsupportedBackend("KairoGPU backend implementation is not linked yet.");
            }
        }

        [[nodiscard]]
        const DeviceDesc& Desc() const noexcept
        {
            return m_desc;
        }

        [[nodiscard]]
        bool IsAvailable() const noexcept
        {
            return m_desc.backend != Backend::None;
        }

    private:
        DeviceDesc m_desc;
    };

    [[nodiscard]]
    inline std::size_t CompiledBackendCount() noexcept
    {
        return 1;
    }

    [[nodiscard]]
    inline bool IsBackendCompiled(Backend backend) noexcept
    {
        return backend == Backend::None;
    }
}
