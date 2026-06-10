module;

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

    struct DeviceCapabilities final
    {
        std::uint64_t maxBufferBytes = 0;
        std::uint32_t maxThreadsPerGroup = 0;
        bool supportsFloat16 = false;
        bool supportsInt8Dot = false;
        bool supportsUnifiedMemory = false;
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

    struct BufferHandle final
    {
        std::uint64_t id = 0;
        BufferDesc desc{};

        [[nodiscard]]
        bool Valid() const noexcept
        {
            return id != 0;
        }
    };

    struct KernelHandle final
    {
        std::uint64_t id = 0;
        KernelDesc desc{};

        [[nodiscard]]
        bool Valid() const noexcept
        {
            return id != 0;
        }
    };

    enum class CommandKind
    {
        Upload,
        Download,
        Dispatch,
        Barrier
    };

    struct Command final
    {
        CommandKind kind = CommandKind::Barrier;
        BufferHandle buffer{};
        KernelHandle kernel{};
        DispatchSize groups{};
    };

    class CommandList final
    {
    public:
        void Upload(BufferHandle buffer)
        {
            m_commands.push_back({ .kind = CommandKind::Upload, .buffer = buffer });
        }

        void Download(BufferHandle buffer)
        {
            m_commands.push_back({ .kind = CommandKind::Download, .buffer = buffer });
        }

        void Dispatch(KernelHandle kernel, DispatchSize groups)
        {
            m_commands.push_back({ .kind = CommandKind::Dispatch, .kernel = kernel, .groups = groups });
        }

        void Barrier()
        {
            m_commands.push_back({ .kind = CommandKind::Barrier });
        }

        [[nodiscard]]
        const std::vector<Command>& Commands() const noexcept
        {
            return m_commands;
        }

        [[nodiscard]]
        bool Empty() const noexcept
        {
            return m_commands.empty();
        }

        void Clear()
        {
            m_commands.clear();
        }

    private:
        std::vector<Command> m_commands;
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

        [[nodiscard]]
        DeviceCapabilities Capabilities() const noexcept
        {
            return {};
        }

        [[nodiscard]]
        BufferHandle CreateBuffer(const BufferDesc& desc)
        {
            if (!IsAvailable())
            {
                throw UnsupportedBackend("Cannot create GPU buffer without a compiled backend.");
            }
            return { .id = ++m_nextResourceId, .desc = desc };
        }

        [[nodiscard]]
        KernelHandle CreateKernel(const KernelDesc& desc)
        {
            if (!IsAvailable())
            {
                throw UnsupportedBackend("Cannot create GPU kernel without a compiled backend.");
            }
            return { .id = ++m_nextResourceId, .desc = desc };
        }

    private:
        DeviceDesc m_desc;
        std::uint64_t m_nextResourceId = 0;
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
