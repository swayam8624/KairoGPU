module;

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(KAIRO_GPU_METAL)
extern "C" {
void* kairo_metal_create_device();
void kairo_metal_destroy_device(void*);
unsigned long long kairo_metal_max_buffer_bytes(void*);
unsigned int kairo_metal_max_threads_per_group(void*);
int kairo_metal_supports_float16(void*);
int kairo_metal_supports_unified_memory(void*);
void* kairo_metal_create_buffer(void*, unsigned long long);
void kairo_metal_destroy_buffer(void*);
}
#endif

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
            if (m_desc.backend == Backend::None)
            {
                return;
            }
#if defined(KAIRO_GPU_METAL)
            if (m_desc.backend == Backend::Metal)
            {
                m_nativeDevice = kairo_metal_create_device();
                if (!m_nativeDevice) throw UnsupportedBackend("Metal is not available on this host.");
                return;
            }
#endif
            throw UnsupportedBackend("Requested KairoGPU backend implementation is not linked.");
        }

        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;

        ~Device()
        {
#if defined(KAIRO_GPU_METAL)
            for (void* buffer : m_nativeBuffers) kairo_metal_destroy_buffer(buffer);
            kairo_metal_destroy_device(m_nativeDevice);
#endif
        }

        [[nodiscard]]
        const DeviceDesc& Desc() const noexcept
        {
            return m_desc;
        }

        [[nodiscard]]
        bool IsAvailable() const noexcept
        {
            return m_nativeDevice != nullptr;
        }

        [[nodiscard]]
        DeviceCapabilities Capabilities() const noexcept
        {
#if defined(KAIRO_GPU_METAL)
            if (m_desc.backend == Backend::Metal && m_nativeDevice)
            {
                return { .maxBufferBytes = kairo_metal_max_buffer_bytes(m_nativeDevice),
                    .maxThreadsPerGroup = kairo_metal_max_threads_per_group(m_nativeDevice),
                    .supportsFloat16 = kairo_metal_supports_float16(m_nativeDevice) != 0,
                    .supportsInt8Dot = false,
                    .supportsUnifiedMemory = kairo_metal_supports_unified_memory(m_nativeDevice) != 0 };
            }
#endif
            return {};
        }

        [[nodiscard]]
        BufferHandle CreateBuffer(const BufferDesc& desc)
        {
            if (!IsAvailable() || desc.byteSize == 0)
            {
                throw UnsupportedBackend("Cannot create a GPU buffer without an available backend and non-zero size.");
            }
#if defined(KAIRO_GPU_METAL)
            if (m_desc.backend == Backend::Metal)
            {
                void* buffer = kairo_metal_create_buffer(m_nativeDevice, static_cast<unsigned long long>(desc.byteSize));
                if (!buffer) throw std::runtime_error("Metal buffer allocation failed.");
                m_nativeBuffers.push_back(buffer);
                return { .id = ++m_nextResourceId, .desc = desc };
            }
#endif
            throw UnsupportedBackend("GPU buffer allocation is unavailable for this backend.");
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
        void* m_nativeDevice = nullptr;
        std::vector<void*> m_nativeBuffers;
    };

    [[nodiscard]]
    inline std::size_t CompiledBackendCount() noexcept
    {
        return 1
#if defined(KAIRO_GPU_METAL)
            + 1
#endif
            ;
    }

    [[nodiscard]]
    inline bool IsBackendCompiled(Backend backend) noexcept
    {
        if (backend == Backend::None) return true;
#if defined(KAIRO_GPU_METAL)
        if (backend == Backend::Metal) return true;
#endif
        return false;
    }
}
