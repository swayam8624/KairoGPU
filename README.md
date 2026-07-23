# KairoGPU

KairoGPU is the GPU backend abstraction for the Kairo ML stack. It defines how
the tensor runtime will talk to Metal, Vulkan, CUDA, or WebGPU without hardcoding
one graphics/compute API into the ML core.

## Problem

GPU acceleration is not a single feature. It requires:

- device discovery,
- buffer allocation and ownership,
- upload/readback synchronization,
- kernel compilation and specialization,
- command recording,
- profiling,
- fallback when a backend is unavailable.

If this leaks directly into the tensor API, every model layer becomes backend
specific.

## Solution

KairoGPU starts with a backend-neutral contract:

- `Backend`: `None`, `Metal`, `Vulkan`, `CUDA`, `WebGPU`.
- `DeviceDesc` and `DeviceCapabilities`.
- `BufferDesc`, `BufferHandle`.
- `KernelDesc`, `KernelHandle`.
- `CommandList` with upload, download, dispatch, and barrier commands.
- an Apple-platform Metal bridge for real device discovery, capabilities, and
  shared-storage buffer allocation.
- a validated Metal Float32 vector-add compute dispatch.
- explicit `UnsupportedBackend` failure for unlinked backends and for kernels
  that do not yet have a validated command implementation.

The current implementation has a bounded Metal resource lifecycle on Apple:
devices and buffers are real Metal objects and are released with `Device`.
Bounded shared-storage `Upload` and `Download` operations are implemented and
tested with a real buffer round trip. `Device::VectorAddFloat` compiles and
submits one fixed Metal compute kernel, waits for its completion, and is tested
through GPU buffer readback. Its compiled pipeline and command queue are cached
and reused across dispatches. This is a deliberate vertical slice, not a
general shader system: it has no generic resource-binding API, asynchronous
queue, matmul, autograd, or profiler.

## Where It Connects

- `KairoMath::Tensor`: eventually dispatches GPU-capable kernels through this
  package.
- `KairoScheduler`: prepares command lists and handles CPU-side async work.
- `KairoSIMD`: remains the optimized CPU fallback.
- `MLLibrary`: uses backend dispatch without changing model code.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build
ctest --test-dir build --output-on-failure
./build/KairoGPUSmoke
```

## Roadmap

1. Resource binding and a broader elementwise kernel library.
2. Tiled matmul, reductions, and explicit asynchronous submission/synchronization.
3. GPU profiling and tensor-runtime backend dispatch.
4. Vulkan/CUDA/WebGPU backends after the first backend is correct.
