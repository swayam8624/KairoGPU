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
- explicit `UnsupportedBackend` failure when a real backend is not linked.

The current implementation is intentionally honest: it exposes the contract and
compiled-backend query surface, but does not pretend to run GPU kernels yet.

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
./build/KairoGPUSmoke
```

## Roadmap

1. Metal backend for Apple Silicon.
2. GPU buffer allocator and staging buffers.
3. Shader/kernel library for tensor elementwise ops and matmul.
4. Command submission and synchronization.
5. GPU profiling.
6. Vulkan/CUDA/WebGPU backends after the first backend is correct.
