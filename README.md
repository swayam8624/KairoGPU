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
- a 16-by-16 threadgroup-tiled Metal Float32 matrix-multiplication dispatch.
- explicit `UnsupportedBackend` failure for unlinked backends and for kernels
  that do not yet have a validated command implementation.

The frozen v1 implementation is intentionally a bounded **Metal compute
runtime**, not a claim of four finished GPU backends. Devices and buffers are
real Metal objects. Buffer handles carry a device-owner identity, destroyed and
foreign handles fail closed, allocation size is checked against device-owned
metadata, and buffer/kernel identifier spaces cannot alias. Shared-storage
`Upload`/`Download`, explicit buffer destruction, Float32 vector add,
element-wise multiply, and row-major 16-by-16 tiled matrix multiplication are
covered by native smoke paths.

`DeviceStats` records live/allocated buffers, upload/download bytes, dispatch
count, and host-observed synchronous dispatch duration. This is useful
regression telemetry but is not presented as GPU hardware timestamp data.
Generic shader binding and asynchronous command submission remain outside the
v1 80% scope.

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

## Frozen V1 And Later Work

Wave B freezes v1 around the validated Metal path. Build the optional benchmark
with `-DKAIRO_GPU_BUILD_BENCHMARK=ON`; it emits
`kairo.gpu.benchmark.v1` JSON for a one-million-element workload.

Generic resource binding, asynchronous submission, hardware timestamp queries,
additional reductions, and Vulkan/CUDA/WebGPU implementations are v2 work.
They are deliberately excluded from the v1 completion denominator rather than
represented as partially implemented backends.

See [STATUS.md](STATUS.md) for the exact scope and limitations.
