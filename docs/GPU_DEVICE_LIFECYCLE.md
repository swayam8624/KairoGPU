# KairoGPU — a bounded compute API from C++ to native Metal

This is the teaching companion to
[`src/KairoGPU.cppm`](../src/KairoGPU.cppm),
[`src/KairoMetal.mm`](../src/KairoMetal.mm),
[`tests/smoke.cpp`](../tests/smoke.cpp) and
[`benchmarks/benchmark.cpp`](../benchmarks/benchmark.cpp).

The core distinction: **a buffer handle is a capability owned by one GPU
device, not a C++ pointer or a renderer texture.** KairoGPU is a compute
library used by selected mathematical/tensor operations. KairoRenderer's
presentation backend is a separate resource and API lifetime.

## 1. The public C++23 module

`export module Kairo.GPU;` defines the backend enumeration, device and
buffer descriptors, capabilities, command descriptions, typed handles and
`Device` lifetime. A Backend enum value is not a compiled implementation.

`Device(Backend::None)` is intentionally unavailable. On builds with
`KAIRO_GPU_METAL`, selecting Metal attempts native
`MTLCreateSystemDefaultDevice`; requesting another unsupported
implementation raises `UnsupportedBackend`. A device's capability
record includes maximum buffer length, workgroup size and selected
data type/unified-memory flags. Do not infer support for arbitrary
CUDA/Vulkan kernels merely because their enums exist.

## 2. C++ handle identity versus native allocation

`BufferHandle` includes a device `ownerId`, a monotonically assigned
buffer ID and a `BufferDesc` with size/usage. `Device::CreateBuffer`
checks the requested size and platform capabilities, creates the native
Metal buffer and stores its pointer in a per-device allocation table.

`DestroyBuffer` rejects a zero, foreign, already destroyed or
descriptor-mismatched handle. IDs are not reused while the Device
lives. Destructing Device releases any still-live native buffers and
then the device. Device is noncopyable so a second C++ object does
not silently share raw GPU ownership.

The C ABI in `KairoMetal.mm` transfers ownership of Objective-C objects
with bridging retain/release; native buffers must not outlive their
MTLDevice or be destroyed while active GPU commands still reference
them. Cross-device BufferHandles are not portable.

## 3. An explicit Float32 compute call

```text
CPU Float32 arrays
   |
Device::CreateBuffer and Upload
   |
Metal C ABI / MTLDevice / MTLBuffer
   |
VectorAddFloat, VectorMultiplyFloat or MatMulFloat
   |
Native Metal compute pipeline / GPU dispatch + completion
   |
Device::Download
   |
Caller-owned CPU Float32 result
```

`VectorAddFloat` and `VectorMultiplyFloat` require three same-length
Float32-capable buffers. `MatMulFloat` uses row-major matrices of
shapes [rows,inner] and [inner,columns], checks multiplication overflow
and buffer byte bounds, and dispatches the selected Metal native kernel.

`Upload` and `Download` transfer bytes; they do not interpret a
tensor's shape, strides or numerical meaning. KairoMath's
`TensorGPU.cppm` must check contiguity and dtype before converting a
Tensor to this raw byte contract. A sliced strided tensor cannot be
transferred as though packed without materialization.

**Exercise:** upload two length-N vectors, run addition, download and
compare against a scalar CPU reference. Repeat for 2×3 times 3×4
matrices, and include invalid sizes/foreign handles.

## 4. Command-list and kernel abstraction limitations

The module exposes `CommandList`, `Command`, `KernelHandle` and
`CreateKernel` descriptions. They do not, by themselves, prove that
arbitrary user shader compilation, deferred command-buffer submission,
multi-queue dependencies or universal kernel dispatch are implemented.
Verify the exact public operations callable in the current source.

Backend availability, feature coverage and semantic correctness are
three independent questions. This library's practical bounded native
implementation centers on selected Metal Float32 operations.

## 5. Measurements and synchronization

`DeviceStats` counts bytes and dispatch calls, while the wrapper times
the native call using CPU `steady_clock` around dispatch. This must
not automatically be labeled **GPU kernel execution latency**. A
native call can include command submission, waits and other host costs.

Benchmark at controlled sizes and distinguish:
- data upload time,
- kernel-only GPU event time (if instrumented),
- GPU-to-CPU readback time,
- total end-to-end latency,
- correctness and numerical tolerance versus CPU reference.

Use warmed pipelines and repeat trials. Unified physical memory on
Apple silicon does not eliminate ownership, command completion or
synchronization requirements.

## 6. Build and validation expectations

Read [`CMakeLists.txt`](../CMakeLists.txt) for the compiled Metal flag,
and [`tests/smoke.cpp`](../tests/smoke.cpp) for actual supported gates.
A test compiled on Linux without Metal cannot establish that the native
Metal kernel path executed. A host with Metal may still lack a requested
feature, buffer allocation or execution resource budget.

**Evidence status:** this is source documentation only. No new Metal
benchmark, GPU completion profile or end-to-end neural-network test
was run by the documentation pass.
