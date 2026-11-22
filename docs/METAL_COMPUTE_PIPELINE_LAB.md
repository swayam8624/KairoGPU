# Metal compute in KairoGPU — executable mental model and source lab

This advanced laboratory builds on [the device-lifecycle guide](GPU_DEVICE_LIFECYCLE.md). Read [`src/KairoGPU.cppm`](../src/KairoGPU.cppm) (C++23 module and validation), [`src/KairoMetal.mm`](../src/KairoMetal.mm) (Objective-C++ Metal bridge), [`tests/smoke.cpp`](../tests/smoke.cpp) (actual checks), and [`benchmarks/benchmark.cpp`](../benchmarks/benchmark.cpp) (measurements). KairoMath consumes selected APIs through its [TensorGPU integration module](https://github.com/swayam8624/KairoMath/blob/master/TensorGPU.cppm).

**Evidence boundary:** this is a source-backed walkthrough, not a fresh host build, device experiment, or GPU profiler trace. Implemented operations are specifically bounded Metal Float32 add, multiply and matrix multiplication, not a generic shader-execution engine.

## 1. Three ownership layers, three different languages

`kairo::gpu::Device` is a C++ owner of one GPU backend. Its constructor creates a globally unique `ownerId` and attempts to create a native device when `Backend::Metal` is compiled. `Backend::None` means no native device and operations cannot allocate GPU buffers. A Vulkan, CUDA or WebGPU enum case is a **description**, not an implemented execution backend.

The `extern "C"` functions in the C++ module cross into `KairoMetal.mm`. Their opaque `void*` handles are **retained Objective-C objects**, not raw CPU tensor data: `CFBridgingRetain` transfers a retained `MTLDevice`/`MTLBuffer` reference to the C++ caller, and `CFBridgingRelease` releases it when done.

A public `BufferHandle` instead contains an integer ID, `ownerId` and `BufferDesc`. It is **not** the native pointer. `Device::NativeBuffer` validates ID range, owner identity, live native pointer and recorded byte size before returning an internal opaque pointer. `DestroyBuffer` releases one buffer, clears its table slot and updates accounting. IDs are not recycled. The `Device` destructor releases remaining buffers before the native device. The owner is noncopyable: a copied `BufferHandle` does not transfer ownership.

Crucially, the validation shown checks the recorded **size**, but does not presently compare every descriptor field (such as usage or debug name) with an immutable stored descriptor. Do not document it as full descriptor-authenticity validation.

## 2. CPU arrays to Metal bytes and back

The correct path for Float32 elementwise addition is:

```text
caller-owned std::array/std::vector<float>
   -> std::span<const std::byte>
   -> Device::Upload(BufferHandle, bytes)
   -> NativeBuffer owner/liveness validation
   -> C ABI kairo_metal_write_buffer
   -> MTLBuffer.contents + memcpy (Shared storage)
   -> Device::VectorAddFloat(a, b, out, count)
   -> native Metal command completion
   -> Device::Download(out, byte span)
   -> caller-owned CPU result
```

`Upload` and `Download` copy **bytes**, not tensors. They check `bytes.size() <= buffer.desc.byteSize` and the native buffer's ownership/size against the internal allocation table. They do not know NHWC, strides, activation units, labels or dtypes. Every caller needs to check `count * sizeof(float)`, the correct element order, and lifetime of the backing spans.

The bridge allocates buffers with `MTLResourceStorageModeShared`. On Apple Silicon, unified physical memory does not make this a portable aliasing contract: CPU and GPU accesses still have a command completion and ordering relationship. Here dispatch calls commit a command buffer and **wait until completed**, so a subsequent download sees the completed result without needing an external asynchronous fence API.

## 3. The vector-add kernel, one thread per index

In `kairo_metal_vector_add`, a first-use `dispatch_once` block compiles embedded Metal Shading Language into an `MTLLibrary`, selects the function `kairo_vector_add`, creates a `MTLComputePipelineState` and a `MTLCommandQueue`. The CPU creates a command buffer and compute encoder, binds three buffers at indices 0, 1 and 2, then passes a `uint32_t` count through `setBytes` at index 3.

Each GPU thread receives `thread_position_in_grid` as an index. If the index is in bounds, it writes `out[i] = lhs[i] + rhs[i]`. Dispatch uses a one-dimensional grid and threadgroup width limited by the pipeline's `maxTotalThreadsPerThreadgroup` and 256. The multiply path follows the same structure but multiplies values.

The C++ wrapper checks `count > 0`, `count <= UINT32_MAX`, and that all three buffer descriptors cover `count * sizeof(float)`. Then it calls `NativeBuffer` on each handle, crosses the C ABI, and measures the call's **host elapsed time**. A failed native operation raises `UnsupportedBackend`; a stale/foreign/undersized resource is invalid argument.

The static pipeline/queue are **process-scoped**, initialized on the first invocation, not stored per `Device` instance. This matters for future support of multiple distinct native Metal devices: inspect resource/device compatibility and do not assume process-global pipeline reuse is universally safe.

## 4. The 16x16 tiled matrix multiplication

`Device::MatMulFloat` expects row-major `A[rows,inner]`, `B[inner,columns]`, `C[rows,columns]`. It validates nonzero dimensions and multiplication overflow in the required byte sizes, checks buffers, then forwards unsigned dimensions to the bridge.

The Metal kernel computes a tile with **256 threads**, arranged as 16x16. Each thread maps its `threadgroup_position_in_grid` and local position to one output (row,column). For every K tile:

1. Load one `A` tile entry into `threadgroup float lhs_tile[16][16]`.
2. Load one `B` tile entry into `rhs_tile[16][16]`, substituting zero beyond logical edges.
3. Call `threadgroup_barrier(mem_flags::mem_threadgroup)` so every thread sees complete local tiles.
4. Accumulate the 16 products into one thread-local scalar.
5. Barrier again before reusing shared tile storage for the next K slice.

Only threads whose row and column are within the true matrix bounds write results. That is why an odd size such as 17x17 exercises out-of-range tile lanes and boundary masking. The smoke test includes this dimension.

**Work it by hand:** for `A[2,3]` and `B[3,2]`, derive `C(0,1) = A(0,0)B(0,1) + A(0,1)B(1,1) + A(0,2)B(2,1)`. Then map it to the shader's `lhs[row*inner+k]` and `rhs[k*columns+column]`. Prove that zero-padding the unfinished part of the 16-wide tile does not alter valid output.

The bridge rejects a pipeline with fewer than 256 maximum threads per group, rather than silently changing tile configuration. The implementation is a useful reference GEMM, not proof of production GEMM speed versus MPS, BLAS or a vendor-tuned library.

## 5. Public contracts that are not a shader runtime

`CommandList` contains records of upload/download/dispatch/barrier descriptions. `CreateKernel` produces an owned `KernelHandle` identifier and description. In the reviewed `Device` code there is **no public Submit(CommandList) implementation**, no general `Dispatch(KernelHandle)` executor, and no externally supplied MSL compilation path. A `CommandList::Barrier()` record is not a native GPU barrier in itself.

Do not use the presence of these types to claim arbitrary user kernels, asynchronous fences, command dependency graphs, GPU graph capture, or graphics rendering.

## 6. What the stats actually measure

`DeviceStats` tracks current buffer count and allocated bytes, cumulative upload/download byte counts, and dispatch counts/durations. `RecordDispatch` receives a `steady_clock` interval around the **synchronous bridge invocation**. That interval may include pipeline initialization, command encoding, queuing, waiting and GPU execution. It is not Metal hardware timestamp duration.

A useful future benchmark should separately time: allocation, upload, pipeline warmup/compilation, dispatch+wait, download and overall end-to-end latency. Compare numeric output with a scalar reference; report input size, CPU/GPU model, OS, compiler, iterations, warmup, summary statistics and measured power/thermal state if relevant. Never report speedup without an equivalently measured CPU comparison.

## 7. Review findings and test gaps

The existing [smoke test](../tests/smoke.cpp) covers compiled backends, a command-list record, native device availability, buffer round-trip and owner/descriptor rejection, arithmetic, tiled matrix multiplication including a 17x17 boundary case, and destroyed/foreign-handle failures when Metal is available.

Additional investigations, not assertions of already fixed bugs:

- Check whether multiple native devices can safely share process-global pipelines and queues.
- Check complete descriptor authenticity where a forged usage/debug-name field should be rejected or explicitly ignored.
- Check `Metal` Float16 capability reporting against actually implemented Half kernels: the bridge returns availability of a device, but the reviewed dispatches are Float32.
- Exercise tiny, large and nonmultiple vector counts, odd matrix dimensions, invalid counts, and all documented native error paths.
- Inspect shader and allocation behavior under repeated concurrent calls, especially global pipeline initialization and device/resource lifetimes.
- For `uint32_t` near-limit dimensions, check grid-ceiling arithmetic independently of buffer-size validation.

## 8. Cross-repository integration exercise

[TensorGPU.cppm](https://github.com/swayam8624/KairoMath/blob/master/TensorGPU.cppm) decides tensor shapes, supported dtypes, contiguity, allocation and CPU/GPU transfer policy. KairoGPU knows only bytes, buffer handles and its three specialized Float32 operations. KairoSIMD is a CPU inner-loop implementation and KairoScheduler manages CPU work; neither substitutes for a Metal command buffer.

Trace `TensorGPU` -> `Device::CreateBuffer` -> `Upload` -> `MatMulFloat` -> `kairo_metal_matmul` -> `Download` -> Tensor. At each arrow, write down who owns the memory and which API guarantees completion. This is the minimum knowledge needed before implementing a new GPU kernel independently.

**Verification status:** explanation and review only. Build, host execution, sanitizer coverage, CPU/GPU parity and performance results require separate evidence.
