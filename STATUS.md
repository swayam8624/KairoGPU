# KairoGPU Status

Wave: B — bounded compute-backend completion  
Frozen v1 target: 80/100  
Source gate: complete for the reduced Metal-v1 scope  
Native gate: Apple Metal smoke + benchmark

## Frozen v1 scope

KairoGPU v1 is deliberately **not** a Metal/Vulkan/CUDA/WebGPU implementation matrix. V1 freezes one production-quality Apple Metal compute path behind a backend-neutral API: device discovery/capabilities, owned buffers, bounded upload/readback, validated Float32 add/multiply/matmul kernels, explicit resource destruction and measurable synchronous dispatch telemetry. Other backends remain v2.

## 80 exit evidence

- Buffer handles carry device ownership and stale/foreign handles fail closed.
- Buffer and kernel identities use separate domains; buffer IDs cannot be displaced by kernel creation.
- Device-side allocation size is authoritative rather than trusting a caller-modified handle descriptor.
- Explicit buffer destruction and double-destroy/stale-use tests exist.
- Metal vector add, vector multiply and tiled matrix multiplication have numerical smoke coverage.
- DeviceStats tracks live/allocated resources, transfer volume and host-observed synchronous dispatch latency.
- A machine-readable Metal benchmark exercises a one-million-element workload.

## Explicit limitations

The dispatch timing is host-observed command completion time, not GPU hardware timestamp-query time. Submission remains synchronous. Generic shader/resource binding, asynchronous queues, Vulkan, CUDA and WebGPU are outside the v1 80% scope and must not be marketed as implemented.
