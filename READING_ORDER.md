# KairoGPU: source reading order

Start with backend-neutral device, buffer, kernel and command-list contracts, then inspect the actual Metal bridge and its compute smoke tests. Vulkan/CUDA/WebGPU enum variants are not proof of implemented backend support.

## Key APIs and interactions

Follow DeviceDesc/Capabilities -> buffer allocation/upload -> validated kernel dispatch -> synchronization/readback; distinguish Metal's shared-buffer dispatch and the explicit UnsupportedBackend path for unimplemented APIs.

## Linked source files (4)

1. [`src/KairoGPU.cppm`](src/KairoGPU.cppm) — read public contracts, lifetime ownership, imports/callers, error behavior and associated tests.
2. [`src/KairoMetal.mm`](src/KairoMetal.mm) — read public contracts, lifetime ownership, imports/callers, error behavior and associated tests.
3. [`tests/smoke.cpp`](tests/smoke.cpp) — read public contracts, lifetime ownership, imports/callers, error behavior and associated tests.
4. [`benchmarks/benchmark.cpp`](benchmarks/benchmark.cpp) — read public contracts, lifetime ownership, imports/callers, error behavior and associated tests.

For setup consult [README.md](README.md), and for downstream consumers inspect [KairoGameEngine](https://github.com/swayam8624/KairoGameEngine). Original code is [MIT licensed](LICENSE) for study, modification and further development; framework binaries/SDKs retain their original licenses. Function-level commentary and full native tests are separate ongoing verification tasks.
