# KairoGPU

KairoGPU is the GPU abstraction package for the Kairo ML stack.

This package is intentionally an interface-first foundation. Real Metal,
Vulkan, CUDA, or WebGPU backends should be implemented deliberately behind this
API after CPU tensor kernels are correct.

Phase 1 scope:

- backend/device descriptors,
- buffer and kernel launch metadata,
- command-list contracts,
- explicit unsupported-operation behavior.
