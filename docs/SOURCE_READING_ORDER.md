# Teaching source map — KairoGPU

Backend-neutral tensor compute contract with bounded real Metal execution.

**Start with the worked guide:** [Metal devices, buffer ownership, Float32 operations and transfer timing](GPU_DEVICE_LIFECYCLE.md). Then study [native command encoding, synchronization and tiled shader execution](METAL_COMPUTE_PIPELINE_LAB.md).

Follow source links sequentially. For each API/function, identify its input contract, owning data structure, called modules, failure modes, outputs, test and downstream consumers.

## Read in this order

1. [`CMakeLists.txt`](../CMakeLists.txt)
2. [`README.md`](../README.md)
3. [`src`](../src)
4. [`tests`](../tests)
5. [`benchmarks`](../benchmarks)
6. [`research`](../research)

## How this repository interacts with others

Tensor/ONNX/Transformers call abstract device/buffer/kernel contracts; Metal currently covers selected operations, not universal GPU training. The backend controls buffer lifetime and synchronization.

## Verification and study rules

- Read the public C++23 module/entrypoint first, then its implementation and tests.
- Explain invariants, ownership, error states, concurrency or numerical assumptions in comments, not trivial syntax.
- A passing CI job covers only tests actually run on that commit/host; source presence is not runtime certification.
- Check [STATUS.md](../STATUS.md) and [STATUS.yaml](../STATUS.yaml) for current explicit verification limitations.
- Original work may be modified and extended under the repository license; third-party content retains upstream terms.


## Complete source-file inventory

After the curated teaching path, follow this deterministic inventory of first-party source, scripts, shaders, and tests. Grouping is a suggested learning order, not an indication that every function is already annotated. Vendored dependencies, ROMs, binaries, data, media, and generated evidence are excluded.

### Entry points and build (1)

1. [`CMakeLists.txt`](../CMakeLists.txt)

### Interfaces and modules (1)

1. [`src/KairoGPU.cppm`](../src/KairoGPU.cppm)

### Implementation and other source (1)

1. [`src/KairoMetal.mm`](../src/KairoMetal.mm)

### Research and benchmarks (1)

1. [`benchmarks/benchmark.cpp`](../benchmarks/benchmark.cpp)

### Verification and tests (1)

1. [`tests/smoke.cpp`](../tests/smoke.cpp)

Snapshot totals: 5 first-party source/build/test files, 14 tracked files overall. Re-audit when files are renamed or added.
