# Frame-Budget-Aware Heterogeneous Inference Research Track

KairoGPU is the primary repository for the heterogeneous execution policy
research track. KairoScheduler supplies CPU execution telemetry; KairoSIMD,
KairoMath Tensor, KairoONNX and KairoTransformers become additional baselines
and workloads when their later waves are locked.

## Application problem

Interactive inference contains a mix of tiny and large tensor operations while
the renderer/game loop has a hard frame deadline. Small kernels can lose on GPU
dispatch/transfer overhead; large matmuls may strongly prefer GPU execution.
Static "GPU for everything" and one fixed size threshold cannot account for
contention or remaining frame budget.

## Research question

Can a backend-neutral runtime choose CPU/SIMD versus Metal execution per
operation or operation group using measured latency, residency/transfer cost and
remaining frame budget, reducing missed frame deadlines without sacrificing
throughput?

## Novel hypothesis

An online deadline-aware cost model with conservative hysteresis and explicit
transfer/residency accounting can outperform CPU-only, GPU-only and static
threshold policies on mixed interactive model workloads.

KairoGPU v1 remains synchronous and bounded. The research implementation must
first establish correct telemetry and async execution without changing the v1
claim.

## Required baselines

CPU scalar/reference; CPU SIMD; Metal-only; fixed tensor-size threshold;
offline-profiled static mapping; adaptive deadline-aware mapping.

## Workloads

MLP, convolutional block, compact attention/decoder block, mixed tiny/large
operator chain, ONNX fixture graphs, and later compact transformer inference.

System conditions include idle renderer, GPU contention, CPU contention, cold
buffers, warm resident buffers, burst inference, and intentionally tiny kernels.

## Metrics

End-to-end inference latency, p50/p95/p99 latency, missed frame deadlines,
throughput, transfer bytes, dispatch count, CPU time, GPU completion time once
timestamp support exists, memory residency and numerical error versus reference.

## Failure criteria

The proposed scheduler fails if decision overhead exceeds savings, migration
thrashes between devices, GPU contention causes unpredictable deadline misses
without recovery, or a static policy dominates all representative mixed
workloads.

## Required figures

Operator/device timeline, latency crossover curves, deadline-miss plot,
transfer-cost ablation, contention results, policy-decision heatmap,
throughput/latency Pareto, and explicit counter-cases.

See `research.yaml` for track state.
