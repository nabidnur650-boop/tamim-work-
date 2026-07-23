# Systems and inference efficiency

On the NVIDIA GB10 training benchmark, M0 processes about 15,337 tokens/s with 2.83 GiB peak memory. M1 processes about 5,211 tokens/s with 5.69 GiB. M2 and M3 process about 8,272 and 8,224 tokens/s with roughly 3.84 GiB.

M3 adds little overhead relative to M2, but both sparse models are substantially slower than dense M0 despite similar active parameter counts. Grouped expert execution, routing, dispatch, and memory movement therefore matter to practical compute claims.

Task inference benchmarks include batch-one and batched normalization and identification. The results show substantial batching gains and distinct latency/memory trade-offs. Any deployment claim should report task, batch size, generated tokens, examples per second, latency, checkpoint hash, and hardware.
