# Optimization, continuation learning rate, and upcycling

Eligible hidden matrices use Muon, while embeddings, normalization parameters, routers, and task heads use AdamW. A separately registered AdamW-only control prevents improvements from being attributed solely to the optimizer split.

Restarting a mature foundation checkpoint at the original high learning rate produced a monotonic validation regression. A validation-only pilot selected Muon 0.001 and AdamW 0.000015 for the continuation. The rejected high-rate run remains in the archive as negative evidence.

Dense-to-MoE transfer compared abrupt bank release, unbanked transfer, random initialization, annealed cross-bank release, and permanent complementary-bank routing. Only the permanent paired-bank strategy met the registered transient and endpoint regression guards. A separate Switch pilot selected auxiliary-balance straight-through routing after a loss-free variant exhibited unacceptable load variation.
