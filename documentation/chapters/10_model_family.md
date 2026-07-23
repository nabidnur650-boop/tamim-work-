# Dense, Switch, standard-MoE, and Boichitro systems

M0 is the dense control. M1 uses Switch top-1 routing and substantially more inactive expert capacity. M2 uses a shared expert plus top-2 of eight routed experts. M3 starts from the M2 continuation and adds the proposed task/dialect routing mechanisms.

The architectures are designed around approximately 83.8 million active parameters per token. Total parameters differ: about 83.8M for M0, 381.1M for M1, and 168.8M for M2/M3. This is active-parameter matching, not wall-clock or energy matching.

The first four M3 blocks remain dense. Early sparse blocks receive causal dialect-evidence signals derived only from the visible prefix. Middle layers use learned load-balance bias. Late sparse blocks receive task-conditioned bias. Shared experts remain active alongside two routed experts.
