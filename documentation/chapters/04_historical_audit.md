# Historical audit and protocol repairs

The project began by auditing two local ZIP archives and two notebooks. The audit found publication-blocking problems in the earlier pipeline: a Vashantor column mismatch, held-out rows reintroduced through derived synthetic data, a missing Barishal validation file, duplicate loading of a regional corpus, template-like synthetic conflicts, uncertain ancestry inside merged BanglaDial material, model–data scale mismatch, and external diagnostics that did not support valid comparative claims.

The repaired workflow treats the original archives as immutable inputs. Every source is handled through an explicit adapter. The old derived archive is quarantined instead of silently reused. Source ancestry, licenses, dialect mapping, row origin, exclusion reason, split membership, semantic groups, and synthetic parentage are recorded in machine-readable manifests.

The monolithic notebooks are retained as historical material, not as the authoritative executable pipeline. Reusable modules under `src/boichitro`, command-line tools, YAML registries, regression tests, immutable hashes, and saved per-example outputs form the reproducible implementation.
