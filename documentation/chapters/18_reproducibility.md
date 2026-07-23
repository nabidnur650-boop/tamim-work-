# Software architecture and reproducibility

The project separates reusable library code under `src/boichitro`, command-line tools under `tools`, YAML configuration under `configs`, regression tests under `tests`, immutable evidence under `reports`, metrics and predictions, and run artifacts.

The recorded regression suite reports 107 passed, 0 failed, and one non-failing forward-compatibility warning. The 44 source figures passed 398/398 checks for paired formats, hashes, resolution, captions, and source data.

Every configuration, Python module, run manifest, training report, and CSV evidence table receives a catalog card later in this monograph. The evidence snapshot copies small human-readable artifacts so the documentation remains inspectable after the original workspace is removed.
