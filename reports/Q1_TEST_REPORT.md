# Q1 validation test report

- Command: `PYTHONPATH=.:src /opt/miniforge3/bin/python -m pytest -q`
- Result: **96 passed**
- Warnings: one PyTorch forward-compatibility warning for GB10 compute capability reporting
- Runtime: 6.13 seconds
- Date: 2026-07-22 (Asia/Seoul)

No test failed. The warning does not change a result; the registered training
process is executing successfully on the same device and software stack.
