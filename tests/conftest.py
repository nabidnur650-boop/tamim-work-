from __future__ import annotations

from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
FULL_DATA_TEST_MODULES = {
    "test_evaluation_protocol.py",
    "test_final_dataset.py",
    "test_local_data_gate.py",
}
REQUIRED_FULL_DATA_FILES = (
    PROJECT / "data/final/v1/normalization_all.parquet",
    PROJECT / "data/final/v1/normalization_test_iid.parquet",
    PROJECT / "data/final/v1/normalization_test_ood.parquet",
    PROJECT / "data/manifests/canonical_rows_preliminary.parquet",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip only full-data checks when this compact repository lacks the data."""
    missing = [path for path in REQUIRED_FULL_DATA_FILES if not path.is_file()]
    if not missing:
        return

    reason = (
        "requires full experiment data excluded from the compact GitHub snapshot; "
        "restore the files documented in README.md to enable this test"
    )
    skip_full_data = pytest.mark.skip(reason=reason)
    for item in items:
        if Path(str(item.fspath)).name in FULL_DATA_TEST_MODULES:
            item.add_marker(skip_full_data)
