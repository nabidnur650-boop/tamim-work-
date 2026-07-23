from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from boichitro.protocol import (  # noqa: E402
    freeze_manifest_path,
    protocol_fingerprints,
    require_frozen_artifact,
    require_protocol_freeze,
)


class ProtocolFreezeTests(unittest.TestCase):
    def test_freeze_detects_post_freeze_config_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "src/package").mkdir(parents=True)
            (project / "tools").mkdir()
            (project / "configs").mkdir()
            (project / "src/package/model.py").write_text("VALUE = 1\n")
            (project / "tools/evaluate.py").write_text("print('ok')\n")
            config = project / "configs/protocol.yaml"
            config.write_text("seed: 17\n")
            fingerprint = protocol_fingerprints(project)
            manifest = {
                "status": "FROZEN",
                "protocol_id": "unit_v1",
                **fingerprint,
                "selected_artifacts_sha256": "0" * 64,
            }
            path = freeze_manifest_path(project, "unit_v1")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(manifest))
            self.assertEqual(
                require_protocol_freeze(project, "unit_v1")["protocol_id"],
                "unit_v1",
            )
            config.write_text("seed: 18\n")
            with self.assertRaisesRegex(RuntimeError, "changed after freeze"):
                require_protocol_freeze(project, "unit_v1")

    def test_frozen_artifact_detects_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            artifact = project / "runs/model.pt"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"frozen")
            freeze = {
                "selected_artifacts": {
                    "runs/model.pt": hashlib.sha256(b"frozen").hexdigest()
                }
            }
            self.assertEqual(
                require_frozen_artifact(project, freeze, artifact),
                hashlib.sha256(b"frozen").hexdigest(),
            )
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "changed after protocol freeze"):
                require_frozen_artifact(project, freeze, artifact)


if __name__ == "__main__":
    unittest.main()
