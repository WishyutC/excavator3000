import json
from pathlib import Path
import tempfile
import unittest

from training_logger import TrainingLogger


class TrainingLoggerTests(unittest.TestCase):
    def test_creates_reproducible_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = TrainingLogger(path=directory, enabled=True)
            manifest = json.loads(logger.manifest_path.read_text(
                encoding="utf-8"
            ))

            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["csv_file"], "training.csv")
            self.assertEqual(
                manifest["csv_fields"],
                list(TrainingLogger.FIELDNAMES)
            )
            self.assertEqual(
                manifest["configuration"]["training"]["episodes"],
                100_000
            )
            self.assertEqual(
                manifest["configuration"]["environment"]["map_selector"],
                "survival_mix"
            )

    def test_resume_does_not_overwrite_original_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            path.write_text('{"original": true}', encoding="utf-8")

            TrainingLogger(path=directory, enabled=True)

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"original": True}
            )


if __name__ == "__main__":
    unittest.main()
