import csv
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

    def test_resume_discards_rows_newer_than_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = TrainingLogger(path=directory, enabled=True)
            for episode in range(1, 6):
                logger.log_episode({"episode": episode})

            removed = logger.reconcile_to_checkpoint(3)

            with logger.path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(removed, 2)
            self.assertEqual([row["episode"] for row in rows], ["1", "2", "3"])
            event_path = Path(directory) / "resume_events.jsonl"
            event = json.loads(event_path.read_text(encoding="utf-8"))
            self.assertEqual(event["checkpoint_episode"], 3)
            self.assertEqual(event["discarded_rows"], 2)


if __name__ == "__main__":
    unittest.main()
