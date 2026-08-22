from copy import deepcopy
import unittest

from config import CONFIG
from progress_tracker import ProgressTracker


def progress_config():
    config = deepcopy(CONFIG["environment"])
    config["progress"].update({
        "enabled": True,
        "checkpoint_radius_m": 0.2,
        "checkpoint_reward": 5.0,
        "distance_reward_scale": 2.0,
        "distance_delta_clip_m": 0.1,
        "waypoints": [[1.0, 0.0], [2.0, 0.0]]
    })
    config["stuck_detection"] = {
        "enabled": True,
        "no_progress_steps": 5,
        "minimum_progress_m": 0.05
    }
    return config


class ProgressTrackerTests(unittest.TestCase):
    def test_disabled_progress_has_no_track_checkpoints(self):
        environment = progress_config()
        environment["progress"]["enabled"] = False
        tracker = ProgressTracker(environment)

        update = tracker.reset((0.0, 0.0, 0.0))

        self.assertEqual(update.checkpoint_count, 0)
        self.assertEqual(update.progress_fraction, 0.0)
        self.assertFalse(tracker.is_stuck(10_000))

    def test_distance_progress_and_ordered_checkpoint_reward(self):
        tracker = ProgressTracker(progress_config())
        tracker.reset((0.0, 0.0, 0.0))

        closer = tracker.update((0.1, 0.0, 0.0), 1)
        checkpoint = tracker.update((0.85, 0.0, 0.0), 2)

        self.assertAlmostEqual(closer.distance_reward, 0.2)
        self.assertEqual(checkpoint.checkpoint_reward, 5.0)
        self.assertEqual(checkpoint.checkpoints_reached, 1)
        self.assertEqual(checkpoint.checkpoint_count, 2)
        self.assertAlmostEqual(checkpoint.progress_fraction, 0.5)

    def test_moving_away_has_bounded_negative_progress(self):
        tracker = ProgressTracker(progress_config())
        tracker.reset((0.0, 0.0, 0.0))

        update = tracker.update((-1.0, 0.0, 0.0), 1)

        self.assertAlmostEqual(update.distance_reward, -0.2)

    def test_stuck_after_configured_steps_without_new_best(self):
        tracker = ProgressTracker(progress_config())
        tracker.reset((0.0, 0.0, 0.0))
        for step in range(1, 5):
            tracker.update((0.0, 0.0, 0.0), step)

        self.assertFalse(tracker.is_stuck(4))
        self.assertTrue(tracker.is_stuck(5))


if __name__ == "__main__":
    unittest.main()
