import unittest

from config import CONFIG
from observation import (
    build_observation,
    proximity_from_distance,
    signed_forward_speed
)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.config = CONFIG["observation"]
        self.identity_orientation = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

    def test_proximity_uses_clear_zero_and_close_one(self):
        self.assertEqual(proximity_from_distance(0.5, 0.5), 0.0)
        self.assertEqual(proximity_from_distance(0.0, 0.5), 1.0)
        self.assertAlmostEqual(proximity_from_distance(0.25, 0.5), 0.5)

    def test_missing_or_out_of_range_distances_are_safe_and_bounded(self):
        self.assertEqual(proximity_from_distance(None, 0.5), 0.0)
        self.assertEqual(proximity_from_distance(-1.0, 0.5), 1.0)
        self.assertEqual(proximity_from_distance(2.0, 0.5), 0.0)

    def test_signed_forward_speed_distinguishes_reverse(self):
        self.assertEqual(
            signed_forward_speed(
                [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                self.identity_orientation
            ),
            -0.1
        )

    def test_forward_projection_respects_robot_heading(self):
        heading_positive_y = [
            0.0, -1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0
        ]

        speed = signed_forward_speed(
            [0.0, 0.08, 0.0, 0.0, 0.0, 0.0],
            heading_positive_y
        )

        self.assertAlmostEqual(speed, 0.08)

    def test_observation_has_ten_bounded_values(self):
        observation = build_observation(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, None, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -10.0],
            self.identity_orientation,
            self.config
        )

        self.assertEqual(len(observation), 10)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in observation[:8]))
        self.assertEqual(observation[8], 1.0)
        self.assertEqual(observation[9], -1.0)

    def test_wrong_sensor_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected 8"):
            build_observation(
                [0.5] * 7,
                [0.0] * 6,
                self.identity_orientation,
                self.config
            )


if __name__ == "__main__":
    unittest.main()
