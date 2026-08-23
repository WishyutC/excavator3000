import math
import random
import unittest

from config import CONFIG
from map_manager import next_map_name, pose_is_clear, sample_spawn_pose


class MapManagerTests(unittest.TestCase):
    def setUp(self):
        self.environment = CONFIG["environment"]
        self.robot = CONFIG["robot"]
        self.arena = self.environment["arena_size_m"]

    def test_two_survival_maps_are_configured_in_same_arena(self):
        maps = self.environment["maps"]

        self.assertEqual(maps["obstacle_field"]["episode_type"], "survival")
        self.assertEqual(maps["chessboard"]["episode_type"], "survival")
        self.assertGreater(len(maps["obstacle_field"]["obstacles"]), 0)
        self.assertGreater(len(maps["chessboard"]["obstacles"]), 0)
        self.assertFalse(maps["obstacle_field"]["goal_enabled"])
        self.assertFalse(maps["chessboard"]["goal_enabled"])

    def test_survival_mix_contains_exactly_three_training_maps(self):
        mix = self.environment["maps"]["survival_mix"]

        self.assertEqual(mix["training_selection"], "random")
        self.assertEqual(mix["evaluation_selection"], "round_robin")
        self.assertEqual(mix["map_pool"], [
            "obstacle_field",
            "tight_corridors",
            "dense_pinch_points"
        ])
        self.assertNotIn("chessboard", mix["map_pool"])

    def test_map_selection_is_balanced_for_eval_and_changes_for_training(self):
        pool = ("one", "two", "three")
        evaluation = [
            next_map_name(pool, None, "round_robin", random.Random(1), index)
            for index in range(6)
        ]
        training_rng = random.Random(7)
        current = "one"
        training = []
        for _ in range(20):
            selected = next_map_name(
                pool, current, "random", training_rng
            )
            training.append(selected)
            self.assertNotEqual(selected, current)
            current = selected

        self.assertEqual(evaluation, ["one", "two", "three"] * 2)
        self.assertGreater(len(set(training)), 1)

    def test_corridor_map_has_tight_but_navigable_lane_spacing(self):
        obstacles = self.environment["maps"]["tight_corridors"]["obstacles"]
        bars = [obstacle for obstacle in obstacles if obstacle["size"][0] > 7.0]
        centers = sorted(obstacle["center"][1] for obstacle in bars)
        clear_lane_widths = [
            right - left - bars[0]["size"][1]
            for left, right in zip(centers, centers[1:])
        ]

        self.assertEqual(len(bars), 5)
        self.assertTrue(all(1.0 <= width <= 1.2 for width in clear_lane_widths))

    def test_obstacle_center_and_outer_wall_are_not_safe_spawns(self):
        map_config = self.environment["maps"]["obstacle_field"]
        obstacle_x, obstacle_y = map_config["obstacles"][0]["center"]

        self.assertFalse(pose_is_clear(
            obstacle_x,
            obstacle_y,
            0.0,
            map_config,
            self.robot,
            self.arena
        ))
        self.assertFalse(pose_is_clear(
            self.arena["x"] / 2.0,
            0.0,
            math.pi / 4.0,
            map_config,
            self.robot,
            self.arena
        ))

    def test_obstacle_field_blocks_every_edge_and_all_four_corners(self):
        obstacles = self.environment["maps"]["obstacle_field"]["obstacles"]
        edge_obstacles = [
            obstacle for obstacle in obstacles
            if max(abs(value) for value in obstacle["center"]) >= 4.0
        ]
        corner_obstacles = [
            obstacle for obstacle in edge_obstacles
            if all(abs(value) >= 4.0 for value in obstacle["center"])
        ]

        self.assertGreaterEqual(len(edge_obstacles), 12)
        self.assertEqual(len(corner_obstacles), 4)
        for axis in (0, 1):
            self.assertTrue(any(
                obstacle["center"][axis] <= -4.0
                for obstacle in edge_obstacles
            ))
            self.assertTrue(any(
                obstacle["center"][axis] >= 4.0
                for obstacle in edge_obstacles
            ))
    def test_random_spawns_are_reproducible_and_clear_on_both_maps(self):
        for map_name in (
            "obstacle_field",
            "tight_corridors",
            "dense_pinch_points",
            "chessboard"
        ):
            map_config = self.environment["maps"][map_name]
            first_rng = random.Random(1234)
            second_rng = random.Random(1234)
            first = sample_spawn_pose(
                map_config, self.robot, self.arena, first_rng
            )
            second = sample_spawn_pose(
                map_config, self.robot, self.arena, second_rng
            )

            self.assertEqual(first, second)
            self.assertTrue(pose_is_clear(
                *first,
                map_config,
                self.robot,
                self.arena
            ))

            rng = random.Random(99)
            for _ in range(100):
                x, y, angle = sample_spawn_pose(
                    map_config, self.robot, self.arena, rng
                )
                self.assertTrue(pose_is_clear(
                    x, y, angle, map_config, self.robot, self.arena
                ))


if __name__ == "__main__":
    unittest.main()
