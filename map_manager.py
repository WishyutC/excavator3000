"""Single-world map selection and collision-safe random spawning."""

import math
import random

from config import CONFIG


GENERATED_NODE_PREFIX = "RL_MAP_"


def next_map_name(pool, current, strategy, random_source, cycle_index=0):
    """Select the next layout for random training or balanced evaluation."""

    names = tuple(pool)
    if not names:
        raise ValueError("A rotating map pool cannot be empty.")
    if strategy == "round_robin":
        return names[int(cycle_index) % len(names)]
    if strategy != "random":
        raise ValueError(f'Unsupported map selection strategy "{strategy}".')
    alternatives = [name for name in names if name != current]
    return random_source.choice(alternatives or list(names))


def _rotated_robot_extents(angle, length_m, width_m):
    """Return axis-aligned half extents for a rotated robot footprint."""

    half_length = float(length_m) / 2.0
    half_width = float(width_m) / 2.0
    return (
        abs(math.cos(angle)) * half_length
        + abs(math.sin(angle)) * half_width,
        abs(math.sin(angle)) * half_length
        + abs(math.cos(angle)) * half_width
    )


def _point_to_box_distance(x, y, obstacle):
    """Distance from a point to an oriented rectangular obstacle."""

    center_x, center_y = map(float, obstacle["center"])
    size_x, size_y = map(float, obstacle["size"])
    angle = float(obstacle.get("angle_rad", 0.0))
    dx = x - center_x
    dy = y - center_y
    cosine = math.cos(angle)
    sine = math.sin(angle)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    outside_x = max(0.0, abs(local_x) - size_x / 2.0)
    outside_y = max(0.0, abs(local_y) - size_y / 2.0)
    return math.hypot(outside_x, outside_y)


def pose_is_clear(x, y, angle, map_config, robot_config, arena_config):
    """Check outer-wall and obstacle clearance for a candidate spawn pose."""

    clearance = float(map_config.get("spawn_clearance_m", 0.0))
    extent_x, extent_y = _rotated_robot_extents(
        angle,
        robot_config["length_m"],
        robot_config["width_m"]
    )
    x_limit = float(arena_config["x"]) / 2.0 - extent_x - clearance
    y_limit = float(arena_config["y"]) / 2.0 - extent_y - clearance
    if abs(x) > x_limit or abs(y) > y_limit:
        return False

    # A circumscribed robot circle is conservative for every heading and
    # avoids spawning any corner of the 25.5 x 14.5 cm body in an obstacle.
    robot_radius = math.hypot(
        float(robot_config["length_m"]) / 2.0,
        float(robot_config["width_m"]) / 2.0
    )
    required_distance = robot_radius + clearance
    return all(
        _point_to_box_distance(float(x), float(y), obstacle)
        > required_distance
        for obstacle in map_config.get("obstacles", [])
    )


def sample_spawn_pose(
    map_config,
    robot_config,
    arena_config,
    random_source=None
):
    """Sample a random, collision-safe XY pose and heading."""

    rng = random_source or random
    attempts = max(1, int(map_config.get("spawn_attempts", 500)))
    half_x = float(arena_config["x"]) / 2.0
    half_y = float(arena_config["y"]) / 2.0

    for _ in range(attempts):
        angle = (
            rng.uniform(-math.pi, math.pi)
            if map_config.get("random_heading", False)
            else 0.0
        )
        x = rng.uniform(-half_x, half_x)
        y = rng.uniform(-half_y, half_y)
        if pose_is_clear(
            x, y, angle, map_config, robot_config, arena_config
        ):
            return x, y, angle

    raise RuntimeError(
        f'Could not find a safe spawn in map "{map_config.get("label", "unknown")}" '
        f"after {attempts} attempts. Reduce obstacle density or clearance."
    )


def _obstacle_node_text(map_name, index, obstacle):
    center_x, center_y = map(float, obstacle["center"])
    size_x, size_y = map(float, obstacle["size"])
    height = float(obstacle.get("height_m", 0.40))
    angle = float(obstacle.get("angle_rad", 0.0))
    color = obstacle.get("color", [0.15, 0.18, 0.22])
    red, green, blue = map(float, color)
    safe_map_name = "".join(
        character if character.isalnum() else "_"
        for character in map_name.upper()
    )
    node_name = f"{GENERATED_NODE_PREFIX}{safe_map_name}_{index:02d}"
    return f"""DEF {node_name} Solid {{
  translation {center_x:.6f} {center_y:.6f} {height / 2.0:.6f}
  rotation 0 0 1 {angle:.6f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {red:.4f} {green:.4f} {blue:.4f}
        roughness 0.75
        metalness 0
      }}
      geometry Box {{ size {size_x:.6f} {size_y:.6f} {height:.6f} }}
    }}
  ]
  name "{node_name}"
  boundingObject Box {{ size {size_x:.6f} {size_y:.6f} {height:.6f} }}
  locked TRUE
}}"""


class MapManager:
    """Activate one configured layout without requiring another world file."""

    def __init__(self, supervisor, environment_config=None):
        self.supervisor = supervisor
        self.environment = environment_config or CONFIG["environment"]
        self.selector_name = self.environment["map_selector"]
        selector = self.environment["maps"][self.selector_name]
        self.map_pool = tuple(selector.get("map_pool", ()))
        unknown = set(self.map_pool) - set(self.environment["maps"])
        if unknown:
            raise ValueError(f"Unknown maps in map_pool: {sorted(unknown)}")
        if self.selector_name in self.map_pool:
            raise ValueError("A map pool cannot contain its own selector.")
        self.random = random.Random(int(CONFIG["training"].get("seed", 0)))
        self.cycle_index = 0
        self.first_reset = True
        if self.map_pool:
            selection = self._selection_strategy()
            self.map_name = next_map_name(
                self.map_pool,
                None,
                selection,
                self.random,
                self.cycle_index
            )
        else:
            self.map_name = self.selector_name
        self.map_config = self.environment["maps"][self.map_name]

    def _selection_strategy(self):
        selector = self.environment["maps"][self.selector_name]
        if CONFIG["program"]["mode"] == "evaluate":
            return selector.get("evaluation_selection", "round_robin")
        return selector.get("training_selection", "random")

    def prepare_episode(self):
        """Rotate the map after each completed episode in a pooled selector."""

        if self.first_reset:
            self.first_reset = False
            return self.map_name
        if not self.map_pool:
            return self.map_name
        strategy = self._selection_strategy()
        if strategy == "round_robin":
            self.cycle_index += 1
        selected = next_map_name(
            self.map_pool,
            self.map_name,
            strategy,
            self.random,
            self.cycle_index
        )
        self.activate(selected)
        return self.map_name

    def activate(self, map_name=None):
        if map_name is not None:
            if map_name not in self.environment["maps"]:
                raise ValueError(f'Unknown map "{map_name}".')
            self.map_name = map_name
            self.map_config = self.environment["maps"][self.map_name]
        self._remove_generated_obstacles()
        race_active = self.map_name == "race_track"
        self._set_race_track_visible(race_active)
        self._set_goal_visible(bool(self.map_config.get("goal_enabled", False)))
        if not race_active:
            self._create_obstacles()
        print(
            f'Map ready | {self.map_name} | '
            f'{len(self.map_config.get("obstacles", []))} obstacles | '
            f'episode {self.map_config.get("episode_type", "goal")}'
        )

    def random_spawn_pose(self, random_source=None):
        return sample_spawn_pose(
            self.map_config,
            CONFIG["robot"],
            self.environment["arena_size_m"],
            random_source=random_source or self.random
        )

    def _root_children(self):
        root = self.supervisor.getRoot()
        children = root.getField("children") if root is not None else None
        if children is None:
            raise RuntimeError("Could not access the Webots world root children.")
        return children

    @staticmethod
    def _node_name(node):
        field = node.getField("name") if node is not None else None
        return field.getSFString() if field is not None else ""

    def _remove_generated_obstacles(self):
        children = self._root_children()
        for index in range(children.getCount() - 1, -1, -1):
            node = children.getMFNode(index)
            if self._node_name(node).startswith(GENERATED_NODE_PREFIX):
                node.remove()

    def _set_race_track_visible(self, visible):
        children = self._root_children()
        target_z = 0.05 if visible else -10.0
        for index in range(children.getCount()):
            node = children.getMFNode(index)
            if not self._node_name(node).startswith("map part"):
                continue
            translation = node.getField("translation")
            if translation is None:
                continue
            position = translation.getSFVec3f()
            translation.setSFVec3f([position[0], position[1], target_z])

    def _set_goal_visible(self, visible):
        goal_def = self.environment["goal"]["def"]
        goal = self.supervisor.getFromDef(goal_def)
        if goal is None:
            if visible:
                raise RuntimeError(f'Could not find goal node DEF "{goal_def}".')
            return
        translation = goal.getField("translation")
        if translation is not None:
            position = translation.getSFVec3f()
            translation.setSFVec3f([
                position[0], position[1], 0.0 if visible else -10.0
            ])

    def _create_obstacles(self):
        children = self._root_children()
        for index, obstacle in enumerate(self.map_config.get("obstacles", [])):
            children.importMFNodeFromString(
                -1,
                _obstacle_node_text(self.map_name, index, obstacle)
            )
