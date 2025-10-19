"""
Advanced Gridworld for Gymnasium
--------------------------------
A configurable Gridworld environment compatible with Gymnasium (step API v0.28+).
Designed to stress-test RL agents across MDP/POMDP variants with stochastic dynamics,
moving obstacles, collectibles, keys/doors, multi-goal objectives, shaped/sparse rewards,
and partial observability.

Key features
============
- Configurable grid size, walls, lava, wind fields, slip probability
- Moving obstacles, optional adversary that chases agent
- Static and dynamic goals (with optional respawn or time-limit)
- Keys & doors mechanic (collect key(s) to unlock door(s))
- Coins (dense reward) vs. terminal goal (sparse reward)
- Curriculum-ready difficulty scaling via set_difficulty()
- Partial observability with egocentric crop & optional one-hot channels
- Multiple observation modes: "rgb", "index", "dict" (tiles + inventory)
- Deterministic seeding and episode-level domain randomization
- Render modes: "ansi", "rgb_array"
"""
import enum
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from gymnasium import spaces


# ----------------------------
# Tile encoding & color table
# ----------------------------
class Tile(enum.IntEnum):
    EMPTY = 0
    WALL = 1
    LAVA = 2
    GOAL = 3
    KEY = 4
    DOOR_LOCKED = 5
    PIZZA = 6
    AGENT = 7  # used only in rendering / observation composition
    MOVING_OBS = 8

# RGB palette (uint8)
PALETTE = np.array([
    [240, 240, 240],  # EMPTY
    [40, 40, 40],      # WALL
    [220, 60, 30],     # LAVA
    [50, 180, 60],     # GOAL
    [30, 144, 255],    # KEY
    [150, 75, 0],      # DOOR_LOCKED
    [255, 215, 0],     # PIZZA
    [80, 80, 255],     # AGENT
    [128, 0, 128],     # MOVING_OBS
], dtype=np.uint8)


# ----------------------------
# Configuration dataclass
# ----------------------------
@dataclass
class GridworldConfig:
    adversary: bool = False  # if True, one obstacle chases the agent
    channels: str = "onehot"  # "onehot" | "index" | "rgb"
    pizza_reward: float = 0.1
    pizzas: int = 0
    collision_penalty: float = -0.2
    domain_randomization: bool = True
    door_requires_keys: int = 1
    door_reward: float = 0.0
    doors: int = 0
    dynamic_obstacles: int = 0
    goal_respawn: bool = False
    goal_reward: float = 1.0
    goal_terminates: bool = True
    goals: int = 1
    height: int = 10
    keys: int = 0
    lava_penalty: float = -1.0
    lava_pools: int = 0
    max_steps: int = 200
    observation_mode: str = "dict"  # "dict" | "tiles" | "rgb"
    render_mode: str | None = None  # "ansi" | "rgb_array"
    respawn_on_lava: bool = False
    seed: int | None = None
    slip_prob: float = 0.0  # stochastic transition (action replaced by random)
    step_penalty: float = 0.0
    truncate_on_collision: bool = False
    view_size: int = 5  # odd -> egocentric crop size (POMDP); <=0 => full obs
    walls_density: float = 0.0  # random walls fraction (excluding borders)
    width: int = 10
    wind_dir: tuple[int, int] = (0, 0)  # (dx, dy)
    wind_strength: float = 0.0  # probability push in wind_dir

    def validate(self) -> None:
        assert self.width >= 4 and self.height >= 4, "grid too small"
        assert self.view_size % 2 == 1 or self.view_size <= 0, "view_size must be odd or <=0"
        assert self.channels in {"onehot", "index", "rgb"}
        assert self.observation_mode in {"dict", "tiles", "rgb"}
        assert 0.0 <= self.slip_prob < 1.0
        assert 0.0 <= self.wind_strength < 1.0
        assert self.goals >= 0 and self.keys >= 0 and self.doors >= 0


# ----------------------------
# Environment
# ----------------------------
class AdvancedGridworldEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 8}

    ACTIONS = {
        0: (0, 0),   # stay
        1: (1, 0),   # right
        2: (-1, 0),  # left
        3: (0, 1),   # down
        4: (0, -1),  # up
    }

    def __init__(self, config: GridworldConfig | None = None):
        super().__init__()
        self.config = config or GridworldConfig()
        self.config.validate()

        self._dynamic_obs_set: set[tuple[int, int]] = set()
        self._rng = np.random.default_rng(self.config.seed)
        self._slip_ep = self.config.slip_prob
        self._wind_dir_ep = self.config.wind_dir
        self.agent_pos = (1, 1)
        self.pizza_positions: list[tuple[int, int]] = []
        self.door_positions: list[tuple[int, int]] = []
        self.dynamic_obs: list[tuple[int, int]] = []
        self.goal_positions: list[tuple[int, int]] = []
        self.grid = np.zeros((self.config.height, self.config.width), dtype=np.int8)
        self.inventory = {"keys": 0}
        self.key_positions: list[tuple[int, int]] = []
        self.step_count = 0

        # Spaces
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        obs_space = self._make_observation_space()
        if self.config.observation_mode == "dict":
            self.observation_space = spaces.Dict({
                "tiles": obs_space,
                "inventory": spaces.Dict({"keys": spaces.Box(low=0, high=255, shape=(), dtype=np.uint8)}),
                "time": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            })
        else:
            self.observation_space = obs_space

    # ------------- Gym API -------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.inventory = {"keys": 0}
        self._generate_layout()
        self._slip_ep, self._wind_dir_ep = self._episode_params()
        obs = self._get_observation()
        info = {"agent_pos": self.agent_pos}
        return obs, info

    def seed(self, seed: int | None = None):
        # Gymnasium prefers seeding in reset(); this is for compatibility
        self._rng = np.random.default_rng(seed)

    def step(self, action: int):
        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        # Slip: replace action with random action with prob slip_prob
        if self._rng.random() < self._slip_ep:
            action = int(self._rng.integers(0, self.action_space.n))

        # Wind: probabilistic push
        dx, dy = self.ACTIONS.get(action, (0, 0))
        if self.config.wind_strength > 0 and self._rng.random() < self.config.wind_strength:
            wdx, wdy = self._wind_dir_ep
            dx += int(np.sign(wdx))
            dy += int(np.sign(wdy))

        new_pos = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        new_pos = self._clip_to_bounds(new_pos)

        # Collision logic
        t = self._tile_at(new_pos)
        collided = False
        if t == Tile.WALL or t == Tile.DOOR_LOCKED:
            collided = True
            reward += self.config.collision_penalty
            if self.config.truncate_on_collision:
                truncated = True
            # stay in place
            new_pos = self.agent_pos
        elif t == Tile.LAVA:
            reward += self.config.lava_penalty
            if self.config.respawn_on_lava:
                new_pos = self._random_empty_cell()
            else:
                terminated = True
        else:
            # collect tokens / interact
            if t == Tile.PIZZA:
                reward += self.config.pizza_reward
                self.pizza_positions.remove(new_pos)
                self.grid[new_pos[1], new_pos[0]] = Tile.EMPTY
            elif t == Tile.KEY:
                self._add_key(1)
                self.key_positions.remove(new_pos)
                self.grid[new_pos[1], new_pos[0]] = Tile.EMPTY
            elif t == Tile.GOAL:
                reward += self.config.goal_reward
                if self.config.goal_terminates:
                    terminated = True
                if self.config.goal_respawn:
                    # respawn goal at a new random location
                    self.grid[new_pos[1], new_pos[0]] = Tile.EMPTY
                    self._spawn_goal(1)
            # obstacle-at-destination check on the layered field
            if new_pos in self._dynamic_obs_set:
                collided = True
                reward += self.config.collision_penalty

                back = (self.agent_pos[0] - dx, self.agent_pos[1] - dy)

                def _safe(p):
                    x, y = p
                    if not (0 <= x < self.config.width and 0 <= y < self.config.height):
                        return False
                    t = Tile(int(self.grid[y, x]))
                    if t in (Tile.WALL, Tile.LAVA, Tile.DOOR_LOCKED):
                        return False
                    if p in self.dynamic_obs:
                        return False
                    return True

                if self.config.truncate_on_collision:
                    truncated = True
                    new_pos = self.agent_pos
                else:
                    new_pos = back if _safe(back) else self.agent_pos

        # Remember the agent’s previous position (for push-backs) then update to new position
        prev_agent_pos = self.agent_pos
        self.agent_pos = new_pos

        # Unlock doors if enough keys and adjacent
        if self.inventory["keys"] >= self.config.door_requires_keys and self.door_positions:
            # check adjacency to a locked door
            for pos in list(self.door_positions):
                if self._is_adjacent(self.agent_pos, pos):
                    # consume keys
                    self.inventory["keys"] -= self.config.door_requires_keys
                    self.grid[pos[1], pos[0]] = Tile.EMPTY
                    self.door_positions.remove(pos)
                    reward += self.config.door_reward

        # Remember previous obstacle positions, then move dynamic obstacles
        prev_obs_positions = list(self.dynamic_obs)
        self._step_dynamic_obstacles()

        # After obstacles move, if one occupies the agent cell, push the agent
        pushed = False
        if self.agent_pos in self.dynamic_obs:
            reward += self.config.collision_penalty
            pushed = True

            # Find (one) colliding obstacle and its movement vector
            coll_idx = None
            for i, pos in enumerate(self.dynamic_obs):
                if pos == self.agent_pos:
                    coll_idx = i
                    break

            if coll_idx is not None:
                ox_prev = prev_obs_positions[coll_idx][0]
                oy_prev = prev_obs_positions[coll_idx][1]
                ox_now, oy_now = self.dynamic_obs[coll_idx]
                odx = ox_now - ox_prev
                ody = oy_now - oy_prev

                # Default candidate: push along obstacle's move direction
                if odx == 0 and ody == 0:
                    # Obstacle didn't move; fall back to pushing the agent back to where it came from
                    cand = prev_agent_pos
                else:
                    cand = (self.agent_pos[0] + odx, self.agent_pos[1] + ody)

                # Helper: is the target cell safe for the agent?
                def _agent_can_occupy(p):
                    x, y = p
                    if not (0 <= x < self.config.width and 0 <= y < self.config.height):
                        return False
                    tile = Tile(int(self.grid[y, x]))
                    if tile in (Tile.WALL, Tile.LAVA, Tile.DOOR_LOCKED):
                        return False
                    # can't move into another obstacle either
                    if p in self.dynamic_obs and p != self.agent_pos:
                        return False
                    return True

                # Try push destination; if not safe, revert to previous cell if safe; else stay
                if _agent_can_occupy(cand):
                    self.agent_pos = cand
                elif _agent_can_occupy(prev_agent_pos):
                    self.agent_pos = prev_agent_pos
                else:
                    # Last-resort escape to avoid persistent co-location.
                    # Fixed neighbor order for determinism:
                    for nx, ny in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        cand2 = (self.agent_pos[0] + nx, self.agent_pos[1] + ny)
                        if _agent_can_occupy(cand2):
                            self.agent_pos = cand2
                            break
                    else:
                        if self.config.truncate_on_collision:
                            truncated = True

            if self.config.truncate_on_collision:
                truncated = True

        # If a push moves the agent onto a pizza/key/goal, process the landing tile
        t_post = self._tile_at(self.agent_pos)
        if t_post == Tile.PIZZA and self.agent_pos in self.pizza_positions:
            reward += self.config.pizza_reward
            self.pizza_positions.remove(self.agent_pos)
            self.grid[self.agent_pos[1], self.agent_pos[0]] = Tile.EMPTY
        elif t_post == Tile.KEY and self.agent_pos in self.key_positions:
            self._add_key(1)
            self.key_positions.remove(self.agent_pos)
            self.grid[self.agent_pos[1], self.agent_pos[0]] = Tile.EMPTY
        elif t_post == Tile.GOAL:
            reward += self.config.goal_reward
            if self.config.goal_terminates:
                terminated = True
            if self.config.goal_respawn:
                self.grid[self.agent_pos[1], self.agent_pos[0]] = Tile.EMPTY
                self._spawn_goal(1)

        # Step penalty
        if not terminated:
            reward += -abs(self.config.step_penalty)

        # Time truncation
        if self.step_count >= self.config.max_steps:
            truncated = True

        obs = self._get_observation()
        info = {
            "agent_pos": self.agent_pos,
            "collided": collided or pushed,
            "slip_prob_ep": self._slip_ep,
            "step": self.step_count,
            "wind_dir_ep": self._wind_dir_ep,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------- Observation helpers -------------

    def _get_observation(self):
        grid = self.grid.copy()
        for (ox, oy) in self.dynamic_obs:  # overlay dynamic obstacles first
            if 0 <= ox < self.config.width and 0 <= oy < self.config.height:
                grid[oy, ox] = Tile.MOVING_OBS
        ax, ay = self.agent_pos
        grid[ay, ax] = Tile.AGENT

        if self.config.view_size > 0:
            v = self.config.view_size
            r = v // 2
            x0, x1 = ax - r, ax + r + 1
            y0, y1 = ay - r, ay + r + 1
            # pad with walls outside bounds for consistent POMDP
            padded = np.pad(
                grid, pad_width=r, mode="constant", constant_values=Tile.WALL
            )
            crop = padded[y0 + r:y1 + r, x0 + r:x1 + r]
        else:
            crop = grid

        if self.config.observation_mode == "rgb":
            rgb = PALETTE[np.clip(crop, 0, len(PALETTE)-1)]
            return rgb

        if self.config.channels == "onehot":
            C = len(Tile)
            obs_tiles = np.eye(C, dtype=np.uint8)[np.clip(crop, 0, C-1)]
        elif self.config.channels == "rgb":
            obs_tiles = PALETTE[np.clip(crop, 0, len(PALETTE)-1)]
        else:  # index
            obs_tiles = crop.astype(np.int8)

        if self.config.observation_mode == "tiles":
            return obs_tiles
        elif self.config.observation_mode == "dict":
            tfrac = np.array(self.step_count / max(1, self.config.max_steps), dtype=np.float32)
            return {
                "tiles": obs_tiles,
                "inventory": {"keys": np.array(min(self.inventory["keys"], 255), dtype=np.uint8)},
                "time": tfrac,
            }
        else:  # fallback
            return obs_tiles

    def _make_observation_space(self) -> spaces.Space:
        H, W = self.config.height, self.config.width
        if self.config.view_size > 0:
            V = self.config.view_size
            obs_shape = (V, V)
        else:
            obs_shape = (H, W)

        if self.config.channels == "onehot":
            C = len(Tile)
            shape = obs_shape + (C,)
            box = spaces.Box(low=0, high=1, shape=shape, dtype=np.uint8)
        elif self.config.channels == "rgb" or self.config.observation_mode == "rgb":
            shape = obs_shape + (3,)
            box = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        else:  # index
            box = spaces.Box(low=0, high=len(Tile) - 1, shape=obs_shape, dtype=np.int8)
        return box

    # ------------- Layout generation -------------

    def _generate_layout(self):
        H, W = self.config.height, self.config.width
        self.grid[:, :] = Tile.EMPTY

        # Border walls
        self.grid[0, :] = Tile.WALL
        self.grid[H-1, :] = Tile.WALL
        self.grid[:, 0] = Tile.WALL
        self.grid[:, W-1] = Tile.WALL

        # Random internal walls
        if self.config.walls_density > 0:
            num_cells = (H-2) * (W-2)
            k = int(self.config.walls_density * num_cells)
            idx = self._rng.choice(num_cells, size=k, replace=False)
            ys = 1 + (idx // (W-2))
            xs = 1 + (idx % (W-2))
            self.grid[ys, xs] = Tile.WALL

        # Lava pools
        for _ in range(max(0, self.config.lava_pools)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.LAVA

        # Coins
        self.pizza_positions = []
        for _ in range(max(0, self.config.pizzas)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.PIZZA
            self.pizza_positions.append(pos)

        # Keys and doors
        self.key_positions = []
        for _ in range(max(0, self.config.keys)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.KEY
            self.key_positions.append(pos)

        self.door_positions = []
        for _ in range(max(0, self.config.doors)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.DOOR_LOCKED
            self.door_positions.append(pos)

        # Goals
        self.goal_positions = []
        self._spawn_goal(max(0, self.config.goals))

        # Agent spawn (avoid hazards)
        self.agent_pos = self._random_empty_cell()

        # Dynamic obstacles (layered; do not write into base grid)
        self.dynamic_obs = []
        for _ in range(max(0, self.config.dynamic_obstacles)):
            pos = self._random_empty_cell()
            self.dynamic_obs.append(pos)
        self._dynamic_obs_set = set(self.dynamic_obs)

        # Ensure agent->key->door->goal is possible (if those objects exist)
        self._ensure_connectivity()

    def _spawn_goal(self, n: int):
        for _ in range(n):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.GOAL
            self.goal_positions.append(pos)

    # ------------- Dynamics helpers -------------

    def _add_key(self, n=1):
        self.inventory["keys"] = int(min(255, self.inventory["keys"] + n))

    def _carve_corridor(self, a: tuple[int, int], b: tuple[int, int], clear_lava=False) -> None:
        """Carve x-then-y corridor by setting blocking tiles to EMPTY; keep borders intact."""
        ax, ay = a
        bx, by = b
        step_x = 1 if bx >= ax else -1
        for x in range(ax, bx + step_x, step_x):
            if 0 < x < self.config.width - 1:
                if self.grid[ay, x] == Tile.WALL or (clear_lava and self.grid[ay, x] == Tile.LAVA):
                    self.grid[ay, x] = Tile.EMPTY
        step_y = 1 if by >= ay else -1
        for y in range(ay, by + step_y, step_y):
            if 0 < y < self.config.height - 1:
                if self.grid[y, bx] == Tile.WALL or (clear_lava and self.grid[y, bx] == Tile.LAVA):
                    self.grid[y, bx] = Tile.EMPTY

    def _clip_to_bounds(self, pos: tuple[int, int]) -> tuple[int, int]:
        x = int(np.clip(pos[0], 0, self.config.width - 1))
        y = int(np.clip(pos[1], 0, self.config.height - 1))
        return (x, y)

    def _ensure_connectivity(self) -> None:
        """Guarantee a simple chain: agent -> key -> door -> goal by carving corridors."""
        waypoints: list[tuple[int, int]] = [self.agent_pos]
        if self.key_positions:
            waypoints.append(self.key_positions[0])
        if self.door_positions:
            waypoints.append(self.door_positions[0])
        if self.goal_positions:
            waypoints.append(self.goal_positions[0])
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            self._carve_corridor(a, b)

    def _episode_params(self):
        """Derive effective episode-specific params without mutating self.config."""
        slip = self.config.slip_prob
        wind_dir = self.config.wind_dir
        if self.config.domain_randomization:
            if self.config.slip_prob > 0:
                slip = float(np.clip(self._rng.normal(self.config.slip_prob, 0.02), 0.0, 0.5))
            if self.config.wind_strength > 0:
                wind_dir = tuple(int(x) for x in self._rng.choice([-1, 0, 1], size=2))
        return slip, wind_dir

    def _is_adjacent(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def _random_empty_cell(self) -> tuple[int, int]:
        H, W = self.config.height, self.config.width
        # retry sampling for robustness
        for _ in range(10000):
            x = int(self._rng.integers(1, W-1))
            y = int(self._rng.integers(1, H-1))
            if self.grid[y, x] == Tile.EMPTY:
                return (x, y)
        # fallback (should be rare)
        empties = np.argwhere(self.grid == Tile.EMPTY)
        if len(empties) == 0:
            return (1, 1)
        y, x = empties[self._rng.integers(0, len(empties))]
        return (int(x), int(y))

    def _step_dynamic_obstacles(self):
        """Advance obstacle layer via random walk (or adversary bias) without touching base tiles."""
        new_positions: list[tuple[int, int]] = []
        for pos in list(self.dynamic_obs):  # iterate a snapshot
            if self.config.adversary:
                # biased move towards agent (Manhattan greedy)
                dx = np.sign(self.agent_pos[0] - pos[0])
                dy = np.sign(self.agent_pos[1] - pos[1])
                candidates = [(dx, 0), (0, dy), (0, 0), (-dx, 0), (0, -dy)]
            else:
                # random walk with stay
                candidates = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
                self._rng.shuffle(candidates)

            moved = pos
            for (cx, cy) in candidates:
                nx, ny = pos[0] + cx, pos[1] + cy
                if not (0 <= nx < self.config.width and 0 <= ny < self.config.height):
                    continue
                tile = Tile(int(self.grid[ny, nx]))
                # can't move into walls, lava, or locked doors; everything else is fine
                if tile not in (Tile.WALL, Tile.LAVA, Tile.DOOR_LOCKED):
                    moved = (nx, ny)
                    break
            # update grid tiles
            new_positions.append(moved)
        self.dynamic_obs = new_positions
        self._dynamic_obs_set = set(new_positions)

    def _tile_at(self, pos: tuple[int, int]) -> Tile:
        x, y = pos
        return Tile(int(self.grid[y, x]))

    # ------------- Difficulty control -------------

    def set_difficulty(self, level: str) -> None:
        """Adjust configuration to a named difficulty preset.
        Presets: "easy", "medium", "hard", "extreme".
        """
        level = level.lower()
        if level == "easy":
            self.config.slip_prob = 0.0
            self.config.dynamic_obstacles = 0
            self.config.walls_density = 0.05
            self.config.pizzas = 3
            self.config.lava_pools = 0
        elif level == "medium":
            self.config.slip_prob = 0.05
            self.config.dynamic_obstacles = 1
            self.config.walls_density = 0.10
            self.config.pizzas = 2
            self.config.lava_pools = 1
        elif level == "hard":
            self.config.slip_prob = 0.1
            self.config.dynamic_obstacles = 2
            self.config.walls_density = 0.15
            self.config.pizzas = 1
            self.config.lava_pools = 2
        elif level == "extreme":
            self.config.slip_prob = 0.2
            self.config.dynamic_obstacles = 3
            self.config.walls_density = 0.20
            self.config.pizzas = 0
            self.config.lava_pools = 3
            self.config.adversary = True
        else:
            raise ValueError(f"Unknown difficulty: {level}")

    # ------------- Rendering -------------

    def render(self):
        mode = self.config.render_mode
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "rgb_array":
            return self._render_rgb()
        else:
            return None

    def _render_ansi(self) -> str:
        glyphs = {
            Tile.EMPTY: " ⬜ ",
            Tile.WALL: " ⬛ ",
            Tile.LAVA: " ♨️ ",
            Tile.GOAL: " 🏁 ",
            Tile.KEY: " 🔑 ",
            Tile.DOOR_LOCKED: " 🚪 ",
            Tile.PIZZA: " 🍕 ",
            Tile.MOVING_OBS: " 😈 ",
            Tile.AGENT: " 😁 ",
        }
        grid = self.grid.copy()
        for (ox, oy) in self.dynamic_obs:  # overlay dynamic obstacles first
            if 0 <= ox < self.config.width and 0 <= oy < self.config.height:
                grid[oy, ox] = Tile.MOVING_OBS
        ax, ay = self.agent_pos
        grid[ay, ax] = Tile.AGENT
        lines = ["".join(glyphs[Tile(int(t))] for t in row) for row in grid]
        out = "\n".join(lines)
        print(out)
        return out

    def _render_rgb(self) -> np.ndarray:
        grid = self.grid.copy()
        for (ox, oy) in self.dynamic_obs:  # overlay dynamic obstacles first
            if 0 <= ox < self.config.width and 0 <= oy < self.config.height:
                grid[oy, ox] = Tile.MOVING_OBS
        ax, ay = self.agent_pos
        grid[ay, ax] = Tile.AGENT
        rgb = PALETTE[np.clip(grid, 0, len(PALETTE)-1)]
        # upscale each cell to 8x8 pixels
        rgb = np.kron(rgb, np.ones((8,8,1), dtype=np.uint8))
        return rgb


# ----------------------------
# Convenience factory
# ----------------------------

def make_env(**kwargs) -> AdvancedGridworldEnv:
    cfg = GridworldConfig(**kwargs)
    return AdvancedGridworldEnv(cfg)


# ----------------------------
# Simple sanity test (manual)
# ----------------------------
if __name__ == "__main__":
    env = make_env(width=8, height=8, view_size=5, pizzas=2, keys=1, doors=1,
                   dynamic_obstacles=1, walls_density=0.1, lava_pools=1,
                   slip_prob=0.05, wind_strength=0.1, wind_dir=(1,0),
                   observation_mode="dict", render_mode="ansi")
    #obs, info = env.reset(seed=0)
    obs, info = env.reset()
    done = False
    total = 0.0
    while True:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        env.render()
        if term or trunc:
            print(f"Episode done, terminated: {term}, truncated: {trunc}, total reward: {total}")
            break
