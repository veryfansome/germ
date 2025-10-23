"""
Advanced Gridworld for Gymnasium
--------------------------------
A configurable Gridworld environment compatible with Gymnasium (step API v0.28+).
Designed to stress-test RL agents.

Key features
============
- Configurable grid size and walls density
- Static and moving obstacles (optionally adversarial)
- 3-slot inventory (Tile-coded) for picking up items that can used on obstacles
- Global wind and terrain-based slip probability (around water)
- Partial observability with egocentric crop & optional one-hot channels
- Multiple observation modes: "rgb", "tiles", "dict" (where "dict" = tiles + inventory + time)
- Tile channels: "onehot", "index", or "rgb"
- Render modes: "ansi", "rgb_array"
- Deterministic seeding and episode-level domain randomization
"""
import enum
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from gymnasium import spaces


# ----------------------------
# Action encoding
# ----------------------------

class Act(enum.IntEnum):
    # Movement / stay (unchanged indices 0..4 to preserve prior behavior)
    STAY = 0
    RIGHT = 1
    LEFT = 2
    DOWN = 3
    UP = 4

    # Use inventory slot s on adjacent tile dir (s in {1,2,3})
    USE1_LEFT = 5
    USE1_RIGHT = 6
    USE1_UP = 7
    USE1_DOWN = 8

    USE2_LEFT = 9
    USE2_RIGHT = 10
    USE2_UP = 11
    USE2_DOWN = 12

    USE3_LEFT = 13
    USE3_RIGHT = 14
    USE3_UP = 15
    USE3_DOWN = 16

    # Drop / switch at current tile, for slot s
    SWAP1 = 17
    SWAP2 = 18
    SWAP3 = 19


# ----------------------------
# Tile encoding & color table
# ----------------------------

class Tile(enum.IntEnum):
    AGENT = 0  # used only in rendering / observation composition
    APPLE = 1  # dense reward, consumed when moved over
    AXE = 2  #  not consumed, can be picked up, can be used to remove trees and moving obstacles
    BUCKET = 3  # not consumed, can be picked up, allows picking up water if held in inventory
    DOOR_LOCKED = 4  # stationary obstacle that can be removed with keys
    EMPTY = 5
    GOAL = 6  # sparse reward
    KEY = 7  # can be picked up, consumed to remove locked doors
    LAVA = 8  # stationary obstacle that can be removed with water
    MOVING_OBS = 9  # mobile obstacle that consumes apples and can be adversarial but can be removed with an axe
    TREE = 10  # stationary obstacle that can be removed with an axe but can drop apples, especially if watered
    WALL = 11
    WATER = 12  # chance to slip when moved over, can be picked up, consumed when used to remove lava or on trees

# RGB palette (uint8)
PALETTE = np.array([
    [ 80,  80, 255],  # AGENT
    [255, 215,   0],  # APPLE
    [160,  82,  45],  # AXE
    [135, 206, 250],  # BUCKET
    [150,  75,   0],  # DOOR_LOCKED
    [240, 240, 240],  # EMPTY
    [ 50, 180,  60],  # GOAL
    [ 30, 144, 255],  # KEY
    [220,  60,  30],  # LAVA
    [128,   0, 128],  # MOVING_OBS
    [ 34, 139,  34],  # TREE
    [ 40,  40,  40],  # WALL
    [ 64, 164, 223],  # WATER
], dtype=np.uint8)


# ----------------------------
# Configuration dataclass
# ----------------------------

@dataclass
class GridworldConfig:
    adversary: bool = False  # if True, one obstacle chases the agent
    apple_reward: float = 0.1
    apples: int = 0
    axes: int = 0
    buckets: int = 0
    channels: str = "onehot"  # "onehot" | "index" | "rgb"
    collision_penalty: float = -0.2
    domain_randomization: bool = True
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
    slip_prob: float = 0.0
    step_penalty: float = 0.0
    tree_drop_chance: float = 0.01  # ambient per-step chance per tree
    tree_drop_chance_watered: float = 0.75  # one-shot boosted chance on watering
    trees: int = 0
    truncate_on_collision: bool = False
    view_size: int = 5  # odd -> egocentric crop size (POMDP); <=0 => full obs
    walls_density: float = 0.0  # random walls fraction (excluding borders)
    water_pools: int = 0
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

    MOVE_DELTAS = {
        Act.STAY: (0, 0),
        Act.RIGHT: (1, 0),
        Act.LEFT: (-1, 0),
        Act.DOWN: (0, 1),
        Act.UP: (0, -1),
    }

    def __init__(self, config: GridworldConfig | None = None):
        super().__init__()
        self.config = config or GridworldConfig()
        self.config.validate()

        self._dynamic_obs_set: set[tuple[int, int]] = set()
        self._rng = np.random.default_rng(self.config.seed)
        self._slip_ep = self.config.slip_prob
        self._wind_dir_ep: tuple[int, int] = self.config.wind_dir
        self._wind_strength_ep: float = self.config.wind_strength
        self.agent_pos = (1, 1)
        self.apple_positions: list[tuple[int, int]] = []
        self.door_positions: list[tuple[int, int]] = []
        self.dynamic_obs: list[tuple[int, int]] = []
        self.goal_positions: list[tuple[int, int]] = []
        self.grid = np.zeros((self.config.height, self.config.width), dtype=np.int8)
        self.inventory_slots: list[int] = [int(Tile.EMPTY), int(Tile.EMPTY), int(Tile.EMPTY)]
        self.tree_positions: list[tuple[int, int]] = []

        self.key_positions: list[tuple[int, int]] = []
        self.step_count = 0

        # Spaces
        self.action_space = spaces.Discrete(len(Act))
        obs_space = self._make_observation_space()
        if self.config.observation_mode == "dict":
            self.observation_space = spaces.Dict({
                "tiles": obs_space,
                # Inventory as 3-slot vector of uint8 tile codes [0..len(Tile)-1]
                "inventory": spaces.Box(low=0, high=len(Tile) - 1, shape=(3,), dtype=np.uint8),
                "time": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            })
        else:
            self.observation_space = obs_space

    # ------------- Gym API -------------

    def close(self):
        pass

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # lets Gymnasium seed np_random
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.step_count = 0
        self.inventory_slots = [int(Tile.EMPTY), int(Tile.EMPTY), int(Tile.EMPTY)]
        self._generate_layout()
        self._slip_ep, self._wind_dir_ep, self._wind_strength_ep = self._episode_params()
        obs = self._get_observation()
        info = {"agent_pos": self.agent_pos}
        return obs, info

    def seed(self, seed: int | None = None):
        # Gymnasium prefers seeding in reset(); this is for compatibility
        self._rng = np.random.default_rng(seed)

    def step(self, action: int):
        self.step_count += 1
        acted_nonmove = False  # True if we executed a use/swap action (consumes step)
        dx = dy = 0
        reward = 0.0
        terminated = False
        truncated = False

        # Remember the agent’s previous position (for push-backs) then update to new position
        prev_agent_pos = self.agent_pos

        # Branch on action family
        try:
            act_enum = Act(int(action))
        except (ValueError, TypeError):
            act_enum = Act.STAY

        if act_enum in self.MOVE_DELTAS:
            # Movement / stay
            dx, dy = self.MOVE_DELTAS[act_enum]
            proposed = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
            proposed = self._clip_to_bounds(proposed)

            # Slip is only possible when moving into or out of water
            src_is_water = (self._tile_at(self.agent_pos) == Tile.WATER)
            dst_is_water = (self._tile_at(proposed) == Tile.WATER)
            if (src_is_water or dst_is_water) and (self._rng.random() < self._slip_ep):
                # randomize among movement actions (0..4)
                action = int(self._rng.integers(0, len(self.MOVE_DELTAS)))
                act_enum = Act(action)
                dx, dy = self.MOVE_DELTAS[act_enum]
                proposed = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
                proposed = self._clip_to_bounds(proposed)

            # Wind: probabilistic push (as a separate single-cell attempt)
            if self._wind_strength_ep > 0 and self._rng.random() < self._wind_strength_ep:
                wdx, wdy = self._wind_dir_ep
                wx, wy = int(np.sign(wdx)), int(np.sign(wdy))
                cand = self._clip_to_bounds((proposed[0] + wx, proposed[1] + wy))
                # allow wind push only into occupiable tiles (not through-walls)
                cand_tile = Tile(int(self.grid[cand[1], cand[0]]))
                if cand_tile not in (Tile.WALL, Tile.LAVA, Tile.DOOR_LOCKED, Tile.TREE):
                    proposed = cand

            new_pos = proposed

        else:
            # Non-movement actions consume the step but do not move the agent
            acted_nonmove = True
            new_pos = self.agent_pos

            # Use slot on adjacent tile?
            sd = self._use_action_to_slot_dir(action)
            if sd is not None:
                s_idx, (ux, uy) = sd
                reward += self._use_slot_on_dir(s_idx, ux, uy)

            # Swap/Drop at current tile?
            elif act_enum in (Act.SWAP1, Act.SWAP2, Act.SWAP3):
                s_idx = {Act.SWAP1: 0, Act.SWAP2: 1, Act.SWAP3: 2}[act_enum]
                reward += self._swap_drop_pick_at_current(s_idx)

            # Unknown action types fall through as no-ops

        # Collision logic
        t = self._tile_at(new_pos)
        collided = False
        if t in (Tile.WALL, Tile.DOOR_LOCKED, Tile.TREE):
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
            if t == Tile.APPLE:
                reward += self.config.apple_reward
                self.apple_positions.remove(new_pos)
                self.grid[new_pos[1], new_pos[0]] = Tile.EMPTY
            elif t == Tile.GOAL:
                reward += self.config.goal_reward
                # remove from bookkeeping + clear cell
                self._remove_goal_at(new_pos)
                self.grid[new_pos[1], new_pos[0]] = Tile.EMPTY
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
                back = prev_agent_pos

                if self.config.truncate_on_collision:
                    truncated = True
                    new_pos = self.agent_pos
                else:
                    new_pos = back if self._agent_can_occupy(back) else self.agent_pos

        # Update to new position
        self.agent_pos = new_pos

        # Remember previous obstacle positions, then move dynamic obstacles
        prev_obs_positions = list(self.dynamic_obs)
        apples_eaten_by_obs = self._step_dynamic_obstacles()

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

                # Try push destination; if not safe, revert to previous cell if safe; else stay
                if self._agent_can_occupy(cand):
                    self.agent_pos = cand
                elif self._agent_can_occupy(prev_agent_pos):
                    self.agent_pos = prev_agent_pos
                else:
                    # Last-resort escape to avoid persistent co-location.
                    # Fixed neighbor order for determinism:
                    for nx, ny in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        cand2 = (self.agent_pos[0] + nx, self.agent_pos[1] + ny)
                        if self._agent_can_occupy(cand2):
                            self.agent_pos = cand2
                            break
                    else:
                        if self.config.truncate_on_collision:
                            truncated = True

            if self.config.truncate_on_collision:
                truncated = True

        # Trees may drop apples with each step
        self._trees_maybe_drop_apples()

        # If a push moves the agent onto an apple/key/goal, process the landing tile
        t_post = self._tile_at(self.agent_pos)
        if t_post == Tile.APPLE and self.agent_pos in self.apple_positions:
            reward += self.config.apple_reward
            self.apple_positions.remove(self.agent_pos)
            self.grid[self.agent_pos[1], self.agent_pos[0]] = Tile.EMPTY
        elif t_post == Tile.GOAL:
            reward += self.config.goal_reward
            # remove from bookkeeping + clear cell
            self._remove_goal_at(self.agent_pos)
            self.grid[self.agent_pos[1], self.agent_pos[0]] = Tile.EMPTY
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
            "acted_nonmove": acted_nonmove,
            "action": int(action),
            "agent_pos": self.agent_pos,
            "apples_eaten_by_obs": apples_eaten_by_obs,
            "collided": collided or pushed,
            "dst_is_water": (self._tile_at(self.agent_pos) == Tile.WATER),
            "slip_prob_ep": self._slip_ep,
            "src_is_water": (self._tile_at(prev_agent_pos) == Tile.WATER),
            "step": self.step_count,
            "wind_dir_ep": self._wind_dir_ep,
            "wind_strength_ep": self._wind_strength_ep,
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
            obs_tiles = crop.astype(np.uint8)

        if self.config.observation_mode == "tiles":
            return obs_tiles
        elif self.config.observation_mode == "dict":
            tfrac = np.array([self.step_count / max(1, self.config.max_steps)], dtype=np.float32)
            return {
                "tiles": obs_tiles,
                "inventory": np.array(self.inventory_slots, dtype=np.uint8),
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
            box = spaces.Box(low=0, high=len(Tile) - 1, shape=obs_shape, dtype=np.uint8)
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
        self._ensure_connectivity()

        # Agent spawn (avoid hazards)
        self.agent_pos = self._random_empty_cell()

        # Lava pools
        for _ in range(max(0, self.config.lava_pools)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.LAVA

        # Trees
        self.tree_positions = []
        for _ in range(max(0, self.config.trees)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.TREE
            self.tree_positions.append(pos)

        # Water pools
        for _ in range(max(0, self.config.water_pools)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.WATER

        # Apple
        self.apple_positions = []
        for _ in range(max(0, self.config.apples)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.APPLE
            self.apple_positions.append(pos)

        # Axes
        for _ in range(max(0, self.config.axes)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.AXE

        # Buckets
        for _ in range(max(0, self.config.buckets)):
            pos = self._random_empty_cell()
            self.grid[pos[1], pos[0]] = Tile.BUCKET

        # Dynamic obstacles (layered; do not write into base grid)
        self.dynamic_obs = []
        for _ in range(max(0, self.config.dynamic_obstacles)):
            pos = self._random_empty_cell()
            self.dynamic_obs.append(pos)
        self._dynamic_obs_set = set(self.dynamic_obs)

    def _remove_goal_at(self, pos: tuple[int, int]) -> None:
        """Remove goal bookkeeping for a goal at pos, if present."""
        try:
            self.goal_positions.remove(pos)
        except ValueError:
            pass

    def _spawn_goal(self, n: int):
        for _ in range(n):
            pos = self._find_goal_spot()
            self.grid[pos[1], pos[0]] = int(Tile.GOAL)
            self.goal_positions.append(pos)

    # ------------- Dynamics helpers -------------

    def _agent_can_occupy(self, p):
        """Is the target cell safe for the agent?"""
        x, y = p
        if not (0 <= x < self.config.width and 0 <= y < self.config.height):
            return False
        tile = Tile(int(self.grid[y, x]))
        if tile in (Tile.WALL, Tile.LAVA, Tile.DOOR_LOCKED, Tile.TREE):
            return False
        # can't move into another obstacle either
        if p in self.dynamic_obs and p != self.agent_pos:
            return False
        return True

    def _attempt_tree_drop_at(self, tree_pos: tuple[int, int], boosted: bool = False):
        """Attempt to drop an apple near a single tree position."""
        p = self.config.tree_drop_chance_watered if boosted else self.config.tree_drop_chance
        if self._rng.random() >= p:
            return
        empties = [(cx, cy) for (cx, cy) in self._neighbors4(tree_pos)
                   if self.grid[cy, cx] == Tile.EMPTY]
        if not empties:
            return
        drop = empties[self._rng.integers(0, len(empties))]
        self.grid[drop[1], drop[0]] = Tile.APPLE
        self.apple_positions.append(drop)

    def _clip_to_bounds(self, pos: tuple[int, int]) -> tuple[int, int]:
        x = int(np.clip(pos[0], 0, self.config.width - 1))
        y = int(np.clip(pos[1], 0, self.config.height - 1))
        return (x, y)

    def _ensure_connectivity(self) -> None:
        """
        Simplified invariant: ensure each GOAL has at least one non-WALL neighbor.
        Walls are immutable; other obstacles are removable and do not matter here.
        If a goal is boxed by walls, relocate it to a valid spot.
        """
        fixed_positions: list[tuple[int, int]] = []
        for pos in list(self.goal_positions):
            x, y = pos
            if not self._has_nonwall_neighbor(pos):
                # relocate
                self.grid[y, x] = int(Tile.EMPTY)
                new_pos = self._find_goal_spot()
                self.grid[new_pos[1], new_pos[0]] = int(Tile.GOAL)
                fixed_positions.append(new_pos)
            else:
                fixed_positions.append(pos)
        self.goal_positions = fixed_positions

    def _episode_params(self):
        """Derive effective episode-specific params without mutating self.config."""
        slip = self.config.slip_prob
        wind_dir = self.config.wind_dir
        wind_strength = self.config.wind_strength
        if self.config.domain_randomization:
            if self.config.slip_prob > 0:
                slip = float(np.clip(self._rng.normal(self.config.slip_prob, 0.02), 0.0, 0.5))
            if self.config.wind_strength > 0:
                wind_dir = tuple(int(x) for x in self._rng.choice([-1, 0, 1], size=2))
                # optional: a tiny jitter to strength as well (bounded)
                wind_strength = float(np.clip(
                    self._rng.normal(self.config.wind_strength, 0.02), 0.0, 1.0))
        return slip, wind_dir, wind_strength

    def _find_goal_spot(self, max_tries: int = 500) -> tuple[int, int]:
        """
        Sample empty cells until we find one that is not boxed by walls.
        Fallback to any empty cell if none found within max_tries (extremely rare
        unless the map is almost all walls).
        """
        last_empty: tuple[int, int] | None = None
        for _ in range(max_tries):
            pos = self._random_empty_cell()
            last_empty = pos
            if self._has_nonwall_neighbor(pos):
                return pos
        # Fallback: accept the last empty even if boxed (the caller will re-validate later)
        return last_empty or (1, 1)

    def _has_nonwall_neighbor(self, pos: tuple[int, int]) -> bool:
        """Return True if at least one 4-neighbor is not a WALL."""
        x, y = pos
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
                if Tile(int(self.grid[ny, nx])) != Tile.WALL:
                    return True
        return False

    def _neighbors4(self, pos: tuple[int, int]):
        x, y = pos
        cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(cx, cy) for (cx, cy) in cand
                if 0 <= cx < self.config.width and 0 <= cy < self.config.height]

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

    def _step_dynamic_obstacles(self) -> int:
        """Advance obstacle layer via random walk (or adversary bias) without touching base tiles."""
        apples_eaten = 0
        next_taken = set()
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
                # can't move into walls, lava, trees, or locked doors; avoid stacking, everything else is fine
                if tile in (Tile.WALL, Tile.LAVA, Tile.DOOR_LOCKED, Tile.TREE):
                    continue
                if (nx, ny) in next_taken:
                    continue  # avoid stacking
                moved = (nx, ny)
                break

            # If the obstacle's destination contains an apple, consume it.
            if self.grid[moved[1], moved[0]] == Tile.APPLE:
                # remove from bookkeeping list if present
                if moved in self.apple_positions:
                    self.apple_positions.remove(moved)
                # clear the base grid cell
                self.grid[moved[1], moved[0]] = int(Tile.EMPTY)
                apples_eaten += 1

            # update positions (avoid stacking handled above)
            next_taken.add(moved)
            new_positions.append(moved)

        self.dynamic_obs = new_positions
        self._dynamic_obs_set = set(new_positions)
        return apples_eaten

    def _tile_at(self, pos: tuple[int, int]) -> Tile:
        x, y = pos
        return Tile(int(self.grid[y, x]))

    def _trees_maybe_drop_apples(self):
        """Ambient apple-drops from all trees (low probability)."""
        if not self.tree_positions:
            return
        for pos in list(self.tree_positions):
            self._attempt_tree_drop_at(pos, boosted=False)

    # ------------- Inventory helpers -------------

    def _slot_empty(self, s: int) -> bool:
        return self.inventory_slots[s] == int(Tile.EMPTY)

    def _take_from_slot(self, s: int) -> int:
        """Remove and return tile code from slot s. Returns Tile.EMPTY if empty."""
        item = self.inventory_slots[s]
        self.inventory_slots[s] = int(Tile.EMPTY)
        return item

    def _swap_drop_pick_at_current(self, s: int) -> float:
        """
        Swap/Drop/Pick at current tile:
          - If slot empty and tile holds a pickable item -> pick it.
          - If slot has item and tile empty -> drop it.
          - If both have items -> swap.
          - If tile holds non-pickable -> do nothing.
        Returns reward delta (0 here; extend if needed).
        """
        if s < 0 or s > 2:
            return 0.0
        ax, ay = self.agent_pos
        t = Tile(int(self.grid[ay, ax]))

        # Define pickable items for inventory (for now: KEY only; extend later)
        pickable = {Tile.KEY, Tile.AXE, Tile.BUCKET}

        # WATER is pickable only if BUCKET is present in inventory
        has_bucket = any(x == int(Tile.BUCKET) for x in self.inventory_slots)
        if has_bucket:
            pickable = pickable | {Tile.WATER}

        slot_empty = self._slot_empty(s)
        tile_has_item = t in pickable

        if slot_empty and tile_has_item:
            # pick up -> remove from grid
            self.inventory_slots[s] = int(t)
            self.grid[ay, ax] = int(Tile.EMPTY)
            # bookkeeping for tracked item types
            if t == Tile.KEY and (ax, ay) in self.key_positions:
                self.key_positions.remove((ax, ay))
            return 0.0

        if (not slot_empty) and t == Tile.EMPTY:
            # drop from slot to tile
            item = self._take_from_slot(s)
            self.grid[ay, ax] = int(item)
            if item == int(Tile.KEY):
                self.key_positions.append((ax, ay))
            return 0.0

        if (not slot_empty) and tile_has_item:
            # swap item in slot with tile item
            item_in_slot = self.inventory_slots[s]
            # place tile item into slot
            self.inventory_slots[s] = int(t)
            # place previous slot item onto tile
            self.grid[ay, ax] = int(item_in_slot)
            # update bookkeeping for tracked types
            if t == Tile.KEY and (ax, ay) in self.key_positions:
                self.key_positions.remove((ax, ay))
            if item_in_slot == int(Tile.KEY):
                self.key_positions.append((ax, ay))
            return 0.0

        # Not pickable or no capacity at chosen slot
        return 0.0

    def _use_action_to_slot_dir(self, action: int):
        mapping = {
            Act.USE1_LEFT: (0, (-1, 0)), Act.USE1_RIGHT: (0, (1, 0)),
            Act.USE1_UP: (0, (0, -1)), Act.USE1_DOWN: (0, (0, 1)),
            Act.USE2_LEFT: (1, (-1, 0)), Act.USE2_RIGHT: (1, (1, 0)),
            Act.USE2_UP: (1, (0, -1)), Act.USE2_DOWN: (1, (0, 1)),
            Act.USE3_LEFT: (2, (-1, 0)), Act.USE3_RIGHT: (2, (1, 0)),
            Act.USE3_UP: (2, (0, -1)), Act.USE3_DOWN: (2, (0, 1)),
        }
        return mapping.get(int(action))

    def _use_slot_on_dir(self, s: int, dx: int, dy: int) -> float:
        """
        Use item in slot s on adjacent tile (dx, dy). Returns reward delta.
        Implemented interactions:
          - KEY on DOOR_LOCKED: unlocks door (consumes key), gives door_reward.
          - AXE on TREE: chops tree (removes it).
          - AXE on MOVING_OBS (if present at target): removes that obstacle.
          - WATER on LAVA: removes lava (consumes water).
          - WATER on TREE: triggers boosted apple drop near tree (consumes water).
        """
        if s < 0 or s > 2:
            return 0.0
        item = self.inventory_slots[s]
        if item == int(Tile.EMPTY):
            return 0.0

        ax, ay = self.agent_pos
        tx, ty = ax + dx, ay + dy
        if not (0 <= tx < self.config.width and 0 <= ty < self.config.height):
            return 0.0

        t = Tile(int(self.grid[ty, tx]))
        reward_delta = 0.0

        # KEY -> unlock adjacent locked door
        if item == int(Tile.KEY) and t == Tile.DOOR_LOCKED:
            self.grid[ty, tx] = int(Tile.EMPTY)
            self._take_from_slot(s)  # consume key
            if (tx, ty) in self.door_positions:
                self.door_positions.remove((tx, ty))
            reward_delta += float(self.config.door_reward)
            return reward_delta

        # AXE -> chop tree or remove moving obstacle at target cell
        if item == int(Tile.AXE):
            # Chop TREE
            if t == Tile.TREE:
                self.grid[ty, tx] = int(Tile.EMPTY)
                if (tx, ty) in self.tree_positions:
                    self.tree_positions.remove((tx, ty))
                return reward_delta
            # Remove MOVING_OBS at target location (layered)
            if (tx, ty) in self.dynamic_obs:
                # delete one instance at that location (if duplicates existed)
                for i, pos in enumerate(list(self.dynamic_obs)):
                    if pos == (tx, ty):
                        del self.dynamic_obs[i]
                        break
                self._dynamic_obs_set = set(self.dynamic_obs)
                return reward_delta

        # WATER -> extinguish lava or water a tree (boost apple drop)
        if item == int(Tile.WATER):
            # Extinguish LAVA
            if t == Tile.LAVA:
                self.grid[ty, tx] = int(Tile.EMPTY)
                self._take_from_slot(s)  # consume water
                return reward_delta
            # Water TREE: boosted apple drop attempt near that tree; tree remains
            if t == Tile.TREE:
                self._take_from_slot(s)  # consume water
                self._attempt_tree_drop_at((tx, ty), boosted=True)
                return reward_delta

        # Placeholder: extend for other item uses here
        return 0.0

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
            Tile.AGENT: " 😁 ",
            Tile.APPLE: " 🍎 ",
            Tile.AXE: " 🪓 ",  # Can be used to chops down trees and remove moving obstacles
            Tile.BUCKET: " 🪣 ",  # Enables picking up water if in inventory
            Tile.DOOR_LOCKED: " 🚪 ",
            Tile.EMPTY: " ⬜ ",
            Tile.GOAL: " 🏁 ",
            Tile.KEY: " 🔑 ",
            Tile.LAVA: " ♨️ ",
            Tile.MOVING_OBS: " 😈 ",
            Tile.TREE: " 🌳 ",  # Occasionally drops apples in adjacent tiles and can be removed with an axe
            Tile.WALL: " 🪨 ",
            Tile.WATER: " 💦 ",  # Risk of slipping, can be picked up, can be used to remove lava tiles, and can be used on trees to increase chance of dropping apples.
        }
        grid = self.grid.copy()
        for (ox, oy) in self.dynamic_obs:  # overlay dynamic obstacles first
            if 0 <= ox < self.config.width and 0 <= oy < self.config.height:
                grid[oy, ox] = Tile.MOVING_OBS
        ax, ay = self.agent_pos
        grid[ay, ax] = Tile.AGENT
        lines = ["".join(glyphs[Tile(int(t))] for t in row) for row in grid]
        out = "\n".join(lines)
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
    env = make_env(width=12, height=12, view_size=5,
                   apples=2, axes=1, buckets=1, doors=1, dynamic_obstacles=1, keys=1,
                   lava_pools=1, trees=2, water_pools=2,
                   walls_density=0.3, slip_prob=0.05, wind_strength=0.1, wind_dir=(1,0),
                   observation_mode="dict", render_mode="ansi")
    #obs, info = env.reset(seed=0)
    obs, info = env.reset()
    done = False
    total = 0.0
    while True:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        print(env.render())
        if term or trunc:
            print(f"Episode done, terminated: {term}, truncated: {trunc}, total reward: {total}")
            break
