import numpy as np
import pytest
from germ.sandbox.rl.gridworld.advanced_env import Tile, make_env


def fresh_env(**overrides):
    """Helper that returns a small env with sensible defaults for tests."""
    cfg = dict(
        width=8, height=8, view_size=5,
        observation_mode="tiles",
        channels="index",
        walls_density=0.0,
        lava_pools=0,
        pizzas=0, keys=0, doors=0, goals=0,
        dynamic_obstacles=0,
        slip_prob=0.0, wind_strength=0.0,
        truncate_on_collision=False,
        goal_terminates=True,
        domain_randomization=False,
        render_mode=None,
    )
    cfg.update(overrides)
    env = make_env(**cfg)
    return env


def test_action_space_sampling_repro_when_seeded():
    env1 = fresh_env(seed=0); env2 = fresh_env(seed=0)
    env1.action_space.seed(999); env2.action_space.seed(999)
    a1 = [env1.action_space.sample() for _ in range(10)]
    a2 = [env2.action_space.sample() for _ in range(10)]
    assert a1 == a2
    env1.close(); env2.close()


@pytest.mark.parametrize("mode", ["tiles", "dict"])
def test_crop_sizes_for_modes(mode):
    for V in (3,5,7):
        env = fresh_env(view_size=V, observation_mode=mode)
        obs, _ = env.reset(seed=0)
        tiles = obs["tiles"] if mode=="dict" else obs
        assert tiles.shape[:2] == (V, V)
        env.close()


def test_domain_randomization_bounds_and_types():
    env = fresh_env(domain_randomization=True, slip_prob=0.1, wind_strength=0.5)
    obs, _ = env.reset(seed=5)
    # In your current implementation, DR mutates config on reset.
    # Bounds we expect:
    assert 0.0 <= env._slip_ep <= 0.5
    # Wind dir must be in {-1,0,1}^2
    assert env._wind_dir_ep[0] in (-1,0,1)
    assert env._wind_dir_ep[1] in (-1,0,1)
    env.close()


def test_door_collision_when_locked():
    env = fresh_env(keys=0, doors=1)
    obs, _ = env.reset(seed=0)
    # Place a locked door above agent
    env.agent_pos = (3, 3)
    env.grid[2, 3] = Tile.DOOR_LOCKED
    prev = env.agent_pos
    obs, r, term, trunc, info = env.step(4)  # up
    assert info["collided"] is True
    assert env.agent_pos == prev
    env.close()


def test_dynamic_obstacle_push_adversary_moves_into_agent_cell():
    # Use adversary mode so obstacle moves greedily into agent
    env = fresh_env(dynamic_obstacles=1, adversary=True)
    obs, _ = env.reset(seed=3)

    # Clear interior and place agent/obstacle in a line
    env.grid[:, :] = Tile.EMPTY
    # Borders remain walls
    H, W = env.config.height, env.config.width
    env.grid[0, :] = env.grid[H-1, :] = Tile.WALL
    env.grid[:, 0] = env.grid[:, W-1] = Tile.WALL

    env.agent_pos = (4, 4)
    env.dynamic_obs = [(3, 4)]  # obstacle left of agent; will move right toward agent

    # Make a step with 'stay' so only obstacle moves
    obs, r, term, trunc, info = env.step(0)
    # After obstacle moves, it should enter the agent cell and push the agent.
    # Agent should end up either to the right (push direction) or previous position if blocked.
    assert info["collided"] is True
    assert env.agent_pos in [(5, 4), (4, 4)]
    env.close()


def test_lava_terminates_when_no_respawn():
    env = fresh_env(lava_pools=1, respawn_on_lava=False)
    obs, _ = env.reset(seed=0)
    # Put lava directly above agent and move up
    env.agent_pos = (3, 3)
    env.grid[2, 3] = Tile.LAVA
    obs, r, term, trunc, info = env.step(4)  # up
    assert term is True
    assert trunc is False
    assert r <= 0.0  # lava penalty + optional step penalty
    env.close()


def test_lava_respawn_when_enabled():
    env = fresh_env(lava_pools=0, respawn_on_lava=True)
    obs, _ = env.reset(seed=0)
    env.agent_pos = (3, 3)
    env.grid[2, 3] = Tile.LAVA
    old = env.agent_pos
    obs, r, term, trunc, info = env.step(4)  # up into lava -> respawn
    assert term is False
    assert trunc is False
    assert env.agent_pos != old
    # New position must be EMPTY
    x, y = env.agent_pos
    assert env.grid[y, x] == Tile.EMPTY
    env.close()


@pytest.mark.parametrize("mode,channels", [
    ("dict", "onehot"),
    ("rgb", "rgb"),
    ("tiles", "index"),
    ("tiles", "onehot"),
])
def test_obs_modes_and_shapes(mode, channels):
    env = fresh_env(observation_mode=mode, channels=channels, view_size=5)
    obs, _ = env.reset(seed=0)
    if mode == "dict":
        assert set(obs.keys()) == {"tiles", "inventory", "time"}
        tiles = obs["tiles"]
        inv = obs["inventory"]
        assert tiles.dtype == np.uint8 and tiles.shape[-1] == len(Tile)
        assert isinstance(inv, dict) and "keys" in inv
        assert inv["keys"].dtype == np.uint8  # scalar array
        assert 0.0 <= float(obs["time"]) <= 1.0
    elif mode == "tiles":
        if channels == "index":
            assert obs.shape == (5, 5)
            assert obs.dtype == np.int8
        elif channels == "onehot":
            assert obs.shape == (5, 5, len(Tile))
            assert obs.dtype == np.uint8
    elif mode == "rgb":
        assert obs.shape == (5, 5, 3)
        assert obs.dtype == np.uint8
    env.close()


def test_obstacle_last_resort_escape_or_truncate():
    # Place agent boxed by three walls but with one safe neighbor; obstacle moves onto it
    env = fresh_env(dynamic_obstacles=1, adversary=True, truncate_on_collision=False)
    obs, _ = env.reset(seed=7)
    H, W = env.config.height, env.config.width
    env.grid[:, :] = Tile.EMPTY
    env.grid[0,:] = env.grid[H-1,:] = Tile.WALL
    env.grid[:,0] = env.grid[:,W-1] = Tile.WALL
    env.agent_pos = (3, 3)
    # block left, up, down; right is free
    env.grid[3,2] = env.grid[2,3] = env.grid[4,3] = Tile.WALL
    env.dynamic_obs = [(2,3)]  # will move right toward agent
    _, _, _, _, info = env.step(0)  # stay; obstacle moves
    # Agent must get pushed to the only free neighbor (3,4) or (4,3) depending on your push order
    assert env.agent_pos in {(4,3), (3,4)}
    env.close()

    # If boxed on all four sides and truncate_on_collision=True -> truncated
    env = fresh_env(dynamic_obstacles=1, adversary=True, truncate_on_collision=True)
    obs, _ = env.reset(seed=6)
    env.grid[:, :] = Tile.WALL
    for y in range(1, H-1):
        for x in range(1, W-1):
            env.grid[y,x] = Tile.WALL
    # carve a 1x1 pocket for agent at (3,3)
    env.grid[3,3] = Tile.EMPTY
    env.agent_pos = (3,3)
    env.dynamic_obs = [(4,3)]  # will step into agent cell but no escape exists
    _, _, _, trunc, _ = env.step(0)
    assert trunc is True
    env.close()


def test_obstacle_over_pizza_preserves_underlying_tile_until_agent_collects():
    env = fresh_env(dynamic_obstacles=1, pizzas=1)
    obs, _ = env.reset(seed=4)
    # Force pizza and obstacle to overlap (layered)
    env.grid[:, :] = Tile.EMPTY
    H, W = env.config.height, env.config.width
    env.grid[0, :] = env.grid[H-1, :] = Tile.WALL
    env.grid[:, 0] = env.grid[:, W-1] = Tile.WALL

    env.agent_pos = (2, 2)
    pizza_pos = (3, 2)
    env.grid[pizza_pos[1], pizza_pos[0]] = Tile.PIZZA
    env.pizza_positions = [pizza_pos]

    # Place obstacle onto the same pizza cell
    env.dynamic_obs = [pizza_pos]

    # Ensure pizza still exists on base grid
    assert env.grid[pizza_pos[1], pizza_pos[0]] == Tile.PIZZA

    # Move agent right to collect (even though obstacle is there, the env will handle push/collect logic)
    obs, r, term, trunc, info = env.step(1)  # right
    # Depending on push resolution, either collected now or not—but if agent ends on pizza cell, it must be removed.
    if env.agent_pos == pizza_pos:
        assert env.grid[pizza_pos[1], pizza_pos[0]] == Tile.EMPTY
        assert pizza_pos not in env.pizza_positions
    env.close()


def test_reset_repro_no_domain_randomization():
    env1 = fresh_env(domain_randomization=False, seed=123)
    env2 = fresh_env(domain_randomization=False, seed=123)
    obs1, _ = env1.reset(); obs2, _ = env2.reset()
    assert np.array_equal(env1.grid, env2.grid)
    assert env1.agent_pos == env2.agent_pos
    # If you expose episode params:
    if hasattr(env1, "_slip_ep") and hasattr(env1, "_wind_dir_ep"):
        assert env1._slip_ep == env2._slip_ep
        assert env1._wind_dir_ep == env2._wind_dir_ep
    env1.close(); env2.close()


def test_time_truncation_hits_max_steps():
    env = fresh_env(max_steps=3)
    obs, _ = env.reset(seed=0)
    # 3 steps → truncated=True on or after the third step depending on your check
    _, _, _, trunc1, _ = env.step(0)
    _, _, _, trunc2, _ = env.step(0)
    _, _, _, trunc3, _ = env.step(0)
    assert trunc1 is False
    assert trunc2 is False
    assert trunc3 is True
    env.close()


def test_truncates_after_single_step_when_max_steps_one():
    env = fresh_env(max_steps=1)
    obs, _ = env.reset(seed=0)
    _, _, _, trunc, _ = env.step(0)
    assert trunc is True
    env.close()


def test_wall_collision_no_move():
    env = make_env(width=6, height=6, walls_density=0.0, observation_mode="tiles")
    obs, _ = env.reset(seed=0)
    # Force agent next to a wall
    env.agent_pos = (1, 1)
    # Up is a wall
    obs, r, term, trunc, info = env.step(4)  # up
    assert info["collided"]
    assert env.agent_pos == (1, 1)


def test_wall_collision_penalty_and_no_move():
    env = fresh_env()
    obs, _ = env.reset(seed=0)
    env.agent_pos = (1, 1)   # up is a wall (border)
    prev = env.agent_pos
    obs, r, term, trunc, info = env.step(4)  # up
    assert info["collided"] is True
    assert env.agent_pos == prev
    # Collision penalty should be <= 0
    assert r <= 0.0
    env.close()
