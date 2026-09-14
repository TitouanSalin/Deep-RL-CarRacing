"""
Environment factory for CarRacing-v3.

Provides functions to create training and evaluation environments
with proper wrappers, seeding, and optional frame stacking.

Wrapper pipeline:
1. ResizeObservation      – RGB (96,96,3) → (84,84,3)
2. VecFrameStack (SB3)    – stacks N frames → (84,84,3*N), applied at vec level
3. VecTransposeImage      – (84,84,C) → (C,84,84) for PyTorch CNN

We use SB3's VecFrameStack instead of gymnasium's FrameStackObservation
for full compatibility with SB3's CNN policies.
RGB is kept (no grayscale conversion) to avoid dimension issues
between gymnasium wrappers and SB3's vectorized environments.
"""

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import ResizeObservation
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


def _make_base_env(
    env_id: str = "CarRacing-v3",
    seed: int = 0,
    render_mode: str | None = None,
) -> gym.Env:
    """
    Create a single CarRacing environment with resize wrapper.

    Frame stacking is handled at the vectorized level by VecFrameStack.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = ResizeObservation(env, shape=(84, 84))  # (96,96,3) → (84,84,3)
    env.reset(seed=seed)
    return env


def make_vec_env(
    env_id: str = "CarRacing-v3",
    seed: int = 0,
    frame_stack: int = 4,
    render_mode: str | None = None,
) -> VecTransposeImage:
    """
    Create a vectorized environment compatible with SB3's CNN policies.

    Parameters
    ----------
    env_id : str
        Gymnasium environment ID.
    seed : int
        Random seed.
    frame_stack : int
        Number of frames to stack.
    render_mode : str or None
        Rendering mode.

    Returns
    -------
    VecTransposeImage
        Vectorized, frame-stacked, transposed environment.
        Observation shape: (frame_stack, 84, 84).
    """
    def _make():
        return _make_base_env(env_id, seed, render_mode)

    vec_env = DummyVecEnv([_make])
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def make_eval_env(
    env_id: str = "CarRacing-v3",
    seed: int = 42,
    frame_stack: int = 4,
) -> VecTransposeImage:
    """
    Create a vectorized evaluation environment (no rendering).

    Uses a different seed than training by default to avoid data leakage.
    """
    return make_vec_env(env_id, seed=seed, frame_stack=frame_stack)
