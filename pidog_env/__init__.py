"""PiDog Gymnasium Environment for Reinforcement Learning."""

from gymnasium.envs.registration import register

from .pidog_env import PiDogEnv

__all__ = ["PiDogEnv"]

# Register the environment
register(
    id="PiDog-v0",
    entry_point="pidog_env:PiDogEnv",
    max_episode_steps=1000,
)
