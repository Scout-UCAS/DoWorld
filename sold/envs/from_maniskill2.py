from envs.compat import OldGymAPIWrapper, RenderAdapter, missing_dependency_message
from envs.wrappers.action_repeat import ActionRepeat
from envs.wrappers.pixels import Pixels
from envs.wrappers.time_limit import TimeLimit
from typing import Tuple


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    """Create a ManiSkill2 environment as a pixel-control task."""
    try:
        import gym
        import mani_skill2.envs  # noqa: F401
    except ImportError as exc:
        raise ImportError(missing_dependency_message(
            "mani_skill2",
            "pip install mani-skill2",
        )) from exc

    kwargs = {
        "obs_mode": "rgbd",
        "reward_mode": "dense",
        "control_mode": "pd_ee_delta_pose",
    }
    try:
        env = gym.make(name, **kwargs)
    except TypeError:
        env = gym.make(name)

    if hasattr(env, "seed"):
        env.seed(seed)
    env = OldGymAPIWrapper(env)
    env = RenderAdapter(env, image_size=image_size, render_mode="rgb_array")
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    env = Pixels(env, image_size)
    return env
