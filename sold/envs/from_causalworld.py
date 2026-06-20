from envs.compat import OldGymAPIWrapper, RenderAdapter, missing_dependency_message
from envs.wrappers.action_repeat import ActionRepeat
from envs.wrappers.pixels import Pixels
from envs.wrappers.time_limit import TimeLimit
from typing import Tuple


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    """Create a CausalWorld task as a pixel-control environment.

    `name` can be a Gym id registered by CausalWorld, or a CausalWorld task generator name when the package exposes the
    classic task API. This adapter is intentionally optional: it does not import CausalWorld unless the suite is used.
    """
    try:
        import gym
        import causal_world  # noqa: F401
    except ImportError as exc:
        raise ImportError(missing_dependency_message(
            "causal_world",
            "pip install causal-world",
        )) from exc

    try:
        env = gym.make(name)
    except Exception:
        try:
            from causal_world.envs import CausalWorld
            from causal_world.task_generators import generate_task
            task = generate_task(task_generator_id=name)
            env = CausalWorld(task=task, skip_frame=1, enable_visualization=False)
        except Exception as exc:
            raise RuntimeError(f"Could not create CausalWorld environment `{name}`.") from exc

    if hasattr(env, "seed"):
        env.seed(seed)
    env = OldGymAPIWrapper(env)
    env = RenderAdapter(env, image_size=image_size)
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    env = Pixels(env, image_size)
    return env
