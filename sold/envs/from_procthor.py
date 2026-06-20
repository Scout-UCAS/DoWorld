import gym
import numpy as np
from envs.compat import OldGymAPIWrapper, missing_dependency_message
from envs.wrappers.action_repeat import ActionRepeat
from envs.wrappers.time_limit import TimeLimit
from typing import Dict, List, Tuple


class ProcTHORGymEnv(gym.Env):
    """Minimal AI2-THOR/ProcTHOR Gym wrapper for image-based navigation experiments."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        scene: str,
        image_size: Tuple[int, int],
        action_names: List[str] | None = None,
    ) -> None:
        try:
            from ai2thor.controller import Controller
        except ImportError as exc:
            raise ImportError(missing_dependency_message(
                "ai2thor",
                "pip install ai2thor prior",
            )) from exc

        self.scene = scene
        self.image_size = image_size
        self.action_names = action_names or [
            "MoveAhead",
            "RotateLeft",
            "RotateRight",
            "LookUp",
            "LookDown",
            "Done",
        ]
        self.controller = Controller(
            scene=scene,
            width=image_size[0],
            height=image_size[1],
            renderDepthImage=False,
            renderInstanceSegmentation=False,
        )
        self.action_space = gym.spaces.Discrete(len(self.action_names))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3,) + tuple(image_size), dtype=np.uint8)
        self._last_frame = None

    def _frame(self) -> np.ndarray:
        frame = self.controller.last_event.frame
        self._last_frame = np.moveaxis(frame, -1, 0).astype(np.uint8)
        return self._last_frame

    def reset(self) -> np.ndarray:
        self.controller.reset(scene=self.scene)
        return self._frame()

    def step(self, action: int):
        action_name = self.action_names[int(action)]
        event = self.controller.step(action=action_name)
        reward = 1.0 if action_name == "Done" and event.metadata.get("lastActionSuccess", False) else 0.0
        done = action_name == "Done"
        info: Dict[str, object] = {
            "success": bool(event.metadata.get("lastActionSuccess", False)) if done else False,
            "last_action_success": bool(event.metadata.get("lastActionSuccess", False)),
        }
        return self._frame(), reward, done, info

    def render(self, mode: str = "rgb_array"):
        if self._last_frame is None:
            return self._frame()
        return np.moveaxis(self._last_frame, 0, -1)


class DiscreteToOneHotAction(gym.ActionWrapper):
    """Expose a Box action space so SOLD's continuous actor can drive discrete ProcTHOR actions."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.num_actions = env.action_space.n
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)

    def action(self, action: np.ndarray) -> int:
        return int(np.asarray(action).argmax())


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    del seed
    env = ProcTHORGymEnv(scene=name, image_size=image_size)
    env = DiscreteToOneHotAction(env)
    env = OldGymAPIWrapper(env)
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    return env
