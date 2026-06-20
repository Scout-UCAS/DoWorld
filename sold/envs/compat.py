import gym
import numpy as np
from typing import Any, Dict, Optional, Tuple


class OldGymAPIWrapper(gym.Wrapper):
    """Normalize Gymnasium/new-Gym APIs to the old Gym API used by this codebase."""

    def reset(self) -> np.ndarray:
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, _ = result
            return obs
        return result

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, bool(terminated or truncated), info
        return result


class RenderAdapter(gym.Wrapper):
    """Provide a stable RGB render interface across simulators."""

    def __init__(
        self,
        env: gym.Env,
        image_size: Tuple[int, int],
        render_mode: Optional[str] = None,
        camera_name: Optional[str] = None,
    ) -> None:
        super().__init__(env)
        self.image_size = image_size
        self.render_mode = render_mode
        self.camera_name = camera_name

    def render(self, mode: str = "rgb_array", size: Optional[Tuple[int, int]] = None):
        size = size or self.image_size
        kwargs: Dict[str, Any] = {}
        if self.camera_name is not None:
            kwargs["camera_name"] = self.camera_name

        try:
            if self.render_mode is not None:
                kwargs["mode"] = self.render_mode
            else:
                kwargs["mode"] = mode
            kwargs["width"], kwargs["height"] = size
            return self.env.render(**kwargs)
        except TypeError:
            pass

        try:
            return self.env.render(mode=mode, size=size)
        except TypeError:
            return self.env.render()


def missing_dependency_message(package_name: str, install_hint: str) -> str:
    return (
        f"Optional benchmark dependency `{package_name}` is not installed. "
        f"Install it first, then rerun this command. Suggested install: {install_hint}"
    )
