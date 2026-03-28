"""Human keyboard control for the race.

Reads Pygame key state and maps to discrete CarRacing actions:
  0 = do nothing
  1 = steer left
  2 = steer right
  3 = accelerate
  4 = brake
"""

from __future__ import annotations

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]


# Default key mapping (arrow keys)
DEFAULT_KEY_MAP = {
    "accelerate": "UP",
    "brake": "DOWN",
    "steer_left": "LEFT",
    "steer_right": "RIGHT",
}

_KEY_LOOKUP = {
    "UP": getattr(pygame, "K_UP", None) if pygame else None,
    "DOWN": getattr(pygame, "K_DOWN", None) if pygame else None,
    "LEFT": getattr(pygame, "K_LEFT", None) if pygame else None,
    "RIGHT": getattr(pygame, "K_RIGHT", None) if pygame else None,
    "W": getattr(pygame, "K_w", None) if pygame else None,
    "S": getattr(pygame, "K_s", None) if pygame else None,
    "A": getattr(pygame, "K_a", None) if pygame else None,
    "D": getattr(pygame, "K_d", None) if pygame else None,
}


class HumanController:
    """Read keyboard input from a Pygame display and convert to discrete actions."""

    def __init__(self, key_map: dict[str, str] | None = None) -> None:
        if pygame is None:
            raise ImportError("pygame is required for human control — pip install pygame")
        self._map = key_map or DEFAULT_KEY_MAP
        self._resolve_keys()
        self.quit_requested = False

    def _resolve_keys(self) -> None:
        self._accel = _KEY_LOOKUP.get(self._map["accelerate"])
        self._brake = _KEY_LOOKUP.get(self._map["brake"])
        self._left = _KEY_LOOKUP.get(self._map["steer_left"])
        self._right = _KEY_LOOKUP.get(self._map["steer_right"])

    def get_action(self) -> int:
        """Poll key state and return the discrete action."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return 0
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit_requested = True
                return 0

        keys = pygame.key.get_pressed()
        if keys[self._accel]:
            return 3  # accelerate
        if keys[self._brake]:
            return 4  # brake
        if keys[self._left]:
            return 1  # steer left
        if keys[self._right]:
            return 2  # steer right
        return 0  # do nothing
