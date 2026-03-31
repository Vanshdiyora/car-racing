"""Human keyboard control for the race.

Returns continuous actions: [steering, gas, brake]
  steering: -1.0 (full left) to +1.0 (full right)
  gas:       0.0 to 1.0
  brake:     0.0 to 1.0

Multiple keys can be held simultaneously (e.g. gas + steer).
Values ramp smoothly for a real driving feel.
"""

from __future__ import annotations

import numpy as np

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

# Smooth ramping parameters
_STEER_SPEED = 0.08      # how fast steering ramps per frame
_STEER_DECAY = 0.12      # how fast steering returns to centre
_GAS_SPEED = 0.10        # how fast gas ramps up
_GAS_DECAY = 0.15        # how fast gas falls off
_BRAKE_SPEED = 0.15      # brake ramp up
_BRAKE_DECAY = 0.20      # brake ramp down

# Speed-aware driving parameters
_HIGH_SPEED_THRESHOLD = 40.0   # above this, steering is progressively reduced
_MAX_SPEED_FOR_SCALE = 80.0    # speed at which steering is most limited
_MIN_STEER_SCALE = 0.35        # minimum steering multiplier at top speed
_TURN_GAS_CUT = 0.6            # gas multiplied by this when steering hard
_TURN_AUTO_BRAKE_STEER = 0.45  # steering magnitude above which auto-brake kicks in
_TURN_AUTO_BRAKE_FORCE = 0.3   # auto-brake strength during sharp high-speed turns


class HumanController:
    """Read keyboard input and produce smooth continuous actions."""

    def __init__(self, key_map: dict[str, str] | None = None) -> None:
        if pygame is None:
            raise ImportError("pygame is required for human control — pip install pygame")
        self._map = key_map or DEFAULT_KEY_MAP
        self._resolve_keys()
        self.quit_requested = False

        # Smooth internal state
        self._steer: float = 0.0
        self._gas: float = 0.0
        self._brake: float = 0.0

    def _resolve_keys(self) -> None:
        self._accel = _KEY_LOOKUP.get(self._map["accelerate"])
        self._brake_key = _KEY_LOOKUP.get(self._map["brake"])
        self._left = _KEY_LOOKUP.get(self._map["steer_left"])
        self._right = _KEY_LOOKUP.get(self._map["steer_right"])

    def get_action(self, speed: float = 0.0) -> np.ndarray:
        """Poll key state and return continuous action [steering, gas, brake].

        Parameters
        ----------
        speed : float
            Current car speed (world units). Used to reduce steering
            sensitivity and apply auto-braking at high speed during turns.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit_requested = True
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        keys = pygame.key.get_pressed()

        # --- Steering (smooth ramp + auto-centre) ---
        if keys[self._left]:
            self._steer = max(-1.0, self._steer - _STEER_SPEED)
        elif keys[self._right]:
            self._steer = min(1.0, self._steer + _STEER_SPEED)
        else:
            # Auto-centre towards 0
            if self._steer > 0:
                self._steer = max(0.0, self._steer - _STEER_DECAY)
            elif self._steer < 0:
                self._steer = min(0.0, self._steer + _STEER_DECAY)

        # --- Gas (smooth ramp) ---
        if keys[self._accel]:
            self._gas = min(1.0, self._gas + _GAS_SPEED)
        else:
            self._gas = max(0.0, self._gas - _GAS_DECAY)

        # --- Brake (smooth ramp) ---
        if keys[self._brake_key]:
            self._brake = min(1.0, self._brake + _BRAKE_SPEED)
        else:
            self._brake = max(0.0, self._brake - _BRAKE_DECAY)

        # --- Speed-aware adjustments ---
        steer_out = self._steer
        gas_out = self._gas
        brake_out = self._brake
        abs_steer = abs(self._steer)

        if speed > _HIGH_SPEED_THRESHOLD:
            # 1) Reduce effective steering at high speed to prevent drift
            speed_ratio = min((speed - _HIGH_SPEED_THRESHOLD)
                              / (_MAX_SPEED_FOR_SCALE - _HIGH_SPEED_THRESHOLD), 1.0)
            steer_scale = 1.0 - speed_ratio * (1.0 - _MIN_STEER_SCALE)
            steer_out = self._steer * steer_scale

            # 2) Cut gas when turning at high speed
            if abs_steer > 0.15:
                turn_factor = min(abs_steer / 0.8, 1.0)  # 0→1 over steering range
                gas_cut = 1.0 - turn_factor * (1.0 - _TURN_GAS_CUT)
                gas_out *= gas_cut

            # 3) Auto-brake during sharp turns at high speed
            if abs_steer > _TURN_AUTO_BRAKE_STEER:
                turn_severity = min((abs_steer - _TURN_AUTO_BRAKE_STEER)
                                    / (1.0 - _TURN_AUTO_BRAKE_STEER), 1.0)
                auto_brake = turn_severity * speed_ratio * _TURN_AUTO_BRAKE_FORCE
                brake_out = max(brake_out, auto_brake)

        return np.array([steer_out, gas_out, brake_out], dtype=np.float32)
