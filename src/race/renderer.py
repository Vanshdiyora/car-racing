"""Pygame-based split-screen renderer for Human vs AI race."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pygame
    import pygame.font
except ImportError:
    pygame = None  # type: ignore[assignment]


class RaceRenderer:
    """Draws the race: two side-by-side views with a HUD overlay."""

    def __init__(
        self,
        width: int = 1200,
        height: int = 600,
        show_hud: bool = True,
        fps: int = 60,
    ) -> None:
        if pygame is None:
            raise ImportError("pygame is required for rendering — pip install pygame")
        pygame.init()
        self._width = width
        self._height = height
        self._show_hud = show_hud
        self._fps = fps
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🏎️  Car Racing — Human vs AI")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 18)
        self._big_font = pygame.font.SysFont("consolas", 32, bold=True)

    def render_frame(
        self,
        human_frame: np.ndarray | None,
        ai_frame: np.ndarray | None,
        human_metrics: dict[str, Any] | None = None,
        ai_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Draw a single frame with optional HUD overlays."""
        self._screen.fill((30, 30, 30))
        half_w = self._width // 2
        panel_h = self._height - 100  # leave room for HUD at bottom

        # Left panel — Human
        if human_frame is not None:
            surf = self._array_to_surface(human_frame, (half_w - 10, panel_h))
            self._screen.blit(surf, (5, 50))
        self._draw_label("HUMAN", half_w // 2 - 40, 10, (100, 200, 255))

        # Right panel — AI
        if ai_frame is not None:
            surf = self._array_to_surface(ai_frame, (half_w - 10, panel_h))
            self._screen.blit(surf, (half_w + 5, 50))
        self._draw_label("AI (PPO)", half_w + half_w // 2 - 50, 10, (255, 100, 100))

        # Divider
        pygame.draw.line(self._screen, (200, 200, 200), (half_w, 0), (half_w, self._height), 2)

        # HUD — bottom bar
        hud_y = self._height - 48
        if self._show_hud:
            # Dark bar background
            pygame.draw.rect(self._screen, (20, 20, 20), (0, hud_y - 5, self._width, 55))
            pygame.draw.line(self._screen, (80, 80, 80), (0, hud_y - 5), (self._width, hud_y - 5), 1)

            if human_metrics:
                self._draw_race_hud(human_metrics, x=10, y=hud_y, colour=(100, 200, 255))
            if ai_metrics:
                self._draw_race_hud(ai_metrics, x=half_w + 10, y=hud_y, colour=(255, 100, 100))

            # Centred timer
            elapsed = human_metrics.get("elapsed", 0.0) if human_metrics else 0.0
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            self._draw_text(f"{mins:02d}:{secs:02d}", half_w - 22, 12, (255, 255, 100), self._font)

        pygame.display.flip()
        self._clock.tick(self._fps)

    def render_results(
        self, human_score: float, ai_score: float, human_summary: dict, ai_summary: dict
    ) -> None:
        """Show a results screen until the user closes it."""
        self._screen.fill((20, 20, 40))
        winner = "HUMAN WINS!" if human_score > ai_score else "AI WINS!" if ai_score > human_score else "TIE!"
        colour = (100, 255, 100) if "HUMAN" in winner else (255, 100, 100) if "AI" in winner else (255, 255, 100)

        self._draw_text(winner, self._width // 2 - 80, 40, colour, self._big_font)
        self._draw_text(f"Human score: {human_score:.0f}", 100, 120, (200, 200, 255))
        self._draw_text(f"AI score:    {ai_score:.0f}", 100, 150, (255, 200, 200))

        y = 200
        for label, summary in [("Human", human_summary), ("AI", ai_summary)]:
            self._draw_text(f"--- {label} ---", 100, y, (255, 255, 255))
            y += 25
            for k, v in summary.items():
                self._draw_text(f"  {k}: {v}", 100, y, (180, 180, 180))
                y += 22
            y += 10

        self._draw_text("Press any key or close window to exit", 100, y + 20, (120, 120, 120))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN):
                    waiting = False

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _array_to_surface(self, arr: np.ndarray, size: tuple[int, int]) -> Any:
        """Convert a numpy RGB array to a scaled Pygame surface."""
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        # Handle stacked frames — take last 3 channels or last frame
        if arr.shape[-1] > 3:
            arr = arr[..., -3:] if arr.shape[-1] >= 3 else arr[..., -1:]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        return pygame.transform.scale(surf, size)

    def _draw_label(self, text: str, x: int, y: int, colour: tuple) -> None:
        self._draw_text(text, x, y, colour, self._big_font)

    def _draw_race_hud(self, metrics: dict[str, Any], x: int, y: int, colour: tuple) -> None:
        """Draw a rich HUD row: Speed | Track% | Reward | Grass indicator."""
        speed = metrics.get("speed", 0.0)
        track_pct = metrics.get("track_pct", 0.0)
        reward = metrics.get("total_reward", 0.0)
        on_grass = metrics.get("on_grass", False)

        # Speed
        self._draw_text(f"SPD:{speed:5.1f}", x, y, colour)
        # Track completion
        self._draw_text(f"Track:{track_pct:5.1f}%", x + 120, y, (200, 200, 200))
        # Reward
        r_col = (100, 255, 100) if reward >= 0 else (255, 80, 80)
        self._draw_text(f"R:{reward:+.0f}", x + 270, y, r_col)
        # Grass warning
        if on_grass:
            self._draw_text("GRASS!", x + 380, y, (255, 50, 50))

    def _draw_hud(self, metrics: dict[str, Any], x: int, y: int) -> None:
        """Legacy HUD — fallback for non-race screens."""
        parts = [f"R:{metrics.get('total_reward', 0):.0f}"]
        parts.append(f"Steps:{metrics.get('steps', 0)}")
        text = "  |  ".join(parts)
        self._draw_text(text, x, y, (200, 200, 200))

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        colour: tuple = (255, 255, 255),
        font: Any = None,
    ) -> None:
        font = font or self._font
        surface = font.render(text, True, colour)
        self._screen.blit(surface, (x, y))
