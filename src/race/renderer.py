"""Pygame-based race renderer — single-view with AI picture-in-picture."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pygame
    import pygame.font
    import pygame.draw
except ImportError:
    pygame = None  # type: ignore[assignment]


class RaceRenderer:
    """Full-screen human view with AI PiP, position bar, and HUD overlay.

    Layout:
    ┌────────────────────────────────────────┐
    │  [Position bar: HUMAN ●────○ AI]       │
    │                                        │
    │         Human full-screen view         │
    │                                   ┌───┐│
    │                                   │AI ││
    │                                   │PiP││
    │                                   └───┘│
    │  SPD:32.1  Track:24.3%  R:+120  00:42  │
    └────────────────────────────────────────┘
    """

    def __init__(
        self,
        width: int = 1200,
        height: int = 700,
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
        pygame.display.set_caption("Car Racing — Human vs AI")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("consolas", 16)
        self._hud_font = pygame.font.SysFont("consolas", 20, bold=True)
        self._big_font = pygame.font.SysFont("consolas", 36, bold=True)
        self._pos_font = pygame.font.SysFont("consolas", 14)

        # PiP dimensions (bottom-right corner)
        self._pip_w = width // 4
        self._pip_h = height // 4
        self._pip_x = width - self._pip_w - 10
        self._pip_y = height - self._pip_h - 55  # above HUD bar

    def render_frame(
        self,
        human_frame: np.ndarray | None,
        ai_frame: np.ndarray | None,
        human_metrics: dict[str, Any] | None = None,
        ai_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Draw single-view frame with PiP and HUD."""
        self._screen.fill((20, 20, 20))

        top_bar_h = 40
        hud_bar_h = 50
        game_h = self._height - top_bar_h - hud_bar_h

        # --- Main view (Human) — full width ---
        if human_frame is not None:
            surf = self._array_to_surface(human_frame, (self._width, game_h))
            self._screen.blit(surf, (0, top_bar_h))

        # --- AI Picture-in-Picture (bottom-right, above HUD) ---
        if ai_frame is not None:
            pip_surf = self._array_to_surface(ai_frame, (self._pip_w, self._pip_h))
            pip_y = top_bar_h + game_h - self._pip_h - 8
            pip_x = self._width - self._pip_w - 8
            # Border
            pygame.draw.rect(
                self._screen, (255, 100, 100),
                (pip_x - 2, pip_y - 2, self._pip_w + 4, self._pip_h + 4), 2,
            )
            self._screen.blit(pip_surf, (pip_x, pip_y))
            # Label
            self._draw_text("AI", pip_x + 4, pip_y + 2, (255, 100, 100), self._pos_font)

        # --- Top bar: position comparison ---
        self._draw_position_bar(human_metrics, ai_metrics, y=0, h=top_bar_h)

        # --- Bottom HUD bar ---
        hud_y = self._height - hud_bar_h
        pygame.draw.rect(self._screen, (15, 15, 15), (0, hud_y, self._width, hud_bar_h))
        pygame.draw.line(self._screen, (60, 60, 60), (0, hud_y), (self._width, hud_y), 1)

        if self._show_hud and human_metrics:
            self._draw_single_hud(human_metrics, ai_metrics, y=hud_y + 8)

        pygame.display.flip()
        self._clock.tick(self._fps)

    # ------------------------------------------------------------------
    # Position bar
    # ------------------------------------------------------------------

    def _draw_position_bar(
        self,
        h_met: dict[str, Any] | None,
        a_met: dict[str, Any] | None,
        y: int,
        h: int,
    ) -> None:
        """Draw a top bar showing track progress for both players."""
        pygame.draw.rect(self._screen, (15, 15, 30), (0, y, self._width, h))

        h_pct = (h_met or {}).get("track_pct", 0.0)
        a_pct = (a_met or {}).get("track_pct", 0.0)

        bar_x = 140
        bar_w = self._width - 280
        bar_y = y + h // 2 - 4
        bar_h = 8

        # Track bar background
        pygame.draw.rect(self._screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h), border_radius=4)

        # Human marker (blue)
        h_pos = bar_x + int(bar_w * min(h_pct, 100.0) / 100.0)
        pygame.draw.circle(self._screen, (80, 180, 255), (h_pos, bar_y + bar_h // 2), 8)
        self._draw_text("H", h_pos - 4, bar_y - 15, (80, 180, 255), self._pos_font)

        # AI marker (red)
        a_pos = bar_x + int(bar_w * min(a_pct, 100.0) / 100.0)
        pygame.draw.circle(self._screen, (255, 80, 80), (a_pos, bar_y + bar_h // 2), 8)
        self._draw_text("AI", a_pos - 6, bar_y + bar_h + 2, (255, 80, 80), self._pos_font)

        # Labels on sides
        self._draw_text("START", bar_x - 50, bar_y - 3, (100, 100, 100), self._pos_font)
        self._draw_text("FINISH", bar_x + bar_w + 8, bar_y - 3, (100, 100, 100), self._pos_font)

        # Position text (P1/P2)
        if h_pct >= a_pct:
            pos_text = "P1"
            pos_col = (80, 255, 80)
        else:
            pos_text = "P2"
            pos_col = (255, 160, 60)
        self._draw_text(pos_text, 10, y + 6, pos_col, self._hud_font)

        # Elapsed timer on right
        elapsed = (h_met or {}).get("elapsed", 0.0)
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        self._draw_text(f"{mins:02d}:{secs:02d}", self._width - 60, y + 6, (255, 255, 100), self._hud_font)

    # ------------------------------------------------------------------
    # Bottom HUD
    # ------------------------------------------------------------------

    def _draw_single_hud(
        self,
        h_met: dict[str, Any],
        a_met: dict[str, Any] | None,
        y: int,
    ) -> None:
        """Draw a single-line HUD for the human + comparison info."""
        speed = h_met.get("speed", 0.0)
        track_pct = h_met.get("track_pct", 0.0)
        reward = h_met.get("total_reward", 0.0)
        on_grass = h_met.get("on_grass", False)

        # Speed with colour (green=fast, yellow=medium, red=slow)
        if speed > 20:
            spd_col = (80, 255, 80)
        elif speed > 8:
            spd_col = (255, 255, 80)
        else:
            spd_col = (255, 80, 80)
        self._draw_text(f"SPD: {speed:.0f}", 15, y, spd_col, self._hud_font)

        # Track %
        self._draw_text(f"Track: {track_pct:.1f}%", 150, y, (200, 200, 200), self._hud_font)

        # Reward
        r_col = (80, 255, 80) if reward >= 0 else (255, 80, 80)
        self._draw_text(f"Score: {reward:+.0f}", 340, y, r_col, self._hud_font)

        # Grass warning
        if on_grass:
            self._draw_text("OFF ROAD!", 530, y, (255, 50, 50), self._hud_font)
        
        # Lap counter
        lap = h_met.get("lap", "")
        if lap:
            self._draw_text(f"Lap {lap}", 660, y, (255, 255, 200), self._hud_font)

        # AI comparison on the right side
        if a_met:
            a_track = a_met.get("track_pct", 0.0)
            gap = track_pct - a_track
            if gap > 0:
                gap_text = f"AHEAD by {gap:.1f}%"
                gap_col = (80, 255, 80)
            elif gap < 0:
                gap_text = f"BEHIND by {abs(gap):.1f}%"
                gap_col = (255, 80, 80)
            else:
                gap_text = "TIED"
                gap_col = (255, 255, 100)
            self._draw_text(gap_text, self._width - 250, y, gap_col, self._hud_font)

    # ------------------------------------------------------------------
    # Results screen
    # ------------------------------------------------------------------

    def render_results(
        self, human_score: float, ai_score: float, human_summary: dict, ai_summary: dict
    ) -> None:
        """Show a results screen until the user closes it."""
        self._screen.fill((20, 20, 40))
        winner = "HUMAN WINS!" if human_score > ai_score else "AI WINS!" if ai_score > human_score else "TIE!"
        colour = (100, 255, 100) if "HUMAN" in winner else (255, 100, 100) if "AI" in winner else (255, 255, 100)

        self._draw_text(winner, self._width // 2 - 100, 40, colour, self._big_font)
        self._draw_text(f"Human score: {human_score:.0f}", 100, 120, (200, 200, 255), self._hud_font)
        self._draw_text(f"AI score:    {ai_score:.0f}", 100, 150, (255, 200, 200), self._hud_font)

        y = 200
        for label, summary in [("Human", human_summary), ("AI", ai_summary)]:
            self._draw_text(f"--- {label} ---", 100, y, (255, 255, 255), self._hud_font)
            y += 28
            for k, v in summary.items():
                self._draw_text(f"  {k}: {v}", 100, y, (180, 180, 180))
                y += 22
            y += 12

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
        if arr.shape[-1] > 3:
            arr = arr[..., -3:] if arr.shape[-1] >= 3 else arr[..., -1:]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
        surf = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        return pygame.transform.scale(surf, size)

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
