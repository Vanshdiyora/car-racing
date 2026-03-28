"""Pygame-based race renderer — same-screen racing with AI car marker + minimap.

Both cars appear on a single view: full-screen follows the human car,
and the AI car is drawn as a coloured arrow marker projected into the
human's camera space.  A corner minimap shows the full track with both
car positions.

Supports UI themes loaded from configs/themes.yaml.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import pygame
    import pygame.font
    import pygame.draw
    import pygame.gfxdraw
except ImportError:
    pygame = None  # type: ignore[assignment]


# Default theme colours (used when no theme is provided)
_DEFAULT_THEME: dict[str, Any] = {
    "background": [20, 20, 20],
    "hud_bar": [15, 15, 15],
    "hud_text": [200, 200, 200],
    "position_bar_bg": [15, 15, 30],
    "track_bar": [50, 50, 50],
    "human_color": [80, 180, 255],
    "ai_color": [255, 80, 80],
    "speed_fast": [80, 255, 80],
    "speed_medium": [255, 255, 80],
    "speed_slow": [255, 80, 80],
    "positive_score": [80, 255, 80],
    "negative_score": [255, 80, 80],
    "grass_warning": [255, 50, 50],
    "pip_border": [255, 100, 100],
    "timer": [255, 255, 100],
    "ahead": [80, 255, 80],
    "behind": [255, 80, 80],
    "tied": [255, 255, 100],
    "font_family": "consolas",
    "font_size_small": 14,
    "font_size_medium": 20,
    "font_size_large": 36,
}

# CarRacing camera constants (must match gymnasium source)
_ZOOM = 2.7
_SCALE = 6.0
_CAM_ZOOM = _ZOOM * _SCALE  # ~16.2 pixels per world unit in the 96×96 frame


class RaceRenderer:
    """Same-screen renderer: human full-view + AI car marker + track minimap."""

    def __init__(
        self,
        width: int = 1200,
        height: int = 700,
        show_hud: bool = True,
        fps: int = 60,
        theme: dict[str, Any] | None = None,
        track_name: str = "",
    ) -> None:
        if pygame is None:
            raise ImportError("pygame is required for rendering — pip install pygame")
        pygame.init()
        self._width = width
        self._height = height
        self._show_hud = show_hud
        self._fps = fps
        self._t = {**_DEFAULT_THEME, **(theme or {})}
        self._track_name = track_name

        self._screen = pygame.display.set_mode((width, height))
        title = "Car Racing — Human vs AI"
        if track_name:
            title += f" — {track_name}"
        pygame.display.set_caption(title)
        self._clock = pygame.time.Clock()

        ff = self._t["font_family"]
        self._font = pygame.font.SysFont(ff, self._t["font_size_small"])
        self._hud_font = pygame.font.SysFont(ff, self._t["font_size_medium"], bold=True)
        self._big_font = pygame.font.SysFont(ff, self._t["font_size_large"], bold=True)
        self._pos_font = pygame.font.SysFont(ff, self._t["font_size_small"])

        # Minimap config
        self._minimap_size = 160
        self._minimap_margin = 10
        self._track_pts: list[tuple[float, float]] | None = None
        self._minimap_surface: Any = None
        self._mm_min_x = 0.0
        self._mm_min_y = 0.0
        self._mm_range = 1.0

    def render_frame(
        self,
        human_frame: np.ndarray | None,
        ai_frame: np.ndarray | None,
        human_metrics: dict[str, Any] | None = None,
        ai_metrics: dict[str, Any] | None = None,
        track_pts: list[tuple[float, float]] | None = None,
    ) -> None:
        """Draw the race frame with AI marker and minimap."""
        self._screen.fill(tuple(self._t["background"]))

        # Cache track waypoints for minimap (sent on first frame)
        if track_pts is not None:
            self._track_pts = track_pts
            self._minimap_surface = None  # rebuild

        top_bar_h = 40
        hud_bar_h = 50
        game_h = self._height - top_bar_h - hud_bar_h

        # --- Main view (Human) — full width ---
        if human_frame is not None:
            surf = self._array_to_surface(human_frame, (self._width, game_h))
            self._screen.blit(surf, (0, top_bar_h))

        # --- Draw AI car marker on the human's view ---
        h_pose = (human_metrics or {}).get("pose")
        a_pose = (ai_metrics or {}).get("pose")
        if h_pose and a_pose and human_frame is not None:
            self._draw_ai_car_marker(
                h_pose, a_pose,
                frame_w=self._width, frame_h=game_h,
                src_frame_h=human_frame.shape[0],
                src_frame_w=human_frame.shape[1],
                offset_y=top_bar_h,
            )

        # --- Minimap (top-right corner, inside game area) ---
        if self._track_pts and h_pose and a_pose:
            self._draw_minimap(
                h_pose, a_pose,
                x=self._width - self._minimap_size - self._minimap_margin,
                y=top_bar_h + self._minimap_margin,
            )

        # --- Top bar: position comparison ---
        self._draw_position_bar(human_metrics, ai_metrics, y=0, h=top_bar_h)

        # --- Bottom HUD bar ---
        hud_y = self._height - hud_bar_h
        pygame.draw.rect(self._screen, tuple(self._t["hud_bar"]), (0, hud_y, self._width, hud_bar_h))
        pygame.draw.line(self._screen, (60, 60, 60), (0, hud_y), (self._width, hud_y), 1)

        if self._show_hud and human_metrics:
            self._draw_single_hud(human_metrics, ai_metrics, y=hud_y + 8)

        pygame.display.flip()
        self._clock.tick(self._fps)

    # ------------------------------------------------------------------
    # AI car marker — projected onto human's camera view
    # ------------------------------------------------------------------

    def _draw_ai_car_marker(
        self,
        h_pose: tuple[float, float, float],
        a_pose: tuple[float, float, float],
        frame_w: int,
        frame_h: int,
        src_frame_h: int,
        src_frame_w: int,
        offset_y: int,
    ) -> None:
        """Project the AI car position into the human camera and draw a marker."""
        hx, hy, ha = h_pose
        ax, ay, aa = a_pose

        # Relative position in world space
        dx = ax - hx
        dy = ay - hy

        # Rotate into human's camera frame (camera rotates by -car_angle)
        cos_a = math.cos(-ha)
        sin_a = math.sin(-ha)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a

        # Scale: CarRacing renders 96×96 with zoom ~16.2 px/world-unit
        # then we scale to display frame size
        scale_x = frame_w / src_frame_w
        scale_y = frame_h / src_frame_h
        px = frame_w / 2.0 + rx * _CAM_ZOOM * scale_x
        py = frame_h / 2.0 - ry * _CAM_ZOOM * scale_y  # y-flip for screen

        # Screen coordinates
        sx = int(px)
        sy = int(py + offset_y)

        # Off-screen? Draw edge indicator instead
        margin = 50
        if (sx < -margin or sx > frame_w + margin
                or sy < offset_y - margin or sy > offset_y + frame_h + margin):
            self._draw_offscreen_indicator(sx, sy, frame_w, frame_h, offset_y)
            return

        # Clamp to visible area
        sx = max(0, min(frame_w - 1, sx))
        sy = max(offset_y, min(offset_y + frame_h - 1, sy))

        # Draw AI car as a rotated triangle (pointing in its heading)
        ai_angle = aa - ha  # relative heading
        size = 14
        cos_r = math.cos(ai_angle + math.pi / 2)
        sin_r = math.sin(ai_angle + math.pi / 2)
        # Triangle vertices: nose + two rear corners
        nose = (sx + int(size * cos_r), sy - int(size * sin_r))
        left = (sx + int(size * 0.6 * math.cos(ai_angle + math.pi / 2 + 2.4)),
                sy - int(size * 0.6 * math.sin(ai_angle + math.pi / 2 + 2.4)))
        right = (sx + int(size * 0.6 * math.cos(ai_angle + math.pi / 2 - 2.4)),
                 sy - int(size * 0.6 * math.sin(ai_angle + math.pi / 2 - 2.4)))

        ai_col = tuple(self._t["ai_color"])
        try:
            pygame.draw.polygon(self._screen, ai_col, [nose, left, right])
        except (ValueError, TypeError):
            pygame.draw.circle(self._screen, ai_col, (sx, sy), 8)
        # White outline for visibility
        pygame.draw.polygon(self._screen, (255, 255, 255), [nose, left, right], 2)
        # Label
        self._draw_text("AI", sx - 8, sy - 22, ai_col, self._pos_font)

    def _draw_offscreen_indicator(
        self, sx: int, sy: int, frame_w: int, frame_h: int, offset_y: int
    ) -> None:
        """Draw a dot at the screen edge pointing towards the off-screen AI car."""
        edge_x = max(20, min(frame_w - 20, sx))
        edge_y = max(offset_y + 20, min(offset_y + frame_h - 20, sy))
        if sx < 0:
            edge_x = 20
        elif sx > frame_w:
            edge_x = frame_w - 20
        if sy < offset_y:
            edge_y = offset_y + 20
        elif sy > offset_y + frame_h:
            edge_y = offset_y + frame_h - 20

        ai_col = tuple(self._t["ai_color"])
        pygame.draw.circle(self._screen, ai_col, (edge_x, edge_y), 8)
        pygame.draw.circle(self._screen, (255, 255, 255), (edge_x, edge_y), 8, 2)
        self._draw_text("AI", edge_x - 8, edge_y - 20, ai_col, self._pos_font)

    # ------------------------------------------------------------------
    # Minimap — track outline + both car dots
    # ------------------------------------------------------------------

    def _draw_minimap(
        self, h_pose: tuple, a_pose: tuple, x: int, y: int
    ) -> None:
        """Draw a small track minimap with both car positions."""
        size = self._minimap_size
        pts = self._track_pts
        if not pts or len(pts) < 4:
            return

        # Build static track surface once
        if self._minimap_surface is None:
            self._minimap_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            self._minimap_surface.fill((0, 0, 0, 140))

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self._mm_min_x = min(xs)
            self._mm_min_y = min(ys)
            self._mm_range = max(max(xs) - self._mm_min_x, max(ys) - self._mm_min_y) * 1.1
            if self._mm_range < 1e-6:
                self._mm_range = 1.0

            # Draw track polyline
            screen_pts = []
            for wx, wy in pts:
                mx = int((wx - self._mm_min_x) / self._mm_range * (size - 20) + 10)
                my = int((1 - (wy - self._mm_min_y) / self._mm_range) * (size - 20) + 10)
                screen_pts.append((mx, my))
            if len(screen_pts) > 2:
                pygame.draw.lines(self._minimap_surface, (100, 100, 100), True, screen_pts, 2)

        # Blit static track
        self._screen.blit(self._minimap_surface, (x, y))

        # Draw car dots (updated every frame)
        def _w2mm(wx: float, wy: float) -> tuple[int, int]:
            mx = int((wx - self._mm_min_x) / self._mm_range * (size - 20) + 10) + x
            my = int((1 - (wy - self._mm_min_y) / self._mm_range) * (size - 20) + 10) + y
            return mx, my

        hmx, hmy = _w2mm(h_pose[0], h_pose[1])
        pygame.draw.circle(self._screen, tuple(self._t["human_color"]), (hmx, hmy), 5)
        pygame.draw.circle(self._screen, (255, 255, 255), (hmx, hmy), 5, 1)

        amx, amy = _w2mm(a_pose[0], a_pose[1])
        pygame.draw.circle(self._screen, tuple(self._t["ai_color"]), (amx, amy), 5)
        pygame.draw.circle(self._screen, (255, 255, 255), (amx, amy), 5, 1)

        # Border
        pygame.draw.rect(self._screen, (80, 80, 80), (x, y, size, size), 1)

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
        pygame.draw.rect(self._screen, tuple(self._t["position_bar_bg"]), (0, y, self._width, h))

        h_pct = (h_met or {}).get("track_pct", 0.0)
        a_pct = (a_met or {}).get("track_pct", 0.0)

        bar_x = 140
        bar_w = self._width - 280
        bar_y = y + h // 2 - 4
        bar_h = 8

        # Track bar background
        pygame.draw.rect(self._screen, tuple(self._t["track_bar"]), (bar_x, bar_y, bar_w, bar_h), border_radius=4)

        # Human marker
        h_pos = bar_x + int(bar_w * min(h_pct, 100.0) / 100.0)
        pygame.draw.circle(self._screen, tuple(self._t["human_color"]), (h_pos, bar_y + bar_h // 2), 8)
        self._draw_text("H", h_pos - 4, bar_y - 15, tuple(self._t["human_color"]), self._pos_font)

        # AI marker
        a_pos = bar_x + int(bar_w * min(a_pct, 100.0) / 100.0)
        pygame.draw.circle(self._screen, tuple(self._t["ai_color"]), (a_pos, bar_y + bar_h // 2), 8)
        self._draw_text("AI", a_pos - 6, bar_y + bar_h + 2, tuple(self._t["ai_color"]), self._pos_font)

        # Labels on sides
        self._draw_text("START", bar_x - 50, bar_y - 3, (100, 100, 100), self._pos_font)
        self._draw_text("FINISH", bar_x + bar_w + 8, bar_y - 3, (100, 100, 100), self._pos_font)

        # Position text (P1/P2)
        if h_pct >= a_pct:
            pos_text = "P1"
            pos_col = tuple(self._t["ahead"])
        else:
            pos_text = "P2"
            pos_col = tuple(self._t["behind"])
        self._draw_text(pos_text, 10, y + 6, pos_col, self._hud_font)

        # Elapsed timer on right
        elapsed = (h_met or {}).get("elapsed", 0.0)
        mins = int(elapsed) // 60
        secs = int(elapsed) % 60
        self._draw_text(f"{mins:02d}:{secs:02d}", self._width - 60, y + 6, tuple(self._t["timer"]), self._hud_font)

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
            spd_col = tuple(self._t["speed_fast"])
        elif speed > 8:
            spd_col = tuple(self._t["speed_medium"])
        else:
            spd_col = tuple(self._t["speed_slow"])
        self._draw_text(f"SPD: {speed:.0f}", 15, y, spd_col, self._hud_font)

        # Track %
        self._draw_text(f"Track: {track_pct:.1f}%", 150, y, tuple(self._t["hud_text"]), self._hud_font)

        # Reward
        r_col = tuple(self._t["positive_score"]) if reward >= 0 else tuple(self._t["negative_score"])
        self._draw_text(f"Score: {reward:+.0f}", 340, y, r_col, self._hud_font)

        # Grass warning
        if on_grass:
            self._draw_text("OFF ROAD!", 530, y, tuple(self._t["grass_warning"]), self._hud_font)
        
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
                gap_col = tuple(self._t["ahead"])
            elif gap < 0:
                gap_text = f"BEHIND by {abs(gap):.1f}%"
                gap_col = tuple(self._t["behind"])
            else:
                gap_text = "TIED"
                gap_col = tuple(self._t["tied"])
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
