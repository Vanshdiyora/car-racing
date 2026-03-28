"""Interactive track editor — click to place waypoints, save to YAML.

Usage:
    python -m src.tools.track_editor
    python -m src.tools.track_editor --load configs/tracks.yaml --track oval
    python -m src.tools.track_editor --output my_track.yaml

Controls:
    Left Click   : Add waypoint
    Right Click  : Remove last waypoint
    C            : Clear all waypoints
    P            : Preview track (smooth spline)
    S            : Save track to YAML file
    L            : Load existing track
    +/-          : Adjust road width
    ESC / Q      : Quit
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pygame
    import pygame.font
except ImportError:
    print("pygame is required: pip install pygame")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml")
    sys.exit(1)

# -- constants --
WIDTH, HEIGHT = 900, 700
BG_COLOR = (30, 35, 40)
GRID_COLOR = (45, 50, 55)
ROAD_PREVIEW_COLOR = (80, 80, 80)
ROAD_FILL_COLOR = (60, 60, 60)
POINT_COLOR = (80, 200, 255)
POINT_HOVER_COLOR = (255, 220, 80)
LINE_COLOR = (60, 140, 200)
SPLINE_COLOR = (100, 255, 100)
TEXT_COLOR = (200, 200, 200)
HELP_COLOR = (120, 120, 120)

# Map screen ↔ world coords
WORLD_SCALE = 2.5  # pixels per world unit
WORLD_OFFSET_X = WIDTH // 2
WORLD_OFFSET_Y = HEIGHT // 2


def screen_to_world(sx: int, sy: int) -> tuple[float, float]:
    wx = (sx - WORLD_OFFSET_X) / WORLD_SCALE
    wy = -(sy - WORLD_OFFSET_Y) / WORLD_SCALE  # y-flip
    return round(wx, 1), round(wy, 1)


def world_to_screen(wx: float, wy: float) -> tuple[int, int]:
    sx = int(wx * WORLD_SCALE + WORLD_OFFSET_X)
    sy = int(-wy * WORLD_SCALE + WORLD_OFFSET_Y)
    return sx, sy


def catmull_rom(pts: list[list[float]], steps: int = 10) -> list[tuple[float, float]]:
    """Catmull-Rom spline through closed loop of points."""
    n = len(pts)
    if n < 4:
        return [(p[0], p[1]) for p in pts]
    result = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        for t in np.linspace(0, 1, steps, endpoint=False):
            tt = t * t
            ttt = tt * t
            x = 0.5 * (
                2 * p1[0]
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * tt
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * ttt
            )
            y = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * tt
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * ttt
            )
            result.append((x, y))
    return result


def draw_grid(screen: pygame.Surface) -> None:
    """Draw a subtle coordinate grid."""
    for wx in range(-150, 151, 25):
        sx, _ = world_to_screen(wx, 0)
        pygame.draw.line(screen, GRID_COLOR, (sx, 0), (sx, HEIGHT))
    for wy in range(-150, 151, 25):
        _, sy = world_to_screen(0, wy)
        pygame.draw.line(screen, GRID_COLOR, (0, sy), (WIDTH, sy))
    # Axes
    cx, cy = world_to_screen(0, 0)
    pygame.draw.line(screen, (70, 70, 70), (cx, 0), (cx, HEIGHT))
    pygame.draw.line(screen, (70, 70, 70), (0, cy), (WIDTH, cy))


def draw_road_preview(screen: pygame.Surface, spline: list, road_w: float) -> None:
    """Draw road width preview along the spline."""
    if len(spline) < 2:
        return
    for i in range(len(spline)):
        x1, y1 = spline[i]
        x2, y2 = spline[(i + 1) % len(spline)]
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.01:
            continue
        nx, ny = -dy / length * road_w, dx / length * road_w
        # Road quad
        p1 = world_to_screen(x1 - nx, y1 - ny)
        p2 = world_to_screen(x1 + nx, y1 + ny)
        p3 = world_to_screen(x2 + nx, y2 + ny)
        p4 = world_to_screen(x2 - nx, y2 - ny)
        pygame.draw.polygon(screen, ROAD_FILL_COLOR, [p1, p2, p3, p4])


def main() -> None:
    parser = argparse.ArgumentParser(description="Track Editor")
    parser.add_argument("--load", help="Load tracks YAML file")
    parser.add_argument("--track", help="Track name to load from file")
    parser.add_argument("--output", default="configs/tracks.yaml", help="Output YAML path")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Track Editor — Click to place waypoints")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    big_font = pygame.font.SysFont("consolas", 22, bold=True)

    waypoints: list[list[float]] = []
    track_name = "my_track"
    road_width = 1.0  # width_scale
    show_preview = True
    saved = False

    # Load existing track if specified
    if args.load and args.track:
        with open(args.load) as f:
            data = yaml.safe_load(f)
        tracks = data.get("tracks", {})
        if args.track in tracks:
            t = tracks[args.track]
            waypoints = [list(p) for p in t["waypoints"]]
            track_name = t.get("name", args.track)
            road_width = t.get("width_scale", 1.0)

    running = True
    while running:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        hover_wx, hover_wy = screen_to_world(mouse_x, mouse_y)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_c:
                    waypoints.clear()
                    saved = False
                elif event.key == pygame.K_p:
                    show_preview = not show_preview
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    road_width = min(2.0, road_width + 0.1)
                elif event.key == pygame.K_MINUS:
                    road_width = max(0.4, road_width - 0.1)
                elif event.key == pygame.K_s:
                    _save_track(args.output, track_name, waypoints, road_width)
                    saved = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click — add point
                    waypoints.append([hover_wx, hover_wy])
                    saved = False
                elif event.button == 3:  # right click — remove last
                    if waypoints:
                        waypoints.pop()
                        saved = False

        # -- draw --
        screen.fill(BG_COLOR)
        draw_grid(screen)

        # Spline preview
        spline = []
        if len(waypoints) >= 4 and show_preview:
            spline = catmull_rom(waypoints, steps=10)
            # Road width
            rw = 6.67 / 2 * road_width  # half of TRACK_WIDTH * scale
            draw_road_preview(screen, spline, rw)
            # Spline line
            screen_pts = [world_to_screen(x, y) for x, y in spline]
            if len(screen_pts) > 1:
                pygame.draw.lines(screen, SPLINE_COLOR, True, screen_pts, 2)

        # Raw waypoint lines
        if len(waypoints) >= 2:
            line_pts = [world_to_screen(p[0], p[1]) for p in waypoints]
            pygame.draw.lines(screen, LINE_COLOR, len(waypoints) >= 3, line_pts, 1)

        # Waypoints
        for i, pt in enumerate(waypoints):
            sx, sy = world_to_screen(pt[0], pt[1])
            col = POINT_COLOR if i > 0 else (255, 100, 80)  # first point = red
            pygame.draw.circle(screen, col, (sx, sy), 6)
            lbl = font.render(str(i + 1), True, TEXT_COLOR)
            screen.blit(lbl, (sx + 8, sy - 8))

        # Cursor crosshair
        pygame.draw.circle(screen, POINT_HOVER_COLOR, (mouse_x, mouse_y), 4, 1)

        # -- HUD --
        y = 8
        _text(screen, big_font, "TRACK EDITOR", 10, y, (100, 200, 255))
        y += 30
        _text(screen, font, f"Points: {len(waypoints)}   Width: {road_width:.1f}x", 10, y)
        y += 22
        _text(screen, font, f"Cursor: ({hover_wx}, {hover_wy})", 10, y)
        y += 22
        if saved:
            _text(screen, font, f"Saved to {args.output}", 10, y, (100, 255, 100))
        y += 30
        _text(screen, font, "LClick:Add  RClick:Undo  C:Clear  S:Save", 10, y, HELP_COLOR)
        y += 20
        _text(screen, font, "P:Preview  +/-:Width  ESC:Quit", 10, y, HELP_COLOR)

        if len(waypoints) < 4:
            _text(screen, font, f"Need {4 - len(waypoints)} more points for preview", 10, y + 30, (255, 200, 80))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _text(
    screen: pygame.Surface, font: pygame.font.Font,
    text: str, x: int, y: int, color: tuple = TEXT_COLOR,
) -> None:
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))


def _save_track(
    path: str, name: str,
    waypoints: list[list[float]], width_scale: float,
) -> None:
    """Append/update track in the YAML file."""
    p = Path(path)
    data: dict = {"tracks": {}}
    if p.exists():
        with open(p) as f:
            data = yaml.safe_load(f) or {"tracks": {}}

    key = name.lower().replace(" ", "_")
    data.setdefault("tracks", {})[key] = {
        "name": name,
        "width_scale": round(width_scale, 2),
        "interpolation_steps": 8,
        "waypoints": [[float(p[0]), float(p[1])] for p in waypoints],
    }

    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
