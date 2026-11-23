#!/usr/bin/env python3
"""
Procedural 75-Second Biological Food Chain Animation & Synth Foley Pipeline
===========================================================================

Sequence: Grass -> Grasshopper -> Frog -> Snake -> Hawk
Duration: Exactly 75.0 seconds (1800 frames @ 24 FPS)
Output:   food_chain_75s.mp4

Everything — visuals (procedural charcoal sketchbook with 3-pass Gaussian line
boil, Bezier curves, L-systems, easing kinematics) and audio (pure NumPy/SciPy
synthesized foley) — is generated entirely from mathematics and code.

Classes:
    - SketchRenderer   : OpenCV + NumPy 3-pass Gaussian line-boil renderer
    - SynthFoley       : Pure NumPy + SciPy procedural sound synthesis primitives
    - AudioFoley       : 75-second timeline-locked stereo audio composer/mixer
    - EntityKinematics : Procedural choreography, Bezier/L-system geometry & easing
    - MovieAssembler   : MoviePy streaming video + audio assembly into MP4
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    from moviepy.editor import AudioFileClip, VideoClip
except ImportError:  # Fallback compatibility for MoviePy 2.x
    from moviepy import AudioFileClip, VideoClip


# =============================================================================
# Global Constants & Sketchbook Palette (BGR for OpenCV)
# =============================================================================

WIDTH = 1280
HEIGHT = 720
FPS = 24
DURATION = 75.0
TOTAL_FRAMES = int(round(DURATION * FPS))  # 1800 frames
SAMPLE_RATE = 44100

# Off-white archival sketchbook & charcoal/graphite palette (OpenCV BGR order)
PALETTE = {
    "paper_bg": (234, 243, 246),       # Warm off-white parchment (RGB: 246, 243, 234)
    "paper_fill": (229, 238, 242),     # Slightly warmer ivory fill for entity bodies
    "wash_light": (208, 216, 220),     # Soft pencil wash
    "wash_med": (176, 182, 188),       # Deeper graphite shadow wash
    "charcoal_dark": (32, 34, 38),     # Deep charcoal ink
    "charcoal_med": (66, 69, 74),      # Mid graphite pencil
    "charcoal_light": (116, 119, 124), # Distant/soft graphite
    "charcoal_faint": (164, 167, 171), # Whisper-light motion/wind lines
}


# =============================================================================
# Mathematical Utilities: Easing Functions, Bezier Curves & L-Systems
# =============================================================================


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if val < lo else (hi if val > hi else val)


def smoothstep(u: float) -> float:
    u = clamp(u)
    return u * u * (3.0 - 2.0 * u)


def ease_in_quad(u: float) -> float:
    u = clamp(u)
    return u * u


def ease_out_quad(u: float) -> float:
    u = clamp(u)
    return 1.0 - (1.0 - u) * (1.0 - u)


def ease_in_out_quad(u: float) -> float:
    u = clamp(u)
    return 2.0 * u * u if u < 0.5 else 1.0 - ((-2.0 * u + 2.0) ** 2) * 0.5


def ease_in_cubic(u: float) -> float:
    u = clamp(u)
    return u * u * u


def ease_out_cubic(u: float) -> float:
    u = clamp(u)
    return 1.0 - (1.0 - u) ** 3


def ease_in_out_cubic(u: float) -> float:
    u = clamp(u)
    return 4.0 * u * u * u if u < 0.5 else 1.0 - ((-2.0 * u + 2.0) ** 3) * 0.5


def ease_in_expo(u: float) -> float:
    u = clamp(u)
    return 0.0 if u <= 0.0 else float(2.0 ** (10.0 * (u - 1.0)))


def ease_out_expo(u: float) -> float:
    u = clamp(u)
    return 1.0 if u >= 1.0 else float(1.0 - 2.0 ** (-10.0 * u))


def quad_bezier(
    p0: Sequence[float], p1: Sequence[float], p2: Sequence[float], steps: int = 16
) -> np.ndarray:
    """Evaluate a quadratic Bezier spline into (steps, 2) float64 points."""
    u = np.linspace(0.0, 1.0, steps, dtype=np.float64)[:, None]
    om = 1.0 - u
    p0a = np.asarray(p0, dtype=np.float64)
    p1a = np.asarray(p1, dtype=np.float64)
    p2a = np.asarray(p2, dtype=np.float64)
    return (om * om) * p0a + (2.0 * om * u) * p1a + (u * u) * p2a


def cubic_bezier(
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    steps: int = 24,
) -> np.ndarray:
    """Evaluate a cubic Bezier spline into (steps, 2) float64 points."""
    u = np.linspace(0.0, 1.0, steps, dtype=np.float64)[:, None]
    om = 1.0 - u
    p0a = np.asarray(p0, dtype=np.float64)
    p1a = np.asarray(p1, dtype=np.float64)
    p2a = np.asarray(p2, dtype=np.float64)
    p3a = np.asarray(p3, dtype=np.float64)
    return (
        (om ** 3) * p0a
        + (3.0 * om * om * u) * p1a
        + (3.0 * om * u * u) * p2a
        + (u ** 3) * p3a
    )


def eval_cubic_bezier_pt(
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    u: float,
) -> np.ndarray:
    """Evaluate a single point on a cubic Bezier curve at parameter u."""
    u = clamp(u)
    om = 1.0 - u
    return (
        (om ** 3) * np.asarray(p0, dtype=np.float64)
        + (3.0 * om * om * u) * np.asarray(p1, dtype=np.float64)
        + (3.0 * om * u * u) * np.asarray(p2, dtype=np.float64)
        + (u ** 3) * np.asarray(p3, dtype=np.float64)
    )


def generate_lsystem(axiom: str, rules: Dict[str, str], iterations: int) -> str:
    """Apply deterministic L-system rewriting rules for botanical structures."""
    current = axiom
    for _ in range(iterations):
        current = "".join(rules.get(ch, ch) for ch in current)
    return current


def rotate_points(
    pts: np.ndarray, angle_rad: float, origin: Tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Rotate (N, 2) points around origin by angle_rad."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    ox, oy = origin
    shifted = pts - np.array([ox, oy], dtype=np.float64)
    rot = np.empty_like(shifted)
    rot[:, 0] = shifted[:, 0] * c - shifted[:, 1] * s + ox
    rot[:, 1] = shifted[:, 0] * s + shifted[:, 1] * c + oy
    return rot


def ellipse_points(
    center: Tuple[float, float],
    axes: Tuple[float, float],
    angle_deg: float = 0.0,
    start_deg: float = 0.0,
    end_deg: float = 360.0,
    steps: int = 34,
) -> np.ndarray:
    """Sample an ellipse arc into (steps, 2) coordinates."""
    thetas = np.linspace(math.radians(start_deg), math.radians(end_deg), steps)
    rx, ry = axes
    pts = np.column_stack((rx * np.cos(thetas), ry * np.sin(thetas)))
    if abs(angle_deg) > 1e-6:
        pts = rotate_points(pts, math.radians(angle_deg))
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    return pts


# =============================================================================
# 1. THE VISUAL ENGINE: SketchRenderer (3-Pass Gaussian Line Boil)
# =============================================================================


class SketchRenderer:
    """
    Procedural hand-drawn sketchbook visual engine built on OpenCV and NumPy.

    Every line/shape is drawn 3 times with slight alpha transparency and
    independent random Gaussian jitter on every (x, y) coordinate on every
    single frame (24 FPS) to create an authentic animated line-boil aesthetic
    on an off-white paper canvas with charcoal/dark-grey lines.
    """

    def __init__(self, width: int = WIDTH, height: int = HEIGHT, seed: int = 42):
        self.width = width
        self.height = height
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.paper_base = self._generate_paper_texture(seed)
        self.canvas = self.paper_base.copy()
        # 3-pass alpha weights and jitter spread multipliers for line boil
        self.pass_alphas = (0.45, 0.37, 0.30)
        self.pass_jitter_mults = (0.75, 1.05, 1.38)

    def _generate_paper_texture(self, seed: int) -> np.ndarray:
        """Synthesize an archival off-white sketchbook paper texture procedurally."""
        rng = np.random.default_rng(seed)
        base = np.full(
            (self.height, self.width, 3), PALETTE["paper_bg"], dtype=np.float32
        )

        # Fine cotton-paper tooth grain
        fine_noise = rng.normal(0.0, 2.1, (self.height, self.width, 1)).astype(
            np.float32
        )
        # Subtle horizontal laid-paper fibers
        fiber_rows = rng.normal(0.0, 1.0, (self.height, 1, 1)).astype(np.float32)
        # Low-frequency paper clouding
        small_cloud = rng.normal(
            0.0, 2.8, (self.height // 16, self.width // 16, 1)
        ).astype(np.float32)
        cloud = cv2.resize(
            small_cloud, (self.width, self.height), interpolation=cv2.INTER_CUBIC
        )[..., None]

        # Soft sketchbook edge vignette
        yy, xx = np.mgrid[0 : self.height, 0 : self.width].astype(np.float32)
        nx = (xx - self.width * 0.5) / (self.width * 0.55)
        ny = (yy - self.height * 0.5) / (self.height * 0.55)
        dist2 = nx * nx + ny * ny
        vignette = -np.clip((dist2 - 0.55) * 9.0, 0.0, 10.5)[..., None]

        textured = np.clip(base + fine_noise + fiber_rows + cloud + vignette, 0, 255)
        return textured.astype(np.uint8)

    def begin_frame(self, frame_idx: int) -> None:
        """Reset canvas to off-white paper and re-seed per-frame Gaussian jitter."""
        self.rng = np.random.default_rng(self.base_seed + frame_idx * 104729 + 17)
        dx = int(self.rng.integers(-1, 2))
        dy = int(self.rng.integers(-1, 2))
        if dx == 0 and dy == 0:
            np.copyto(self.canvas, self.paper_base)
        else:
            self.canvas = np.roll(self.paper_base, shift=(dy, dx), axis=(0, 1))

    def get_rgb_frame(self) -> np.ndarray:
        """Return the current rendered frame in RGB uint8 format for MoviePy."""
        return cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB)

    def _compute_roi(
        self, pts: np.ndarray, pad: int
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute tight clipped bounding box [x0, y0, x1, y1] for fast ROI alpha blend."""
        if pts.size == 0:
            return None
        xmin = int(math.floor(float(np.min(pts[:, 0])))) - pad
        ymin = int(math.floor(float(np.min(pts[:, 1])))) - pad
        xmax = int(math.ceil(float(np.max(pts[:, 0])))) + pad
        ymax = int(math.ceil(float(np.max(pts[:, 1])))) + pad
        x0 = max(0, xmin)
        y0 = max(0, ymin)
        x1 = min(self.width, xmax)
        y1 = min(self.height, ymax)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _blend_roi_draw(
        self,
        pts: np.ndarray,
        pad: int,
        alpha: float,
        draw_fn: Callable[[np.ndarray, int, int], None],
    ) -> None:
        """Execute an OpenCV drawing callback on a minimal ROI and alpha-blend back."""
        if alpha <= 0.005:
            return
        roi_box = self._compute_roi(pts, pad)
        if roi_box is None:
            return
        x0, y0, x1, y1 = roi_box
        dst = self.canvas[y0:y1, x0:x1]
        if alpha >= 0.995:
            draw_fn(dst, x0, y0)
        else:
            overlay = dst.copy()
            draw_fn(overlay, x0, y0)
            cv2.addWeighted(overlay, alpha, dst, 1.0 - alpha, 0.0, dst)

    def fill_polygon(
        self,
        points: np.ndarray,
        color: Tuple[int, int, int] = PALETTE["paper_fill"],
        alpha: float = 0.96,
        jitter: float = 0.7,
    ) -> None:
        """Fill a closed polygon (ivory body mask or graphite wash) with soft boil."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] < 3 or alpha <= 0.005:
            return
        jpts = pts + self.rng.normal(0.0, jitter, pts.shape) if jitter > 0 else pts
        pad = int(math.ceil(jitter * 3.0)) + 4

        def _draw(buf: np.ndarray, x0: int, y0: int) -> None:
            local = np.round(jpts - np.array([x0, y0])).astype(np.int32)
            cv2.fillPoly(buf, [local], color, lineType=cv2.LINE_AA)

        self._blend_roi_draw(jpts, pad, clamp(alpha), _draw)

    def draw_polylines_batch(
        self,
        polylines: Sequence[np.ndarray],
        color: Tuple[int, int, int] = PALETTE["charcoal_dark"],
        thickness: float = 2.0,
        jitter: float = 1.35,
        closed: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """
        Render a group of polylines sharing style using the 3-pass Gaussian
        line-boil effect. Each polyline receives independent per-vertex and
        per-stroke Gaussian jitter on all 3 passes, blended with slight alpha.
        """
        valid = [
            np.asarray(p, dtype=np.float64)
            for p in polylines
            if p is not None and len(p) >= 2
        ]
        if not valid or opacity <= 0.005:
            return

        base_thick = max(1, int(round(thickness)))
        pad = base_thick + int(math.ceil(jitter * 4.5)) + 5

        all_pts = np.vstack(valid)
        roi_box = self._compute_roi(all_pts, pad)
        if roi_box is None:
            return
        x0, y0, x1, y1 = roi_box
        dst = self.canvas[y0:y1, x0:x1]
        offset = np.array([x0, y0], dtype=np.float64)

        for p_idx in range(3):
            sigma = jitter * self.pass_jitter_mults[p_idx]
            alpha = clamp(self.pass_alphas[p_idx] * opacity)
            if alpha <= 0.005:
                continue
            pass_thick = max(1, base_thick if p_idx < 2 else max(1, base_thick - 1))
            jittered_local = []
            for pts in valid:
                drift = self.rng.normal(0.0, sigma * 0.45, (1, 2))
                vjit = self.rng.normal(0.0, sigma * 0.85, pts.shape)
                local = np.round((pts + drift + vjit) - offset).astype(np.int32)
                jittered_local.append(local)

            overlay = dst.copy()
            cv2.polylines(
                overlay, jittered_local, closed, color, pass_thick, lineType=cv2.LINE_AA
            )
            cv2.addWeighted(overlay, alpha, dst, 1.0 - alpha, 0.0, dst)

    def draw_polyline(
        self,
        points: np.ndarray,
        color: Tuple[int, int, int] = PALETTE["charcoal_dark"],
        thickness: float = 2.0,
        jitter: float = 1.35,
        closed: bool = False,
        opacity: float = 1.0,
    ) -> None:
        """Draw a single polyline with the 3-pass Gaussian line-boil effect."""
        self.draw_polylines_batch(
            [points],
            color=color,
            thickness=thickness,
            jitter=jitter,
            closed=closed,
            opacity=opacity,
        )

    def draw_line(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        color: Tuple[int, int, int] = PALETTE["charcoal_dark"],
        thickness: float = 2.0,
        jitter: float = 1.35,
        opacity: float = 1.0,
        subdivisions: int = 4,
    ) -> None:
        """Draw a straight sketch line with 3-pass Gaussian line boil."""
        p0a = np.asarray(p0, dtype=np.float64)
        p1a = np.asarray(p1, dtype=np.float64)
        length = float(np.linalg.norm(p1a - p0a))
        n_pts = max(2, min(subdivisions + 1, int(length / 45.0) + 2))
        u = np.linspace(0.0, 1.0, n_pts)[:, None]
        pts = (1.0 - u) * p0a + u * p1a
        self.draw_polyline(
            pts,
            color=color,
            thickness=thickness,
            jitter=jitter,
            closed=False,
            opacity=opacity,
        )

    def draw_bezier(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        p2: Sequence[float],
        color: Tuple[int, int, int] = PALETTE["charcoal_dark"],
        thickness: float = 2.0,
        jitter: float = 1.35,
        steps: int = 14,
        opacity: float = 1.0,
    ) -> None:
        """Draw a quadratic Bezier spline with 3-pass Gaussian line boil."""
        pts = quad_bezier(p0, p1, p2, steps=steps)
        self.draw_polyline(
            pts, color=color, thickness=thickness, jitter=jitter, opacity=opacity
        )

    def draw_cubic_bezier(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        p2: Sequence[float],
        p3: Sequence[float],
        color: Tuple[int, int, int] = PALETTE["charcoal_dark"],
        thickness: float = 2.0,
        jitter: float = 1.35,
        steps: int = 20,
        opacity: float = 1.0,
    ) -> None:
        """Draw a cubic Bezier spline with 3-pass Gaussian line boil."""
        pts = cubic_bezier(p0, p1, p2, p3, steps=steps)
        self.draw_polyline(
            pts, color=color, thickness=thickness, jitter=jitter, opacity=opacity
        )

    def draw_ellipse(
        self,
        center: Tuple[float, float],
        axes: Tuple[float, float],
        angle_deg: float = 0.0,
        color: Tuple[int, int, int] = PALETTE["charcoal_dark"],
        fill_color: Optional[Tuple[int, int, int]] = None,
        fill_alpha: float = 0.96,
        thickness: float = 2.2,
        jitter: float = 1.35,
        opacity: float = 1.0,
        steps: int = 32,
    ) -> None:
        """Draw a sketched ellipse with optional fill and 3-pass Gaussian line boil."""
        if opacity <= 0.005:
            return
        pts = ellipse_points(center, axes, angle_deg=angle_deg, steps=steps)[:-1]
        if fill_color is not None:
            self.fill_polygon(
                pts, color=fill_color, alpha=fill_alpha * opacity, jitter=jitter * 0.5
            )
        self.draw_polyline(
            pts,
            color=color,
            thickness=thickness,
            jitter=jitter,
            closed=True,
            opacity=opacity,
        )

    def draw_circle(
        self,
        center: Tuple[float, float],
        radius: float,
        color: Tuple[int, int, int] = PALETTE["charcoal_dark"],
        fill_color: Optional[Tuple[int, int, int]] = None,
        fill_alpha: float = 0.96,
        thickness: float = 2.0,
        jitter: float = 1.2,
        opacity: float = 1.0,
    ) -> None:
        """Draw a sketched circle with optional fill and 3-pass Gaussian line boil."""
        steps = max(14, min(32, int(radius * 1.4) + 10))
        self.draw_ellipse(
            center=center,
            axes=(radius, radius),
            angle_deg=0.0,
            color=color,
            fill_color=fill_color,
            fill_alpha=fill_alpha,
            thickness=thickness,
            jitter=jitter,
            opacity=opacity,
            steps=steps,
        )

    def draw_polygon(
        self,
        points: np.ndarray,
        color: Tuple[int, int, int] = PALETTE["charcoal_dark"],
        fill_color: Optional[Tuple[int, int, int]] = None,
        fill_alpha: float = 0.96,
        thickness: float = 2.0,
        jitter: float = 1.35,
        opacity: float = 1.0,
    ) -> None:
        """Draw a closed polygon with optional fill and 3-pass Gaussian line boil."""
        if opacity <= 0.005:
            return
        pts = np.asarray(points, dtype=np.float64)
        if fill_color is not None:
            self.fill_polygon(
                pts, color=fill_color, alpha=fill_alpha * opacity, jitter=jitter * 0.45
            )
        self.draw_polyline(
            pts,
            color=color,
            thickness=thickness,
            jitter=jitter,
            closed=True,
            opacity=opacity,
        )

    def draw_ellipse_hatching(
        self,
        center: Tuple[float, float],
        axes: Tuple[float, float],
        angle_deg: float = 0.0,
        hatch_angle_deg: float = -35.0,
        spacing: float = 9.0,
        region_y_range: Tuple[float, float] = (0.05, 0.92),
        color: Tuple[int, int, int] = PALETTE["charcoal_light"],
        thickness: float = 1.2,
        jitter: float = 1.0,
        opacity: float = 0.58,
    ) -> None:
        """Add batched procedural charcoal cross-hatching lines inside an ellipse."""
        if opacity <= 0.005:
            return
        rx, ry = axes
        if rx < 5 or ry < 5:
            return
        rot_rad = math.radians(angle_deg)
        c, s = math.cos(rot_rad), math.sin(rot_rad)
        h_rad = math.radians(hatch_angle_deg)

        y_lo = ry * region_y_range[0]
        y_hi = ry * region_y_range[1]
        y_mid = 0.5 * (y_lo + y_hi)
        ts = np.linspace(-rx, rx, 22)
        ch, sh = math.cos(h_rad), math.sin(h_rad)

        lines: List[np.ndarray] = []
        for x_off in np.arange(-rx * 0.95, rx * 0.95, spacing):
            lx = x_off + ts * ch
            ly = y_mid + ts * sh
            inside = (
                ((lx / (rx * 0.93)) ** 2 + (ly / (ry * 0.93)) ** 2 <= 1.0)
                & (ly >= y_lo)
                & (ly <= y_hi)
            )
            idx = np.flatnonzero(inside)
            if idx.size >= 2:
                p0_l = np.array([lx[idx[0]], ly[idx[0]]])
                p1_l = np.array([lx[idx[-1]], ly[idx[-1]]])
                if np.linalg.norm(p1_l - p0_l) > 6.0:
                    p0 = [
                        center[0] + p0_l[0] * c - p0_l[1] * s,
                        center[1] + p0_l[0] * s + p0_l[1] * c,
                    ]
                    p1 = [
                        center[0] + p1_l[0] * c - p1_l[1] * s,
                        center[1] + p1_l[0] * s + p1_l[1] * c,
                    ]
                    lines.append(np.array([p0, p1], dtype=np.float64))

        if lines:
            self.draw_polylines_batch(
                lines,
                color=color,
                thickness=thickness,
                jitter=jitter,
                closed=False,
                opacity=opacity,
            )


# =============================================================================
# 2. THE AUDIO ENGINE: SynthFoley & AudioFoley (Pure NumPy + SciPy Synthesis)
# =============================================================================


class SynthFoley:
    """
    Pure procedural foley synthesizer using NumPy waveform generation and
    SciPy Butterworth signal filtering. Zero external audio samples.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, seed: int = 1337):
        self.sr = sample_rate
        self.rng = np.random.default_rng(seed)

    def _butter_filter(
        self,
        sig: np.ndarray,
        cutoff: float | Tuple[float, float],
        btype: str,
        order: int = 4,
    ) -> np.ndarray:
        nyq = 0.5 * self.sr
        if isinstance(cutoff, tuple):
            wn = (
                max(1e-4, min(0.999, cutoff[0] / nyq)),
                max(1e-4, min(0.999, cutoff[1] / nyq)),
            )
        else:
            wn = max(1e-4, min(0.999, cutoff / nyq))
        sos = signal.butter(order, wn, btype=btype, output="sos")
        return signal.sosfilt(sos, sig)

    def _fade(
        self, sig: np.ndarray, fade_in_s: float = 0.008, fade_out_s: float = 0.015
    ) -> np.ndarray:
        out = sig.copy()
        n = len(out)
        fi = min(n // 2, max(1, int(fade_in_s * self.sr)))
        fo = min(n // 2, max(1, int(fade_out_s * self.sr)))
        out[:fi] *= np.linspace(0.0, 1.0, fi)
        out[-fo:] *= np.linspace(1.0, 0.0, fo)
        return out

    def synth_grass_wind(
        self, duration: float, env_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Grass (Wind): Low-pass filtered white noise with a slow oscillating
        amplitude envelope and gentle rustle overtone layer.
        """
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        white = self.rng.standard_normal(n)

        # Primary warm low-pass wind bed
        low_wind = self._butter_filter(white, cutoff=420.0, btype="low", order=4)
        # Secondary grassy rustle (mid-low band evoking blades brushing in breeze)
        white2 = self.rng.standard_normal(n)
        rustle = self._butter_filter(white2, cutoff=(480.0, 1700.0), btype="band", order=3)

        # Slow oscillating amplitude LFOs simulating organic meadow breezes
        lfo_slow = 0.58 + 0.42 * np.sin(2.0 * np.pi * 0.19 * t - 0.8)
        lfo_gust = 0.72 + 0.28 * np.sin(2.0 * np.pi * 0.083 * t + 0.4)
        lfo_rustle = 0.50 + 0.50 * np.sin(2.0 * np.pi * 0.31 * t + 1.5)

        wind = low_wind * lfo_slow * lfo_gust * 1.45 + rustle * lfo_rustle * 0.24
        if env_fn is not None:
            wind *= env_fn(t)
        return self._fade(wind, 0.15, 0.25)

    def synth_grasshopper_hop(self, duration: float = 0.50) -> np.ndarray:
        """Boing/hop synth: springy pitch-bent triangle/sine with launch flick."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr

        freq = 175.0 + 460.0 * (1.0 - np.exp(-t * 9.5)) + 34.0 * np.sin(2.0 * np.pi * 24.0 * t)
        phase = 2.0 * np.pi * np.cumsum(freq) / self.sr
        tone = 0.72 * np.sin(phase) + 0.28 * signal.sawtooth(phase, width=0.5)
        env = np.exp(-t * 6.5) * (1.0 - np.exp(-t * 95.0))

        click = self._butter_filter(self.rng.standard_normal(n), (800.0, 4200.0), "band", 3)
        click *= np.exp(-t * 48.0) * 0.40

        return self._fade(tone * env * 0.85 + click, 0.003, 0.03)

    def synth_grasshopper_chirp(
        self, duration: float = 0.38, pulses: int = 4, base_hz: float = 4350.0
    ) -> np.ndarray:
        """Grasshopper: High-pitch square-wave chirps with stridulation pulsing."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr

        pulse_rate = pulses / duration
        pulse_phase = (t * pulse_rate) % 1.0
        pulse_env = np.where(
            pulse_phase < 0.64,
            np.sin(np.pi * pulse_phase / 0.64) ** 1.25,
            0.0,
        )
        tooth_mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 58.0 * t)

        freq = base_hz + 400.0 * pulse_phase
        phase = 2.0 * np.pi * np.cumsum(freq) / self.sr
        sq = signal.square(phase)
        sq_shaped = self._butter_filter(sq, (2200.0, 9600.0), "band", 3)

        macro_env = np.sin(np.pi * np.clip(t / duration, 0.0, 1.0)) ** 0.5
        out = sq_shaped * pulse_env * tooth_mod * macro_env * 0.58
        return self._fade(out, 0.005, 0.015)

    def synth_grasshopper_crunch(self, duration: float = 0.17) -> np.ndarray:
        """Grasshopper: Fast, rhythmic sawtooth 'crunching' sound for grass bites."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr

        f1 = 520.0 + 220.0 * np.sin(2.0 * np.pi * 34.0 * t)
        f2 = 1260.0 - 400.0 * (t / duration)
        f3 = 2250.0 + 440.0 * np.cos(2.0 * np.pi * 64.0 * t)
        saw = (
            0.48 * signal.sawtooth(2.0 * np.pi * np.cumsum(f1) / self.sr)
            + 0.34 * signal.sawtooth(2.0 * np.pi * np.cumsum(f2) / self.sr)
            + 0.18 * signal.sawtooth(2.0 * np.pi * np.cumsum(f3) / self.sr)
        )

        grain_env = np.zeros(n, dtype=np.float64)
        for pos_frac, amp in ((0.05, 1.0), (0.33, 0.85), (0.63, 0.68)):
            dt = t - pos_frac * duration
            mask = dt >= 0.0
            grain_env[mask] += amp * np.exp(-dt[mask] * 48.0) * (1.0 - np.exp(-dt[mask] * 420.0))

        snap = self._butter_filter(self.rng.standard_normal(n), (1100.0, 6200.0), "band", 3)
        out = (0.78 * saw + 0.22 * snap) * grain_env
        out = self._butter_filter(out, (350.0, 5900.0), "band", 3)
        return self._fade(out * 1.15, 0.002, 0.012)

    def synth_frog_hop_thud(self, duration: float = 0.24, weight: float = 1.0) -> np.ndarray:
        """Heavy soft thud when the large frog lands in the grass."""
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        freq = 128.0 * np.exp(-t * 16.0) + 38.0
        phase = 2.0 * np.pi * np.cumsum(freq) / self.sr
        thump = np.sin(phase) * np.exp(-t * 15.0)
        grass_brush = self._butter_filter(self.rng.standard_normal(n), 750.0, "low", 3)
        grass_brush *= np.exp(-t * 22.0) * 0.28
        return self._fade((thump * 0.72 + grass_brush) * weight, 0.002, 0.025)

    def synth_frog_ribbit(
        self, duration: float = 0.56, base_freq: float = 102.0, intensity: float = 1.0
    ) -> np.ndarray:
        """
        Frog: True FM (Frequency Modulation) synthesis for a low, vibrating
        two-syllable 'rib-bit' croak.
        """
        n = int(round(duration * self.sr))
        out = np.zeros(n, dtype=np.float64)

        syllables = [
            (0.00, 0.16, base_freq * 0.92, base_freq * 1.55, 34.0, 4.2, 0.78),
            (0.20, 0.52, base_freq * 1.05, base_freq * 1.88, 39.0, 5.4, 1.00),
        ]
        for s0, s1, f_lo, f_hi, f_mod, mod_idx, gain in syllables:
            i0 = int(round(s0 * self.sr))
            i1 = min(n, int(round(s1 * self.sr)))
            if i1 <= i0:
                continue
            ls = np.linspace(0.0, 1.0, i1 - i0)

            fc = f_lo + (f_hi - f_lo) * (np.sin(np.pi * (ls ** 0.75)))
            fm1 = f_mod + 12.0 * ls
            fm2 = fc * 2.02

            ph_c = 2.0 * np.pi * np.cumsum(fc) / self.sr
            ph_m1 = 2.0 * np.pi * np.cumsum(fm1) / self.sr
            ph_m2 = 2.0 * np.pi * np.cumsum(fm2) / self.sr

            idx_env = mod_idx * (0.45 + 0.55 * np.sin(np.pi * ls))
            fm_wave = np.sin(ph_c + idx_env * np.sin(ph_m1) + 1.35 * np.sin(ph_m2))

            amp_env = (np.sin(np.pi * ls) ** 0.7) * (0.65 + 0.35 * np.sin(ph_m1))
            out[i0:i1] += fm_wave * amp_env * gain

        out = self._butter_filter(out, (95.0, 1250.0), "band", 3)
        return self._fade(out * 0.85 * intensity, 0.008, 0.025)

    def synth_tongue_smack(self, duration: float = 0.18) -> np.ndarray:
        """
        Frog Tongue Hit: Sharp, high-amplitude transient 'smack' combining
        broadband whip snap, steep pitch-drop transient, and sticky wet pop.
        """
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr

        burst = self.rng.standard_normal(n)
        smack_hi = self._butter_filter(burst, (900.0, 9500.0), "band", 3)
        smack_hi *= np.exp(-t * 95.0) * 1.48

        f_snap = 2150.0 * np.exp(-t * 140.0) + 125.0
        snap_tone = np.sin(2.0 * np.pi * np.cumsum(f_snap) / self.sr)
        snap_tone *= np.exp(-t * 48.0) * 1.15

        dt_pop = np.maximum(0.0, t - 0.014)
        f_pop = 420.0 + 680.0 * (1.0 - np.exp(-dt_pop * 120.0))
        pop = np.sin(2.0 * np.pi * np.cumsum(f_pop) / self.sr)
        pop *= np.where(t >= 0.014, np.exp(-dt_pop * 58.0), 0.0) * 0.55

        out = smack_hi + snap_tone + pop
        return self._fade(np.tanh(out * 1.25) * 0.98, 0.0006, 0.02)

    def synth_snake_hiss(
        self, duration: float, env_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Snake: High-pass filtered white noise (sibilant hiss) with breathy
        undulating modulation and slight high-frequency resonance.
        """
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        white = self.rng.standard_normal(n)

        hp = self._butter_filter(white, cutoff=3900.0, btype="high", order=5)
        sib = self._butter_filter(white, cutoff=(4800.0, 7800.0), btype="band", order=3)

        breath = 0.68 + 0.32 * np.sin(2.0 * np.pi * 0.95 * t + 0.3)
        hiss = (hp * 0.65 + sib * 0.55) * breath * 0.42

        if env_fn is not None:
            hiss *= env_fn(t)
        return self._fade(hiss, 0.08, 0.05)

    def synth_snake_lunge_gulp(self, duration: float = 2.3) -> np.ndarray:
        """
        Snake Strike & Swallow: Heavy low-frequency thud followed immediately
        by a deep low sine-wave gulp.
        """
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        out = np.zeros(n, dtype=np.float64)

        # Violent strike impact thud
        f_thud = 135.0 * np.exp(-t * 18.0) + 34.0
        thud = np.sin(2.0 * np.pi * np.cumsum(f_thud) / self.sr) * np.exp(-t * 9.5)
        strike_air = self._butter_filter(self.rng.standard_normal(n), (180.0, 1400.0), "band", 3)
        strike_air *= np.exp(-t * 28.0) * 0.38
        out += thud * 0.98 + strike_air

        # Low sine-wave gulp (two visceral peristaltic swallow waves)
        for g_start, g_len, f_start, f_dip, f_end, gain in (
            (0.28, 0.78, 108.0, 44.0, 74.0, 0.88),
            (1.02, 0.88, 86.0, 38.0, 56.0, 0.64),
        ):
            i0 = int(round(g_start * self.sr))
            i1 = min(n, int(round((g_start + g_len) * self.sr)))
            if i1 <= i0:
                continue
            u = np.linspace(0.0, 1.0, i1 - i0)
            freq = np.where(
                u < 0.55,
                f_start + (f_dip - f_start) * np.sin(0.5 * np.pi * (u / 0.55)),
                f_dip + (f_end - f_dip) * np.sin(0.5 * np.pi * ((u - 0.55) / 0.45)),
            )
            ph = 2.0 * np.pi * np.cumsum(freq) / self.sr
            gulp = 0.82 * np.sin(ph) + 0.18 * np.sin(2.0 * ph)
            env = (np.sin(np.pi * u) ** 0.85) * (0.75 + 0.25 * np.sin(3.0 * np.pi * u))
            out[i0:i1] += gulp * env * gain

        return self._fade(out, 0.003, 0.08)

    def synth_hawk_screech(self, duration: float = 2.0) -> np.ndarray:
        """
        Hawk: Piercing, descending pitch-bent sawtooth wave (raptor screech)
        with raspy vibrato and resonant formant shaping.
        """
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        u = t / duration

        pitch_curve = np.where(
            u < 0.08,
            2800.0 + 560.0 * (u / 0.08),
            3360.0 * ((1460.0 / 3360.0) ** ((u - 0.08) / 0.92)),
        )
        vibrato = (58.0 + 35.0 * u) * np.sin(2.0 * np.pi * (36.0 - 8.0 * u) * t)
        f1 = pitch_curve + vibrato
        f2 = f1 * 0.992

        ph1 = 2.0 * np.pi * np.cumsum(f1) / self.sr
        ph2 = 2.0 * np.pi * np.cumsum(f2) / self.sr
        saw = 0.65 * signal.sawtooth(ph1) + 0.35 * signal.sawtooth(ph2)

        screech = self._butter_filter(saw, (1050.0, 6800.0), "band", 4)
        env = np.where(
            u < 0.06,
            (u / 0.06) ** 0.6,
            np.exp(-1.85 * (u - 0.06)) * (0.85 + 0.15 * np.sin(2.0 * np.pi * 14.0 * t)),
        )
        return self._fade(screech * env * 1.15, 0.008, 0.12)

    def synth_hawk_whoosh(self, duration: float = 8.8, peak_time: float = 3.25) -> np.ndarray:
        """
        Hawk: Massive heavy filtered noise wind whoosh peaking at the talon
        strike and pulsing with heavy wingbeats during ascent.
        """
        n = int(round(duration * self.sr))
        t = np.arange(n, dtype=np.float64) / self.sr
        white = self.rng.standard_normal(n)

        sub_roar = self._butter_filter(white, 340.0, "low", 4)
        mid_rush = self._butter_filter(white, (260.0, 1300.0), "band", 3)
        air_slice = self._butter_filter(white, (1100.0, 3900.0), "band", 3)

        before = np.exp(-((t - peak_time) / 1.15) ** 2)
        after = np.exp(-np.maximum(0.0, t - peak_time) * 0.68)
        macro_env = np.where(t <= peak_time, before, after)

        slice_env = np.exp(-((t - peak_time) / 0.55) ** 2)

        dt_climb = np.maximum(0.0, t - (peak_time + 0.15))
        wing_beats = np.where(
            (t > peak_time + 0.15) & (t < peak_time + 3.6),
            0.55 + 0.45 * (np.maximum(0.0, np.sin(2.0 * np.pi * 2.9 * dt_climb)) ** 1.5),
            1.0,
        )

        dt_hit = np.maximum(0.0, t - peak_time)
        hit_thump = np.where(
            t >= peak_time,
            np.sin(2.0 * np.pi * 68.0 * dt_hit) * np.exp(-dt_hit * 18.0) * 0.65,
            0.0,
        )

        whoosh = (
            sub_roar * macro_env * wing_beats * 1.45
            + mid_rush * macro_env * wing_beats * 0.80
            + air_slice * slice_env * 0.50
            + hit_thump
        )
        return self._fade(whoosh * 0.95, 0.25, 0.65)


class AudioFoley(SynthFoley):
    """
    75-second synchronized stereo soundtrack composer and mixer.

    Consumes the exact event schedule from EntityKinematics so every chirp,
    crunch, ribbit, tongue smack, hiss, silence, gulp, screech, and whoosh
    aligns sample-accurately with the 24 FPS animation frames.
    """

    def __init__(
        self,
        timeline: Dict[str, object],
        duration: float = DURATION,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(sample_rate=sample_rate, seed=2026)
        self.timeline = timeline
        self.duration = duration
        self.total_samples = int(round(duration * sample_rate))

    def _mix_stereo(
        self,
        master: np.ndarray,
        mono_clip: np.ndarray,
        start_time: float,
        pan: float = 0.0,
        gain: float = 1.0,
    ) -> None:
        """Mix a mono synthesized clip into stereo master with constant-power panning."""
        start_idx = int(round(start_time * self.sr))
        if start_idx >= self.total_samples:
            return
        clip = mono_clip * gain
        if start_idx < 0:
            clip = clip[-start_idx:]
            start_idx = 0
        end_idx = min(self.total_samples, start_idx + len(clip))
        length = end_idx - start_idx
        if length <= 0:
            return

        theta = (clamp(pan, -1.0, 1.0) + 1.0) * (math.pi * 0.25)
        master[start_idx:end_idx, 0] += clip[:length] * math.cos(theta)
        master[start_idx:end_idx, 1] += clip[:length] * math.sin(theta)

    def synthesize_track(self) -> np.ndarray:
        """Build the complete 75.0-second stereo float64 master track."""
        master = np.zeros((self.total_samples, 2), dtype=np.float64)

        silence_start = float(self.timeline["snake_silence_start"])
        lunge_start = float(self.timeline["snake_lunge_start"])

        # 1. Ambient Grass Wind Bed across the 75s timeline
        def wind_master_env(t: np.ndarray) -> np.ndarray:
            s1 = np.clip((t - 0.15) / 13.0, 0.0, 1.0)
            env = 0.04 + 0.56 * (s1 ** 1.30)
            env = np.where(t > 15.0, env * (0.68 + 0.32 * np.exp(-(t - 15.0) * 0.4)), env)
            # Sudden silence right before snake lunge
            cut_down = np.clip((silence_start - t) / 0.10, 0.0, 1.0)
            in_silence = (t >= silence_start) & (t < lunge_start + 2.4)
            env = np.where(t >= silence_start - 0.10, env * cut_down, env)
            env = np.where(in_silence, 0.0, env)
            # Quiet breeze returns briefly before hawk arrives
            post_gulp = np.clip((t - 57.8) / 1.8, 0.0, 1.0)
            env = np.where(t >= 57.8, 0.28 * post_gulp, env)
            # Fading silence into final 3 seconds
            fade_end = np.clip((74.6 - t) / 5.1, 0.0, 1.0) ** 1.8
            env = np.where(t >= 69.5, env * fade_end, env)
            env = np.where(t >= 74.7, 0.0, env)
            return env

        wind_l = self.synth_grass_wind(self.duration, env_fn=wind_master_env)
        wind_r = self.synth_grass_wind(self.duration, env_fn=wind_master_env)
        master[:, 0] += wind_l * 0.85
        master[:, 1] += wind_r * 0.85

        # 2. Scene 2: Grasshopper (15s - 30s) — Boing Hop, Square Chirps & Sawtooth Crunches
        hop_t = float(self.timeline["gh_hop_start"])
        self._mix_stereo(master, self.synth_grasshopper_hop(0.52), hop_t, pan=0.45, gain=0.82)
        self._mix_stereo(
            master,
            self.synth_frog_hop_thud(0.14, weight=0.25),
            float(self.timeline["gh_hop_end"]),
            pan=0.28,
            gain=0.65,
        )

        for idx, chirp_t in enumerate(self.timeline["gh_chirps"]):
            pan = 0.22 - 0.03 * idx
            base_hz = 4250.0 + (idx % 3) * 180.0
            self._mix_stereo(
                master,
                self.synth_grasshopper_chirp(duration=0.38, pulses=4, base_hz=base_hz),
                float(chirp_t),
                pan=pan,
                gain=0.88,
            )

        for idx, crunch_t in enumerate(self.timeline["gh_crunches"]):
            pan = 0.25 - 0.008 * idx
            self._mix_stereo(
                master,
                self.synth_grasshopper_crunch(duration=0.17),
                float(crunch_t),
                pan=pan,
                gain=0.92,
            )

        # 3. Scene 3: Frog (30s - 45s) — Heavy Hops, FM Ribbits & Sharp Tongue Smack
        for hop_land_t, pan, weight in self.timeline["frog_hop_landings"]:
            self._mix_stereo(
                master,
                self.synth_frog_hop_thud(duration=0.25, weight=weight),
                float(hop_land_t),
                pan=float(pan),
                gain=0.82,
            )

        for rib_t, base_f, inten in self.timeline["frog_ribbits"]:
            self._mix_stereo(
                master,
                self.synth_frog_ribbit(duration=0.56, base_freq=float(base_f), intensity=float(inten)),
                float(rib_t),
                pan=-0.42,
                gain=0.88,
            )

        tongue_hit_t = float(self.timeline["tongue_hit_time"])
        self._mix_stereo(
            master,
            self.synth_tongue_smack(duration=0.19),
            tongue_hit_t,
            pan=0.05,
            gain=1.10,
        )

        # 4. Scene 4: Snake (45s - 60s) — High-Pass Hiss, Sudden Silence, Thud/Gulp
        hiss_start = float(self.timeline["snake_hiss_start"])
        hiss_dur = max(0.5, silence_start - hiss_start)

        def hiss_env(t: np.ndarray) -> np.ndarray:
            u = t / hiss_dur
            build = np.clip(u / 0.35, 0.0, 1.0) ** 0.9
            coil_swell = 1.0 + 0.65 * np.clip((u - 0.68) / 0.28, 0.0, 1.0)
            sharp_stop = np.clip((1.0 - u) / 0.022, 0.0, 1.0)
            return build * coil_swell * sharp_stop

        hiss = self.synth_snake_hiss(hiss_dur, env_fn=hiss_env)
        self._mix_stereo(master, hiss, hiss_start, pan=0.38, gain=0.95)

        strike_hit_t = float(self.timeline["snake_strike_hit"])
        self._mix_stereo(
            master,
            self.synth_snake_lunge_gulp(duration=2.3),
            strike_hit_t,
            pan=-0.12,
            gain=0.98,
        )

        # 5. Scene 5: Hawk (60s - 75s) — Descending Screech, Heavy Whoosh & Fading Silence
        screech_t = float(self.timeline["hawk_screech_time"])
        self._mix_stereo(
            master,
            self.synth_hawk_screech(duration=2.0),
            screech_t,
            pan=-0.25,
            gain=1.05,
        )

        whoosh_start = float(self.timeline["hawk_whoosh_start"])
        talon_hit_t = float(self.timeline["hawk_talon_hit"])
        whoosh = self.synth_hawk_whoosh(
            duration=8.8, peak_time=max(0.6, talon_hit_t - whoosh_start)
        )
        self._mix_stereo(master, whoosh, whoosh_start, pan=0.05, gain=1.02)

        # Master soft-clipping limiter, headroom normalization & clean endpoints
        master = np.tanh(master * 0.95)
        peak = float(np.max(np.abs(master)))
        if peak > 1e-6:
            master = master * (0.90 / peak)

        fi = int(0.15 * self.sr)
        fo = int(1.60 * self.sr)
        master[:fi] *= np.linspace(0.0, 1.0, fi)[:, None]
        master[-fo:] *= (np.linspace(1.0, 0.0, fo) ** 2.0)[:, None]
        master[-int(0.25 * self.sr) :] = 0.0
        return master

    def export_wav(self, wav_path: str) -> str:
        """Synthesize and write 16-bit stereo PCM WAV file using SciPy."""
        stereo = self.synthesize_track()
        pcm16 = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(wav_path, self.sr, pcm16)
        return wav_path


# =============================================================================
# 3. CHOREOGRAPHY & PROCEDURAL GEOMETRY: EntityKinematics
# =============================================================================


@dataclass
class GrassBlade:
    x0: float
    y0: float
    height: float
    lean: float
    arch_bias: float          # Extra mid-curve bend so blades arch organically
    thickness: float
    depth_layer: int          # 0=back, 1=mid, 2=front (shorter foreground tufts)
    t_start: float
    grow_dur: float
    wind_freq: float
    wind_phase: float
    wind_amp: float
    eat_times: List[float]


@dataclass
class LSystemPlant:
    x0: float
    y0: float
    lstring: str
    step_len: float
    angle_deg: float
    t_start: float
    grow_dur: float
    wind_phase: float
    depth_layer: int
    eat_times: List[float]


@dataclass
class Feather:
    spawn_time: float
    x0: float
    y0: float
    fall_speed: float
    drift_amp: float
    drift_omega: float
    drift_phase: float
    wind_bias: float
    length: float
    tilt_amp: float
    lstring: str


class EntityKinematics:
    """
    Controls the entire 75-second 5-stage biological food chain choreography
    using Bezier curves, L-systems, parabolic physics, sinusoidal kinematics,
    and easing functions.
    """

    def __init__(
        self,
        width: int = WIDTH,
        height: int = HEIGHT,
        fps: int = FPS,
        duration: float = DURATION,
        seed: int = 7500,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.ground_y = height - 44.0  # y = 676 baseline
        self.rng = np.random.default_rng(seed)

        self.timeline = self._build_timeline()
        self.grass_blades: List[GrassBlade] = self._init_grass_blades()
        self.lsystem_plants: List[LSystemPlant] = self._init_lsystem_plants()
        self.ground_strokes: List[Tuple[float, float, float, float]] = self._init_ground_strokes()
        self.feathers: List[Feather] = self._init_feathers()

    def _build_timeline(self) -> Dict[str, object]:
        """Define frame-accurate timestamps for all 5 scenes."""
        gh_bouts = [
            (18.60, 20.10, 795.0, [19.05, 19.38, 19.72]),
            (20.75, 22.25, 762.0, [21.18, 21.52, 21.85]),
            (22.50, 23.95, 730.0, [22.92, 23.25, 23.58]),
            (24.65, 26.10, 700.0, [25.08, 25.40, 25.74]),
            (26.40, 28.10, 678.0, [26.82, 27.14, 27.46, 27.78]),
        ]
        all_crunches: List[float] = []
        for _, _, _, ct in gh_bouts:
            all_crunches.extend(ct)

        # Frog tongue strike: strictly < 5 frames (4 frames: 966, 967, 968, 969)
        tongue_frame_start = 966
        tongue_hit_frame = 967
        tongue_gone_frame = 969

        return {
            # Scene 2: Grasshopper
            "gh_hop_start": 15.20,
            "gh_hop_end": 16.85,
            "gh_chirps": [17.45, 18.08, 20.32, 24.22, 29.05],
            "gh_bouts": gh_bouts,
            "gh_crunches": all_crunches,
            # Scene 3: Frog
            "frog_hop1": (30.20, 31.60),
            "frog_hop2": (32.15, 33.55),
            "frog_hop3": (43.10, 44.20),
            "frog_hop_landings": [
                (31.60, -0.68, 1.00),
                (33.55, -0.46, 1.05),
                (44.20, -0.32, 0.60),
            ],
            "frog_ribbits": [
                (34.75, 102.0, 1.00),
                (37.15, 96.0, 1.08),
                (42.05, 110.0, 0.72),
            ],
            "tongue_frame_start": tongue_frame_start,
            "tongue_hit_frame": tongue_hit_frame,
            "tongue_gone_frame": tongue_gone_frame,
            "tongue_hit_time": tongue_hit_frame / float(self.fps),
            # Scene 4: Snake
            "snake_enter_start": 45.00,
            "snake_hiss_start": 45.35,
            "snake_coil_start": 51.40,
            "snake_coil_end": 54.10,
            "snake_silence_start": 53.50,
            "snake_lunge_start": 55.20,
            "snake_strike_hit": 55.56,
            "snake_lunge_end": 55.70,
            # Scene 5: Hawk
            "hawk_shadow_start": 60.80,
            "hawk_screech_time": 61.55,
            "hawk_whoosh_start": 62.10,
            "hawk_dive_start": 62.20,
            "hawk_talon_hit": 65.35,
            "hawk_ascent_start": 65.62,
            "hawk_exit_time": 68.75,
            "meadow_fade_start": 69.25,
            "meadow_fade_end": 71.75,
        }

    def _init_ground_strokes(self) -> List[Tuple[float, float, float, float]]:
        """Generate sketchy horizontal charcoal earth lines along the meadow floor."""
        strokes = []
        for _ in range(28):
            x0 = float(self.rng.uniform(15.0, self.width - 140.0))
            length = float(self.rng.uniform(55.0, 195.0))
            y0 = float(self.rng.uniform(self.ground_y - 4.0, self.height - 10.0))
            t_s = float(self.rng.uniform(0.45, 3.8))
            strokes.append((x0, x0 + length, y0, t_s))
        return strokes

    def _init_grass_blades(self) -> List[GrassBlade]:
        """Create 78 gracefully curved Bezier grass blades across 3 depth layers."""
        blades: List[GrassBlade] = []
        gh_bouts = self.timeline["gh_bouts"]

        xs = np.linspace(14.0, self.width - 14.0, 78) + self.rng.uniform(-9.0, 9.0, 78)
        for i, x0 in enumerate(xs):
            layer = i % 3  # 0=tall back, 1=medium mid, 2=shorter crisp foreground tufts
            y0 = float(self.ground_y + 18.0 + layer * 10.0 + self.rng.uniform(-5.0, 5.0))
            if layer == 0:
                height = float(self.rng.uniform(140.0, 235.0))
                thickness = 1.4
            elif layer == 1:
                height = float(self.rng.uniform(110.0, 185.0))
                thickness = 1.9
            else:
                # Shorter foreground grass tufts frame creatures' feet without blocking faces!
                height = float(self.rng.uniform(58.0, 112.0))
                thickness = 2.3

            lean = float(self.rng.uniform(-76.0, 76.0))
            # Give each blade an unmistakable curved Bezier arch rather than a straight line
            if abs(lean) < 22.0:
                lean = math.copysign(float(self.rng.uniform(26.0, 64.0)), lean if lean != 0 else 1.0)
            arch_bias = float(self.rng.uniform(-22.0, 22.0))
            t_start = float(self.rng.uniform(0.45, 6.2))
            grow_dur = float(self.rng.uniform(3.4, 5.4))
            wind_freq = float(self.rng.uniform(1.15, 1.85))
            wind_phase = float(x0 * 0.0085 + self.rng.uniform(0.0, 1.2))
            wind_amp = float(self.rng.uniform(16.0, 28.0))

            eat_times: List[float] = []
            for _, _, bout_x, crunches in gh_bouts:
                if (bout_x - 74.0) <= x0 <= (bout_x + 14.0):
                    eat_times.extend(crunches)
            eat_times = sorted(set(eat_times))

            blades.append(
                GrassBlade(
                    x0=float(x0),
                    y0=y0,
                    height=height,
                    lean=lean,
                    arch_bias=arch_bias,
                    thickness=thickness,
                    depth_layer=layer,
                    t_start=t_start,
                    grow_dur=grow_dur,
                    wind_freq=wind_freq,
                    wind_phase=wind_phase,
                    wind_amp=wind_amp,
                    eat_times=eat_times,
                )
            )
        return blades

    def _init_lsystem_plants(self) -> List[LSystemPlant]:
        """Generate 11 branching botanical meadow stalks via L-system rewriting."""
        rules_a = {"X": "F[+X][-X]FX", "F": "FF"}
        rules_b = {"X": "F[-X][+X]F", "F": "FF"}
        lstr_a = generate_lsystem("X", rules_a, iterations=2)
        lstr_b = generate_lsystem("X", rules_b, iterations=2)

        gh_bouts = self.timeline["gh_bouts"]
        x_positions = [82, 192, 330, 468, 578, 670, 756, 898, 1018, 1130, 1215]
        plants: List[LSystemPlant] = []
        for idx, xp in enumerate(x_positions):
            x0 = float(xp + self.rng.uniform(-14.0, 14.0))
            y0 = float(self.ground_y + 20.0 + (idx % 2) * 8.0)
            eat_times: List[float] = []
            for _, _, bout_x, crunches in gh_bouts:
                if (bout_x - 66.0) <= x0 <= (bout_x + 14.0):
                    eat_times.extend(crunches)
            plants.append(
                LSystemPlant(
                    x0=x0,
                    y0=y0,
                    lstring=lstr_a if idx % 2 == 0 else lstr_b,
                    step_len=float(self.rng.uniform(28.0, 38.0)),
                    angle_deg=float(self.rng.uniform(21.0, 29.0)),
                    t_start=float(self.rng.uniform(1.6, 5.8)),
                    grow_dur=float(self.rng.uniform(4.2, 5.8)),
                    wind_phase=float(x0 * 0.008),
                    depth_layer=idx % 2,
                    eat_times=sorted(set(eat_times)),
                )
            )
        return plants

    def _init_feathers(self) -> List[Feather]:
        """Initialize procedural hawk feathers for the strike burst and final 3s drift."""
        feather_lstr = generate_lsystem("A", {"A": "S[+B][-B]A"}, iterations=5)
        talon_t = float(self.timeline["hawk_talon_hit"])

        # Spread gracefully across the full width so the final 3s empty screen is poetic
        configs = [
            # (spawn_t, x0, y0, fall_speed, drift_amp, omega, phase, wind_bias, len, tilt)
            (talon_t + 0.05, 445.0, 515.0, 18.0, 62.0, 1.32, 0.2, -16.0, 56.0, 0.44),
            (talon_t + 0.08, 690.0, 485.0, 20.0, 70.0, 1.20, 1.8, 18.0, 62.0, 0.48),
            (talon_t + 0.12, 575.0, 445.0, 22.0, 56.0, 1.48, 3.1, -6.0, 48.0, 0.40),
            (talon_t + 0.55, 760.0, 325.0, 37.0, 78.0, 1.18, 0.9, 12.0, 64.0, 0.46),
            (talon_t + 1.05, 510.0, 220.0, 44.0, 82.0, 1.28, 2.4, -18.0, 60.0, 0.50),
            (67.20, 880.0, 95.0, 55.0, 90.0, 1.36, 4.1, -14.0, 68.0, 0.46),
            (67.95, 360.0, 55.0, 58.0, 84.0, 1.25, 1.1, 14.0, 58.0, 0.44),
            (68.55, 635.0, 30.0, 60.0, 92.0, 1.14, 5.2, -8.0, 66.0, 0.52),
        ]
        return [
            Feather(
                spawn_time=c[0],
                x0=c[1],
                y0=c[2],
                fall_speed=c[3],
                drift_amp=c[4],
                drift_omega=c[5],
                drift_phase=c[6],
                wind_bias=c[7],
                length=c[8],
                tilt_amp=c[9],
                lstring=feather_lstr,
            )
            for c in configs
        ]

    # -------------------------------------------------------------------------
    # Scene-Wide Environmental Modulators
    # -------------------------------------------------------------------------

    def _meadow_opacity(self, t: float) -> float:
        fade_s = float(self.timeline["meadow_fade_start"])
        fade_e = float(self.timeline["meadow_fade_end"])
        if t < fade_s:
            return 1.0
        if t >= fade_e:
            return 0.0
        return 1.0 - smoothstep((t - fade_s) / (fade_e - fade_s))

    def _wind_gust(self, t: float) -> float:
        base = 0.80 + 0.45 * math.sin(0.55 * t) + 0.25 * math.sin(1.23 * t + 0.9)
        sil_s = float(self.timeline["snake_silence_start"])
        lunge_s = float(self.timeline["snake_lunge_start"])
        if sil_s <= t <= lunge_s:
            return 0.14
        return base

    def _eat_remaining_ratio(self, eat_times: List[float], t: float) -> float:
        if not eat_times or t < eat_times[0]:
            return 1.0
        bites = 0.0
        for et in eat_times:
            if t >= et:
                bites += clamp((t - et) / 0.14)
        return clamp(1.0 - bites * 0.34, 0.0, 1.0)

    def _entity_grass_deflection(self, x0: float, t: float) -> float:
        defl = 0.0
        for land_t, land_x in ((31.60, 95.0), (33.55, 300.0), (44.20, 385.0)):
            dt = t - land_t
            if 0.0 <= dt < 1.2:
                dist = x0 - land_x
                if abs(dist) < 155.0:
                    push = math.exp(-dt * 3.8) * (1.0 - abs(dist) / 155.0)
                    defl += math.copysign(44.0 * push, dist if abs(dist) > 2 else 1.0)

        hit_t = float(self.timeline["hawk_talon_hit"])
        if hit_t - 0.65 <= t <= hit_t + 2.8:
            pulse = math.exp(-((t - (hit_t + 0.15)) / 0.85) ** 2)
            dist = x0 - 595.0
            if abs(dist) < 440.0:
                radial = (1.0 - abs(dist) / 440.0) ** 0.7
                defl += math.copysign(105.0 * pulse * radial, dist if abs(dist) > 2 else 1.0)
        return defl

    # -------------------------------------------------------------------------
    # Drawing Scene 1 & Persistent Meadow (Batched Bezier Grass + L-Systems)
    # -------------------------------------------------------------------------

    def _draw_meadow(
        self, renderer: SketchRenderer, t: float, layer_filter: Optional[int] = None
    ) -> None:
        opacity = self._meadow_opacity(t)
        if opacity <= 0.005 or t < 0.40:
            return

        wind_g = self._wind_gust(t)

        # Sketchy ground baseline strokes (batched for speed)
        if layer_filter in (None, 0):
            g_lines = []
            for gx0, gx1, gy, ts in self.ground_strokes:
                if t < ts:
                    continue
                prog = ease_out_cubic((t - ts) / 1.8)
                cur_x1 = gx0 + (gx1 - gx0) * prog
                g_lines.append(np.array([[gx0, gy], [0.5 * (gx0 + cur_x1), gy], [cur_x1, gy]]))
            if g_lines:
                renderer.draw_polylines_batch(
                    g_lines,
                    color=PALETTE["charcoal_light"],
                    thickness=1.2,
                    jitter=1.1,
                    opacity=0.52 * opacity,
                )

        # L-System botanical meadow stalks (batched segments per depth layer)
        lsys_segments: List[np.ndarray] = []
        lsys_seeds: List[Tuple[float, float, float]] = []
        for plant in self.lsystem_plants:
            if layer_filter is not None and plant.depth_layer != layer_filter:
                continue
            if t < plant.t_start:
                continue
            grow = ease_out_cubic((t - plant.t_start) / plant.grow_dur)
            remain = self._eat_remaining_ratio(plant.eat_times, t)
            eff_scale = grow * remain
            if eff_scale <= 0.04:
                continue
            self._collect_lsystem_plant(plant, t, eff_scale, wind_g, lsys_segments, lsys_seeds)

        if lsys_segments:
            lcol = (
                PALETTE["charcoal_light"]
                if layer_filter == 0
                else PALETTE["charcoal_med"]
            )
            renderer.draw_polylines_batch(
                lsys_segments,
                color=lcol,
                thickness=1.5,
                jitter=1.05,
                opacity=0.80 * opacity,
            )
            # Small botanical seed pods at branch tips
            seed_polys = [
                ellipse_points((sx, sy), (sr, sr), steps=10)[:-1]
                for sx, sy, sr in lsys_seeds
            ]
            if seed_polys:
                renderer.draw_polylines_batch(
                    seed_polys,
                    color=lcol,
                    thickness=1.1,
                    jitter=0.75,
                    closed=True,
                    opacity=0.72 * opacity,
                )

        # Curved Bezier grass splines (batched per depth layer for fast 3-pass boil)
        layer_colors = (
            PALETTE["charcoal_light"],
            PALETTE["charcoal_med"],
            PALETTE["charcoal_dark"],
        )
        layer_thick = (1.5, 2.0, 2.4)
        layer_op = (0.72, 0.86, 0.96)

        layers_to_draw = [layer_filter] if layer_filter is not None else [0, 1, 2]
        for lyr in layers_to_draw:
            blade_curves: List[np.ndarray] = []
            for blade in self.grass_blades:
                if blade.depth_layer != lyr or t < blade.t_start:
                    continue
                grow = ease_out_cubic((t - blade.t_start) / blade.grow_dur)
                remain = self._eat_remaining_ratio(blade.eat_times, t)
                h_frac = grow * remain
                if h_frac <= 0.03:
                    continue

                cur_h = blade.height * h_frac
                sway = (
                    blade.wind_amp
                    * wind_g
                    * h_frac
                    * (
                        math.sin(blade.wind_freq * t + blade.wind_phase)
                        + 0.34 * math.sin(2.15 * blade.wind_freq * t + 0.6 * blade.wind_phase)
                    )
                )
                defl = self._entity_grass_deflection(blade.x0, t) * h_frac
                # True sweeping curved Bezier arch (stem rises tall, tip curves smoothly)
                p0 = (blade.x0, blade.y0)
                p1 = (
                    blade.x0 + (blade.lean * 0.20 + blade.arch_bias + sway * 0.34 + defl * 0.38) * h_frac,
                    blade.y0 - cur_h * 0.68,
                )
                p2 = (
                    blade.x0 + blade.lean * h_frac + sway + defl,
                    blade.y0 - cur_h + abs(blade.lean * 0.14 + defl * 0.22) * h_frac,
                )
                blade_curves.append(quad_bezier(p0, p1, p2, steps=11))

            if blade_curves:
                renderer.draw_polylines_batch(
                    blade_curves,
                    color=layer_colors[lyr],
                    thickness=layer_thick[lyr],
                    jitter=1.18,
                    closed=False,
                    opacity=layer_op[lyr] * opacity,
                )

    def _collect_lsystem_plant(
        self,
        plant: LSystemPlant,
        t: float,
        scale: float,
        wind_g: float,
        out_segments: List[np.ndarray],
        out_seeds: List[Tuple[float, float, float]],
    ) -> None:
        """Turtle-interpret an L-system botanical stalk into continuous branch polylines."""
        x, y = plant.x0, plant.y0
        angle = -math.pi * 0.5
        stack: List[Tuple[float, float, float, int]] = []
        depth = 0
        defl = self._entity_grass_deflection(plant.x0, t) * 0.0065
        wind_rad = (
            0.105 * wind_g * math.sin(1.25 * t + plant.wind_phase) + defl
        ) * scale
        d_ang = math.radians(plant.angle_deg)

        cur_branch: List[Tuple[float, float]] = [(x, y)]
        for ch in plant.lstring:
            if ch == "F":
                seg_len = plant.step_len * scale * (0.74 ** depth)
                eff_angle = angle + wind_rad * (depth + 1)
                nx = x + seg_len * math.cos(eff_angle)
                ny = y + seg_len * math.sin(eff_angle)
                cur_branch.append((nx, ny))
                x, y = nx, ny
            elif ch == "+":
                angle += d_ang
            elif ch == "-":
                angle -= d_ang
            elif ch == "[":
                if len(cur_branch) >= 2:
                    out_segments.append(np.asarray(cur_branch, dtype=np.float64))
                stack.append((x, y, angle, depth))
                depth += 1
                cur_branch = [(x, y)]
            elif ch == "]":
                if len(cur_branch) >= 2:
                    out_segments.append(np.asarray(cur_branch, dtype=np.float64))
                if scale > 0.45 and depth >= 1:
                    out_seeds.append((x, y, 2.8 * scale))
                if stack:
                    x, y, angle, depth = stack.pop()
                cur_branch = [(x, y)]
        if len(cur_branch) >= 2:
            out_segments.append(np.asarray(cur_branch, dtype=np.float64))

    def _draw_wind_wisps(self, renderer: SketchRenderer, t: float) -> None:
        """Draw subtle drifting Bezier breeze ribbons during Scene 1."""
        if not (3.5 <= t <= 15.8):
            return
        scene_env = smoothstep((t - 3.5) / 2.2) * (1.0 - smoothstep((t - 13.8) / 2.0))
        for i in range(3):
            period = 4.6 + i * 0.9
            u = ((t + i * 1.7) % period) / period
            x_base = -120.0 + u * (self.width + 260.0)
            y_base = 175.0 + i * 115.0 + 24.0 * math.sin(u * math.pi * 2.5 + i)
            alpha = math.sin(u * math.pi) * 0.46 * scene_env
            p0 = (x_base - 105.0, y_base + 10.0)
            p1 = (x_base - 25.0, y_base - 24.0)
            p2 = (x_base + 48.0, y_base + 20.0)
            p3 = (x_base + 115.0, y_base - 16.0)
            renderer.draw_cubic_bezier(
                p0,
                p1,
                p2,
                p3,
                color=PALETTE["charcoal_faint"],
                thickness=1.5,
                jitter=1.4,
                steps=16,
                opacity=alpha,
            )

    # -------------------------------------------------------------------------
    # Scene 2 Entity: Geometric Grasshopper (15.0s - 40.375s)
    # -------------------------------------------------------------------------

    def _grasshopper_pose(self, t: float, frame_idx: int) -> Optional[Dict[str, float]]:
        """Compute exact grasshopper parabolic hop, head-down eating dip & chomp."""
        hop_s = float(self.timeline["gh_hop_start"])
        hop_e = float(self.timeline["gh_hop_end"])
        gone_f = int(self.timeline["tongue_gone_frame"])
        hit_f = int(self.timeline["tongue_hit_frame"])

        if t < hop_s or frame_idx >= gone_f:
            return None

        land_x, land_y = 805.0, 584.0
        x, y = land_x, land_y
        # OpenCV rotation angle: grasshopper faces LEFT (head at -x).
        # Because +y is DOWN, ang > 0 lifts the left-facing head UP; ang < 0 dips head DOWN!
        ang = 0.06
        leg_extend = 0.0
        wing_flutter = 0.0
        jaw_open = 0.0
        chirp_vib = 0.0
        nervous = 0.0

        # 1. Mathematically perfect parabolic hop entry (15.20s - 16.85s)
        if hop_s <= t <= hop_e:
            u = (t - hop_s) / (hop_e - hop_s)
            x0, y0 = 1375.0, 550.0
            x1, y1 = land_x, land_y
            arc_h = 345.0
            x = x0 + (x1 - x0) * u
            y = y0 + (y1 - y0) * u - 4.0 * arc_h * u * (1.0 - u)
            # Tangent along parabola: nose tilts UP while rising, DOWN while descending
            dx_du = x1 - x0
            dy_du = (y1 - y0) - 4.0 * arc_h * (1.0 - 2.0 * u)
            ang = clamp(math.atan2(-dy_du, -dx_du) * 0.42, -0.45, 0.48)
            leg_extend = 1.0 - 0.50 * math.sin(u * math.pi)
            wing_flutter = 0.5 + 0.5 * math.sin(t * 78.0)

        # 2. Landing shock absorption squash (16.85s - 17.30s)
        elif hop_e < t <= 17.30:
            u = (t - hop_e) / 0.45
            squash = math.sin(u * math.pi) * math.exp(-u * 2.5)
            y = land_y + 16.0 * squash
            ang = -0.14 * squash

        # 3. Feeding bouts & stepping left (17.30s - 40.3s)
        else:
            gh_bouts = self.timeline["gh_bouts"]
            cur_x = land_x
            head_dip = 0.0
            for b_s, b_e, target_x, crunches in gh_bouts:
                if t >= b_e:
                    cur_x = target_x
                elif t >= b_s:
                    step_u = ease_in_out_quad((t - b_s) / 0.32)
                    cur_x = cur_x + (target_x - cur_x) * step_u
                    u_b = (t - b_s) / (b_e - b_s)
                    if u_b < 0.25:
                        head_dip = ease_in_out_cubic(u_b / 0.25)
                    elif u_b < 0.82:
                        head_dip = 1.0
                    else:
                        head_dip = 1.0 - ease_in_out_cubic((u_b - 0.82) / 0.18)
                    for ct in crunches:
                        if 0.0 <= t - ct <= 0.18:
                            jaw_open = max(jaw_open, math.sin(math.pi * (t - ct) / 0.18))
                    break

            x = cur_x
            # Negative angle dips the left-facing head DOWNWARD into the grass splines!
            ang = 0.06 - 0.46 * head_dip - 0.06 * jaw_open
            y = land_y + 20.0 * head_dip + 2.5 * math.sin(t * 2.6)

            for ct in self.timeline["gh_chirps"]:
                if 0.0 <= t - ct <= 0.38:
                    chirp_vib = math.sin(2.0 * math.pi * 24.0 * (t - ct))

            if t >= 33.5:
                nervous = clamp((t - 33.5) / 2.5)
                ang += 0.06 * nervous

        # Yanked back into frog's mouth on Frame 968
        if frame_idx == hit_f + 1:
            mouth_pos = self._frog_mouth_pos(t)
            x = mouth_pos[0] + (x - mouth_pos[0]) * 0.48
            y = mouth_pos[1] + (y - mouth_pos[1]) * 0.48
            ang = -0.65

        return {
            "x": x,
            "y": y,
            "ang": ang,
            "leg_extend": leg_extend,
            "wing_flutter": wing_flutter,
            "jaw_open": jaw_open,
            "chirp_vib": chirp_vib,
            "nervous": nervous,
        }

    def _draw_grasshopper(self, renderer: SketchRenderer, t: float, frame_idx: int) -> None:
        pose = self._grasshopper_pose(t, frame_idx)
        if pose is None:
            return

        cx, cy = pose["x"], pose["y"]
        ang = pose["ang"]
        leg_ext = pose["leg_extend"]
        wing_fl = pose["wing_flutter"]
        jaw = pose["jaw_open"]
        chirp = pose["chirp_vib"]
        nerv = pose["nervous"]
        sc = 1.22  # Crisp heroic scale

        ca, sa = math.cos(ang), math.sin(ang)

        def to_world(lx: float, ly: float) -> Tuple[float, float]:
            sx, sy = lx * sc, ly * sc
            return (cx + sx * ca - sy * sa, cy + sx * sa + sy * ca)

        # Parabolic hop dashed trajectory trail during entry
        hop_s = float(self.timeline["gh_hop_start"])
        hop_e = float(self.timeline["gh_hop_end"])
        if hop_s <= t <= hop_e + 0.35:
            trail_alpha = 0.45 * (1.0 - smoothstep((t - hop_e) / 0.35))
            cur_u = clamp((t - hop_s) / (hop_e - hop_s))
            if cur_u > 0.05:
                us = np.linspace(max(0.0, cur_u - 0.38), cur_u - 0.03, 14)
                tx = 1375.0 + (805.0 - 1375.0) * us
                ty = 550.0 + (584.0 - 550.0) * us - 4.0 * 345.0 * us * (1.0 - us)
                dashes = [
                    np.array([[tx[k], ty[k]], [tx[k + 1], ty[k + 1]]])
                    for k in range(0, len(us) - 1, 2)
                ]
                renderer.draw_polylines_batch(
                    dashes,
                    color=PALETTE["charcoal_faint"],
                    thickness=1.5,
                    jitter=1.2,
                    opacity=trail_alpha,
                )

        # 1. Far-side legs (behind body)
        far_hip = to_world(10.0, -4.0)
        far_knee = to_world(44.0 + 18.0 * leg_ext, -56.0 + 26.0 * leg_ext + chirp * 7.0)
        far_ankle = to_world(76.0 + 36.0 * leg_ext, 34.0 + 16.0 * leg_ext)
        far_legs = [np.array([far_hip, far_knee, far_ankle])]
        for lx_hip, lx_foot in ((-26.0, -42.0), (-6.0, -14.0)):
            far_legs.append(
                np.array(
                    [
                        to_world(lx_hip, 12.0),
                        to_world(lx_hip - 9.0, 26.0),
                        to_world(lx_foot, 46.0 - 14.0 * leg_ext),
                    ]
                )
            )
        renderer.draw_polylines_batch(
            far_legs, color=PALETTE["charcoal_light"], thickness=1.9, jitter=1.05
        )

        # 2. Segmented Elongated Abdomen
        abd_c = to_world(34.0, 4.0)
        renderer.draw_ellipse(
            abd_c,
            axes=(44.0 * sc, 14.0 * sc),
            angle_deg=math.degrees(ang) + 8.0,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["paper_fill"],
            thickness=2.4,
            jitter=1.25,
        )
        rib_lines = [
            np.array(
                [
                    to_world(seg_x, -6.0 + (seg_x - 14.0) * 0.11),
                    to_world(seg_x - 2.0, 14.0 - (seg_x - 14.0) * 0.04),
                ]
            )
            for seg_x in (14.0, 26.0, 38.0, 50.0, 61.0)
        ]
        renderer.draw_polylines_batch(
            rib_lines, color=PALETTE["charcoal_med"], thickness=1.3, jitter=0.9
        )

        # 3. Folded / Fluttering Bezier Wing
        wing_lift = -26.0 * wing_fl
        w0 = to_world(-8.0, -10.0)
        w1 = to_world(30.0, -24.0 + wing_lift)
        w2 = to_world(82.0, -4.0 + wing_lift * 1.15)
        w3 = to_world(44.0, 5.0 + wing_lift * 0.45)
        wing_poly = np.vstack((quad_bezier(w0, w1, w2, 12), quad_bezier(w2, w3, w0, 10)[1:]))
        renderer.draw_polygon(
            wing_poly,
            color=PALETTE["charcoal_med"],
            fill_color=PALETTE["wash_light"],
            fill_alpha=0.55,
            thickness=1.9,
            jitter=1.15,
        )

        # 4. Geometric Thorax Shield
        thor_c = to_world(-14.0, -2.0)
        renderer.draw_ellipse(
            thor_c,
            axes=(22.0 * sc, 17.0 * sc),
            angle_deg=math.degrees(ang) - 10.0,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["paper_fill"],
            thickness=2.6,
            jitter=1.25,
        )
        renderer.draw_ellipse_hatching(
            thor_c,
            axes=(22.0 * sc, 17.0 * sc),
            angle_deg=math.degrees(ang) - 10.0,
            spacing=8.0,
            region_y_range=(0.0, 0.85),
            opacity=0.45,
        )

        # 5. Angled Geometric Head + Eye + Articulated Mandibles + Long Antennae
        head_c = to_world(-38.0, -5.0)
        renderer.draw_ellipse(
            head_c,
            axes=(13.5 * sc, 19.0 * sc),
            angle_deg=math.degrees(ang) + 16.0,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["paper_fill"],
            thickness=2.5,
            jitter=1.2,
        )
        eye_c = to_world(-41.0, -12.0)
        renderer.draw_ellipse(
            eye_c,
            axes=(5.2 * sc, 7.4 * sc),
            angle_deg=math.degrees(ang) + 12.0,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["charcoal_dark"],
            thickness=1.4,
            jitter=0.8,
        )
        renderer.draw_circle(
            to_world(-42.5, -14.0),
            radius=1.8 * sc,
            color=PALETTE["paper_bg"],
            fill_color=PALETTE["paper_bg"],
            thickness=1.0,
            jitter=0.4,
        )

        # Mandibles opening/closing during grass crunches
        m_root = to_world(-42.0, 12.0)
        m_left = to_world(-49.0 - 6.0 * jaw, 22.0 + 3.0 * jaw)
        m_right = to_world(-35.0 + 5.0 * jaw, 23.0 + 2.0 * jaw)
        renderer.draw_polylines_batch(
            [np.array([m_root, m_left]), np.array([m_root, m_right])],
            color=PALETTE["charcoal_dark"],
            thickness=2.2,
            jitter=0.9,
        )

        # Clipped sketch grass bits flicking off during chomps
        if jaw > 0.30:
            bits = []
            for k in range(3):
                ang_p = -2.2 + k * 0.78 + 0.2 * math.sin(t * 25.0 + k)
                r_p = 14.0 + 22.0 * jaw + k * 6.0
                p_bit = (m_root[0] + r_p * math.cos(ang_p), m_root[1] + r_p * math.sin(ang_p))
                p_bit2 = (p_bit[0] + 9.0 * math.cos(ang_p + 0.6), p_bit[1] + 9.0 * math.sin(ang_p + 0.6))
                bits.append(np.array([p_bit, p_bit2]))
            renderer.draw_polylines_batch(bits, PALETTE["charcoal_med"], thickness=1.7, jitter=1.0)

        # Sweeping Bezier antennae
        ant_sway = 8.0 * math.sin(t * 3.8) + 4.5 * nerv * math.sin(t * 34.0)
        ant_curves = [
            quad_bezier(
                to_world(-46.0, dy_a),
                to_world(-68.0, dy_a - 24.0 + ant_sway * (0.8 + 0.3 * idx)),
                to_world(-96.0, dy_a - 14.0 + ant_sway * (1.2 + 0.3 * idx)),
                10,
            )
            for idx, dy_a in enumerate((-18.0, -14.0))
        ]
        renderer.draw_polylines_batch(ant_curves, PALETTE["charcoal_dark"], thickness=1.5, jitter=1.0)

        # 6. Near-side Fore & Mid Angled Geometric Legs
        near_legs = []
        for lx_hip, lx_knee, lx_foot in ((-22.0, -34.0, -48.0), (-4.0, -8.0, -16.0)):
            near_legs.append(
                np.array(
                    [
                        to_world(lx_hip, 11.0),
                        to_world(lx_knee, 27.0),
                        to_world(lx_foot, 49.0 - 16.0 * leg_ext),
                        to_world(lx_foot - 7.0, 50.0 - 16.0 * leg_ext),
                    ]
                )
            )
        renderer.draw_polylines_batch(near_legs, PALETTE["charcoal_dark"], thickness=2.2, jitter=1.05)

        # 7. Iconic Near-Side Hind Jumping Leg
        hip = to_world(6.0, 2.0)
        knee = to_world(38.0 + 22.0 * leg_ext, -54.0 + 28.0 * leg_ext + chirp * 9.0)
        ankle = to_world(66.0 + 44.0 * leg_ext, 46.0 + 14.0 * leg_ext)
        foot = to_world(76.0 + 48.0 * leg_ext, 49.0 + 14.0 * leg_ext)

        fem_vec = np.array(knee) - np.array(hip)
        fem_len = max(1.0, float(np.linalg.norm(fem_vec)))
        fem_n = np.array([-fem_vec[1], fem_vec[0]]) / fem_len
        fem_poly = np.array(
            [
                np.array(hip) + fem_n * (6.5 * sc),
                np.array(hip) + fem_vec * 0.42 + fem_n * (9.5 * sc),
                np.array(knee) + fem_n * (2.5 * sc),
                np.array(knee) - fem_n * (2.5 * sc),
                np.array(hip) + fem_vec * 0.42 - fem_n * (5.0 * sc),
                np.array(hip) - fem_n * (5.5 * sc),
            ]
        )
        renderer.draw_polygon(
            fem_poly,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["paper_fill"],
            thickness=2.4,
            jitter=1.2,
        )
        renderer.draw_polyline(
            np.array([knee, ankle, foot]),
            color=PALETTE["charcoal_dark"],
            thickness=2.3,
            jitter=1.1,
        )

        # Visual chirp vibration arcs radiating above wing
        if abs(chirp) > 0.1:
            rc = to_world(26.0, -34.0)
            arcs = [
                ellipse_points(rc, (rad, rad), start_deg=225, end_deg=315, steps=10)
                for rad in (20.0, 32.0)
            ]
            renderer.draw_polylines_batch(
                arcs,
                color=PALETTE["charcoal_med"],
                thickness=1.4,
                jitter=1.1,
                opacity=0.62 * abs(chirp),
            )

    # -------------------------------------------------------------------------
    # Scene 3 Entity: Large Wide Oval Frog (30.0s - 55.60s)
    # -------------------------------------------------------------------------

    def _frog_pose(self, t: float) -> Optional[Dict[str, float]]:
        """Compute frog hops, squash-and-stretch, vocal-sac inflation & posture."""
        if t < 30.15 or t >= float(self.timeline["snake_strike_hit"]) + 0.02:
            return None

        h1_s, h1_e = self.timeline["frog_hop1"]
        h2_s, h2_e = self.timeline["frog_hop2"]
        h3_s, h3_e = self.timeline["frog_hop3"]

        x, y = 300.0, 608.0
        squash = 0.0
        pitch = 0.0
        leg_ext = 0.0
        throat = 0.0
        blink = 0.0
        fear = 0.0

        def eval_hop(u: float, x0: float, x1: float, arc_h: float) -> Tuple[float, float, float, float]:
            hx = x0 + (x1 - x0) * u
            hy = 610.0 - 4.0 * arc_h * u * (1.0 - u)
            hpitch = -0.35 * (1.0 - 2.0 * u)
            hext = math.sin(u * math.pi)
            return hx, hy, hpitch, hext

        if t < h1_s:
            return None
        elif h1_s <= t <= h1_e:
            u = (t - h1_s) / (h1_e - h1_s)
            x, y, pitch, leg_ext = eval_hop(u, -185.0, 95.0, 185.0)
            squash = -0.14 * leg_ext
        elif h1_e < t < h2_s:
            x, y = 95.0, 610.0
            u = (t - h1_e) / (h2_s - h1_e)
            squash = 0.24 * math.sin(min(1.0, u * 2.2) * math.pi) * math.exp(-u * 2.5)
        elif h2_s <= t <= h2_e:
            u = (t - h2_s) / (h2_e - h2_s)
            x, y, pitch, leg_ext = eval_hop(u, 95.0, 300.0, 195.0)
            squash = -0.15 * leg_ext
        elif h2_e < t < h3_s:
            x, y = 300.0, 608.0
            if t < h2_e + 0.65:
                u = (t - h2_e) / 0.65
                squash = 0.26 * math.sin(u * math.pi) * math.exp(-u * 2.4)
            y += 2.2 * math.sin(t * 2.8)
            if 39.20 <= t <= 40.25:
                cu = smoothstep((t - 39.20) / 0.90)
                squash += 0.14 * cu
                y += 8.0 * cu
                pitch = 0.08 * cu
            if 40.55 <= t <= 41.35:
                gu = (t - 40.55) / 0.80
                throat = max(throat, 0.52 * math.sin(gu * math.pi))
            if 41.35 <= t <= 41.60:
                blink = math.sin(math.pi * (t - 41.35) / 0.25)
        elif h3_s <= t <= h3_e:
            u = (t - h3_s) / (h3_e - h3_s)
            x, y, pitch, leg_ext = eval_hop(u, 300.0, 385.0, 95.0)
            squash = -0.10 * leg_ext
        else:
            x, y = 385.0, 608.0
            if t < h3_e + 0.45:
                u = (t - h3_e) / 0.45
                squash = 0.16 * math.sin(u * math.pi) * math.exp(-u * 2.5)
            y += 2.0 * math.sin(t * 2.8)
            if t >= 50.5:
                fear = smoothstep((t - 50.5) / 2.2)
                squash += 0.12 * fear
                y += 7.0 * fear

        # Vocal sac inflation synced to FM ribbits
        for rib_t, _, inten in self.timeline["frog_ribbits"]:
            if 0.0 <= t - rib_t <= 0.56:
                ru = (t - rib_t) / 0.56
                pulse = (
                    0.78 * math.exp(-((ru - 0.16) / 0.11) ** 2)
                    + 1.12 * math.exp(-((ru - 0.62) / 0.20) ** 2)
                )
                throat = max(throat, pulse * inten)

        return {
            "x": x,
            "y": y,
            "squash": squash,
            "pitch": pitch,
            "leg_ext": leg_ext,
            "throat": throat,
            "blink": blink,
            "fear": fear,
        }

    def _frog_mouth_pos(self, t: float) -> Tuple[float, float]:
        pose = self._frog_pose(t)
        if pose is None:
            return (380.0, 595.0)
        return (pose["x"] + 82.0, pose["y"] - 6.0)

    def _draw_frog(self, renderer: SketchRenderer, t: float, frame_idx: int) -> None:
        pose = self._frog_pose(t)
        if pose is None:
            return

        cx, cy = pose["x"], pose["y"]
        sq = pose["squash"]
        pitch = pose["pitch"]
        leg_ext = pose["leg_ext"]
        throat = pose["throat"]
        blink = pose["blink"]
        fear = pose["fear"]

        # Exact 4-frame tongue strike (< 5 frames!)
        f_start = int(self.timeline["tongue_frame_start"])
        f_hit = int(self.timeline["tongue_hit_frame"])

        tongue_ext = 0.0
        mouth_open = 0.0
        if frame_idx == f_start:
            tongue_ext, mouth_open = 0.62, 1.0
        elif frame_idx == f_hit:
            tongue_ext, mouth_open = 1.00, 1.0
        elif frame_idx == f_hit + 1:
            tongue_ext, mouth_open = 0.48, 0.85

        rx = 94.0 * (1.0 + 0.22 * sq)
        ry = 60.0 * (1.0 - 0.20 * sq)
        ca, sa = math.cos(pitch), math.sin(pitch)

        def to_w(lx: float, ly: float) -> Tuple[float, float]:
            return (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)

        if leg_ext < 0.7:
            sh_scale = 1.0 - 0.55 * leg_ext
            renderer.draw_ellipse(
                (cx, self.ground_y + 12.0),
                axes=(rx * 0.92 * sh_scale, 13.0 * sh_scale),
                color=PALETTE["charcoal_faint"],
                fill_color=PALETTE["wash_light"],
                fill_alpha=0.55,
                thickness=1.0,
                jitter=1.4,
            )

        # 1. Far hind leg
        far_hip = to_w(-52.0, 22.0)
        far_knee = to_w(-88.0 - 52.0 * leg_ext, 10.0 + 30.0 * leg_ext)
        far_foot = to_w(-46.0 - 95.0 * leg_ext, 58.0 + 26.0 * leg_ext)
        renderer.draw_polyline(
            np.array([far_hip, far_knee, far_foot]),
            color=PALETTE["charcoal_light"],
            thickness=3.2,
            jitter=1.2,
        )

        # 2. Inflatable Vocal Sac bulging prominently during FM ribbits
        if throat > 0.04:
            sac_c = to_w(64.0 + 14.0 * throat, 26.0 + 12.0 * throat)
            renderer.draw_ellipse(
                sac_c,
                axes=(30.0 + 28.0 * throat, 20.0 + 20.0 * throat),
                angle_deg=math.degrees(pitch) + 14.0,
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["paper_bg"],
                thickness=2.2,
                jitter=1.3,
            )
            # Subtle croak sound-wave ripples
            if throat > 0.35:
                rib_arcs = [
                    ellipse_points(
                        to_w(95.0, 14.0), (r, r), start_deg=-42, end_deg=42, steps=10
                    )
                    for r in (26.0, 42.0)
                ]
                renderer.draw_polylines_batch(
                    rib_arcs,
                    color=PALETTE["charcoal_med"],
                    thickness=1.5,
                    jitter=1.1,
                    opacity=0.55 * min(1.0, throat),
                )

        # 3. Large Wide Oval Body
        renderer.draw_ellipse(
            (cx, cy),
            axes=(rx, ry),
            angle_deg=math.degrees(pitch) - 10.0,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["paper_fill"],
            thickness=3.0,
            jitter=1.45,
        )
        renderer.draw_ellipse_hatching(
            (cx, cy),
            axes=(rx, ry),
            angle_deg=math.degrees(pitch) - 10.0,
            hatch_angle_deg=-32.0,
            spacing=11.0,
            region_y_range=(0.18, 0.90),
            color=PALETTE["charcoal_light"],
            thickness=1.2,
            opacity=0.52,
        )
        for sx, sy, srx, sry in (
            (-32.0, -28.0, 17.0, 10.0),
            (6.0, -34.0, 14.0, 8.5),
            (-56.0, -8.0, 12.0, 8.0),
            (-10.0, -10.0, 11.0, 7.0),
        ):
            renderer.draw_ellipse(
                to_w(sx, sy),
                axes=(srx, sry),
                angle_deg=math.degrees(pitch) - 12.0,
                color=PALETTE["charcoal_med"],
                fill_color=PALETTE["wash_light"],
                fill_alpha=0.58,
                thickness=1.4,
                jitter=1.0,
            )

        # 4. Prominent Perched Frog Eyes
        for ex, ey, er in ((30.0, -54.0, 17.5), (56.0, -49.0, 19.0)):
            ec = to_w(ex, ey)
            renderer.draw_circle(
                ec,
                radius=er,
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["paper_fill"],
                thickness=2.4,
                jitter=1.1,
            )
            if blink > 0.45:
                renderer.draw_line(
                    (ec[0] - er * 0.72, ec[1]),
                    (ec[0] + er * 0.72, ec[1]),
                    PALETTE["charcoal_dark"],
                    thickness=2.2,
                )
            else:
                p_dx = 4.5 + 1.5 * fear
                p_r = 7.2 + 2.2 * fear
                renderer.draw_ellipse(
                    (ec[0] + p_dx, ec[1] + 1.0),
                    axes=(p_r * 1.1, p_r * 0.8),
                    color=PALETTE["charcoal_dark"],
                    fill_color=PALETTE["charcoal_dark"],
                    thickness=1.2,
                    jitter=0.7,
                )
                renderer.draw_circle(
                    (ec[0] + p_dx - 2.4, ec[1] - 2.2),
                    radius=2.3,
                    color=PALETTE["paper_bg"],
                    fill_color=PALETTE["paper_bg"],
                    thickness=1.0,
                    jitter=0.4,
                )

        # 5. Wide Frog Mouth
        m_back = to_w(24.0, 8.0)
        m_front = to_w(rx * 0.95, -5.0)
        if mouth_open > 0.1:
            jaw_drop = 28.0 * mouth_open
            m_upper = to_w(rx * 0.94, -14.0 * mouth_open)
            m_lower = to_w(rx * 0.86, 12.0 + jaw_drop)
            mouth_poly = np.array([m_back, m_upper, to_w(rx * 0.78, 4.0), m_lower])
            renderer.draw_polygon(
                mouth_poly,
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["wash_med"],
                thickness=2.4,
                jitter=1.1,
            )
        else:
            renderer.draw_bezier(
                m_back, to_w(62.0, 10.0), m_front, PALETTE["charcoal_dark"], thickness=2.6, jitter=1.1
            )

        # 6. Muscular Z-Folded Near Hind Leg
        thigh_c = to_w(-52.0 - 24.0 * leg_ext, 24.0 + 12.0 * leg_ext)
        renderer.draw_ellipse(
            thigh_c,
            axes=(44.0, 23.0),
            angle_deg=math.degrees(pitch) - 24.0 - 25.0 * leg_ext,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["paper_fill"],
            thickness=2.6,
            jitter=1.3,
        )
        knee = to_w(-82.0 - 46.0 * leg_ext, 18.0 + 26.0 * leg_ext)
        ankle = to_w(-36.0 - 92.0 * leg_ext, 55.0 + 28.0 * leg_ext)
        renderer.draw_line(knee, ankle, PALETTE["charcoal_dark"], thickness=5.2, jitter=1.3)
        toes = [
            np.array([ankle, (ankle[0] + td_x - 24.0 * leg_ext, ankle[1] + td_y + 8.0 * leg_ext)])
            for td_x, td_y in ((24.0, 4.0), (18.0, 10.0), (6.0, 11.0))
        ]
        renderer.draw_polylines_batch(toes, PALETTE["charcoal_dark"], thickness=2.0, jitter=1.0)

        # 7. Sturdy Front Foreleg & Splayed Toes
        hand = to_w(64.0, 58.0 - 10.0 * leg_ext)
        renderer.draw_polyline(
            np.array([to_w(38.0, 18.0), to_w(54.0, 40.0), hand]),
            color=PALETTE["charcoal_dark"],
            thickness=3.6,
            jitter=1.2,
        )
        f_toes = [
            np.array([hand, (hand[0] + dx_t, hand[1] + dy_t)])
            for dx_t, dy_t in ((14.0, 2.0), (10.0, 8.0), (1.0, 9.0))
        ]
        renderer.draw_polylines_batch(f_toes, PALETTE["charcoal_dark"], thickness=2.0, jitter=0.9)

        # 8. THE LIGHTNING TONGUE STRIKE (< 5 frames: straight line + circle on end!)
        if tongue_ext > 0.01:
            mouth_origin = to_w(rx * 0.80, 0.0)
            target = (678.0, 578.0)
            tip = (
                mouth_origin[0] + (target[0] - mouth_origin[0]) * tongue_ext,
                mouth_origin[1] + (target[1] - mouth_origin[1]) * tongue_ext,
            )
            renderer.draw_line(
                mouth_origin,
                tip,
                color=PALETTE["charcoal_dark"],
                thickness=5.2,
                jitter=1.25,
                subdivisions=6,
            )
            renderer.draw_circle(
                tip,
                radius=15.5,
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["wash_med"],
                thickness=2.6,
                jitter=1.2,
            )
            if frame_idx == f_hit:
                bursts = [
                    np.array(
                        [
                            (tip[0] + 21.0 * math.cos(a), tip[1] + 21.0 * math.sin(a)),
                            (tip[0] + 46.0 * math.cos(a), tip[1] + 46.0 * math.sin(a)),
                        ]
                    )
                    for a in (k * math.pi / 4.0 + 0.2 for k in range(8))
                ]
                renderer.draw_polylines_batch(
                    bursts, PALETTE["charcoal_dark"], thickness=2.3, jitter=1.2
                )

    # -------------------------------------------------------------------------
    # Scene 4 Entity: Thick Undulating Bezier Snake (45.0s - 68.8s)
    # -------------------------------------------------------------------------

    def _snake_slither_spine(
        self, head_x: float, head_y: float, t: float, total_len: float = 535.0, n_pts: int = 44
    ) -> np.ndarray:
        """Build undulating Bezier-smoothed sine-wave spine extending rightward from head."""
        ctrl_s = np.linspace(0.0, 1.0, 7)
        ctrl_x = head_x + ctrl_s * total_len
        wave_amp = 34.0 * np.sin(np.pi * np.clip(ctrl_s * 1.15, 0.0, 1.0))
        ctrl_y = head_y + wave_amp * np.sin(ctrl_s * 9.5 - t * 6.8)
        ctrl_y[0] -= 20.0
        ctrl_y[1] -= 10.0

        seg1 = cubic_bezier(
            (ctrl_x[0], ctrl_y[0]),
            (ctrl_x[1], ctrl_y[1]),
            (ctrl_x[2], ctrl_y[2]),
            (ctrl_x[3], ctrl_y[3]),
            steps=n_pts // 2,
        )
        seg2 = cubic_bezier(
            (ctrl_x[3], ctrl_y[3]),
            (ctrl_x[4], ctrl_y[4]),
            (ctrl_x[5], ctrl_y[5]),
            (ctrl_x[6], ctrl_y[6]),
            steps=n_pts - n_pts // 2 + 1,
        )
        return np.vstack((seg1, seg2[1:]))

    def _snake_coiled_spine(self, t: float, n_pts: int = 44) -> np.ndarray:
        """Build a tight menacing S-coiled Bezier strike posture on the right-center."""
        tremor = 2.0 * math.sin(t * 28.0) if t >= 54.0 else 0.0
        p0 = (700.0 + tremor, 502.0)   # High poised viper head aimed at frog (385, 608)
        p1 = (825.0, 460.0)            # Proud arched S-neck
        p2 = (730.0, 628.0)            # Front coil base
        p3 = (855.0, 634.0)            # Mid-coil anchor
        p4 = (1005.0, 634.0)           # Broad rear coil loop
        p5 = (955.0, 548.0)            # Stacked upper coil
        p6 = (812.0, 594.0)            # Tucked tail tip

        seg1 = cubic_bezier(p0, p1, p2, p3, steps=n_pts // 2)
        seg2 = cubic_bezier(p3, p4, p5, p6, steps=n_pts - n_pts // 2 + 1)
        return np.vstack((seg1, seg2[1:]))

    def _snake_lunged_spine(self, lunge_u: float, t: float, n_pts: int = 44) -> np.ndarray:
        """Build the explosively uncoiled lunging spine engulfing the frog."""
        settle = smoothstep((t - 56.0) / 3.0) if t > 56.0 else 0.0
        hx = 700.0 + (355.0 - 700.0) * lunge_u + 58.0 * settle
        hy = 502.0 + (602.0 - 502.0) * lunge_u + 14.0 * settle

        relax_wave = 26.0 * settle * math.sin(t * 1.4)
        p0 = (hx, hy)
        p1 = (hx + 150.0, hy - 22.0 + 30.0 * settle)
        p2 = (hx + 280.0, hy + 28.0 - 18.0 * settle + relax_wave * 0.4)
        p3 = (hx + 360.0, 624.0)
        p4 = (hx + 440.0, 608.0 - 18.0 * settle)
        p5 = (hx + 505.0, 638.0)
        p6 = (hx + 560.0, 624.0)

        seg1 = cubic_bezier(p0, p1, p2, p3, steps=n_pts // 2)
        seg2 = cubic_bezier(p3, p4, p5, p6, steps=n_pts - n_pts // 2 + 1)
        return np.vstack((seg1, seg2[1:]))

    def _snake_snatched_spine(
        self, grab_pos: Tuple[float, float], t: float, n_pts: int = 44
    ) -> np.ndarray:
        """Build a writhing, dangling Bezier spine suspended from the hawk's talons."""
        gx, gy = grab_pos
        s = np.linspace(0.0, 1.0, n_pts)
        grab_s = 0.42
        ds = s - grab_s
        drop_y = np.abs(ds) * 335.0
        thrash_x = np.sign(ds) * (np.abs(ds) * 145.0) + 56.0 * np.abs(ds) * np.sin(
            s * 11.5 - t * 18.5
        )
        thrash_y = 28.0 * np.abs(ds) * np.cos(s * 9.0 - t * 15.5)
        return np.column_stack((gx + thrash_x, gy + drop_y + thrash_y))

    def _snake_state(self, t: float) -> Optional[Dict[str, object]]:
        enter_s = float(self.timeline["snake_enter_start"])
        coil_s = float(self.timeline["snake_coil_start"])
        coil_e = float(self.timeline["snake_coil_end"])
        lunge_s = float(self.timeline["snake_lunge_start"])
        lunge_e = float(self.timeline["snake_lunge_end"])
        talon_t = float(self.timeline["hawk_talon_hit"])
        exit_t = float(self.timeline["hawk_exit_time"])

        if t < enter_s or t > exit_t + 0.15:
            return None

        jaw_open = 0.0
        tongue_flick = 0.0
        bulge_s = -1.0
        bulge_amp = 0.0

        if t < coil_s:
            u = ease_out_quad((t - enter_s) / (coil_s - enter_s))
            head_x = 1475.0 + (780.0 - 1475.0) * u
            spine = self._snake_slither_spine(head_x, 616.0, t)
            if (t % 2.1) < 0.35:
                tongue_flick = math.sin(math.pi * ((t % 2.1) / 0.35))
        elif t < coil_e:
            u = ease_in_out_cubic((t - coil_s) / (coil_e - coil_s))
            s_slith = self._snake_slither_spine(780.0, 616.0, t)
            s_coil = self._snake_coiled_spine(t)
            spine = (1.0 - u) * s_slith + u * s_coil
            if 52.0 <= t <= 53.3:
                tongue_flick = max(0.0, math.sin((t - 52.0) * 8.0))
        elif t < lunge_s:
            spine = self._snake_coiled_spine(t)
        elif t < lunge_e:
            u = (t - lunge_s) / (lunge_e - lunge_s)
            # Whip-fast lunge reaching the frog cleanly by t=55.56s!
            lunge_u = ease_out_cubic(clamp(u * 1.15))
            s_coil = self._snake_coiled_spine(t)
            s_lunge = self._snake_lunged_spine(lunge_u, t)
            spine = (1.0 - lunge_u) * s_coil + lunge_u * s_lunge
            jaw_open = math.sin(clamp(u * 1.10) * math.pi)
        elif t < talon_t:
            spine = self._snake_lunged_spine(1.0, t)
            gulp_u = clamp((t - lunge_e) / 3.8)
            bulge_s = 0.08 + 0.30 * ease_out_cubic(gulp_u)
            bulge_amp = 29.0 * smoothstep((t - (lunge_e - 0.05)) / 0.28)
            if t > 63.6:
                alarm = smoothstep((t - 63.6) / 0.85)
                spine[:8, 1] -= np.linspace(42.0, 0.0, 8) * alarm
        else:
            hawk_pose = self._hawk_pose(t)
            if hawk_pose is None:
                return None
            talon_pos = (hawk_pose["talon_x"], hawk_pose["talon_y"])
            u_lift = smoothstep((t - talon_t) / 0.32)
            s_ground = self._snake_lunged_spine(1.0, talon_t)
            s_hang = self._snake_snatched_spine(talon_pos, t)
            spine = (1.0 - u_lift) * s_ground + u_lift * s_hang
            bulge_s = 0.38
            bulge_amp = 26.0

        return {
            "spine": spine,
            "jaw_open": jaw_open,
            "tongue_flick": tongue_flick,
            "bulge_s": bulge_s,
            "bulge_amp": bulge_amp,
        }

    def _draw_snake(self, renderer: SketchRenderer, t: float) -> None:
        state = self._snake_state(t)
        if state is None:
            return

        spine: np.ndarray = state["spine"]  # type: ignore[assignment]
        jaw_open = float(state["jaw_open"])
        tongue_flick = float(state["tongue_flick"])
        bulge_s = float(state["bulge_s"])
        bulge_amp = float(state["bulge_amp"])

        n = len(spine)
        s = np.linspace(0.0, 1.0, n)

        diffs = np.gradient(spine, axis=0)
        norms = np.maximum(np.linalg.norm(diffs, axis=1, keepdims=True), 1e-6)
        tangents = diffs / norms
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

        # Thick muscular Bezier body + traveling frog-shaped oval bulge!
        base_w = 26.0 * np.sin(np.pi * np.clip(s * 0.88 + 0.08, 0.02, 0.99)) ** 0.55
        base_w[:4] *= np.array([0.62, 0.78, 0.90, 0.96])
        base_w *= np.clip((1.0 - s) * 4.5, 0.06, 1.0)
        if bulge_s > 0.0:
            base_w += bulge_amp * np.exp(-((s - bulge_s) / 0.095) ** 2)

        left_edge = spine + normals * base_w[:, None]
        right_edge = spine - normals * base_w[:, None]
        body_poly = np.vstack((left_edge, right_edge[::-1]))

        renderer.fill_polygon(body_poly, color=PALETTE["paper_fill"], alpha=0.97, jitter=0.8)

        # Batched dorsal diamond saddle scales along the Bezier spine
        diamonds = []
        for k in range(3, n - 3, 3):
            w_k = float(base_w[k]) * 0.62
            c_pt = spine[k]
            t_vec = tangents[k]
            n_vec = normals[k]
            d_poly = np.array(
                [
                    c_pt + t_vec * 10.5,
                    c_pt + n_vec * w_k,
                    c_pt - t_vec * 10.5,
                    c_pt - n_vec * w_k,
                ]
            )
            renderer.fill_polygon(d_poly, color=PALETTE["wash_light"], alpha=0.52, jitter=0.4)
            diamonds.append(d_poly)
        if diamonds:
            renderer.draw_polylines_batch(
                diamonds, color=PALETTE["charcoal_med"], thickness=1.35, jitter=0.9, closed=True
            )

        # Bold 3-pass boiled charcoal upper & lower contours
        renderer.draw_polylines_batch(
            [left_edge, right_edge],
            color=PALETTE["charcoal_dark"],
            thickness=3.0,
            jitter=1.4,
        )
        renderer.draw_polyline(
            spine[2:-2], PALETTE["charcoal_light"], thickness=1.1, jitter=1.1, opacity=0.45
        )

        # Menacing Angular Viper Head at spine[0]
        h_pos = spine[0]
        fwd = -tangents[0]
        up = np.array([-fwd[1], fwd[0]])
        if up[1] > 0:
            up = -up

        if jaw_open > 0.08:
            # Dramatic hinged wide-open viper strike jaws with dark gullet & curved fangs!
            jaw_ang = 0.62 * jaw_open
            c_j, s_j = math.cos(jaw_ang), math.sin(jaw_ang)
            u_fwd = fwd * c_j + up * s_j
            l_fwd = fwd * c_j - up * s_j

            u_tip = h_pos + u_fwd * 52.0
            l_tip = h_pos + l_fwd * 48.0
            gullet = np.array([h_pos - fwd * 4.0, u_tip, h_pos + fwd * 14.0, l_tip])
            renderer.fill_polygon(gullet, color=PALETTE["wash_med"], alpha=0.78, jitter=0.6)

            upper_jaw = np.array(
                [
                    h_pos - fwd * 12.0 + up * 16.0,
                    h_pos + u_fwd * 24.0 + up * 17.0,
                    u_tip + up * 4.0,
                    u_tip - up * 5.0,
                    h_pos - fwd * 4.0 + up * 2.0,
                ]
            )
            lower_jaw = np.array(
                [
                    h_pos - fwd * 10.0 - up * 13.0,
                    h_pos + l_fwd * 22.0 - up * 13.0,
                    l_tip - up * 4.0,
                    l_tip + up * 4.0,
                    h_pos - fwd * 4.0 - up * 2.0,
                ]
            )
            renderer.draw_polygon(
                upper_jaw, PALETTE["charcoal_dark"], PALETTE["paper_fill"], thickness=2.8
            )
            renderer.draw_polygon(
                lower_jaw, PALETTE["charcoal_dark"], PALETTE["paper_fill"], thickness=2.8
            )
            # Paired curved venom fangs
            f1 = u_tip - u_fwd * 10.0 - up * 3.0
            f2 = f1 - up * 15.0 - fwd * 3.0
            renderer.draw_line(f1, f2, PALETTE["charcoal_dark"], thickness=2.4, jitter=0.7)
            # Crisp eye on upper skull
            eye_pt = h_pos + u_fwd * 18.0 + up * 9.0
            renderer.draw_circle(
                (float(eye_pt[0]), float(eye_pt[1])),
                radius=4.5,
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["charcoal_dark"],
                thickness=1.2,
                jitter=0.5,
            )
            # Dynamic lunge motion speed-lines
            slines = [
                np.array(
                    [
                        h_pos - fwd * (50.0 + k * 24.0) + up * ((k - 1.5) * 28.0),
                        h_pos - fwd * (135.0 + k * 24.0) + up * ((k - 1.5) * 28.0),
                    ]
                )
                for k in range(4)
            ]
            renderer.draw_polylines_batch(
                slines, PALETTE["charcoal_med"], thickness=1.7, jitter=1.3, opacity=0.68
            )
        else:
            head_poly = np.array(
                [
                    h_pos - fwd * 14.0 + up * 16.0,
                    h_pos + fwd * 14.0 + up * 20.0,
                    h_pos + fwd * 45.0 + up * 3.0,
                    h_pos + fwd * 42.0 - up * 7.0,
                    h_pos + fwd * 10.0 - up * 16.0,
                    h_pos - fwd * 14.0 - up * 13.0,
                ]
            )
            renderer.draw_polygon(
                head_poly,
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["paper_fill"],
                thickness=2.8,
                jitter=1.25,
            )
            eye_pt = h_pos + fwd * 17.0 + up * 7.5
            renderer.draw_ellipse(
                (float(eye_pt[0]), float(eye_pt[1])),
                axes=(6.8, 4.6),
                angle_deg=math.degrees(math.atan2(fwd[1], fwd[0])),
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["paper_bg"],
                thickness=1.8,
                jitter=0.7,
            )
            renderer.draw_line(
                eye_pt + up * 4.2, eye_pt - up * 4.2, PALETTE["charcoal_dark"], thickness=2.0, jitter=0.4
            )
            if tongue_flick > 0.05:
                t_len = 36.0 * tongue_flick
                snout = h_pos + fwd * 44.0
                mid_t = snout + fwd * t_len + up * (5.0 * math.sin(t * 36.0))
                fork1 = mid_t + fwd * 13.0 + up * 8.5
                fork2 = mid_t + fwd * 13.0 - up * 8.5
                renderer.draw_polylines_batch(
                    [np.array([snout, mid_t, fork1]), np.array([mid_t, fork2])],
                    color=PALETTE["charcoal_dark"],
                    thickness=1.6,
                    jitter=0.85,
                )

    # -------------------------------------------------------------------------
    # Scene 5 Entity: Massive Sharp-V Apex Hawk & Falling Feathers (60.8s - 75s)
    # -------------------------------------------------------------------------

    def _hawk_pose(self, t: float) -> Optional[Dict[str, float]]:
        dive_s = float(self.timeline["hawk_dive_start"])
        hit_t = float(self.timeline["hawk_talon_hit"])
        asc_s = float(self.timeline["hawk_ascent_start"])
        exit_t = float(self.timeline["hawk_exit_time"])

        if t < dive_s or t > exit_t:
            return None

        strike_x, strike_y = 595.0, 502.0
        flap = 0.0
        dive_tuck = 0.0
        body_angle = 0.0
        talon_reach = 1.0
        talon_clench = 0.0

        # 1. Steep predatory Bezier dive from top edge (62.20s - 65.35s)
        if t < hit_t:
            u = (t - dive_s) / (hit_t - dive_s)
            e_u = 0.65 * ease_in_quad(u) + 0.35 * u
            pos = eval_cubic_bezier_pt(
                (170.0, -320.0),
                (315.0, 45.0),
                (488.0, 340.0),
                (strike_x, strike_y),
                e_u,
            )
            x, y = float(pos[0]), float(pos[1])
            flare = smoothstep((u - 0.68) / 0.30)
            dive_tuck = 1.0 - flare
            flap = -0.20 * dive_tuck + 0.38 * flare
            body_angle = 0.36 * (1.0 - flare) + 0.04 * flare
            talon_reach = 0.40 + 0.60 * flare
            talon_clench = 0.0

        # 2. Talon lock-on micro-compression (65.35s - 65.62s)
        elif t < asc_s:
            u = (t - hit_t) / (asc_s - hit_t)
            x = strike_x + 12.0 * u
            y = strike_y + 10.0 * math.sin(u * math.pi)
            flap = 0.45 - 0.85 * u
            body_angle = -0.08 * u
            talon_reach = 1.0
            talon_clench = 1.0

        # 3. Fast ease-in acceleration arc ripping snake completely off top edge!
        else:
            u = (t - asc_s) / (exit_t - asc_s)
            acc_u = 0.65 * ease_in_cubic(u) + 0.35 * ease_in_expo(u)
            pos = eval_cubic_bezier_pt(
                (strike_x + 12.0, strike_y),
                (735.0, 390.0),
                (895.0, 50.0),
                (1065.0, -580.0),
                acc_u,
            )
            x, y = float(pos[0]), float(pos[1])
            flap = math.sin(2.0 * math.pi * 2.85 * (t - asc_s) - 0.8)
            body_angle = -0.14 - 0.14 * ease_out_quad(u)
            talon_reach = 0.96
            talon_clench = 1.0

        talon_x = x + 6.0 * math.cos(body_angle)
        talon_y = y + 95.0 * talon_reach
        return {
            "x": x,
            "y": y,
            "angle": body_angle,
            "flap": flap,
            "dive_tuck": dive_tuck,
            "talon_reach": talon_reach,
            "talon_clench": talon_clench,
            "talon_x": talon_x,
            "talon_y": talon_y,
        }

    def _draw_hawk_shadow(self, renderer: SketchRenderer, t: float) -> None:
        sh_s = float(self.timeline["hawk_shadow_start"])
        hit_t = float(self.timeline["hawk_talon_hit"])
        asc_s = float(self.timeline["hawk_ascent_start"])
        if not (sh_s <= t <= asc_s + 1.4):
            return

        if t <= hit_t:
            u = ease_in_quad((t - sh_s) / (hit_t - sh_s))
            sx = 240.0 + (595.0 - 240.0) * u
            rx = 80.0 + 215.0 * u
            ry = 14.0 + 26.0 * u
            alpha = 0.12 + 0.42 * u
        else:
            u = clamp((t - hit_t) / 1.6)
            sx = 595.0 + 280.0 * u
            rx = 295.0 * (1.0 - 0.45 * u)
            ry = 40.0 * (1.0 - 0.55 * u)
            alpha = 0.54 * (1.0 - u)

        renderer.draw_ellipse(
            (sx, self.ground_y + 14.0),
            axes=(rx, ry),
            color=PALETTE["charcoal_faint"],
            fill_color=PALETTE["wash_med"],
            fill_alpha=alpha,
            thickness=1.0,
            jitter=2.2,
            opacity=alpha,
        )

    def _build_raptor_wing(
        self,
        to_w: Callable[[float, float], Tuple[float, float]],
        side: float,
        span: float,
        elbow_y: float,
        tip_y: float,
        flap: float,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Construct a broad, muscular sharp-V raptor wing polygon with sculpted
        serrated primary flight-feather finger tips and covert plumage lines.
        """
        w_shld = to_w(side * 24.0, -18.0)
        w_wrist = to_w(side * (span * 0.44), elbow_y - 26.0)
        w_outer = to_w(side * (span * 0.86), tip_y - 10.0)
        lead_curve = quad_bezier(w_shld, w_wrist, w_outer, 14)

        # 5 sculpted serrated primary wingtip finger feathers integrated into polygon
        finger_pts: List[Tuple[float, float]] = []
        for idx in range(5):
            f_frac = idx / 4.0
            tip_x = side * (span * (1.03 - 0.09 * f_frac))
            tip_yy = tip_y + idx * 24.0 - 16.0 * flap
            notch_x = side * (span * (0.84 - 0.08 * f_frac))
            notch_yy = tip_y + idx * 23.0 + 18.0 - 10.0 * flap
            finger_pts.append(to_w(tip_x, tip_yy))
            finger_pts.append(to_w(notch_x, notch_yy))

        # Broad muscular secondary trailing edge sweeping back into the body
        sec_start = finger_pts[-1]
        sec_mid = to_w(side * (span * 0.42), elbow_y + 112.0 - 25.0 * flap)
        sec_in = to_w(side * 16.0, 34.0)
        trail_curve = quad_bezier(sec_start, sec_mid, sec_in, 12)

        wing_poly = np.vstack((lead_curve, np.asarray(finger_pts), trail_curve[1:]))

        # Layered secondary/covert plumage lines inside the broad wing
        plumage: List[np.ndarray] = []
        covert_arc = quad_bezier(
            to_w(side * 28.0, 2.0),
            to_w(side * (span * 0.45), elbow_y + 38.0),
            to_w(side * (span * 0.80), tip_y + 42.0),
            12,
        )
        plumage.append(covert_arc)

        for k in range(1, 7):
            u_f = k / 7.5
            p_top = lead_curve[min(len(lead_curve) - 1, int(u_f * len(lead_curve)))]
            p_bot = trail_curve[max(0, len(trail_curve) - 1 - int(u_f * len(trail_curve)))]
            plumage.append(np.array([p_top, p_bot], dtype=np.float64))

        return wing_poly, plumage

    def _draw_hawk(self, renderer: SketchRenderer, t: float) -> None:
        pose = self._hawk_pose(t)
        if pose is None:
            return

        cx, cy = pose["x"], pose["y"]
        ang = pose["angle"]
        flap = pose["flap"]
        tuck = pose["dive_tuck"]
        reach = pose["talon_reach"]
        clench = pose["talon_clench"]
        ca, sa = math.cos(ang), math.sin(ang)

        def to_w(lx: float, ly: float) -> Tuple[float, float]:
            return (cx + lx * ca - ly * sa, cy + lx * sa + ly * ca)

        # High-speed dive wind streaks
        if tuck > 0.25:
            streaks = []
            for side in (-1.0, 1.0):
                wt = to_w(side * 245.0, -175.0)
                streaks.append(np.array([wt, (wt[0] - 60.0, wt[1] - 125.0)]))
            renderer.draw_polylines_batch(
                streaks, PALETTE["charcoal_faint"], thickness=1.7, jitter=1.4, opacity=0.62 * tuck
            )

        # 1. Broad Fanned Raptor Tail Feathers
        tail_root = to_w(-58.0, -6.0)
        for k in range(-3, 4):
            fa = math.radians(-162.0 + k * 9.5) + ang
            t_tip = (tail_root[0] + 114.0 * math.cos(fa), tail_root[1] + 114.0 * math.sin(fa))
            t_side = (
                tail_root[0] + 102.0 * math.cos(fa + 0.10),
                tail_root[1] + 102.0 * math.sin(fa + 0.10),
            )
            renderer.draw_polygon(
                np.array([tail_root, t_tip, t_side]),
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["paper_fill"],
                thickness=1.8,
                jitter=1.15,
            )

        # 2. MASSIVE SHARP-V WINGSPAN (broad sculpted raptor wings!)
        span = 385.0 - 110.0 * tuck
        tip_y = -188.0 - 70.0 * tuck - 108.0 * flap
        elbow_y = -76.0 - 30.0 * tuck - 48.0 * flap

        for side in (-1.0, 1.0):
            wing_poly, plumage = self._build_raptor_wing(to_w, side, span, elbow_y, tip_y, flap)
            renderer.draw_polygon(
                wing_poly,
                color=PALETTE["charcoal_dark"],
                fill_color=PALETTE["paper_fill"],
                thickness=3.2,
                jitter=1.55,
            )
            renderer.draw_polylines_batch(
                plumage,
                color=PALETTE["charcoal_light"],
                thickness=1.35,
                jitter=1.1,
                opacity=0.68,
            )

        # 3. Powerful Aerodynamic Raptor Torso
        renderer.draw_ellipse(
            (cx, cy),
            axes=(68.0, 39.0),
            angle_deg=math.degrees(ang) + 8.0,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["paper_fill"],
            thickness=3.0,
            jitter=1.4,
        )
        renderer.draw_ellipse_hatching(
            (cx, cy),
            axes=(68.0, 39.0),
            angle_deg=math.degrees(ang) + 8.0,
            spacing=9.5,
            region_y_range=(0.05, 0.85),
            opacity=0.55,
        )

        # 4. Fierce Raptor Head + Piercing Eye + Sharp Curved Hooked Beak!
        head_c = to_w(66.0, -8.0)
        renderer.draw_ellipse(
            head_c,
            axes=(29.0, 22.0),
            angle_deg=math.degrees(ang) + 12.0,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["paper_fill"],
            thickness=2.8,
            jitter=1.25,
        )
        bk_top_root = to_w(86.0, -15.0)
        bk_ctrl = to_w(124.0, -13.0)
        bk_tip = to_w(119.0, 17.0)  # Razor hooked raptor beak!
        bk_bot_root = to_w(84.0, 5.0)
        beak_poly = np.vstack(
            (quad_bezier(bk_top_root, bk_ctrl, bk_tip, 12), quad_bezier(bk_tip, to_w(104.0, 5.0), bk_bot_root, 10)[1:])
        )
        renderer.draw_polygon(
            beak_poly,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["wash_med"],
            thickness=2.7,
            jitter=1.1,
        )
        eye_c = to_w(74.0, -12.0)
        renderer.draw_circle(
            eye_c,
            radius=5.4,
            color=PALETTE["charcoal_dark"],
            fill_color=PALETTE["charcoal_dark"],
            thickness=1.2,
            jitter=0.6,
        )
        renderer.draw_circle(
            (eye_c[0] - 1.5, eye_c[1] - 1.5),
            radius=1.7,
            color=PALETTE["paper_bg"],
            fill_color=PALETTE["paper_bg"],
            thickness=1.0,
            jitter=0.3,
        )
        renderer.draw_line(
            to_w(61.0, -22.0), to_w(86.0, -16.0), PALETTE["charcoal_dark"], thickness=2.8, jitter=0.9
        )

        # 5. Outstretched Feathered Legs & Razor-Curved Talons (drawn on top of snake grip!)
        claws: List[np.ndarray] = []
        for leg_dx in (-15.0, 15.0):
            hip = to_w(leg_dx, 26.0)
            knee = to_w(leg_dx + 6.0, 56.0 * reach)
            foot = to_w(leg_dx - 1.0, 94.0 * reach)
            renderer.draw_line(hip, knee, PALETTE["charcoal_dark"], thickness=5.4, jitter=1.2)
            renderer.draw_line(knee, foot, PALETTE["charcoal_dark"], thickness=3.8, jitter=1.1)

            splay = 1.0 - 0.55 * clench
            for claw_side in (-1.0, 0.0, 1.0):
                c1 = (foot[0] + claw_side * 19.0 * splay + 6.0, foot[1] + 15.0)
                c2 = (
                    foot[0] + claw_side * 12.0 * splay - claw_side * 11.0 * clench,
                    foot[1] + 30.0 - 7.0 * clench,
                )
                claws.append(quad_bezier(foot, c1, c2, steps=8))
        renderer.draw_polylines_batch(claws, PALETTE["charcoal_dark"], thickness=2.8, jitter=0.9)

    def _draw_feathers(self, renderer: SketchRenderer, t: float) -> None:
        """
        Draw procedural drifting hawk feathers (curved Bezier rachis + L-system
        lateral barbs) falling gently with horizontal sine drift — starring solo
        on an empty sketchbook screen during the final 3 seconds (72.0s - 75.0s).
        """
        for f in self.feathers:
            if t < f.spawn_time:
                continue
            age = t - f.spawn_time
            sin_val = math.sin(f.drift_omega * age + f.drift_phase)
            cos_val = math.cos(f.drift_omega * age + f.drift_phase)

            fx = f.x0 + f.drift_amp * sin_val + f.wind_bias * age
            fy = f.y0 + f.fall_speed * age + 8.0 * math.sin(2.0 * f.drift_omega * age)
            if fy > self.height + 75.0:
                continue

            fade_bottom = clamp((self.height + 35.0 - fy) / 75.0)
            fade_end = clamp((self.duration - t) / 0.85) if t > self.duration - 0.85 else 1.0
            op = 0.90 * fade_bottom * fade_end
            if op <= 0.01:
                continue

            tilt = f.tilt_amp * cos_val
            ca, sa = math.cos(tilt), math.sin(tilt)
            half_l = f.length * 0.5

            q0 = (fx - half_l * ca, fy - half_l * sa)
            q1 = (fx - 6.0 * sa, fy + 6.0 * ca)
            q2 = (fx + half_l * ca, fy + half_l * sa)
            quill = quad_bezier(q0, q1, q2, 10)
            renderer.draw_polyline(
                quill, PALETTE["charcoal_dark"], thickness=1.9, jitter=1.1, opacity=op
            )

            barbs: List[np.ndarray] = []
            barb_idx = 0
            for ch in f.lstring:
                if ch == "S":
                    barb_idx += 1
                elif ch == "B" and 1 <= barb_idx <= 7:
                    u_q = clamp(barb_idx / 8.5, 0.14, 0.90)
                    pt_i = min(len(quill) - 1, int(u_q * (len(quill) - 1)))
                    root = quill[pt_i]
                    vane_w = (f.length * 0.24) * math.sin(u_q * math.pi)
                    for side in (-1.0, 1.0):
                        b_ang = tilt + side * 0.84
                        b_tip = (
                            float(root[0]) + vane_w * math.cos(b_ang),
                            float(root[1]) + vane_w * math.sin(b_ang),
                        )
                        barbs.append(np.array([root, b_tip], dtype=np.float64))
            if barbs:
                renderer.draw_polylines_batch(
                    barbs,
                    color=PALETTE["charcoal_med"],
                    thickness=1.2,
                    jitter=0.95,
                    opacity=op * 0.84,
                )

    # -------------------------------------------------------------------------
    # Master Frame Dispatcher (0s -> 75s)
    # -------------------------------------------------------------------------

    def render_frame(self, renderer: SketchRenderer, frame_idx: int) -> None:
        """Render one complete frame of the 75-second food chain choreography."""
        t = frame_idx / float(self.fps)
        renderer.begin_frame(frame_idx)

        self._draw_wind_wisps(renderer, t)

        # Back & mid grass meadow behind creatures
        self._draw_meadow(renderer, t, layer_filter=0)
        self._draw_meadow(renderer, t, layer_filter=1)

        # Sweeping hawk ground shadow (Scene 5)
        self._draw_hawk_shadow(renderer, t)

        # Food-chain entities in causal occlusion order
        self._draw_grasshopper(renderer, t, frame_idx)
        self._draw_frog(renderer, t, frame_idx)
        self._draw_snake(renderer, t)
        self._draw_hawk(renderer, t)

        # Short crisp foreground grass tufts framing the bottom edge
        self._draw_meadow(renderer, t, layer_filter=2)

        # Procedural drifting feathers (solo on empty screen in final 3 seconds)
        self._draw_feathers(renderer, t)


# =============================================================================
# 4. ASSEMBLY & OUTPUT: MovieAssembler
# =============================================================================


class MovieAssembler:
    """
    Combines the OpenCV/NumPy SketchRenderer frames (24 FPS, 1800 frames) and
    the SciPy/NumPy AudioFoley track into food_chain_75s.mp4 using MoviePy.
    """

    def __init__(
        self,
        output_path: str = "food_chain_75s.mp4",
        width: int = WIDTH,
        height: int = HEIGHT,
        fps: int = FPS,
        duration: float = DURATION,
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.total_frames = int(round(duration * fps))

        self.renderer = SketchRenderer(width=width, height=height)
        self.kinematics = EntityKinematics(
            width=width, height=height, fps=fps, duration=duration
        )
        self.audio_engine = AudioFoley(
            timeline=self.kinematics.timeline,
            duration=duration,
            sample_rate=SAMPLE_RATE,
        )

    def _make_frame(self, t: float) -> np.ndarray:
        """MoviePy frame callback mapping continuous time t to exact frame index."""
        frame_idx = min(max(0, int(round(t * self.fps))), self.total_frames - 1)
        self.kinematics.render_frame(self.renderer, frame_idx)
        return self.renderer.get_rgb_frame()

    def build(self) -> str:
        """Synthesize audio, stream all 1800 procedural frames, and write final MP4."""
        tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix="_food_chain.wav")
        os.close(tmp_wav_fd)

        audio_clip = None
        video_clip = None
        try:
            print(f"[1/3] Synthesizing {self.duration:.1f}s procedural foley soundtrack...")
            self.audio_engine.export_wav(tmp_wav_path)

            print(
                f"[2/3] Rendering {self.total_frames} procedural sketch frames "
                f"({self.width}x{self.height} @ {self.fps} FPS)..."
            )
            video_clip = VideoClip(self._make_frame, duration=self.duration)
            audio_clip = AudioFileClip(tmp_wav_path)

            if hasattr(video_clip, "set_audio"):
                final_clip = video_clip.set_audio(audio_clip)
            else:
                final_clip = video_clip.with_audio(audio_clip)

            print(f"[3/3] Encoding MP4 to {self.output_path} ...")
            final_clip.write_videofile(
                self.output_path,
                fps=self.fps,
                codec="libx264",
                audio_codec="aac",
                audio_bitrate="192k",
                preset="medium",
                ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18"],
                threads=os.cpu_count() or 2,
            )
            print(
                f"Done! Rendered {self.output_path} "
                f"({self.duration:.1f}s, {self.total_frames} frames)."
            )
            return self.output_path
        finally:
            if audio_clip is not None:
                audio_clip.close()
            if video_clip is not None:
                video_clip.close()
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)


if __name__ == "__main__":
    assembler = MovieAssembler(output_path="food_chain_75s.mp4")
    assembler.build()
