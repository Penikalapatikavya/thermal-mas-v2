"""Synthetic thermal camera: segmentation + depth -> pseudo-thermal image.

Pure numpy, no Webots imports, so it can be tested offline.
Pseudo-thermal: physically motivated (Stefan-Boltzmann), not radiometric.
"""
from dataclasses import dataclass, field
import numpy as np

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant


@dataclass(frozen=True)
class Material:
    name: str
    temperature_k: float
    emissivity: float
    texture_k: float = 0.5  # per-pixel temperature variation


# Segmentation colour (R, G, B) -> material. Night-time urban scene.
MATERIALS = {
    (255, 0, 0): Material("person", 306.0, 0.98, 0.8),
    (51, 51, 51): Material("road", 291.0, 0.95, 0.6),
    (79, 148, 229): Material("building", 289.0, 0.85, 0.5),
    (245, 209, 13): Material("crosswalk_paint", 290.0, 0.90, 0.4),
    (255, 255, 255): Material("markings_signs", 290.0, 0.90, 0.4),
    (166, 166, 166): Material("crash_barrier", 287.0, 0.30, 0.3),
    (0, 42, 0): Material("tree", 286.0, 0.97, 0.8),
    (64, 64, 64): Material("traffic_light_pole", 288.0, 0.90, 0.3),
    (240, 138, 28): Material("traffic_cone", 287.0, 0.95, 0.3),
    (110, 28, 25): Material("car", 303.0, 0.90, 2.0),
}
UNTAGGED_GROUND = Material("untagged_ground", 290.0, 0.95, 0.6)
SKY = Material("sky", 260.0, 1.0, 0.0)


@dataclass
class ThermalConfig:
    ambient_k: float = 288.0        # air temperature
    reflected_k: float = 275.0      # what low-emissivity surfaces reflect (night sky + surroundings)
    sky_depth_m: float = 99.0       # untagged pixel at/beyond this depth = sky
    netd_k: float = 0.05            # sensor noise, like a real LWIR camera
    attenuation_per_m: float = 0.0  # fog: 0 = clear air
    display_range_k: tuple = field(default=(280.0, 310.0))


def temperature_map(seg, depth_m, cfg, rng, car_boxes=None):
    h, w = seg.shape[:2]
    temp = np.full((h, w), UNTAGGED_GROUND.temperature_k, np.float32)
    emis = np.full((h, w), UNTAGGED_GROUND.emissivity, np.float32)
    tex = np.full((h, w), UNTAGGED_GROUND.texture_k, np.float32)

    rgb = seg[:, :, :3].astype(np.int32)
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    for (r, g, b), m in MATERIALS.items():
        mask = packed == ((r << 16) | (g << 8) | b)
        temp[mask], emis[mask], tex[mask] = m.temperature_k, m.emissivity, m.texture_k

    far = ~np.isfinite(depth_m) | (depth_m >= cfg.sky_depth_m)
    sky = (packed == 0) & far
    temp[sky], emis[sky], tex[sky] = SKY.temperature_k, SKY.emissivity, SKY.texture_k

    if car_boxes:
        car_key = (110 << 16) | (28 << 8) | 25
        for x0, y0, x1, y1 in car_boxes:
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            m = packed[y0:y1, x0:x1] == car_key
            if not m.any():
                continue
            v = (np.arange(y0, y1) - y0) / max(1, (y1 - y0 - 1))
            v = np.broadcast_to(v[:, None], m.shape)
            t_band = np.where(v < 0.35, cfg.ambient_k - 4.0,
                              np.where(v < 0.75, cfg.ambient_k + 1.0, cfg.ambient_k + 15.0))
            e_band = np.where(v < 0.35, 0.85, np.where(v < 0.75, 0.90, 0.95))
            temp[y0:y1, x0:x1][m] = t_band[m]
            emis[y0:y1, x0:x1][m] = e_band[m]
            tex[y0:y1, x0:x1][m] = 1.0

    temp += rng.normal(0.0, 1.0, (h, w)).astype(np.float32) * tex
    return temp, emis, sky


def synthesize(seg, depth_m, cfg=None, rng=None, car_boxes=None):
    """Return a float32 (H, W) thermal image in [0, 1]. Higher = hotter."""
    cfg = cfg or ThermalConfig()
    rng = rng or np.random.default_rng()
    depth = np.asarray(depth_m, np.float32)

    temp, emis, sky = temperature_map(seg, depth, cfg, rng, car_boxes)
    rad = emis * SIGMA * temp**4 + (1 - emis) * SIGMA * cfg.reflected_k**4

    if cfg.attenuation_per_m > 0:
        d = np.where(np.isfinite(depth), depth, cfg.sky_depth_m)
        tau = np.exp(-cfg.attenuation_per_m * np.clip(d, 0, None))
        air = SIGMA * cfg.ambient_k**4
        rad = np.where(sky, rad, rad * tau + air * (1 - tau))

    t_lo, t_hi = cfg.display_range_k
    r_lo, r_hi = SIGMA * t_lo**4, SIGMA * t_hi**4
    if cfg.netd_k > 0:
        per_kelvin = 4 * SIGMA * ((t_lo + t_hi) / 2) ** 3
        rad = rad + rng.normal(0.0, cfg.netd_k * per_kelvin, rad.shape)

    return np.clip((rad - r_lo) / (r_hi - r_lo), 0, 1).astype(np.float32)


def to_uint8(thermal):
    """White-hot 8-bit image for saving/viewing."""
    return (np.clip(thermal, 0, 1) * 255).astype(np.uint8)
