"""Local radar geometry: geodetic -> ENU -> (slant range, azimuth, elevation).

Flat-earth ENU relative to the radar site (E = R*cos(lat0)*dlon, N = R*dlat,
U = alt - site_alt) -- the same approximation WHACK01 uses for motion
quantities, accurate to well under the measurement noise at the <=100 km
ranges simulated here.
"""

from typing import Tuple

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def enu_from_geodetic(lat_deg, lon_deg, alt_m,
                      site_lat_deg: float, site_lon_deg: float, site_alt_m: float
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """East/North/Up in metres relative to the radar site."""
    lat0 = np.radians(site_lat_deg)
    east = EARTH_RADIUS_M * np.cos(lat0) * np.radians(np.asarray(lon_deg) - site_lon_deg)
    north = EARTH_RADIUS_M * np.radians(np.asarray(lat_deg) - site_lat_deg)
    up = np.asarray(alt_m) - site_alt_m
    return east, north, up


def polar_from_enu(east, north, up) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(slant_range_m, azimuth_deg in [0,360), elevation_deg).

    Azimuth is compass-style: 0 = north, 90 = east -- matching the heading
    convention used throughout WHACK01.
    """
    ground = np.hypot(east, north)
    slant = np.hypot(ground, up)
    azimuth = np.degrees(np.arctan2(east, north)) % 360.0
    elevation = np.degrees(np.arctan2(up, ground))
    return slant, azimuth, elevation
