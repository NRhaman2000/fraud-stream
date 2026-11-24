"""Great-circle distance, used for 'how far from home / from the last purchase' features."""

from __future__ import annotations

import math
from typing import Final

EARTH_RADIUS_KM: Final = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two (lat, lon) points in degrees. Symmetric, >= 0."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    # Clamp guards against a > 1 from floating-point error at antipodal points.
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, a))))
