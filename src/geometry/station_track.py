"""A station track: the path a laser tracker can stand on around a large part
in a factory cell -- pure geometry, no physics.

**The physical picture.** A large part (a wing box, a fuselage section) sits
on a jig. Trackers are tripod- or stand-mounted on the floor in two rows, one
along each long side of the part, and can go round either end of it. This
module describes that as a closed **stadium-shaped path** around the part's
plan-view footprint: two straight rows joined by a semicircular loop at each
end. It is an explicit description of the cell (part footprint, standoffs,
blocked zones), so that a real cell layout can be filled in later; nothing here
is hidden inside an experiment function. Nothing in this module imports from
`planning`.

**Coordinates.** The part's long axis is world x. The footprint runs from
`part_x_min_m` to `part_x_max_m`, is `part_width_m` wide and is centred on
`part_centre_y_m`. "Left" is +y and "right" is -y, looking along +x from above.

  y_left_row  = centre_y + width/2 + standoff_left      (the +y row)
  y_right_row = centre_y - width/2 - standoff_right     (the -y row)

**Arc length.** A position on the track is its arc length `s_m` from the
start, in [0, perimeter_m). **s = 0 is at the -x end of the -y row** (the
join between the -y row and the -x loop); s increases **anticlockwise seen
from above**: along the -y row towards +x, round the +x loop, back along the
+y row towards -x, round the -x loop. Distance travelled along the track
between two stations is simply the difference of their `s_m` (the shorter way
round the closed path is `min(d, perimeter - d)`), which is what a future
relocation-time model needs.

  segment            s range
  "-y row"           [0, L)
  "+x loop"          [L, L + pi R)
  "+y row"           [L + pi R, 2L + pi R)
  "-x loop"          [2L + pi R, 2L + 2 pi R)

with L = x_max - x_min and R the loop radius below.

**Unequal standoffs.** With equal standoffs each end loop is a semicircle
centred on the midpoint of that tip of the footprint, radius R = width/2 +
standoff. If the two standoffs differ, the two rows are at different distances
from the part's centreline and a semicircle about the tip midpoint cannot join
them. The simplest closed path that stays continuous, with no corner a tracker
cannot be carried round, is a semicircle centred on the midpoint *between the
two rows* at that end, of radius R = (distance between the rows) / 2 = width/2
+ (standoff_left + standoff_right)/2. It reduces to the symmetric case when
the standoffs are equal. Its centre is displaced from the tip midpoint by
(standoff_left - standoff_right)/2 in y. The clearance between the loop and the
footprint is never less than the smaller standoff (equal to it at the joins).

**Blocked intervals.** `blocked_intervals_m` is a list of `(s_start, s_end)`
arc-length intervals where no station may stand (an aisle, a jig leg, a safety
zone). An interval with `s_start > s_end` (after reducing both modulo the
perimeter) wraps past s = 0.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

SEGMENT_NAMES = ("-y row", "+x loop", "+y row", "-x loop")


@dataclass(frozen=True)
class StationTrack:
    """A closed stadium-shaped path around a part's footprint (see the module
    docstring for the coordinate and arc-length conventions).

    `standoff_m` is the distance from each long face to its row; give
    `standoff_left_m` and/or `standoff_right_m` to make the two sides differ.
    """

    part_x_min_m: float
    part_x_max_m: float
    part_width_m: float
    standoff_m: float
    part_centre_y_m: float = 0.0
    standoff_left_m: Optional[float] = None
    standoff_right_m: Optional[float] = None
    blocked_intervals_m: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self):
        if self.part_x_max_m <= self.part_x_min_m:
            raise ValueError("part_x_max_m must be greater than part_x_min_m")
        if self.part_width_m <= 0.0:
            raise ValueError("part_width_m must be positive")
        if self.left_standoff_m <= 0.0 or self.right_standoff_m <= 0.0:
            raise ValueError("standoffs must be positive")

    # -- derived dimensions ------------------------------------------------
    @property
    def left_standoff_m(self) -> float:
        return self.standoff_m if self.standoff_left_m is None else self.standoff_left_m

    @property
    def right_standoff_m(self) -> float:
        return self.standoff_m if self.standoff_right_m is None else self.standoff_right_m

    @property
    def row_length_m(self) -> float:
        """L: the straight length of each row (the part's length)."""
        return self.part_x_max_m - self.part_x_min_m

    @property
    def y_left_row_m(self) -> float:
        return self.part_centre_y_m + self.part_width_m / 2.0 + self.left_standoff_m

    @property
    def y_right_row_m(self) -> float:
        return self.part_centre_y_m - self.part_width_m / 2.0 - self.right_standoff_m

    @property
    def loop_radius_m(self) -> float:
        """R: half the distance between the two rows."""
        return (self.y_left_row_m - self.y_right_row_m) / 2.0

    @property
    def loop_centre_y_m(self) -> float:
        """y of both loop centres: midway between the rows."""
        return (self.y_left_row_m + self.y_right_row_m) / 2.0

    @property
    def perimeter_m(self) -> float:
        return 2.0 * self.row_length_m + 2.0 * np.pi * self.loop_radius_m

    @property
    def tip_plus_x_s_m(self) -> float:
        """s at the apex of the +x loop (its furthest point from the part)."""
        return self.row_length_m + 0.5 * np.pi * self.loop_radius_m

    @property
    def tip_minus_x_s_m(self) -> float:
        """s at the apex of the -x loop."""
        return 2.0 * self.row_length_m + 1.5 * np.pi * self.loop_radius_m

    # -- position and segment ---------------------------------------------
    def _breakpoints(self) -> Tuple[float, float, float]:
        length, half_loop = self.row_length_m, np.pi * self.loop_radius_m
        return length, length + half_loop, 2.0 * length + half_loop

    def position_xy_m(self, s_m: float) -> np.ndarray:
        """Plan-view (x, y) at arc length `s_m`. Periodic: `s_m` and
        `s_m + perimeter_m` give the same point."""
        s = float(s_m) % self.perimeter_m
        b1, b2, b3 = self._breakpoints()
        radius, cy = self.loop_radius_m, self.loop_centre_y_m
        if s < b1:  # -y row, heading +x
            return np.array([self.part_x_min_m + s, self.y_right_row_m])
        if s < b2:  # +x loop, from the -y row up round to the +y row
            phi = -np.pi / 2.0 + (s - b1) / radius
            return np.array([self.part_x_max_m + radius * np.cos(phi), cy + radius * np.sin(phi)])
        if s < b3:  # +y row, heading -x
            return np.array([self.part_x_max_m - (s - b2), self.y_left_row_m])
        phi = np.pi / 2.0 + (s - b3) / radius  # -x loop, from the +y row down round to the -y row
        return np.array([self.part_x_min_m + radius * np.cos(phi), cy + radius * np.sin(phi)])

    def segment(self, s_m: float) -> str:
        """Which part of the track `s_m` lies on: one of `SEGMENT_NAMES`."""
        s = float(s_m) % self.perimeter_m
        return SEGMENT_NAMES[int(np.searchsorted(self._breakpoints(), s, side="right"))]

    # -- blocked zones -----------------------------------------------------
    def is_blocked(self, s_m: float) -> bool:
        """True if `s_m` lies in any blocked interval (wrapping intervals
        included; interval ends are inclusive)."""
        s = float(s_m) % self.perimeter_m
        for start, end in self.blocked_intervals_m:
            a, b = start % self.perimeter_m, end % self.perimeter_m
            if (a <= s <= b) if a <= b else (s >= a or s <= b):
                return True
        return False

    def travel_distance_m(self, s_from_m: float, s_to_m: float) -> float:
        """Shortest distance along the track between two stations."""
        d = abs(float(s_to_m) - float(s_from_m)) % self.perimeter_m
        return min(d, self.perimeter_m - d)
