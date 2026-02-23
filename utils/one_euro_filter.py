"""
One Euro Filter — adaptive low-pass filter for hand landmark smoothing.

The One Euro Filter (Casiez et al., 2012) solves the jitter-lag tradeoff
that simple EMA cannot fix:

  - When the signal is SLOW/STILL  → heavy smoothing (low cutoff, ~1 Hz)
  - When the signal is FAST/MOVING → light smoothing (high cutoff, responsive)

This is achieved by making the cutoff frequency proportional to the
filtered speed (derivative) of the signal.

Reference: One Euro Filter (https://gery.casiez.net/1euro/)
Recommended values from original paper + MediaPipe hand tracking practice:
  min_cutoff = 1.0  (lower = smoother when still, but more lag on start)
  beta       = 0.007 (higher = faster response when moving)
  d_cutoff   = 1.0  (derivative smoothing, usually keep at 1.0)
"""

import math


class LowPassFilter:
    """Simple first-order low-pass filter (EMA with dynamic alpha)."""

    def __init__(self, cutoff_hz, sample_rate=30.0):
        self._alpha = self._compute_alpha(cutoff_hz, sample_rate)
        self._prev = None
        self._initialized = False

    @staticmethod
    def _compute_alpha(cutoff_hz, sample_rate):
        tau = 1.0 / (2 * math.pi * cutoff_hz)
        te = 1.0 / sample_rate
        return 1.0 / (1.0 + tau / te)

    def set_cutoff(self, cutoff_hz, sample_rate=30.0):
        self._alpha = self._compute_alpha(cutoff_hz, sample_rate)

    def filter(self, value):
        if not self._initialized:
            self._prev = value
            self._initialized = True
            return value
        result = self._alpha * value + (1.0 - self._alpha) * self._prev
        self._prev = result
        return result

    def last_value(self):
        return self._prev


class OneEuroFilter:
    """
    One Euro Filter for a single scalar signal.

    Parameters (tuned for 30fps MediaPipe hand tracking):
      min_cutoff : float = 1.0
          Minimum cutoff frequency in Hz. Controls jitter when still.
          Lower = smoother/less jitter. Typical range: 0.5–2.0

      beta : float = 0.007
          Speed coefficient. Controls lag when moving fast.
          Higher = faster response. Typical range: 0.001–0.1

      d_cutoff : float = 1.0
          Cutoff for derivative filter. Usually leave at 1.0.
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0, sample_rate=30.0):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._sample_rate = sample_rate
        self._x_filter = LowPassFilter(min_cutoff, sample_rate)
        self._dx_filter = LowPassFilter(d_cutoff, sample_rate)
        self._prev_value = None

    def filter(self, value):
        # Estimate derivative
        if self._prev_value is None:
            self._prev_value = value
            dx = 0.0
        else:
            dx = (value - self._prev_value) * self._sample_rate

        # Filter the derivative
        edx = self._dx_filter.filter(dx)

        # Adaptive cutoff: speed-proportional
        cutoff = self._min_cutoff + self._beta * abs(edx)
        self._x_filter.set_cutoff(cutoff, self._sample_rate)

        # Filter the value
        result = self._x_filter.filter(value)
        self._prev_value = value
        return result

    def reset(self):
        """Reset filter state (call when hand disappears)."""
        self._prev_value = None
        self._x_filter._initialized = False
        self._dx_filter._initialized = False


class HandLandmarkFilter:
    """
    Applies One Euro Filter to all 21 hand landmarks (x, y, z).

    Parameters tuned for MediaPipe hand tracking at ~30fps:
      min_cutoff = 1.0  (good jitter suppression when still)
      beta       = 0.007 (fast response when intentionally moving)

    Usage:
        filt = HandLandmarkFilter()
        smoothed = filt.apply(raw_landmarks)  # returns list of smoothed LM
        filt.reset()  # call when hand leaves frame
    """

    # Number of landmarks (MediaPipe uses 21)
    NUM_LANDMARKS = 21

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0, sample_rate=30.0):
        # Create x, y, z filters for each landmark
        self._filters_x = [
            OneEuroFilter(min_cutoff, beta, d_cutoff, sample_rate)
            for _ in range(self.NUM_LANDMARKS)
        ]
        self._filters_y = [
            OneEuroFilter(min_cutoff, beta, d_cutoff, sample_rate)
            for _ in range(self.NUM_LANDMARKS)
        ]
        self._filters_z = [
            OneEuroFilter(min_cutoff, beta, d_cutoff, sample_rate)
            for _ in range(self.NUM_LANDMARKS)
        ]

    def apply(self, raw_landmarks):
        """
        Filter all landmarks and return smoothed versions.

        Args:
            raw_landmarks: List of 21 MediaPipe landmarks with .x .y .z

        Returns:
            List of 21 simple objects with smoothed .x .y .z attributes
        """
        smoothed = []
        for i, lm in enumerate(raw_landmarks):
            s = _SmoothLM(
                x=self._filters_x[i].filter(lm.x),
                y=self._filters_y[i].filter(lm.y),
                z=self._filters_z[i].filter(getattr(lm, 'z', 0.0)),
            )
            smoothed.append(s)
        return smoothed

    def reset(self):
        """Reset all filters (call when hand leaves the frame)."""
        for f in self._filters_x + self._filters_y + self._filters_z:
            f.reset()


class _SmoothLM:
    """Lightweight landmark container holding smoothed x, y, z."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
