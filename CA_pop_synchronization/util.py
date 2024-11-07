import cmath
import numpy as np
from typing import Sequence


def wrap_angle(angle_rads: float, domain: str) -> float:
    if domain == '-pi to pi':
        return ((angle_rads + np.pi) % (2 * np.pi)) - np.pi
    elif domain == '0 to 2pi':
        return angle_rads % (2 * np.pi)
    else:
        raise ValueError(f"The value of domain '{domain}' is not acceptable.")


def compute_average_phasor(angles: Sequence[float]) -> float:
    return np.mean(np.exp(1j * np.array(angles)))


def compute_average_angle(angles: Sequence[float]) -> float:
    return cmath.phase(compute_average_phasor(angles))


def compute_order_parameter(angles: Sequence[float]) -> float:
    return np.abs(compute_average_phasor(angles))