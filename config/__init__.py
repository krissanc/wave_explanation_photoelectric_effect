"""Configuration module for photoelectric wave simulation."""

from .parameters import (
    PhysicsParameters,
    SimulationParameters,
    SweepParameters,
    DEFAULT_PHYSICS,
    DEFAULT_SIMULATION,
    DEFAULT_SWEEP,
    HIGH_SENSITIVITY,
    PHOTOELECTRIC_TUNED,
    STRONG_BINDING,
    WEAK_BINDING,
    LONG_SIMULATION,
    HIGH_PRECISION_SWEEP,
    QUICK_SWEEP,
    get_parameter_set,
    estimate_threshold_frequency
)

__all__ = [
    'PhysicsParameters',
    'SimulationParameters', 
    'SweepParameters',
    'DEFAULT_PHYSICS',
    'DEFAULT_SIMULATION',
    'DEFAULT_SWEEP',
    'HIGH_SENSITIVITY',
    'PHOTOELECTRIC_TUNED',
    'STRONG_BINDING',
    'WEAK_BINDING',
    'LONG_SIMULATION',
    'HIGH_PRECISION_SWEEP',
    'QUICK_SWEEP',
    'get_parameter_set',
    'estimate_threshold_frequency'
]

