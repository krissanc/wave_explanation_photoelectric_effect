"""
Parameter Configuration Module

Defines default parameters and provides parameter sets for different
experimental scenarios. These parameters control the physics model
and can be tuned to reproduce photoelectric effect characteristics.

Key Parameters:
    omega0 : Natural binding frequency (electron-nucleus coupling)
    damping: Energy loss rate
    alpha  : Internal store decay rate
    beta   : Frequency-dependent coupling strength
    
The photoelectric threshold emerges from the balance between:
    - Store charging: βf²xu (grows with frequency squared)
    - Store decay: αs (constant decay rate)
    
Critical frequency roughly where: βf²⟨xu⟩ ~ α⟨s⟩
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhysicsParameters:
    """
    Physical parameters for the atom model.
    
    These control the dynamics of the bound electron and its
    interaction with the electromagnetic wave.
    """
    
    # Binding parameters
    omega0: float = 1.0       # Natural frequency (binding strength)
    damping: float = 0.01     # Velocity damping coefficient
    
    # Internal store parameters (KEY for frequency threshold)
    alpha: float = 0.5        # Store decay rate
    beta: float = 1.0         # Motion-store coupling strength
    
    # Wave parameters
    amplitude: float = 0.01   # Input wave amplitude
    
    # Escape criteria
    escape_position: float = 5.0    # Position threshold for escape
    escape_energy: Optional[float] = None  # Energy threshold (optional)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'PhysicsParameters':
        """Create from dictionary."""
        return cls(**d)
    
    def describe(self) -> str:
        """Human-readable description of parameters."""
        return f"""
Physics Parameters:
  Binding:
    ω₀ (natural freq)  = {self.omega0}
    d  (damping)       = {self.damping}
  
  Internal Store:
    α  (decay rate)    = {self.alpha}
    β  (coupling)      = {self.beta}
  
  Wave:
    A  (amplitude)     = {self.amplitude}
  
  Escape:
    X_esc (position)   = {self.escape_position}
    E_esc (energy)     = {self.escape_energy}
"""


@dataclass
class SimulationParameters:
    """Parameters controlling the simulation itself (not physics)."""
    
    t_max: float = 200.0      # Maximum simulation time
    dt: float = 0.001         # Integration time step
    store_every: int = 10     # Data storage frequency
    
    # Initial conditions
    x0: float = 0.001         # Initial position perturbation
    v0: float = 0.0           # Initial velocity
    s0: float = 0.0           # Initial store value
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class SweepParameters:
    """Parameters for frequency/amplitude sweeps."""
    
    # Frequency sweep
    freq_min: float = 0.1
    freq_max: float = 3.0
    freq_steps: int = 30
    
    # Amplitude sweep (optional)
    amp_min: float = 0.001
    amp_max: float = 0.1
    amp_steps: int = 20
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# PRE-CONFIGURED PARAMETER SETS
# =============================================================================

# Default parameters - TUNED to show clear photoelectric threshold behavior
# With these parameters, threshold frequency is approximately f_th ≈ 0.12-0.15
# Below threshold: no escape (electron remains bound)
# Above threshold: escape occurs, with escape time decreasing as f increases
DEFAULT_PHYSICS = PhysicsParameters(
    omega0=1.0,         # Natural binding frequency
    damping=0.0001,     # Very low damping - allows oscillation growth
    alpha=0.05,         # Low store decay rate
    beta=100.0,         # Strong frequency-dependent coupling (key for threshold)
    amplitude=0.2,      # Moderate wave amplitude
    escape_position=5.0 # Position threshold for escape
)
DEFAULT_SIMULATION = SimulationParameters()
DEFAULT_SWEEP = SweepParameters()


# High sensitivity - very easy to see threshold behavior
HIGH_SENSITIVITY = PhysicsParameters(
    omega0=1.0,
    damping=0.002,    # Very low damping
    alpha=0.2,        # Slow store decay
    beta=3.0,         # Strong coupling
    amplitude=0.05,
    escape_position=3.0  # Lower escape threshold
)


# Photoelectric-like - tuned to show clear threshold behavior
PHOTOELECTRIC_TUNED = PhysicsParameters(
    omega0=1.0,
    damping=0.005,
    alpha=0.25,
    beta=2.5,
    amplitude=0.05,
    escape_position=5.0
)


# Strong binding - for studying tightly bound electrons
STRONG_BINDING = PhysicsParameters(
    omega0=2.0,       # Higher natural frequency
    damping=0.02,
    alpha=0.6,
    beta=1.2,
    amplitude=0.01,
    escape_position=5.0
)


# Weak binding - loosely bound electron
WEAK_BINDING = PhysicsParameters(
    omega0=0.5,
    damping=0.01,
    alpha=0.3,
    beta=0.6,
    amplitude=0.01,
    escape_position=5.0
)


# Long time simulation for detailed dynamics study
LONG_SIMULATION = SimulationParameters(
    t_max=500.0,
    dt=0.001,
    store_every=20,
    x0=0.001
)


# High precision for accurate threshold determination
HIGH_PRECISION_SWEEP = SweepParameters(
    freq_min=0.1,
    freq_max=3.0,
    freq_steps=100,  # More frequency points
    amp_steps=30
)


# Quick sweep for rapid exploration
QUICK_SWEEP = SweepParameters(
    freq_min=0.5,
    freq_max=2.5,
    freq_steps=15,
    amp_steps=10
)


def get_parameter_set(name: str) -> tuple:
    """
    Get a named parameter set.
    
    Args:
        name: One of 'default', 'high_sensitivity', 'photoelectric',
              'strong_binding', 'weak_binding'
              
    Returns:
        Tuple of (PhysicsParameters, SimulationParameters, SweepParameters)
    """
    physics_sets = {
        'default': DEFAULT_PHYSICS,
        'high_sensitivity': HIGH_SENSITIVITY,
        'photoelectric': PHOTOELECTRIC_TUNED,
        'strong_binding': STRONG_BINDING,
        'weak_binding': WEAK_BINDING
    }
    
    name_lower = name.lower()
    
    if name_lower in physics_sets:
        logger.info(f"Loading parameter set: {name}")
        return physics_sets[name_lower], DEFAULT_SIMULATION, DEFAULT_SWEEP
    else:
        logger.warning(f"Unknown parameter set '{name}', using default")
        return DEFAULT_PHYSICS, DEFAULT_SIMULATION, DEFAULT_SWEEP


def estimate_threshold_frequency(params: PhysicsParameters) -> float:
    """
    Estimate the threshold frequency based on parameters.
    
    This is a rough analytical estimate based on the balance between
    store charging and decay. The actual threshold emerges from the
    full nonlinear dynamics and may differ.
    
    The store equation is: ds/dt = -αs + βf²xu
    
    For the store to grow on average, we need:
    βf²⟨xu⟩_avg > α⟨s⟩_avg
    
    This gives a rough threshold: f_th ~ √(α/β) * correction_factor
    
    Args:
        params: Physics parameters
        
    Returns:
        Estimated threshold frequency
    """
    # Very rough estimate - actual threshold depends on nonlinear dynamics
    raw_estimate = (params.alpha / params.beta) ** 0.5
    
    # Correction factor from numerical observations (tune empirically)
    correction = 1.2
    
    estimate = raw_estimate * correction
    
    logger.info(
        f"Estimated threshold frequency: {estimate:.3f} "
        f"(raw: {raw_estimate:.3f})"
    )
    
    return estimate

