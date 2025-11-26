"""
Physics Module - Core Differential Equations for Wave-Photoelectric Model

This module defines the state-space model for a bound electron interacting
with an incoming electromagnetic wave, with an internal "store" mechanism
that creates frequency-dependent threshold behavior.

State Variables:
    x(t) : electron displacement from equilibrium
    v(t) : electron velocity (dx/dt)
    s(t) : internal store / gain variable

Input:
    u(t) = A * sin(2πft) : incoming wave

Governing Equations:
    dx/dt = v
    dv/dt = -ω₀²x - dv + s*u(t)
    ds/dt = -αs + β*f²*x*u(t)

Physical Interpretation:
    - ω₀²x : binding force (electron-nucleus spring)
    - dv   : damping (energy loss mechanisms)
    - s*u  : amplified wave driving (internal gain)
    - αs   : store relaxation (energy dissipation in store)
    - βf²xu: store charging (frequency-dependent coupling)

The f² term in the store equation is KEY - it creates a frequency threshold
where below a critical frequency, the store cannot maintain itself, but above
that frequency, the store grows leading to electron escape.
"""

import numpy as np
import logging

# Configure module logger
logger = logging.getLogger(__name__)


class AtomModel:
    """
    1D toy atom model with internal store mechanism.
    
    This class encapsulates the physics of a bound electron interacting
    with an electromagnetic wave through a frequency-dependent internal
    gain mechanism.
    """
    
    def __init__(
        self,
        omega0: float = 1.0,    # Natural binding frequency
        damping: float = 0.01,  # Velocity damping coefficient
        alpha: float = 0.5,     # Store decay rate
        beta: float = 1.0,      # Motion-store coupling strength
        amplitude: float = 0.01 # Input wave amplitude
    ):
        """
        Initialize the atom model with physical parameters.
        
        Args:
            omega0: Natural frequency of bound electron (binding strength)
            damping: Damping coefficient for electron velocity
            alpha: Internal store decay rate (how fast store depletes)
            beta: Coupling coefficient between motion and store charging
            amplitude: Amplitude of incoming electromagnetic wave
        """
        self.omega0 = omega0
        self.omega0_sq = omega0 ** 2
        self.damping = damping
        self.alpha = alpha
        self.beta = beta
        self.amplitude = amplitude
        
        logger.info(
            f"AtomModel initialized: omega0={omega0:.4f}, d={damping:.4f}, "
            f"alpha={alpha:.4f}, beta={beta:.4f}, A={amplitude:.6f}"
        )
    
    def wave_input(self, t: float, frequency: float) -> float:
        """
        Calculate the input wave value at time t.
        
        u(t) = A * sin(2πft)
        
        Args:
            t: Current time
            frequency: Wave frequency
            
        Returns:
            Wave amplitude at time t
        """
        return self.amplitude * np.sin(2.0 * np.pi * frequency * t)
    
    def derivatives(
        self, 
        t: float, 
        state: np.ndarray, 
        frequency: float
    ) -> np.ndarray:
        """
        Calculate state derivatives for the system.
        
        This is the core physics: the right-hand side of our ODEs.
        
        Args:
            t: Current time
            state: Current state vector [x, v, s]
            frequency: Input wave frequency
            
        Returns:
            Derivative vector [dx/dt, dv/dt, ds/dt]
        """
        x, v, s = state
        
        # Input wave
        u = self.wave_input(t, frequency)
        
        # Position derivative: dx/dt = v
        dx_dt = v
        
        # Velocity derivative: dv/dt = -ω₀²x - d*v + s*u
        # This is a driven damped harmonic oscillator with state-dependent gain
        dv_dt = -self.omega0_sq * x - self.damping * v + s * u
        
        # Store derivative: ds/dt = -αs + βf² * |v*u|
        # Modified to use POWER absorption: when velocity and wave force align,
        # energy is being pumped into the electron, charging the store.
        # The f² factor creates the frequency threshold.
        # Using abs() ensures the store always charges when there's power flow.
        freq_sq = frequency ** 2
        power_absorption = abs(v * u)  # Energy being pumped in
        ds_dt = -self.alpha * s + self.beta * freq_sq * power_absorption
        
        return np.array([dx_dt, dv_dt, ds_dt])
    
    def calculate_energy(self, state: np.ndarray) -> dict:
        """
        Calculate various energy components of the system.
        
        Args:
            state: Current state vector [x, v, s]
            
        Returns:
            Dictionary with energy components
        """
        x, v, s = state
        
        kinetic = 0.5 * v ** 2
        potential = 0.5 * self.omega0_sq * x ** 2
        total_mechanical = kinetic + potential
        
        return {
            'kinetic': kinetic,
            'potential': potential,
            'mechanical': total_mechanical,
            'store': s,
            'position': x,
            'velocity': v
        }
    
    def __repr__(self) -> str:
        return (
            f"AtomModel(omega0={self.omega0}, d={self.damping}, "
            f"alpha={self.alpha}, beta={self.beta}, A={self.amplitude})"
        )


class EscapeDetector:
    """
    Detects when an electron has escaped its bound state.
    
    Escape can be defined by:
    1. Position exceeding threshold: |x| > X_esc
    2. Kinetic energy exceeding threshold: ½v² > E_esc
    """
    
    def __init__(
        self,
        position_threshold: float = 5.0,
        energy_threshold: float = None,  # Optional: use ½v² > E_esc
        use_energy: bool = False
    ):
        """
        Initialize escape detector.
        
        Args:
            position_threshold: Position beyond which electron is "escaped"
            energy_threshold: Kinetic energy threshold for escape
            use_energy: If True, use energy criterion instead of position
        """
        self.position_threshold = position_threshold
        self.energy_threshold = energy_threshold
        self.use_energy = use_energy
        
        logger.info(
            f"EscapeDetector initialized: X_esc={position_threshold}, "
            f"E_esc={energy_threshold}, use_energy={use_energy}"
        )
    
    def check_escape(self, state: np.ndarray) -> tuple:
        """
        Check if electron has escaped.
        
        Args:
            state: Current state vector [x, v, s]
            
        Returns:
            Tuple of (escaped: bool, reason: str or None)
        """
        x, v, s = state
        
        # Position-based escape
        if abs(x) > self.position_threshold:
            return True, f"position (|x|={abs(x):.4f} > {self.position_threshold})"
        
        # Energy-based escape (optional)
        if self.use_energy and self.energy_threshold is not None:
            kinetic = 0.5 * v ** 2
            if kinetic > self.energy_threshold:
                return True, f"energy (KE={kinetic:.4f} > {self.energy_threshold})"
        
        return False, None
    
    def check_numerical_instability(self, state: np.ndarray) -> bool:
        """
        Check for numerical instability (NaN or very large values).
        
        Args:
            state: Current state vector
            
        Returns:
            True if instability detected
        """
        if np.any(np.isnan(state)):
            logger.warning("NaN detected in state vector")
            return True
        
        if np.any(np.abs(state) > 1e10):
            logger.warning("Very large values detected in state vector")
            return True
        
        return False

