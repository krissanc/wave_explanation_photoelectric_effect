"""
Simulation Module - Main Simulation Engine

This module ties together the physics model and integrator to run
complete simulations of the wave-photoelectric interaction.

Key functionality:
- Single frequency simulation with escape detection
- Parameter sweeps (frequency, amplitude, etc.)
- Result collection and analysis preparation
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .physics import AtomModel, EscapeDetector
from .integrator import RK4Integrator

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    
    # Time parameters
    t_max: float = 150.0      # Maximum simulation time
    dt: float = 0.0005        # Integration time step (small for accuracy)
    
    # Initial conditions
    x0: float = 0.05          # Initial displacement (needs non-zero to start)
    v0: float = 0.0           # Initial velocity
    s0: float = 0.0           # Initial store value
    
    # Data storage
    store_every: int = 20     # Store state every N steps
    
    def __post_init__(self):
        logger.info(
            f"SimulationConfig: t_max={self.t_max}, dt={self.dt}, "
            f"x0={self.x0}, store_every={self.store_every}"
        )


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    
    # Input parameters
    frequency: float
    amplitude: float
    config: SimulationConfig
    
    # Outcome
    escaped: bool
    escape_time: Optional[float]
    escape_reason: Optional[str]
    
    # Time series data
    times: np.ndarray = field(repr=False)
    positions: np.ndarray = field(repr=False)
    velocities: np.ndarray = field(repr=False)
    stores: np.ndarray = field(repr=False)
    
    # Metadata
    computation_time_ms: float = 0.0
    n_steps: int = 0
    
    def get_final_state(self) -> Dict:
        """Get the final state of the simulation."""
        return {
            'x': self.positions[-1] if len(self.positions) > 0 else None,
            'v': self.velocities[-1] if len(self.velocities) > 0 else None,
            's': self.stores[-1] if len(self.stores) > 0 else None,
            't': self.times[-1] if len(self.times) > 0 else None
        }
    
    def get_max_amplitude(self) -> float:
        """Get maximum displacement reached."""
        return np.max(np.abs(self.positions)) if len(self.positions) > 0 else 0.0
    
    def get_max_store(self) -> float:
        """Get maximum store value reached."""
        return np.max(self.stores) if len(self.stores) > 0 else 0.0


class Simulation:
    """
    Main simulation engine.
    
    Orchestrates the physics model, integrator, and escape detection
    to run complete simulations of the wave-photoelectric interaction.
    """
    
    def __init__(
        self,
        atom_model: AtomModel,
        escape_detector: EscapeDetector,
        config: SimulationConfig = None
    ):
        """
        Initialize simulation engine.
        
        Args:
            atom_model: The atom physics model
            escape_detector: Detector for electron escape
            config: Simulation configuration (uses defaults if None)
        """
        self.atom = atom_model
        self.detector = escape_detector
        self.config = config if config is not None else SimulationConfig()
        self.integrator = RK4Integrator(dt=self.config.dt)
        
        logger.info(f"Simulation engine initialized with {atom_model}")
    
    def run(self, frequency: float) -> SimulationResult:
        """
        Run a single simulation at the given frequency.
        
        Args:
            frequency: Input wave frequency
            
        Returns:
            SimulationResult with all simulation data
        """
        import time
        start_time = time.perf_counter()
        
        logger.info(f"Starting simulation at f={frequency:.4f}")
        
        # Initial state
        y0 = np.array([
            self.config.x0,
            self.config.v0,
            self.config.s0
        ])
        
        # Track escape
        escaped = False
        escape_time = None
        escape_reason = None
        
        def escape_callback(t: float, y: np.ndarray) -> bool:
            """Callback to check for escape during integration."""
            nonlocal escaped, escape_time, escape_reason
            
            # Check numerical stability
            if self.detector.check_numerical_instability(y):
                escaped = True
                escape_time = t
                escape_reason = "numerical_instability"
                return True
            
            # Check escape condition
            is_escaped, reason = self.detector.check_escape(y)
            if is_escaped:
                escaped = True
                escape_time = t
                escape_reason = reason
                logger.info(f"ESCAPE detected at t={t:.4f}: {reason}")
                return True
            
            return False
        
        # Run integration
        # Note: frequency is passed as *args to be forwarded to deriv_func
        result = self.integrator.integrate(
            0.0,                        # t0
            y0,                         # y0
            self.config.t_max,          # t_end
            self.atom.derivatives,      # deriv_func
            frequency,                  # *args: forwarded to derivatives()
            callback=escape_callback,
            store_every=self.config.store_every
        )
        
        # Extract time series
        times = result['times']
        states = result['states']
        
        computation_time = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            f"Simulation complete: f={frequency:.4f}, "
            f"escaped={escaped}, escape_time={escape_time}, "
            f"compute_time={computation_time:.1f}ms"
        )
        
        return SimulationResult(
            frequency=frequency,
            amplitude=self.atom.amplitude,
            config=self.config,
            escaped=escaped,
            escape_time=escape_time,
            escape_reason=escape_reason,
            times=times,
            positions=states[:, 0],
            velocities=states[:, 1],
            stores=states[:, 2],
            computation_time_ms=computation_time,
            n_steps=result['n_steps']
        )
    
    def run_frequency_sweep(
        self,
        frequencies: np.ndarray,
        progress_callback: Optional[callable] = None
    ) -> List[SimulationResult]:
        """
        Run simulations across a range of frequencies.
        
        Args:
            frequencies: Array of frequencies to test
            progress_callback: Optional callback(i, n, result) for progress
            
        Returns:
            List of SimulationResult for each frequency
        """
        results = []
        n_freqs = len(frequencies)
        
        logger.info(f"Starting frequency sweep: {n_freqs} frequencies")
        logger.info(f"Frequency range: [{frequencies[0]:.4f}, {frequencies[-1]:.4f}]")
        
        for i, freq in enumerate(frequencies):
            result = self.run(freq)
            results.append(result)
            
            if progress_callback is not None:
                progress_callback(i + 1, n_freqs, result)
        
        # Summary statistics
        n_escaped = sum(1 for r in results if r.escaped)
        logger.info(
            f"Frequency sweep complete: {n_escaped}/{n_freqs} escaped"
        )
        
        return results
    
    def run_amplitude_sweep(
        self,
        frequency: float,
        amplitudes: np.ndarray
    ) -> List[SimulationResult]:
        """
        Run simulations across a range of amplitudes at fixed frequency.
        
        This tests the photoelectric effect prediction that threshold
        depends on frequency, not amplitude.
        
        Args:
            frequency: Fixed wave frequency
            amplitudes: Array of amplitudes to test
            
        Returns:
            List of SimulationResult for each amplitude
        """
        results = []
        original_amplitude = self.atom.amplitude
        
        logger.info(
            f"Starting amplitude sweep at f={frequency}: "
            f"{len(amplitudes)} amplitudes"
        )
        
        for amp in amplitudes:
            self.atom.amplitude = amp
            result = self.run(frequency)
            results.append(result)
        
        # Restore original amplitude
        self.atom.amplitude = original_amplitude
        
        return results


def create_default_simulation(
    omega0: float = 1.0,
    damping: float = 0.01,
    alpha: float = 0.5,
    beta: float = 1.0,
    amplitude: float = 0.01,
    escape_threshold: float = 5.0,
    t_max: float = 200.0,
    dt: float = 0.001
) -> Simulation:
    """
    Factory function to create a simulation with default parameters.
    
    This is a convenience function for quick experimentation.
    
    Args:
        omega0: Natural binding frequency
        damping: Velocity damping coefficient
        alpha: Store decay rate
        beta: Motion-store coupling
        amplitude: Input wave amplitude
        escape_threshold: Position threshold for escape
        t_max: Maximum simulation time
        dt: Integration time step
        
    Returns:
        Configured Simulation object
    """
    atom = AtomModel(
        omega0=omega0,
        damping=damping,
        alpha=alpha,
        beta=beta,
        amplitude=amplitude
    )
    
    detector = EscapeDetector(position_threshold=escape_threshold)
    
    config = SimulationConfig(t_max=t_max, dt=dt)
    
    return Simulation(atom, detector, config)

