"""
Numerical Integration Module

Implements various numerical integrators for solving the ODEs,
with the RK4 (Runge-Kutta 4th order) as the primary method.

The integrators are designed for:
- Accuracy in capturing the resonance dynamics
- Efficiency for parameter sweeps
- Stability for long-time simulations
"""

import numpy as np
import logging
from typing import Callable, Tuple, Optional

logger = logging.getLogger(__name__)


class RK4Integrator:
    """
    4th-order Runge-Kutta integrator.
    
    This is the workhorse integrator for the simulation - it provides
    a good balance of accuracy and computational cost.
    
    The RK4 method evaluates the derivative at four points per step:
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)
    y_new = y + (dt/6)(k1 + 2k2 + 2k3 + k4)
    
    Error is O(dt^5) per step, O(dt^4) globally.
    """
    
    def __init__(self, dt: float = 0.001):
        """
        Initialize RK4 integrator.
        
        Args:
            dt: Time step size (smaller = more accurate but slower)
        """
        self.dt = dt
        logger.debug(f"RK4Integrator initialized with dt={dt}")
    
    def step(
        self, 
        t: float, 
        y: np.ndarray, 
        deriv_func: Callable,
        *args
    ) -> Tuple[float, np.ndarray]:
        """
        Perform one RK4 integration step.
        
        Args:
            t: Current time
            y: Current state vector
            deriv_func: Function that computes derivatives
            *args: Additional arguments to pass to deriv_func
            
        Returns:
            Tuple of (new_time, new_state)
        """
        dt = self.dt
        
        # Four derivative evaluations
        k1 = deriv_func(t, y, *args)
        k2 = deriv_func(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
        k3 = deriv_func(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
        k4 = deriv_func(t + dt, y + dt * k3, *args)
        
        # Weighted average
        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t_new = t + dt
        
        return t_new, y_new
    
    def integrate(
        self,
        t0: float,
        y0: np.ndarray,
        t_end: float,
        deriv_func: Callable,
        *args,
        callback: Optional[Callable] = None,
        store_every: int = 1
    ) -> dict:
        """
        Integrate from t0 to t_end.
        
        Args:
            t0: Initial time
            y0: Initial state
            t_end: End time
            deriv_func: Derivative function
            *args: Additional args for deriv_func
            callback: Optional function called each step (for escape detection)
                      Should return True to stop integration
            store_every: Store state every N steps (for memory efficiency)
            
        Returns:
            Dictionary with 'times', 'states', 'stopped_early', 'stop_time'
        """
        t = t0
        y = y0.copy()
        
        # Pre-allocate storage (estimate)
        n_steps = int((t_end - t0) / self.dt) + 1
        n_stored = n_steps // store_every + 1
        
        times = np.zeros(n_stored)
        states = np.zeros((n_stored, len(y0)))
        
        times[0] = t
        states[0] = y
        
        step_count = 0
        store_idx = 1
        stopped_early = False
        stop_time = None
        
        logger.debug(f"Starting integration from t={t0} to t={t_end}")
        
        while t < t_end:
            # Advance one step
            t, y = self.step(t, y, deriv_func, *args)
            step_count += 1
            
            # Store if needed
            if step_count % store_every == 0 and store_idx < n_stored:
                times[store_idx] = t
                states[store_idx] = y
                store_idx += 1
            
            # Check callback (escape detection)
            if callback is not None:
                should_stop = callback(t, y)
                if should_stop:
                    logger.debug(f"Integration stopped early at t={t:.6f}")
                    stopped_early = True
                    stop_time = t
                    # Store final state
                    if store_idx < n_stored:
                        times[store_idx] = t
                        states[store_idx] = y
                        store_idx += 1
                    break
        
        # Trim arrays to actual size
        times = times[:store_idx]
        states = states[:store_idx]
        
        logger.debug(
            f"Integration complete: {step_count} steps, "
            f"{store_idx} stored points, stopped_early={stopped_early}"
        )
        
        return {
            'times': times,
            'states': states,
            'stopped_early': stopped_early,
            'stop_time': stop_time,
            'n_steps': step_count
        }


class AdaptiveRK45Integrator:
    """
    Adaptive step-size RK4/5 integrator (Dormand-Prince).
    
    This integrator automatically adjusts the step size to maintain
    a target error tolerance, which is useful for:
    - Regions where the solution changes rapidly (near resonance)
    - Long-time simulations where efficiency matters
    
    Note: More complex than fixed-step RK4, use only if needed.
    """
    
    def __init__(
        self, 
        dt_initial: float = 0.01,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        dt_min: float = 1e-10,
        dt_max: float = 1.0
    ):
        """
        Initialize adaptive integrator.
        
        Args:
            dt_initial: Initial step size guess
            rtol: Relative error tolerance
            atol: Absolute error tolerance
            dt_min: Minimum allowed step size
            dt_max: Maximum allowed step size
        """
        self.dt = dt_initial
        self.rtol = rtol
        self.atol = atol
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # Dormand-Prince coefficients (simplified version)
        # Full implementation would have the complete Butcher tableau
        logger.debug(f"AdaptiveRK45 initialized: rtol={rtol}, atol={atol}")
    
    def step(
        self, 
        t: float, 
        y: np.ndarray, 
        deriv_func: Callable,
        *args
    ) -> Tuple[float, np.ndarray, float]:
        """
        Perform one adaptive step with error estimation.
        
        Returns:
            Tuple of (new_time, new_state, new_dt)
        """
        # For now, fall back to fixed RK4 with occasional step adjustment
        # A full implementation would use embedded RK4/5 for error estimation
        
        dt = self.dt
        
        # RK4 step
        k1 = deriv_func(t, y, *args)
        k2 = deriv_func(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
        k3 = deriv_func(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
        k4 = deriv_func(t + dt, y + dt * k3, *args)
        
        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
        # Simple step size adjustment based on state magnitude
        # (placeholder for proper error estimation)
        max_deriv = np.max(np.abs(k1))
        if max_deriv > 0:
            suggested_dt = 0.01 / max_deriv
            suggested_dt = np.clip(suggested_dt, self.dt_min, self.dt_max)
            # Smooth adjustment
            self.dt = 0.9 * self.dt + 0.1 * suggested_dt
        
        return t + dt, y_new, self.dt


class EulerIntegrator:
    """
    Simple Euler integrator (1st order).
    
    Included for comparison and educational purposes.
    NOT recommended for production use due to low accuracy.
    """
    
    def __init__(self, dt: float = 0.0001):
        self.dt = dt
        logger.warning("EulerIntegrator is for testing only - use RK4 for accuracy")
    
    def step(
        self, 
        t: float, 
        y: np.ndarray, 
        deriv_func: Callable,
        *args
    ) -> Tuple[float, np.ndarray]:
        """Simple forward Euler step."""
        dy = deriv_func(t, y, *args)
        y_new = y + self.dt * dy
        return t + self.dt, y_new

