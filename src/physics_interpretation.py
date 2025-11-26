"""
Physics Interpretation Module

This module provides tools for interpreting the model parameters
in terms of real physical quantities, and for exploring what the
internal "store" variable might represent.

The Central Question:
    If light is truly a wave, what internal mechanism of matter
    creates the photoelectric threshold?

Our Model Parameters:
    ω₀ : Natural binding frequency (electron-nucleus)
    d  : Damping (energy dissipation)
    α  : Store decay rate (decoherence)
    β  : Frequency coupling (how f² charges the store)
    s  : Internal store (THE MYSTERY - what is it?)

This module explores mappings between these abstract parameters
and physical properties of real materials.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Physical constants (SI units)
ELECTRON_MASS = 9.109e-31      # kg
ELECTRON_CHARGE = 1.602e-19    # C
HBAR = 1.055e-34               # J·s
EPSILON_0 = 8.854e-12          # F/m
SPEED_OF_LIGHT = 3.0e8         # m/s
PLANCK = 6.626e-34             # J·s


@dataclass
class MaterialProperties:
    """
    Physical properties of a material relevant to photoelectric effect.
    
    These are the REAL quantities we want to connect to model parameters.
    """
    
    name: str
    work_function_eV: float          # Minimum energy to extract electron
    plasma_frequency_Hz: float       # Collective electron oscillation
    electron_density_m3: float       # Free electron density
    fermi_energy_eV: float          # Highest occupied energy at T=0
    coherence_time_s: float         # How long phase coherence lasts
    mean_free_path_m: float         # Average distance between collisions
    
    def work_function_J(self) -> float:
        return self.work_function_eV * ELECTRON_CHARGE
    
    def threshold_frequency_Hz(self) -> float:
        """Classical threshold frequency from work function."""
        return self.work_function_J() / PLANCK
    
    def threshold_wavelength_nm(self) -> float:
        """Threshold wavelength."""
        f = self.threshold_frequency_Hz()
        return (SPEED_OF_LIGHT / f) * 1e9


# Common materials data
MATERIALS = {
    'sodium': MaterialProperties(
        name='Sodium (Na)',
        work_function_eV=2.36,
        plasma_frequency_Hz=5.7e15,
        electron_density_m3=2.5e28,
        fermi_energy_eV=3.2,
        coherence_time_s=1e-14,
        mean_free_path_m=3e-8
    ),
    'potassium': MaterialProperties(
        name='Potassium (K)',
        work_function_eV=2.29,
        plasma_frequency_Hz=4.4e15,
        electron_density_m3=1.3e28,
        fermi_energy_eV=2.1,
        coherence_time_s=1.5e-14,
        mean_free_path_m=3.5e-8
    ),
    'copper': MaterialProperties(
        name='Copper (Cu)',
        work_function_eV=4.65,
        plasma_frequency_Hz=1.6e16,
        electron_density_m3=8.5e28,
        fermi_energy_eV=7.0,
        coherence_time_s=5e-15,
        mean_free_path_m=4e-8
    ),
    'gold': MaterialProperties(
        name='Gold (Au)',
        work_function_eV=5.1,
        plasma_frequency_Hz=1.4e16,
        electron_density_m3=5.9e28,
        fermi_energy_eV=5.5,
        coherence_time_s=3e-15,
        mean_free_path_m=3.7e-8
    ),
    'cesium': MaterialProperties(
        name='Cesium (Cs)',
        work_function_eV=2.14,  # Lowest known work function
        plasma_frequency_Hz=3.0e15,
        electron_density_m3=0.9e28,
        fermi_energy_eV=1.6,
        coherence_time_s=2e-14,
        mean_free_path_m=4e-8
    )
}


@dataclass
class ModelParameters:
    """Our model's abstract parameters."""
    omega0: float   # Natural frequency (dimensionless in model)
    damping: float  # Damping coefficient
    alpha: float    # Store decay rate
    beta: float     # Frequency coupling
    amplitude: float # Wave amplitude


def material_to_model_parameters(
    material: MaterialProperties,
    reference_frequency: float = 1e15  # Hz, for normalization
) -> ModelParameters:
    """
    Map physical material properties to model parameters.
    
    This is a KEY function - it connects real physics to our model.
    
    Mappings:
        ω₀ ~ work_function / ℏ (binding strength)
        d  ~ 1 / (mean_free_path × v_fermi) (collision rate)
        α  ~ 1 / coherence_time (decoherence rate)
        β  ~ electron_density × coupling_constant (collective effect strength)
    
    Args:
        material: MaterialProperties for a real material
        reference_frequency: Frequency for normalization
        
    Returns:
        ModelParameters for simulation
    """
    # Normalize everything to reference frequency
    omega_ref = 2 * np.pi * reference_frequency
    
    # ω₀: Binding frequency from work function
    # E = ℏω implies ω = E/ℏ
    omega0_physical = material.work_function_J() / HBAR
    omega0 = omega0_physical / omega_ref
    
    # d: Damping from collision rate
    # Fermi velocity ~ sqrt(2·E_F/m)
    v_fermi = np.sqrt(2 * material.fermi_energy_eV * ELECTRON_CHARGE / ELECTRON_MASS)
    collision_rate = v_fermi / material.mean_free_path_m
    damping = collision_rate / omega_ref
    
    # α: Store decay from decoherence
    alpha = 1.0 / (material.coherence_time_s * omega_ref)
    
    # β: Coupling strength from electron density
    # Higher density = more collective enhancement = stronger coupling
    # Normalize to typical metal density
    reference_density = 5e28  # m^-3
    density_factor = material.electron_density_m3 / reference_density
    # Scale factor tuned to give reasonable threshold
    beta = 100.0 * density_factor
    
    # Amplitude: normalized
    amplitude = 0.2
    
    logger.info(
        f"Material {material.name} -> Model: "
        f"omega0={omega0:.4f}, d={damping:.6f}, "
        f"alpha={alpha:.4f}, beta={beta:.2f}"
    )
    
    return ModelParameters(
        omega0=omega0,
        damping=damping,
        alpha=alpha,
        beta=beta,
        amplitude=amplitude
    )


def model_to_physical_threshold(
    params: ModelParameters,
    reference_frequency: float = 1e15
) -> Dict:
    """
    Calculate physical threshold frequency from model parameters.
    
    Args:
        params: Model parameters
        reference_frequency: Reference frequency in Hz
        
    Returns:
        Dictionary with physical predictions
    """
    # Threshold in model units: f_th ~ sqrt(alpha/beta)
    threshold_model = np.sqrt(params.alpha / params.beta) * 1.2  # Correction factor
    
    # Convert to physical units
    threshold_Hz = threshold_model * reference_frequency
    threshold_wavelength_nm = (SPEED_OF_LIGHT / threshold_Hz) * 1e9
    threshold_energy_eV = (PLANCK * threshold_Hz) / ELECTRON_CHARGE
    
    return {
        'threshold_model_units': threshold_model,
        'threshold_Hz': threshold_Hz,
        'threshold_wavelength_nm': threshold_wavelength_nm,
        'threshold_energy_eV': threshold_energy_eV
    }


@dataclass
class StoreInterpretation:
    """
    Possible physical interpretations of the store variable s.
    """
    
    name: str
    description: str
    physical_quantity: str
    units: str
    how_it_charges: str
    how_it_decays: str
    experimental_signature: str


# Possible interpretations of the mysterious "store" variable
STORE_INTERPRETATIONS = [
    StoreInterpretation(
        name="Coherent Polarization",
        description="""
        The store represents the coherent polarization of the electron cloud.
        When the EM wave drives the electrons, they oscillate together (coherently).
        This coherent motion creates a collective dipole moment that enhances
        the local field experienced by any single electron.
        """,
        physical_quantity="Collective dipole moment per electron",
        units="C·m / electron",
        how_it_charges="Synchronized electron oscillation builds collective dipole",
        how_it_decays="Electron-electron scattering randomizes phases",
        experimental_signature="Enhanced absorption at resonance, narrowband response"
    ),
    
    StoreInterpretation(
        name="Surface Plasmon Amplitude",
        description="""
        The store represents the amplitude of surface plasmon oscillations.
        Plasmons are collective oscillations of the electron gas.
        When the EM wave frequency approaches the plasmon frequency,
        energy accumulates in this collective mode.
        """,
        physical_quantity="Plasmon mode amplitude",
        units="Number of plasmon quanta",
        how_it_charges="EM wave drives plasmon resonance",
        how_it_decays="Plasmon damping (Landau damping, surface scattering)",
        experimental_signature="Sharp resonance at plasma frequency"
    ),
    
    StoreInterpretation(
        name="Local Field Enhancement",
        description="""
        In a crystal, the local electric field differs from the applied field
        due to the polarization of surrounding atoms (Lorentz local field).
        The store represents this enhancement factor.
        """,
        physical_quantity="Local field enhancement factor",
        units="Dimensionless ratio",
        how_it_charges="Polarization of surrounding medium builds up",
        how_it_decays="Relaxation of polarization",
        experimental_signature="Dielectric constant frequency dependence"
    ),
    
    StoreInterpretation(
        name="Phase Coherence",
        description="""
        The store tracks the phase relationship between electron motion
        and the driving wave. When they're in phase, energy transfers
        efficiently. The store measures this phase lock quality.
        """,
        physical_quantity="Phase correlation coefficient",
        units="Dimensionless (-1 to 1)",
        how_it_charges="Motion synchronizes with drive",
        how_it_decays="Random perturbations (thermal noise, collisions)",
        experimental_signature="Interference effects, coherent control possible"
    ),
    
    StoreInterpretation(
        name="Virtual State Population",
        description="""
        Quantum mechanically, the electron can be in a superposition of 
        bound and free states. The store represents the population of
        intermediate 'virtual' states that facilitate the transition.
        """,
        physical_quantity="Probability amplitude squared",
        units="Dimensionless probability",
        how_it_charges="Multiphoton-like transitions through virtual states",
        how_it_decays="Decoherence destroys superposition",
        experimental_signature="Intensity-dependent transition rates"
    )
]


def analyze_store_physics(
    simulation_result,
    interpretation: str = "all"
) -> Dict:
    """
    Analyze simulation results through different physical interpretations.
    
    Args:
        simulation_result: Result from simulation run
        interpretation: Which interpretation to use, or "all"
        
    Returns:
        Analysis dictionary with physical interpretations
    """
    times = simulation_result.times
    positions = simulation_result.positions
    velocities = simulation_result.velocities
    stores = simulation_result.stores
    
    analysis = {
        'max_store': np.max(stores),
        'mean_store': np.mean(stores[len(stores)//2:]),  # Second half
        'store_buildup_time': None,
        'phase_correlation': None
    }
    
    # Find when store reaches 50% of max
    if np.max(stores) > 0.1:
        half_max = np.max(stores) / 2
        above_half = np.where(stores > half_max)[0]
        if len(above_half) > 0:
            analysis['store_buildup_time'] = times[above_half[0]]
    
    # Compute phase correlation between position and velocity
    # In resonance, v and x are 90° out of phase
    # Correlation of x and v should be near zero at resonance
    if len(positions) > 10:
        corr = np.corrcoef(positions[10:], velocities[10:])[0, 1]
        analysis['phase_correlation'] = corr
    
    return analysis


def explain_threshold_physics() -> str:
    """
    Generate a detailed explanation of why frequency threshold emerges.
    
    Returns:
        Formatted explanation string
    """
    explanation = """
======================================================================
          WHY DOES A FREQUENCY THRESHOLD EMERGE?                  
======================================================================

The threshold emerges from a RATE COMPETITION in the store dynamics:

    ds/dt = -alpha*s + beta*f^2*|v*u|
            ---------   --------------
                |             |
              DECAY       CHARGING

----------------------------------------------------------------------
AT LOW FREQUENCY (f < f_threshold):
----------------------------------------------------------------------
  * Charging rate (proportional to f^2) is SMALL
  * Decay rate alpha is CONSTANT
  * DECAY WINS -> s -> 0
  * No amplification -> electron stays bound

  Physical picture: The wave is too slow. Between peaks,
  the internal coherence decays. Each cycle starts fresh.

----------------------------------------------------------------------
AT HIGH FREQUENCY (f > f_threshold):
----------------------------------------------------------------------
  * Charging rate (proportional to f^2) is LARGE
  * Decay rate alpha is CONSTANT
  * CHARGING WINS -> s grows
  * Amplification increases -> positive feedback -> escape

  Physical picture: The wave pumps energy faster than it
  leaks out. Coherence builds. Oscillations grow.

----------------------------------------------------------------------
THE THRESHOLD CONDITION:
----------------------------------------------------------------------
    
    At threshold: charging rate ~ decay rate
    
    beta * f_th^2 * <|vu|> ~ alpha * <s>
    
    Solving: f_threshold ~ sqrt(alpha/beta)

----------------------------------------------------------------------
THE KEY INSIGHT:
----------------------------------------------------------------------
    
    The threshold is determined by MATERIAL PROPERTIES (alpha, beta),
    not by the light itself!
    
    * alpha (decay rate) = how fast coherence is lost
    * beta (coupling) = how strongly frequency couples to internal state
    
    Different materials have different alpha, beta -> different thresholds.
    This is exactly what we observe: different metals have different
    work functions (photoelectric thresholds).

======================================================================
"""
    return explanation


def compare_materials_threshold() -> str:
    """
    Compare threshold predictions for different materials.
    
    Returns:
        Formatted comparison table
    """
    lines = [
        "====================================================================",
        "         MATERIAL COMPARISON: THRESHOLD PREDICTIONS             ",
        "====================================================================",
        " Material    | Work Func | Wavelength  | Our Model | Match?     ",
        "             | (eV)      | (nm) actual | (nm) pred |            ",
        "--------------------------------------------------------------------",
    ]
    
    for name, mat in MATERIALS.items():
        params = material_to_model_parameters(mat)
        prediction = model_to_physical_threshold(params)
        
        actual_nm = mat.threshold_wavelength_nm()
        predicted_nm = prediction['threshold_wavelength_nm']
        
        # Check if within 50%
        match = "~OK" if 0.5 < predicted_nm/actual_nm < 2.0 else "XXX"
        
        lines.append(
            f" {mat.name[:11]:<11} | {mat.work_function_eV:^9.2f} | "
            f"{actual_nm:^11.1f} | {predicted_nm:^9.1f} | {match:^10}"
        )
    
    lines.append("====================================================================")
    
    return "\n".join(lines)


def get_physical_meaning_summary() -> str:
    """
    Get a summary of physical meanings of all parameters.
    
    Returns:
        Formatted summary string
    """
    summary = """
======================================================================
           PHYSICAL MEANING OF MODEL PARAMETERS                   
======================================================================

  omega0 - BINDING STRENGTH
  -------------------------
  * Natural oscillation frequency of bound electron
  * Related to work function: phi ~ hbar * omega0
  * Set by atomic potential well shape
  * Higher omega0 = harder to escape = higher threshold

----------------------------------------------------------------------

  d (damping) - ENERGY DISSIPATION
  ---------------------------------
  * Rate of energy loss from electron motion
  * Electron-phonon scattering
  * Electron-electron scattering
  * Must be SMALL for threshold to exist

----------------------------------------------------------------------

  alpha - COHERENCE DECAY RATE
  ----------------------------
  * How fast the "internal store" depletes
  * Decoherence rate of electronic state
  * Higher alpha = higher threshold (harder to build coherence)
  * Related to: temperature, disorder, electron-phonon coupling

----------------------------------------------------------------------

  beta - FREQUENCY COUPLING
  -------------------------
  * How strongly f^2 couples to store charging
  * Related to electron density (collective effects)
  * Higher beta = lower threshold (easier to charge store)
  * May relate to: dipole moment, polarizability, density

----------------------------------------------------------------------

  s (store) - THE MYSTERY VARIABLE
  ---------------------------------
  * Internal gain/amplification state
  * POSSIBLE INTERPRETATIONS:
    1. Coherent electronic polarization
    2. Surface plasmon amplitude
    3. Local field enhancement factor
    4. Phase coherence between electron and wave
    5. Population of virtual/intermediate states

  * This is the KEY to understanding what's happening!

======================================================================
"""
    return summary


if __name__ == "__main__":
    # Demo the interpretations
    print(explain_threshold_physics())
    print()
    print(get_physical_meaning_summary())
    print()
    print(compare_materials_threshold())

