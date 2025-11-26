"""
Crystal Lattice Black Box Model

THE VISION:
-----------
Light hits a small region of metal crystal lattice. This region is our
"black box" - a 3D cutout containing:
    - Bound electrons
    - Ion cores (nuclei)
    - The electromagnetic field
    - Collective excitations (plasmons, phonons)

Our state-space model describes what happens inside this box.
The goal is to derive the COMPLETE system of equations and understand
the physical meaning of each term.

APPROACH:
---------
Like the fox-rabbit population model using differential equations,
we have multiple interacting quantities:
    - Electron position x(t)
    - Electron velocity v(t)  
    - Internal store s(t)
    - Local electric field E_local(t)
    - Lattice displacement u_lattice(t)
    - Polarization P(t)

These are COUPLED - each affects the others. We need to find
the complete system of equations that governs their evolution.

FROM STATE MATRIX TO PHYSICAL INSIGHT:
--------------------------------------
The state matrix contains all the physics. By analyzing it,
we can extract:
    - What forces act on what
    - What energy flows where
    - What creates the frequency threshold
    - What the "store" physically represents
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# THE BLACK BOX: CRYSTAL LATTICE REGION
# =============================================================================

@dataclass
class CrystalRegion:
    """
    A small region of crystal lattice - our "black box".
    
    Physical structure:
    - N_atoms: number of atoms in region
    - a: lattice constant (spacing between atoms)
    - Z: atomic number (protons per atom)
    - n_free: number of free electrons per atom
    
    This defines the stage on which the photoelectric effect plays out.
    """
    
    # Lattice geometry
    lattice_constant_m: float = 3.5e-10    # Typical metal ~3.5 Angstroms
    region_size_m: float = 1e-8            # 10 nm region
    
    # Atomic properties
    atomic_number: int = 29                 # Copper
    free_electrons_per_atom: float = 1.0    # One free electron per atom
    
    # Derived quantities
    @property
    def n_atoms(self) -> int:
        """Number of atoms in region."""
        return int((self.region_size_m / self.lattice_constant_m) ** 3)
    
    @property
    def electron_density(self) -> float:
        """Free electron density (m^-3)."""
        atoms_per_volume = 1.0 / (self.lattice_constant_m ** 3)
        return atoms_per_volume * self.free_electrons_per_atom
    
    @property
    def plasma_frequency(self) -> float:
        """Plasma frequency of electron gas (rad/s)."""
        e = 1.602e-19
        m_e = 9.109e-31
        eps_0 = 8.854e-12
        return np.sqrt(self.electron_density * e**2 / (eps_0 * m_e))
    
    @property
    def debye_length(self) -> float:
        """Debye screening length (m)."""
        # At room temperature
        k_B = 1.38e-23
        T = 300  # K
        e = 1.602e-19
        eps_0 = 8.854e-12
        return np.sqrt(eps_0 * k_B * T / (self.electron_density * e**2))


# =============================================================================
# THE COMPLETE STATE VECTOR
# =============================================================================

@dataclass
class FullState:
    """
    Complete state of the crystal region.
    
    This is the FULL description of what's happening in our black box.
    Each variable represents a physical quantity that evolves in time.
    """
    
    # Electron state (our original model)
    x: float          # Electron displacement from equilibrium (m)
    v: float          # Electron velocity (m/s)
    s: float          # Internal store/gain (dimensionless)
    
    # Local electromagnetic field
    E_local: float    # Local electric field (V/m)
    B_local: float    # Local magnetic field (T)
    
    # Lattice state
    u_lattice: float  # Lattice ion displacement (m)
    v_lattice: float  # Lattice velocity (m/s)
    
    # Collective excitations
    P: float          # Polarization (C/m^2)
    n_plasmon: float  # Plasmon occupation number
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for matrix operations."""
        return np.array([
            self.x, self.v, self.s,
            self.E_local, self.B_local,
            self.u_lattice, self.v_lattice,
            self.P, self.n_plasmon
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'FullState':
        """Create from numpy array."""
        return cls(
            x=arr[0], v=arr[1], s=arr[2],
            E_local=arr[3], B_local=arr[4],
            u_lattice=arr[5], v_lattice=arr[6],
            P=arr[7], n_plasmon=arr[8]
        )


# =============================================================================
# THE SYSTEM OF EQUATIONS
# =============================================================================

class CrystalLatticeEquations:
    """
    The complete system of differential equations governing the black box.
    
    This is like the fox-rabbit system, but for a crystal:
    
    dx/dt = f1(x, v, s, E, B, u, P, ...)
    dv/dt = f2(x, v, s, E, B, u, P, ...)
    ds/dt = f3(x, v, s, E, B, u, P, ...)
    dE/dt = f4(x, v, s, E, B, u, P, ...)
    ...
    
    The coupling between these equations creates the emergent behavior.
    """
    
    def __init__(self, crystal: CrystalRegion):
        """
        Initialize with crystal parameters.
        
        Args:
            crystal: The crystal region (our black box geometry)
        """
        self.crystal = crystal
        
        # Physical constants
        self.e = 1.602e-19        # Electron charge (C)
        self.m_e = 9.109e-31      # Electron mass (kg)
        self.eps_0 = 8.854e-12    # Permittivity of free space
        self.c = 3.0e8            # Speed of light (m/s)
        
        # Model parameters (to be connected to physical quantities)
        self.omega_0 = 1e15       # Electron binding frequency (rad/s)
        self.gamma_e = 1e12       # Electron damping rate (1/s)
        self.omega_lattice = 1e13 # Lattice vibration frequency (rad/s)
        self.gamma_lattice = 1e11 # Lattice damping rate (1/s)
        
        # Store dynamics parameters
        self.alpha = 1e13         # Store decay rate (1/s)
        self.beta = 1e-6          # Store coupling coefficient
        
        # Coupling parameters
        self.g_ep = 1e-20         # Electron-plasmon coupling
        self.g_el = 1e-10         # Electron-lattice coupling
        
        logger.info(f"CrystalLatticeEquations initialized for {crystal.n_atoms} atoms")
    
    def derive_equations(self) -> str:
        """
        Derive and document the complete system of equations.
        
        Returns the equations in human-readable form.
        """
        equations = """
========================================================================
    COMPLETE SYSTEM OF EQUATIONS FOR CRYSTAL LATTICE BLACK BOX
========================================================================

The state vector is: Y = [x, v, s, E_local, B_local, u_lattice, v_lattice, P, n_plasmon]

------------------------------------------------------------------------
1. ELECTRON POSITION (dx/dt = v)
------------------------------------------------------------------------
   dx/dt = v
   
   Simple kinematic relation.

------------------------------------------------------------------------
2. ELECTRON VELOCITY (Newton's 2nd Law + Forces)
------------------------------------------------------------------------
   m_e * dv/dt = F_binding + F_damping + F_electric + F_store + F_lattice
   
   Expanded:
   dv/dt = -omega_0^2 * x           [Binding to nucleus: spring force]
           - gamma_e * v             [Damping: collisions/radiation]
           + (e/m_e) * E_local       [Electric force from local field]
           + s * E_local             [Store-amplified response]
           + g_el * u_lattice        [Coupling to lattice motion]
   
   PHYSICAL MEANING:
   - Binding: Electron wants to stay near nucleus
   - Damping: Energy loss to environment
   - Electric: Wave pushes electron
   - Store: AMPLIFICATION - the key to threshold!
   - Lattice: Electron feels lattice vibrations

------------------------------------------------------------------------
3. INTERNAL STORE (The Mystery Variable)
------------------------------------------------------------------------
   ds/dt = -alpha * s + beta * f^2 * |v * E_local|
   
   Or in terms of power absorption:
   ds/dt = -alpha * s + beta * omega^2 * |Power_absorbed|
   
   PHYSICAL MEANING:
   - Decay (-alpha*s): Store naturally depletes
   - Charging (beta*f^2*...): Frequency-dependent energy accumulation
   
   WHAT IS s PHYSICALLY?
   Candidates:
   a) Coherent polarization of electron cloud
   b) Population of excited states
   c) Local field enhancement factor
   d) Phase coherence with driving wave
   e) Plasmon amplitude

------------------------------------------------------------------------
4. LOCAL ELECTRIC FIELD (Maxwell + Polarization)
------------------------------------------------------------------------
   dE_local/dt = c^2 * curl(B) - (1/eps_0) * dP/dt - (sigma/eps_0) * E_local
   
   In our 1D model:
   dE_local/dt = -omega_p^2 * x      [Electron displacement creates field]
                 - gamma_p * E_local  [Field damping]
                 + (dE_input/dt)      [Incoming wave]
   
   PHYSICAL MEANING:
   - Moving charges create changing fields
   - Polarization affects local field
   - External wave drives the system

------------------------------------------------------------------------
5. LOCAL MAGNETIC FIELD (Maxwell)
------------------------------------------------------------------------
   dB_local/dt = -curl(E)
   
   In 1D: dB_local/dt ~ -dE_local/dx
   
   PHYSICAL MEANING:
   - Changing E creates B
   - B affects electron through Lorentz force (small effect at optical freq)

------------------------------------------------------------------------
6. LATTICE DISPLACEMENT (Ion Motion)
------------------------------------------------------------------------
   M_ion * d^2u_lattice/dt^2 = -K * u_lattice - Gamma_L * du/dt + F_electron
   
   Or: d(v_lattice)/dt = -omega_L^2 * u_lattice 
                         - gamma_L * v_lattice
                         + (Z*e^2/M) * (x - u_lattice)  [Electron pulls on ion]
   
   PHYSICAL MEANING:
   - Ions vibrate around equilibrium
   - Electron motion couples to lattice
   - This is PHONON physics

------------------------------------------------------------------------
7. LATTICE VELOCITY
------------------------------------------------------------------------
   d(u_lattice)/dt = v_lattice
   
   Kinematic relation.

------------------------------------------------------------------------
8. POLARIZATION (Collective Electron Response)
------------------------------------------------------------------------
   dP/dt = n_e * e * v + relaxation_terms
   
   Or: dP/dt = eps_0 * chi * dE/dt - P/tau_relax
   
   PHYSICAL MEANING:
   - Polarization = collective electron displacement
   - Creates the dielectric response
   - Has its own dynamics (relaxation)

------------------------------------------------------------------------
9. PLASMON OCCUPATION (Collective Oscillations)
------------------------------------------------------------------------
   d(n_plasmon)/dt = generation - decay
                   = g_ep * |E_local|^2 * n_e - gamma_plasmon * n_plasmon
   
   PHYSICAL MEANING:
   - Plasmons are collective electron oscillations
   - They're generated by fields, decay over time
   - Could be related to our "store" variable!

========================================================================
                    THE COUPLING MATRIX
========================================================================

We can write this as: dY/dt = A * Y + B * u(t) + nonlinear_terms

Where A is the coupling matrix showing how each variable affects each other:

         x    v    s    E    B    u_L  v_L  P    n_pl
    x [  0    1    0    0    0    0    0    0    0   ]
    v [ -w0^2 -g_e s/m  e/m  0   g_el  0    0    0   ]
    s [  0   f^2   -a   f^2  0    0    0    0    0   ]
    E [ -wp^2 0    0   -gp   c    0    0   -1/e  0   ]
    B [  0    0    0   -1    0    0    0    0    0   ]
   u_L[  0    0    0    0    0    0    1    0    0   ]
   v_L[ g_el 0    0    0    0   -wL^2 -gL  0    0   ]
    P [  0   n_e  0   chi   0    0    0   -1/t  0   ]
   n_pl[ 0    0    0   g_ep  0    0    0    0   -g_pl]

The STRUCTURE of this matrix reveals the physics!
- Off-diagonal terms = couplings between subsystems
- The feedback loop through 's' creates the threshold

========================================================================
"""
        return equations
    
    def state_evolution(
        self, 
        t: float, 
        state: np.ndarray, 
        omega: float,
        E_input_amplitude: float
    ) -> np.ndarray:
        """
        Calculate the time derivative of the full state.
        
        This is the right-hand side of: dY/dt = F(Y, t)
        
        Args:
            t: Current time
            state: Current state vector [x, v, s, E, B, u_L, v_L, P, n_pl]
            omega: Angular frequency of input wave
            E_input_amplitude: Amplitude of input electric field
            
        Returns:
            Time derivative of state vector
        """
        # Unpack state
        x, v, s, E_local, B_local, u_L, v_L, P, n_pl = state
        
        # Input wave
        E_input = E_input_amplitude * np.sin(omega * t)
        
        # Effective local field (input + polarization effects)
        E_eff = E_input + E_local
        
        # 1. dx/dt = v
        dx_dt = v
        
        # 2. dv/dt (electron motion equation)
        F_binding = -self.omega_0**2 * x
        F_damping = -self.gamma_e * v
        F_electric = (self.e / self.m_e) * E_eff
        F_store = s * E_eff  # Store amplifies response
        F_lattice = self.g_el * (u_L - x)  # Coupling to lattice
        
        dv_dt = F_binding + F_damping + F_electric + F_store + F_lattice
        
        # 3. ds/dt (internal store - the key equation!)
        f = omega / (2 * np.pi)  # Frequency in Hz
        power_absorbed = abs(v * E_eff)  # Power = F * v
        ds_dt = -self.alpha * s + self.beta * (f**2) * power_absorbed
        
        # 4. dE_local/dt (field dynamics)
        omega_p = self.crystal.plasma_frequency
        gamma_p = 1e13  # Field damping
        dE_local_dt = -omega_p**2 * x - gamma_p * E_local
        
        # 5. dB_local/dt (simplified: B tracks E with phase lag)
        dB_local_dt = -E_local / (self.c * 1e-9)  # Simplified
        
        # 6. d(u_L)/dt = v_L
        du_L_dt = v_L
        
        # 7. d(v_L)/dt (lattice motion)
        M_ratio = 1836 * self.crystal.atomic_number  # Ion/electron mass ratio
        F_restoring = -self.omega_lattice**2 * u_L
        F_damping_L = -self.gamma_lattice * v_L
        F_from_electron = (self.g_el / M_ratio) * (x - u_L)
        dv_L_dt = F_restoring + F_damping_L + F_from_electron
        
        # 8. dP/dt (polarization dynamics)
        n_e = self.crystal.electron_density
        tau_relax = 1e-14  # Relaxation time
        dP_dt = n_e * self.e * v - P / tau_relax
        
        # 9. d(n_pl)/dt (plasmon dynamics)
        gamma_pl = 1e13  # Plasmon decay rate
        generation = self.g_ep * E_eff**2 * n_e
        dn_pl_dt = generation - gamma_pl * n_pl
        
        return np.array([
            dx_dt, dv_dt, ds_dt,
            dE_local_dt, dB_local_dt,
            du_L_dt, dv_L_dt,
            dP_dt, dn_pl_dt
        ])
    
    def get_coupling_matrix(self, omega: float) -> np.ndarray:
        """
        Extract the linearized coupling matrix A.
        
        Near equilibrium: dY/dt ≈ A * Y
        
        The eigenvalues of A determine stability.
        The eigenvectors show the coupled modes.
        
        Args:
            omega: Frequency (affects some couplings)
            
        Returns:
            9x9 coupling matrix
        """
        f = omega / (2 * np.pi)
        omega_p = self.crystal.plasma_frequency
        
        # Build the matrix
        A = np.zeros((9, 9))
        
        # Row 0: dx/dt = v
        A[0, 1] = 1.0
        
        # Row 1: dv/dt = ...
        A[1, 0] = -self.omega_0**2  # Binding
        A[1, 1] = -self.gamma_e     # Damping
        A[1, 2] = 1.0               # Store effect (linearized)
        A[1, 3] = self.e / self.m_e # Electric field
        A[1, 5] = self.g_el         # Lattice coupling
        
        # Row 2: ds/dt = ...
        A[2, 1] = self.beta * f**2  # Velocity coupling
        A[2, 2] = -self.alpha       # Decay
        A[2, 3] = self.beta * f**2  # Field coupling
        
        # Row 3: dE/dt = ...
        A[3, 0] = -omega_p**2       # Electron position
        A[3, 3] = -1e13             # Field damping
        
        # Row 4: dB/dt
        A[4, 3] = -1.0 / (self.c * 1e-9)
        
        # Row 5: du_L/dt = v_L
        A[5, 6] = 1.0
        
        # Row 6: dv_L/dt
        A[6, 0] = self.g_el / (1836 * self.crystal.atomic_number)
        A[6, 5] = -self.omega_lattice**2
        A[6, 6] = -self.gamma_lattice
        
        # Row 7: dP/dt
        n_e = self.crystal.electron_density
        A[7, 1] = n_e * self.e
        A[7, 7] = -1e14  # Relaxation
        
        # Row 8: dn_pl/dt
        A[8, 8] = -1e13  # Plasmon decay
        
        return A
    
    def analyze_stability(self, omega: float) -> Dict:
        """
        Analyze the stability of the system at given frequency.
        
        Eigenvalues tell us:
        - Negative real part = stable (decays)
        - Positive real part = unstable (grows) = ESCAPE!
        
        The transition from stable to unstable is the THRESHOLD.
        
        Args:
            omega: Angular frequency
            
        Returns:
            Analysis results
        """
        A = self.get_coupling_matrix(omega)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Check stability
        max_real = np.max(np.real(eigenvalues))
        is_stable = max_real < 0
        
        # Find the most unstable mode
        most_unstable_idx = np.argmax(np.real(eigenvalues))
        most_unstable_mode = eigenvectors[:, most_unstable_idx]
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'max_real_eigenvalue': max_real,
            'is_stable': is_stable,
            'most_unstable_mode': most_unstable_mode,
            'frequency': omega / (2 * np.pi)
        }
    
    def find_threshold_frequency(
        self, 
        omega_min: float = 1e14, 
        omega_max: float = 1e16,
        n_points: int = 100
    ) -> Dict:
        """
        Find the threshold frequency where system becomes unstable.
        
        This is where the photoelectric effect "turns on".
        
        Args:
            omega_min: Minimum frequency to test
            omega_max: Maximum frequency to test
            n_points: Number of frequency points
            
        Returns:
            Threshold analysis
        """
        omegas = np.linspace(omega_min, omega_max, n_points)
        max_reals = []
        
        for omega in omegas:
            analysis = self.analyze_stability(omega)
            max_reals.append(analysis['max_real_eigenvalue'])
        
        max_reals = np.array(max_reals)
        
        # Find where it crosses zero (stable -> unstable)
        crossings = np.where(np.diff(np.sign(max_reals)))[0]
        
        if len(crossings) > 0:
            # Interpolate to find threshold
            idx = crossings[0]
            # Linear interpolation
            omega_threshold = omegas[idx] + (omegas[idx+1] - omegas[idx]) * \
                              (-max_reals[idx]) / (max_reals[idx+1] - max_reals[idx])
        else:
            omega_threshold = None
        
        return {
            'frequencies': omegas / (2 * np.pi),
            'max_eigenvalues': max_reals,
            'threshold_frequency_Hz': omega_threshold / (2 * np.pi) if omega_threshold else None,
            'threshold_wavelength_nm': (3e8 / (omega_threshold / (2 * np.pi))) * 1e9 if omega_threshold else None
        }


# =============================================================================
# THE INTERDEPENDENCY ANALYSIS (Like Fox-Rabbit)
# =============================================================================

def analyze_interdependencies() -> str:
    """
    Analyze how different parts of the system depend on each other.
    
    Like in predator-prey models:
    - Foxes eat rabbits -> rabbit population affects fox growth
    - Rabbits are eaten by foxes -> fox population affects rabbit decline
    
    In our system:
    - Electron motion creates fields
    - Fields drive electron motion
    - Store amplifies the response
    - etc.
    """
    analysis = """
========================================================================
            INTERDEPENDENCY ANALYSIS: WHAT AFFECTS WHAT
========================================================================

Like the fox-rabbit predator-prey model, our system has coupled
dynamics where each quantity affects others.

PREDATOR-PREY ANALOGY:
    dR/dt = a*R - b*R*F     (Rabbits grow, get eaten by foxes)
    dF/dt = c*R*F - d*F     (Foxes eat rabbits, die naturally)

OUR SYSTEM:
    Multiple coupled subsystems with feedback loops.

------------------------------------------------------------------------
                        DEPENDENCY GRAPH
------------------------------------------------------------------------

    INCOMING WAVE
         |
         v
    +--------------------+
    | LOCAL FIELD E(t)   |<---------+
    +--------------------+          |
         |                          |
         v                          |
    +--------------------+     +--------------------+
    | ELECTRON x(t),v(t) |<--->| STORE s(t)         |
    +--------------------+     | (THE AMPLIFIER)    |
         |    ^                +--------------------+
         |    |                     ^
         v    |                     |
    +--------------------+          |
    | LATTICE u(t)       |----------+
    | (PHONONS)          |
    +--------------------+
         |
         v
    +--------------------+
    | POLARIZATION P(t)  |
    +--------------------+
         |
         v
    +--------------------+
    | PLASMONS n(t)      |
    +--------------------+

------------------------------------------------------------------------
                    FEEDBACK LOOPS
------------------------------------------------------------------------

LOOP 1: MAIN AMPLIFICATION LOOP (Creates threshold!)
    E_field -> electron_motion -> store_charges -> amplifies_response -> E_field
    
    This is POSITIVE FEEDBACK when f > f_threshold
    This is NEGATIVE FEEDBACK when f < f_threshold
    
    The f^2 factor determines which regime we're in!

LOOP 2: FIELD-POLARIZATION LOOP
    E_field -> polarization -> modifies_E_field
    
    This is the dielectric response.

LOOP 3: ELECTRON-LATTICE LOOP
    electron_moves -> pulls_on_ions -> ions_move -> affects_electron
    
    This creates phonon-assisted processes.

LOOP 4: COLLECTIVE MODE LOOP
    field_fluctuations -> plasmon_excitation -> field_modification
    
    This is collective electron behavior.

------------------------------------------------------------------------
                    ENERGY FLOW ANALYSIS
------------------------------------------------------------------------

Energy enters through: INCOMING WAVE
Energy stored in:
    - Electron kinetic energy (1/2 m v^2)
    - Electron potential energy (1/2 m omega_0^2 x^2)
    - Field energy (epsilon_0 E^2 / 2)
    - Lattice vibration energy
    - STORE s (internal coherent excitation)
    - Plasmon energy

Energy leaves through:
    - Damping (heat)
    - Radiation (scattered wave)
    - ESCAPE (electron leaves!)

THE THRESHOLD CONDITION:
    Energy_in(from wave) > Energy_out(to damping)
    This requires: f > f_threshold

------------------------------------------------------------------------
                    THE STORE: CENTRAL MYSTERY
------------------------------------------------------------------------

The store variable s is the KEY to understanding the threshold.

What s DOES mathematically:
    - Amplifies the electron's response to the field
    - Is charged by frequency-dependent power absorption
    - Decays naturally over time

What s COULD BE physically:
    
    1. COHERENT SUPERPOSITION
       s ~ |psi_excited|^2
       Population of excited electronic states that hasn't decohered
       
    2. LOCAL FIELD ENHANCEMENT
       s ~ (E_local / E_applied) - 1
       How much the local field exceeds the applied field
       
    3. PHASE COHERENCE
       s ~ <cos(phi_electron - phi_wave)>
       Correlation between electron phase and wave phase
       
    4. PLASMON AMPLITUDE
       s ~ sqrt(n_plasmon)
       Amplitude of collective oscillation mode
       
    5. POLARIZATION MEMORY
       s ~ integral(P dt)
       Accumulated polarization effect

Each interpretation makes TESTABLE PREDICTIONS!

========================================================================
"""
    return analysis


def get_testable_predictions() -> str:
    """
    List testable predictions from the model.
    """
    predictions = """
========================================================================
                    TESTABLE PREDICTIONS
========================================================================

Our wave-only model makes specific predictions that differ from the
standard photon picture. Testing these could validate or refine the model.

------------------------------------------------------------------------
1. TEMPERATURE DEPENDENCE
------------------------------------------------------------------------
   Prediction: Threshold frequency INCREASES with temperature
   
   Reason: Higher T -> more phonons -> higher alpha (decoherence)
           Higher alpha -> higher threshold (harder to maintain store)
   
   Photon model: Threshold should be temperature-independent
                 (photon energy doesn't change with T)
   
   TEST: Measure threshold vs temperature for clean metal surface

------------------------------------------------------------------------
2. SURFACE CLEANLINESS EFFECT
------------------------------------------------------------------------
   Prediction: Cleaner surfaces have SHARPER thresholds
   
   Reason: Surface defects -> higher damping -> broader transition
   
   TEST: Compare threshold sharpness for UHV-prepared vs air-exposed

------------------------------------------------------------------------
3. CRYSTAL ORIENTATION DEPENDENCE
------------------------------------------------------------------------
   Prediction: Different crystal faces have different thresholds
   
   Reason: Different orientations -> different electron dynamics
           -> different alpha, beta parameters
   
   TEST: Measure threshold for (100), (110), (111) faces

------------------------------------------------------------------------
4. COHERENCE LENGTH EFFECT
------------------------------------------------------------------------
   Prediction: Larger illumination area -> lower threshold
   
   Reason: More atoms participate -> larger beta (collective effect)
   
   TEST: Measure threshold vs beam spot size

------------------------------------------------------------------------
5. PULSE DURATION EFFECT
------------------------------------------------------------------------
   Prediction: For very short pulses, threshold should INCREASE
   
   Reason: Need time to build up store; ultrafast pulses don't allow this
   
   Photon model: Threshold unchanged (only photon energy matters)
   
   TEST: Measure threshold vs pulse duration (fs to ns)

------------------------------------------------------------------------
6. TWO-FREQUENCY EXPERIMENT
------------------------------------------------------------------------
   Prediction: Two beams at f1 < f_th and f2 < f_th could together
               cause emission if f1 + f2 > effective_threshold
   
   Reason: Both contribute to store charging; combined effect exceeds decay
   
   TEST: Illuminate with two sub-threshold frequencies simultaneously

------------------------------------------------------------------------
7. INTENSITY-DEPENDENT THRESHOLD (subtle effect)
------------------------------------------------------------------------
   Prediction: Very slight threshold decrease at extreme intensities
   
   Reason: Nonlinear terms become relevant; multi-pathway charging
   
   Photon model: Threshold strictly independent of intensity
   
   TEST: Precision measurement of threshold vs intensity over 6+ decades

------------------------------------------------------------------------
8. MAGNETIC FIELD EFFECT
------------------------------------------------------------------------
   Prediction: Strong B-field should affect threshold
   
   Reason: Lorentz force modifies electron trajectory -> affects coupling
   
   TEST: Measure threshold in strong magnetic field (several Tesla)

========================================================================
"""
    return predictions


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    # Create a copper crystal region
    crystal = CrystalRegion(
        lattice_constant_m=3.6e-10,
        region_size_m=1e-8,
        atomic_number=29,
        free_electrons_per_atom=1.0
    )
    
    print(f"Crystal region: {crystal.n_atoms} atoms")
    print(f"Electron density: {crystal.electron_density:.2e} m^-3")
    print(f"Plasma frequency: {crystal.plasma_frequency/(2*np.pi):.2e} Hz")
    print()
    
    # Create the equation system
    equations = CrystalLatticeEquations(crystal)
    
    # Print the equations
    print(equations.derive_equations())
    
    # Print interdependency analysis
    print(analyze_interdependencies())
    
    # Print testable predictions
    print(get_testable_predictions())

