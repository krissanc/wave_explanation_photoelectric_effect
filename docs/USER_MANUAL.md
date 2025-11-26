# User Manual: Wave-Based Photoelectric Effect Explorer

## Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/light_wave_photoelectric_effect.git
cd light_wave_photoelectric_effect

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Interactive Notebook

The easiest way to explore the model is through the Jupyter notebook:

```bash
jupyter notebook explore.ipynb
```

Or in VS Code/Cursor:
1. Open `explore.ipynb`
2. Run cells sequentially (Shift+Enter)

### 3. Running from Command Line

```bash
# Quick test - single simulation
python main.py --experiment quick

# Frequency sweep - find threshold
python main.py --experiment sweep

# Test amplitude independence
python main.py --experiment amplitude

# Full photoelectric comparison
python main.py --experiment full
```

---

## Understanding the Model

### The Core Idea

Light hits a crystal lattice "black box". Inside this box:
- Electrons oscillate in response to the wave
- An internal "store" mechanism accumulates energy
- The store charging rate depends on **frequency squared (f²)**
- This creates a frequency threshold - just like the photoelectric effect!

### State Variables

| Variable | Symbol | Meaning |
|----------|--------|---------|
| Position | x(t) | Electron displacement from equilibrium |
| Velocity | v(t) | Electron velocity |
| Store | s(t) | Internal gain/coherence mechanism |

### The Key Equations

```
dx/dt = v                              [Kinematics]
dv/dt = -ω₀²x - γv + s·u(t)           [Driven oscillator with gain]
ds/dt = -αs + β·f²·|v·u|              [Store dynamics - THE KEY!]
```

The **f²** term creates the threshold:
- Low f → charging < decay → no escape
- High f → charging > decay → escape!

---

## Notebook Sections Guide

### Section 1-5: Basic Model

Run these to understand the fundamental model:
- **Section 1**: Set up parameters
- **Section 2**: Frequency sweep to find threshold
- **Section 3**: Test amplitude independence (key photoelectric property!)
- **Section 4**: Parameter exploration
- **Section 5**: Summary

### Section 6: Crystal Lattice Black Box

The complete physical model with 9 state variables:
- Electron dynamics (x, v)
- Internal store (s)
- Local fields (E, B)
- Lattice vibrations (u_L, v_L)
- Polarization (P)
- Plasmons (n_pl)

**Key outputs:**
- Coupling matrix visualization
- Eigenvalue stability analysis
- Threshold from linear stability

### Section 7: Wave Emission (Compton Foundation)

Accelerating electrons radiate! This section shows:
- Input wave → electron oscillation → emitted wave
- The foundation for explaining Compton scattering with waves

### Section 8: Multi-Frequency Driving

**Critical test!** Can two sub-threshold frequencies together cause emission?

This is a testable prediction that differs from the photon model!

### Section 9: Different Materials

Compare model predictions for real metals:
- Cesium (Cs): lowest work function
- Sodium (Na), Potassium (K): alkali metals
- Copper (Cu), Gold (Au): transition metals

### Section 10-11: Predictions and Summary

- Testable predictions that differ from photon model
- Complete equation summary
- The revolutionary insight

---

## Parameter Tuning Guide

### Main Parameters

| Parameter | Symbol | Effect |
|-----------|--------|--------|
| omega0 | ω₀ | Binding strength (higher = harder to escape) |
| damping | γ | Energy loss (lower = easier buildup) |
| alpha | α | Store decay rate (higher = higher threshold) |
| beta | β | Frequency coupling (higher = lower threshold) |
| amplitude | A | Wave amplitude (threshold is independent!) |

### Threshold Formula

```
f_threshold ≈ √(α/β)
```

### Tuning for Photoelectric Behavior

For clear threshold behavior:
1. Set damping very low (0.0001)
2. Set beta high (100)
3. Set alpha moderate (0.05)
4. Amplitude should NOT affect threshold

Example good parameters:
```python
params = PhysicsParameters(
    omega0=1.0,
    damping=0.0001,
    alpha=0.05,
    beta=100.0,
    amplitude=0.2,
    escape_position=5.0
)
```

---

## Running Specific Experiments

### Experiment 1: Find the Threshold

```python
from src.simulation import create_default_simulation
from src.analysis import find_threshold_frequency

sim = create_default_simulation(
    omega0=1.0, damping=0.0001, alpha=0.05, beta=100.0,
    amplitude=0.2, escape_threshold=5.0, t_max=150.0
)

frequencies = np.linspace(0.05, 0.3, 40)
results = [sim.run(f) for f in frequencies]
threshold = find_threshold_frequency(results)

print(f"Threshold: {threshold.threshold_frequency}")
```

### Experiment 2: Test Amplitude Independence

```python
amplitudes = [0.001, 0.005, 0.01, 0.05, 0.1]

for amp in amplitudes:
    sim = create_default_simulation(..., amplitude=amp)
    # Find threshold for each amplitude
    # They should all be the same!
```

### Experiment 3: Multi-Frequency Test

```python
from explore import MultiFrequencyAtomModel

model = MultiFrequencyAtomModel(...)

# Single frequencies (below threshold)
result_f1 = model.simulate([0.08], [0.2])
result_f2 = model.simulate([0.09], [0.2])

# Both together
result_combined = model.simulate([0.08, 0.09], [0.2, 0.2])

# Does combined escape when individuals don't?
```

### Experiment 4: Material Comparison

```python
from src.physics_interpretation import MATERIALS, material_to_model_parameters

for name, material in MATERIALS.items():
    params = material_to_model_parameters(material)
    # Run simulation with these parameters
    # Compare to known work function
```

---

## Understanding the Output

### Simulation Result Fields

```python
result.times       # Time array
result.positions   # Electron position x(t)
result.velocities  # Electron velocity v(t)
result.stores      # Store value s(t)
result.escaped     # True if electron escaped
result.escape_time # When escape occurred (if it did)
result.frequency   # Input frequency
```

### Threshold Analysis Fields

```python
threshold.threshold_frequency    # The threshold value
threshold.threshold_uncertainty  # Error estimate
threshold.below_threshold       # Frequencies that stayed bound
threshold.above_threshold       # Frequencies that escaped
```

---

## Troubleshooting

### No Escape Detected

If electrons never escape:
- Increase simulation time (t_max)
- Decrease damping
- Increase beta
- Decrease alpha
- Increase amplitude (though threshold shouldn't change!)

### Threshold Varies with Amplitude

If threshold changes with amplitude:
- Decrease damping further
- Check that alpha/beta ratio is appropriate
- Ensure simulation time is long enough

### Numerical Instability

If you see NaN or very large values:
- Decrease time step (dt)
- Decrease amplitude
- Check for parameter combinations causing runaway

---

## The Big Picture

### What This Model Shows

1. **Frequency threshold can emerge from wave mechanics** - no photons needed!
2. **Amplitude independence arises naturally** from the internal dynamics
3. **The "quantum" might be in matter, not light**

### Testable Predictions

| Our Model Predicts | Photon Model Predicts |
|-------------------|----------------------|
| Threshold increases with temperature | No change |
| Threshold affected by pulse duration | No change |
| Two sub-threshold frequencies together might work | Never works |
| Crystal orientation matters | Doesn't matter |

### The Revolutionary Claim

> **"The quantum is not a property of light at all, but rather how MATTER organizes its internal degrees of freedom!"**

---

## API Reference

### Main Classes

```python
# Core atom model
from src.physics import AtomModel, EscapeDetector

# Simulation engine
from src.simulation import Simulation, create_default_simulation

# Analysis tools
from src.analysis import find_threshold_frequency, analyze_escape_times

# Visualization
from src.visualization import plot_time_series, plot_frequency_sweep

# Crystal lattice model
from src.crystal_lattice_model import CrystalRegion, CrystalLatticeEquations

# Compton scattering
from src.compton import ComptonAtomModel, ComptonSimulation

# Material properties
from src.physics_interpretation import MATERIALS, material_to_model_parameters
```

### Key Functions

```python
# Create a simulation
sim = create_default_simulation(omega0, damping, alpha, beta, amplitude, ...)

# Run simulation at a frequency
result = sim.run(frequency)

# Find threshold from results
threshold = find_threshold_frequency(results_list)

# Material to model parameters
params = material_to_model_parameters(MATERIALS['copper'])
```

---

## Contributing

This is an exploratory scientific project. Contributions welcome:
- New testable predictions
- Connections to stochastic electrodynamics
- 3D extensions
- Comparison with experimental data

---

## Citation

If you use this work in research, please cite:

```
Wave-Based Photoelectric Effect Model
Exploring quantum-like threshold behavior from classical wave mechanics
https://github.com/YOUR_USERNAME/light_wave_photoelectric_effect
```

---

## License

MIT License - Free for research and educational use.

