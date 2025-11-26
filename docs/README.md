# Light Wave Photoelectric Effect Simulation

## Overview

This project explores whether the photoelectric effect can emerge from **pure wave mechanics** with an internal resonance mechanism, rather than requiring particle-based (photon) explanations.

The core idea: if we model an atom with an internal "store" or "gain" variable that responds to incoming electromagnetic waves in a **frequency-dependent** way, can we reproduce the key characteristics of the photoelectric effect?

## The Model

### State Variables

We model a 1D "toy atom" with three state variables:

- **x(t)**: Electron displacement from equilibrium (position)
- **v(t)**: Electron velocity
- **s(t)**: Internal store/gain (active response mechanism)

### Input Wave

```
u(t) = A·sin(2πft)
```

where `A` is amplitude and `f` is frequency.

### Governing Equations

```
dx/dt = v
dv/dt = -ω₀²x - d·v + s·u(t)
ds/dt = -α·s + β·f²·x·u(t)
```

### Physical Interpretation

| Term | Meaning |
|------|---------|
| `-ω₀²x` | Binding force (electron-nucleus spring) |
| `-d·v` | Damping (energy dissipation) |
| `s·u(t)` | Amplified wave driving (internal gain) |
| `-α·s` | Store relaxation (store decay) |
| `β·f²·x·u` | **Store charging (frequency-dependent!)** |

### The Key Insight: Frequency Threshold

The **f²** term in the store equation is critical:

- At **low frequencies**: The store charging rate `β·f²·⟨xu⟩` cannot overcome the decay rate `α·s`
  → Store decays to zero → Electron stays bound

- At **high frequencies**: The charging rate exceeds decay
  → Store grows → Driving amplifies → Electron escapes

This creates a **frequency threshold** analogous to the photoelectric effect!

## Project Structure

```
light_wave_photoelectric_effect/
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── physics.py            # Core physics model (ODEs)
│   ├── integrator.py         # RK4 numerical integrator
│   ├── simulation.py         # Simulation engine
│   ├── analysis.py           # Analysis tools
│   └── visualization.py      # Plotting functions
├── config/
│   ├── __init__.py
│   └── parameters.py         # Parameter configurations
├── results/                   # Output directory
├── docs/
│   └── README.md             # This file
└── explore.ipynb             # Interactive Jupyter notebook
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test

```bash
python main.py --experiment quick
```

### Frequency Sweep

```bash
python main.py --experiment sweep
```

### Amplitude Independence Test

```bash
python main.py --experiment amplitude
```

### Full Photoelectric Comparison

```bash
python main.py --experiment full
```

### Different Parameter Sets

```bash
python main.py --params photoelectric
python main.py --params high_sensitivity
python main.py --params strong_binding
```

## Key Results to Look For

### 1. Frequency Threshold Exists
Below a critical frequency, electrons never escape. Above it, they do.

### 2. Amplitude Independence
The threshold frequency should be (nearly) independent of wave amplitude.
This is a key characteristic of the real photoelectric effect!

### 3. Escape Time Behavior
Above threshold, escape time should decrease as frequency increases.

## Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Natural frequency | ω₀ | 1.0 | Binding strength |
| Damping | d | 0.01 | Energy loss rate |
| Store decay | α | 0.5 | How fast store depletes |
| Coupling | β | 1.0 | Motion-store coupling |
| Amplitude | A | 0.01 | Wave amplitude |
| Escape threshold | X_esc | 5.0 | Position for escape |

## Tuning Guide

To achieve photoelectric-like behavior:

1. **α/β ratio** determines approximate threshold: `f_th ~ √(α/β)`
2. **Low damping** allows store to build up
3. **β large enough** for store to charge above threshold
4. **α not too small** so below-threshold decays properly

## Recent Developments

### 1. Crystal Lattice Black Box Model

The project now includes a comprehensive model of the crystal lattice as a "black box" containing:
- Electron dynamics (position, velocity)
- Internal store/coherence mechanism  
- Local electromagnetic fields
- Lattice vibrations (phonons)
- Collective excitations (plasmons)
- Polarization dynamics

See `src/crystal_lattice_model.py` for the full 9-variable state-space model.

### 2. Complete System of Equations

The model now includes multiple equation types working together:
- **Linear DEs**: Kinematic relations (dx/dt = v)
- **Harmonic oscillator**: Damped driven oscillation
- **Polynomial (f²)**: The KEY threshold-creating term
- **Product terms**: Coupling between subsystems (s·E, v·E)
- **Nonlinear terms**: Power absorption |v·E|

### 3. Compton Scattering Extension

Wave emission from accelerating electrons is now modeled! See `src/compton.py`:
- Larmor radiation from accelerating charges
- Doppler-based frequency shifts
- Store-modulated output waves
- Analysis of Compton-like wavelength shifts

### 4. Multi-Frequency Driving

The model now supports multiple simultaneous driving frequencies:
- Test whether two sub-threshold frequencies together can cause escape
- This is a KEY testable prediction that differs from the photon model!

### 5. Material Comparison

Map model parameters to real materials (Na, K, Cu, Au, Cs) and compare:
- Work functions → binding frequency ω₀
- Coherence times → store decay α
- Electron density → coupling β

## Testable Predictions

Our wave-only model makes several predictions that **differ from the photon model**:

| Prediction | Wave Model | Photon Model |
|------------|------------|--------------|
| **Temperature dependence** | Threshold increases with T | No change |
| **Pulse duration** | Threshold increases for ultrashort pulses | No change |
| **Two sub-threshold freqs** | Could cause emission together | Never causes emission |
| **Surface cleanliness** | Affects threshold sharpness | No effect |
| **Crystal orientation** | Different faces = different thresholds | Orientation-independent |
| **Extreme intensity** | Slight threshold decrease | Strictly independent |

These predictions could be tested experimentally to validate or refine the model.

## Future Directions

- Extend to 3D crystal lattice geometry
- Include relativistic corrections for high frequencies
- Model specific materials with ab-initio parameters
- Investigate connections to stochastic electrodynamics (SED)
- Explore implications for light propagation medium

## Physics Notes

This model explores the possibility that quantum-mechanical phenomena like the photoelectric effect might emerge from classical wave mechanics with appropriate internal structure. The key insight is that:

1. **Frequency-dependent coupling** can create threshold behavior
2. **Internal active elements** (the store) can amplify responses
3. **Nonlinear dynamics** can lead to bifurcations (escape vs. bound)

This doesn't claim to replace quantum mechanics, but explores what aspects of photoelectric behavior can emerge from wave-only models.

## Physics Analysis

See [PHYSICS_ANALYSIS.md](PHYSICS_ANALYSIS.md) for a deep dive into:
- Why the frequency threshold emerges
- Physical interpretation of each parameter
- What the mysterious "store" variable might represent
- Connection to real material properties

## The Key Insight

> **"The quantum is not in the light, but in the response."**

Our model suggests that the photoelectric threshold emerges from MATERIAL PROPERTIES, not from light being quantized. The internal dynamics of matter (the store variable, coherence, coupling) determine whether and when electrons escape.

## License

MIT License - Free for research and educational use.

