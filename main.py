"""
Light Wave Photoelectric Effect Simulation
==========================================

Main entry point for running experiments that investigate whether
the photoelectric effect can emerge from pure wave mechanics with
an internal resonance/gain mechanism.

This simulation implements a toy 1D "atom" model where:
- An electron is bound by a spring-like potential
- An electromagnetic wave drives the electron
- An internal "store" amplifies the response in a frequency-dependent way
- Above a threshold frequency, the store grows → electron escapes
- Below threshold, the store decays → electron stays bound

The goal is to show that:
1. There IS a frequency threshold (photoelectric effect)
2. The threshold is nearly INDEPENDENT of wave amplitude
3. Escape time depends mainly on frequency, not amplitude

Usage:
    python main.py                      # Run default frequency sweep
    python main.py --experiment quick   # Quick test run
    python main.py --experiment full    # Comprehensive analysis
    python main.py --params high_sensitivity  # Use different parameter set
"""

import numpy as np
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.physics import AtomModel, EscapeDetector
from src.simulation import Simulation, SimulationConfig, create_default_simulation
from src.analysis import (
    find_threshold_frequency, 
    analyze_escape_times,
    analyze_amplitude_independence,
    compare_to_photoelectric_effect
)
from src.visualization import (
    plot_frequency_sweep,
    plot_time_series,
    plot_phase_portrait,
    plot_comprehensive_analysis,
    plot_amplitude_comparison,
    plot_energy_evolution
)
from src.compton import (
    ComptonAtomModel,
    ComptonSimulation,
    run_compton_experiment,
    analyze_compton_shift,
    plot_compton_results
)
from config.parameters import (
    PhysicsParameters,
    SimulationParameters,
    SweepParameters,
    get_parameter_set,
    estimate_threshold_frequency,
    DEFAULT_PHYSICS,
    PHOTOELECTRIC_TUNED,
    HIGH_SENSITIVITY
)


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None):
    """Configure logging for the simulation."""
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_quick_test(show_plots: bool = True):
    """
    Quick test to verify the simulation works.
    
    Runs a small frequency sweep and shows basic results.
    """
    logger = logging.getLogger("quick_test")
    logger.info("=" * 60)
    logger.info("QUICK TEST: Verifying simulation setup")
    logger.info("=" * 60)
    
    # Create simulation with TUNED parameters that show clear threshold
    # These parameters produce a threshold around f ≈ 0.12
    # Key: very low damping, low alpha, high beta
    sim = create_default_simulation(
        omega0=1.0,
        damping=0.0001,    # Near-zero damping allows oscillation growth
        alpha=0.05,        # Slow store decay
        beta=100.0,        # Strong frequency-dependent coupling
        amplitude=0.2,     # Moderate amplitude
        escape_threshold=5.0,
        t_max=150.0,       # Allow time for dynamics to develop
        dt=0.0005          # Small timestep for accuracy
    )
    
    # Run frequency sweep spanning the threshold
    frequencies = np.linspace(0.05, 0.5, 15)
    
    logger.info(f"Running frequency sweep: {len(frequencies)} frequencies")
    logger.info(f"Frequency range: [{frequencies[0]:.2f}, {frequencies[-1]:.2f}]")
    
    results = []
    for i, f in enumerate(frequencies):
        logger.info(f"[{i+1}/{len(frequencies)}] Simulating f = {f:.3f}...")
        result = sim.run(f)
        results.append(result)
        
        status = "ESCAPE" if result.escaped else "NO ESCAPE"
        time_str = f"t={result.escape_time:.2f}" if result.escaped else "---"
        logger.info(f"  -> {status} ({time_str})")
    
    # Find threshold
    threshold = find_threshold_frequency(results)
    logger.info("")
    logger.info(f"THRESHOLD FREQUENCY: {threshold.threshold_frequency:.4f}")
    logger.info(f"  Uncertainty: ±{threshold.threshold_uncertainty:.4f}")
    logger.info(f"  Confidence: {threshold.confidence:.2f}")
    
    # Summary
    n_escaped = sum(1 for r in results if r.escaped)
    logger.info("")
    logger.info(f"Summary: {n_escaped}/{len(results)} simulations resulted in escape")
    
    if show_plots:
        plot_frequency_sweep(results, threshold.threshold_frequency, show=True)
    
    return results, threshold


def run_frequency_sweep_experiment(
    params: PhysicsParameters = DEFAULT_PHYSICS,
    sweep: SweepParameters = None,
    show_plots: bool = True,
    save_results: bool = True
):
    """
    Run a comprehensive frequency sweep experiment.
    
    This is the main experiment for demonstrating the frequency threshold.
    """
    logger = logging.getLogger("freq_sweep")
    logger.info("=" * 60)
    logger.info("FREQUENCY SWEEP EXPERIMENT")
    logger.info("=" * 60)
    
    if sweep is None:
        sweep = SweepParameters(
            freq_min=0.1,
            freq_max=3.0,
            freq_steps=40
        )
    
    # Log parameters
    logger.info(params.describe())
    
    # Estimate threshold
    estimated_thresh = estimate_threshold_frequency(params)
    logger.info(f"Estimated threshold (analytical): {estimated_thresh:.3f}")
    
    # Create simulation
    atom = AtomModel(
        omega0=params.omega0,
        damping=params.damping,
        alpha=params.alpha,
        beta=params.beta,
        amplitude=params.amplitude
    )
    detector = EscapeDetector(position_threshold=params.escape_position)
    config = SimulationConfig(t_max=200.0, dt=0.001, store_every=10)
    sim = Simulation(atom, detector, config)
    
    # Generate frequency array
    frequencies = np.linspace(sweep.freq_min, sweep.freq_max, sweep.freq_steps)
    
    logger.info(f"Sweeping {len(frequencies)} frequencies: [{sweep.freq_min}, {sweep.freq_max}]")
    logger.info("")
    
    # Run sweep with progress
    results = []
    for i, f in enumerate(frequencies):
        result = sim.run(f)
        results.append(result)
        
        # Progress logging
        status = "ESCAPE" if result.escaped else "BOUND"
        time_str = f"t={result.escape_time:.1f}" if result.escaped else "---"
        logger.info(
            f"[{i+1:3d}/{len(frequencies)}] f={f:.4f} → {status:6s} ({time_str})"
        )
    
    # Analyze results
    logger.info("")
    logger.info("=" * 40)
    logger.info("ANALYSIS")
    logger.info("=" * 40)
    
    threshold = find_threshold_frequency(results)
    escape_analysis = analyze_escape_times(results)
    
    logger.info(f"Threshold Frequency: {threshold.threshold_frequency:.4f}")
    logger.info(f"  Uncertainty: ±{threshold.threshold_uncertainty:.4f}")
    logger.info(f"  Method: {threshold.method}")
    logger.info(f"  Confidence: {threshold.confidence:.2f}")
    
    n_escaped = len(threshold.frequencies_above)
    n_bound = len(threshold.frequencies_below)
    logger.info(f"Escaped: {n_escaped}, Bound: {n_bound}")
    
    if 'min_escape_time' in escape_analysis:
        logger.info(f"Escape time range: [{escape_analysis['min_escape_time']:.2f}, "
                   f"{escape_analysis['max_escape_time']:.2f}]")
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_results:
        # Save summary
        summary_file = results_dir / f"freq_sweep_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Frequency Sweep Results\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(params.describe())
            f.write(f"\nThreshold: {threshold.threshold_frequency:.4f} ± {threshold.threshold_uncertainty:.4f}\n")
            f.write(f"Escaped: {n_escaped}, Bound: {n_bound}\n\n")
            f.write("Frequency | Escaped | Escape Time\n")
            f.write("-" * 40 + "\n")
            for r in results:
                t_str = f"{r.escape_time:.2f}" if r.escape_time else "---"
                f.write(f"{r.frequency:8.4f} | {'Yes' if r.escaped else 'No ':3s}     | {t_str}\n")
        logger.info(f"Results saved to {summary_file}")
    
    # Plots
    if show_plots:
        logger.info("")
        logger.info("Generating plots...")
        
        plot_frequency_sweep(
            results, 
            threshold.threshold_frequency,
            save_path=results_dir / f"freq_sweep_{timestamp}.png" if save_results else None,
            show=True
        )
        
        plot_comprehensive_analysis(
            results,
            threshold.threshold_frequency,
            save_path=results_dir / f"comprehensive_{timestamp}.png" if save_results else None,
            show=True
        )
        
        # Show example time series for escaped and bound
        escaped_results = [r for r in results if r.escaped]
        bound_results = [r for r in results if not r.escaped]
        
        if escaped_results:
            plot_time_series(
                escaped_results[0],
                save_path=results_dir / f"escaped_timeseries_{timestamp}.png" if save_results else None,
                show=True
            )
        
        if bound_results:
            plot_time_series(
                bound_results[-1],  # Highest frequency that didn't escape
                save_path=results_dir / f"bound_timeseries_{timestamp}.png" if save_results else None,
                show=True
            )
    
    return results, threshold


def run_amplitude_independence_test(
    params: PhysicsParameters = DEFAULT_PHYSICS,
    show_plots: bool = True
):
    """
    Test whether threshold is independent of amplitude.
    
    This is a KEY test for photoelectric-like behavior:
    - In the real photoelectric effect, threshold depends on frequency, not intensity
    - If our model shows amplitude independence, it captures this key physics
    """
    logger = logging.getLogger("amp_test")
    logger.info("=" * 60)
    logger.info("AMPLITUDE INDEPENDENCE TEST")
    logger.info("=" * 60)
    logger.info("Testing whether threshold depends on wave amplitude...")
    logger.info("")
    
    # Test multiple amplitudes with TUNED parameters
    # Using parameters that show clear threshold behavior
    amplitudes = [0.05, 0.1, 0.2, 0.3, 0.4]  # Range around working amplitude
    frequencies = np.linspace(0.05, 0.4, 20)  # Span the threshold region
    
    results_by_amplitude = {}
    
    for amp in amplitudes:
        logger.info(f"Testing amplitude A = {amp}")
        
        # Use TUNED parameters that produce threshold behavior
        atom = AtomModel(
            omega0=1.0,
            damping=0.0001,   # Very low damping
            alpha=0.05,       # Low store decay
            beta=100.0,       # Strong coupling
            amplitude=amp
        )
        detector = EscapeDetector(position_threshold=5.0)
        config = SimulationConfig(t_max=150.0, dt=0.0005)
        sim = Simulation(atom, detector, config)
        
        results = []
        for f in frequencies:
            result = sim.run(f)
            results.append(result)
        
        results_by_amplitude[amp] = results
        
        # Find threshold for this amplitude
        thresh = find_threshold_frequency(results)
        n_escaped = sum(1 for r in results if r.escaped)
        logger.info(f"  -> Threshold: {thresh.threshold_frequency:.4f}, "
                   f"Escaped: {n_escaped}/{len(results)}")
    
    # Analyze amplitude independence
    logger.info("")
    logger.info("=" * 40)
    logger.info("AMPLITUDE INDEPENDENCE ANALYSIS")
    logger.info("=" * 40)
    
    analysis = analyze_amplitude_independence(results_by_amplitude)
    
    if 'thresholds' in analysis:
        logger.info(f"Mean threshold: {analysis['threshold_mean']:.4f}")
        logger.info(f"Std deviation: {analysis['threshold_std']:.4f}")
        logger.info(f"Variation: {analysis['threshold_variation']:.2%}")
        logger.info(f"Amplitude range tested: {analysis['amplitude_range']:.1f}x")
        
        if analysis.get('power_law_slope') is not None:
            logger.info(f"Power law slope: {analysis['power_law_slope']:.4f}")
            logger.info("  (slope ~ 0 means amplitude-independent)")
        
        if analysis['is_amplitude_independent']:
            logger.info("")
            logger.info("[OK] RESULT: Threshold is AMPLITUDE INDEPENDENT")
            logger.info("  This matches photoelectric effect behavior!")
        else:
            logger.info("")
            logger.info("[!] RESULT: Threshold varies with amplitude")
            logger.info("  Need to tune parameters for better photoelectric behavior")
    
    if show_plots:
        plot_amplitude_comparison(results_by_amplitude, show=True)
    
    return results_by_amplitude, analysis


def run_full_photoelectric_comparison(show_plots: bool = True):
    """
    Run comprehensive comparison with photoelectric effect.
    
    Tests all key characteristics:
    1. Frequency threshold exists
    2. Threshold is amplitude-independent
    3. Escape time decreases with frequency above threshold
    """
    logger = logging.getLogger("full_comparison")
    logger.info("=" * 60)
    logger.info("FULL PHOTOELECTRIC EFFECT COMPARISON")
    logger.info("=" * 60)
    logger.info("")
    
    # Use tuned parameters
    params = PHOTOELECTRIC_TUNED
    logger.info("Using PHOTOELECTRIC_TUNED parameters")
    logger.info(params.describe())
    
    # 1. Frequency sweep
    logger.info("Step 1: Frequency Sweep")
    logger.info("-" * 40)
    
    frequencies = np.linspace(0.1, 3.0, 50)
    
    atom = AtomModel(
        omega0=params.omega0,
        damping=params.damping,
        alpha=params.alpha,
        beta=params.beta,
        amplitude=params.amplitude
    )
    detector = EscapeDetector(position_threshold=params.escape_position)
    config = SimulationConfig(t_max=200.0, dt=0.001, store_every=10)
    sim = Simulation(atom, detector, config)
    
    freq_results = sim.run_frequency_sweep(frequencies)
    threshold = find_threshold_frequency(freq_results)
    
    logger.info(f"Threshold: {threshold.threshold_frequency:.4f}")
    
    # 2. Amplitude sweep
    logger.info("")
    logger.info("Step 2: Amplitude Independence Test")
    logger.info("-" * 40)
    
    amplitudes = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results_by_amplitude = {}
    
    for amp in amplitudes:
        sim.atom.amplitude = amp
        results = sim.run_frequency_sweep(frequencies)
        results_by_amplitude[amp] = results
        t = find_threshold_frequency(results)
        logger.info(f"A={amp:.4f} → f_th={t.threshold_frequency:.4f}")
    
    # Reset amplitude
    sim.atom.amplitude = params.amplitude
    
    # 3. Full comparison
    logger.info("")
    logger.info("Step 3: Photoelectric Effect Comparison")
    logger.info("-" * 40)
    
    comparison = compare_to_photoelectric_effect(freq_results, results_by_amplitude)
    logger.info(comparison.summary())
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(results_dir / f"photoelectric_comparison_{timestamp}.txt", 'w') as f:
        f.write(comparison.summary())
    
    if show_plots:
        plot_comprehensive_analysis(
            freq_results,
            threshold.threshold_frequency,
            save_path=results_dir / f"photoelectric_analysis_{timestamp}.png",
            show=True
        )
        plot_amplitude_comparison(
            results_by_amplitude,
            save_path=results_dir / f"amplitude_comparison_{timestamp}.png",
            show=True
        )
    
    return comparison


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description="Light Wave Photoelectric Effect Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Run default frequency sweep
    python main.py --experiment quick       # Quick test
    python main.py --experiment full        # Full photoelectric comparison
    python main.py --params photoelectric   # Use photoelectric-tuned params
    python main.py --no-plots               # Run without showing plots
        """
    )
    
    parser.add_argument(
        "--experiment", "-e",
        choices=["quick", "sweep", "amplitude", "full", "compton"],
        default="sweep",
        help="Type of experiment to run (compton = Compton scattering test)"
    )
    
    parser.add_argument(
        "--params", "-p",
        choices=["default", "high_sensitivity", "photoelectric", "strong_binding", "weak_binding"],
        default="default",
        help="Parameter set to use"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Don't show plots (useful for batch runs)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger("main")
    logger.info("=" * 60)
    logger.info("LIGHT WAVE PHOTOELECTRIC EFFECT SIMULATION")
    logger.info("=" * 60)
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Parameters: {args.params}")
    logger.info(f"Show plots: {not args.no_plots}")
    logger.info("")
    
    # Get parameters
    physics_params, sim_params, sweep_params = get_parameter_set(args.params)
    
    show_plots = not args.no_plots
    
    # Run experiment
    if args.experiment == "quick":
        results, threshold = run_quick_test(show_plots=show_plots)
        
    elif args.experiment == "sweep":
        results, threshold = run_frequency_sweep_experiment(
            params=physics_params,
            show_plots=show_plots
        )
        
    elif args.experiment == "amplitude":
        results, analysis = run_amplitude_independence_test(
            params=physics_params,
            show_plots=show_plots
        )
        
    elif args.experiment == "full":
        comparison = run_full_photoelectric_comparison(show_plots=show_plots)
    
    elif args.experiment == "compton":
        logger.info("Running Compton scattering experiment...")
        frequencies = np.linspace(0.1, 0.5, 12)
        results, analysis = run_compton_experiment(
            frequencies=frequencies,
            show_plots=show_plots
        )
        logger.info("")
        logger.info("COMPTON RESULTS:")
        logger.info(f"  Mean frequency shift: {analysis['mean_shift']:.4f}")
        logger.info(f"  Shift trend with frequency: {'Increasing' if analysis.get('shift_increases_with_freq', False) else 'Decreasing/Constant'}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

