"""
Visualization Module - Plotting Tools for Simulation Results

Creates visualizations to understand:
- Time evolution of state variables (x, v, s)
- Phase portraits (x vs v)
- Store dynamics (s over time)
- Frequency sweep results (escape vs frequency)
- Energy flow diagrams
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
from typing import List, Optional, Tuple
from pathlib import Path

from .simulation import SimulationResult

logger = logging.getLogger(__name__)

# Set up matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')

# Custom color palette
COLORS = {
    'position': '#2E86AB',      # Blue
    'velocity': '#A23B72',      # Magenta
    'store': '#F18F01',         # Orange
    'escape': '#C73E1D',        # Red
    'no_escape': '#3E8914',     # Green
    'threshold': '#6A0572',     # Purple
    'energy': '#1B998B',        # Teal
}


def setup_figure_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_time_series(
    result: SimulationResult,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot time series of all state variables for a single simulation.
    
    Creates a 3-panel plot showing:
    - Position x(t)
    - Velocity v(t)
    - Store s(t)
    
    Args:
        result: SimulationResult to plot
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        matplotlib Figure object
    """
    setup_figure_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    t = result.times
    
    # Position
    ax1 = axes[0]
    ax1.plot(t, result.positions, color=COLORS['position'], label='x(t)')
    ax1.set_ylabel('Position x')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if result.escaped:
        ax1.axvline(x=result.escape_time, color=COLORS['escape'], 
                    linestyle='--', label=f'Escape (t={result.escape_time:.2f})')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Simulation at f = {result.frequency:.4f}')
    
    # Velocity
    ax2 = axes[1]
    ax2.plot(t, result.velocities, color=COLORS['velocity'], label='v(t)')
    ax2.set_ylabel('Velocity v')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    # Store
    ax3 = axes[2]
    ax3.plot(t, result.stores, color=COLORS['store'], label='s(t)')
    ax3.set_ylabel('Internal Store s')
    ax3.set_xlabel('Time t')
    ax3.legend(loc='upper right')
    
    # Mark escape on all panels
    if result.escaped:
        for ax in axes:
            ax.axvline(x=result.escape_time, color=COLORS['escape'], 
                      linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved time series plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_phase_portrait(
    result: SimulationResult,
    include_store: bool = True,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot phase portrait (x vs v) with optional store coloring.
    
    The phase portrait shows the trajectory in position-velocity space,
    which reveals the dynamics of the bound electron.
    
    - Closed orbits = bound state
    - Spiraling outward = escape
    - Color can encode the store value
    
    Args:
        result: SimulationResult to plot
        include_store: If True, color trajectory by store value
        save_path: Optional save path
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    setup_figure_style()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = result.positions
    v = result.velocities
    s = result.stores
    
    if include_store and len(s) > 0:
        # Color by store value
        points = np.array([x, v]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        
        # Normalize store for coloring
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-10)
        
        lc = LineCollection(segments, cmap='plasma', alpha=0.8)
        lc.set_array(s_norm[:-1])
        lc.set_linewidth(1.5)
        
        line = ax.add_collection(lc)
        cbar = plt.colorbar(line, ax=ax, label='Normalized Store s')
    else:
        ax.plot(x, v, color=COLORS['position'], alpha=0.7, linewidth=1)
    
    # Mark start and end
    ax.plot(x[0], v[0], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(x[-1], v[-1], 'ro', markersize=10, label='End', zorder=5)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Velocity v')
    ax.set_title(f'Phase Portrait at f = {result.frequency:.4f}')
    ax.legend()
    ax.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved phase portrait to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_frequency_sweep(
    results: List[SimulationResult],
    threshold_freq: Optional[float] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot results of frequency sweep showing escape vs no-escape.
    
    Creates a visualization showing:
    - Which frequencies lead to escape
    - Escape times for escaped simulations
    - Threshold frequency if provided
    
    Args:
        results: List of SimulationResults from frequency sweep
        threshold_freq: Optional threshold frequency to mark
        save_path: Optional save path
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    setup_figure_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Sort by frequency
    sorted_results = sorted(results, key=lambda r: r.frequency)
    
    freqs = np.array([r.frequency for r in sorted_results])
    escaped = np.array([r.escaped for r in sorted_results])
    escape_times = np.array([
        r.escape_time if r.escape_time is not None else np.nan 
        for r in sorted_results
    ])
    
    # Top panel: Escape/No-Escape binary
    ax1 = axes[0]
    for i, (f, esc) in enumerate(zip(freqs, escaped)):
        color = COLORS['escape'] if esc else COLORS['no_escape']
        marker = 'v' if esc else 'o'
        ax1.scatter(f, 1 if esc else 0, c=color, s=100, marker=marker, 
                   edgecolor='white', linewidth=1, zorder=3)
    
    ax1.set_ylabel('Escaped')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['No', 'Yes'])
    ax1.set_title('Frequency Sweep: Escape Threshold')
    
    if threshold_freq is not None:
        ax1.axvline(x=threshold_freq, color=COLORS['threshold'], 
                   linestyle='--', linewidth=2, label=f'Threshold ≈ {threshold_freq:.3f}')
        ax1.legend()
    
    # Bottom panel: Escape times
    ax2 = axes[1]
    valid_mask = ~np.isnan(escape_times)
    ax2.scatter(freqs[valid_mask], escape_times[valid_mask], 
               c=COLORS['store'], s=80, edgecolor='white', linewidth=1)
    
    # Plot no-escape as max time
    max_time = np.nanmax(escape_times) if np.any(valid_mask) else 100
    no_escape_mask = ~escaped
    ax2.scatter(freqs[no_escape_mask], 
               np.full(np.sum(no_escape_mask), max_time * 1.1),
               c=COLORS['no_escape'], s=80, marker='x', label='No escape')
    
    ax2.set_xlabel('Frequency f')
    ax2.set_ylabel('Escape Time t')
    ax2.set_title('Escape Time vs Frequency')
    ax2.legend()
    
    if threshold_freq is not None:
        ax2.axvline(x=threshold_freq, color=COLORS['threshold'], 
                   linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved frequency sweep plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_amplitude_comparison(
    results_by_amplitude: dict,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare behavior across different amplitudes.
    
    Shows whether threshold frequency is independent of amplitude
    (key photoelectric effect characteristic).
    
    Args:
        results_by_amplitude: Dict mapping amplitude -> list of results
        save_path: Optional save path
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    setup_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cmap = plt.cm.viridis
    amplitudes = sorted(results_by_amplitude.keys())
    colors = cmap(np.linspace(0.2, 0.8, len(amplitudes)))
    
    # Left: Escape vs frequency for each amplitude
    ax1 = axes[0]
    for amp, color in zip(amplitudes, colors):
        results = sorted(results_by_amplitude[amp], key=lambda r: r.frequency)
        freqs = [r.frequency for r in results]
        escaped = [1 if r.escaped else 0 for r in results]
        ax1.plot(freqs, escaped, 'o-', color=color, label=f'A={amp:.4f}', 
                markersize=6, alpha=0.7)
    
    ax1.set_xlabel('Frequency f')
    ax1.set_ylabel('Escaped (0/1)')
    ax1.set_title('Escape vs Frequency for Different Amplitudes')
    ax1.legend(title='Amplitude')
    
    # Right: Threshold vs amplitude
    ax2 = axes[1]
    thresholds = []
    uncertainties = []
    
    from .analysis import find_threshold_frequency
    
    for amp in amplitudes:
        results = results_by_amplitude[amp]
        thresh = find_threshold_frequency(results)
        thresholds.append(thresh.threshold_frequency)
        uncertainties.append(thresh.threshold_uncertainty)
    
    ax2.errorbar(amplitudes, thresholds, yerr=uncertainties, 
                fmt='o-', color=COLORS['threshold'], 
                markersize=10, capsize=5, linewidth=2)
    ax2.set_xlabel('Amplitude A')
    ax2.set_ylabel('Threshold Frequency')
    ax2.set_title('Threshold vs Amplitude\n(Should be flat for photoelectric effect)')
    
    # Add horizontal line at mean threshold
    mean_thresh = np.mean(thresholds)
    ax2.axhline(y=mean_thresh, color='gray', linestyle='--', 
               label=f'Mean = {mean_thresh:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved amplitude comparison to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_energy_evolution(
    result: SimulationResult,
    omega0: float = 1.0,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot energy components over time.
    
    Shows:
    - Kinetic energy: ½v²
    - Potential energy: ½ω₀²x²
    - Total mechanical energy
    - Internal store s
    
    Args:
        result: SimulationResult to analyze
        omega0: Natural frequency for potential energy calculation
        save_path: Optional save path
        show: Whether to display
        
    Returns:
        matplotlib Figure
    """
    setup_figure_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    t = result.times
    kinetic = 0.5 * result.velocities ** 2
    potential = 0.5 * (omega0 ** 2) * result.positions ** 2
    total = kinetic + potential
    
    # Top: Energy components
    ax1 = axes[0]
    ax1.plot(t, kinetic, color=COLORS['velocity'], label='Kinetic ½v²', alpha=0.8)
    ax1.plot(t, potential, color=COLORS['position'], label=f'Potential ½ω₀²x²', alpha=0.8)
    ax1.plot(t, total, color=COLORS['energy'], label='Total Mechanical', linewidth=2)
    
    ax1.set_ylabel('Energy')
    ax1.set_title(f'Energy Evolution at f = {result.frequency:.4f}')
    ax1.legend()
    
    # Bottom: Store
    ax2 = axes[1]
    ax2.plot(t, result.stores, color=COLORS['store'], linewidth=2)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Internal Store s')
    ax2.set_title('Internal Store Evolution')
    
    # Mark escape
    if result.escaped:
        for ax in axes:
            ax.axvline(x=result.escape_time, color=COLORS['escape'], 
                      linestyle='--', alpha=0.7, label='Escape')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved energy plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_comprehensive_analysis(
    results: List[SimulationResult],
    threshold_freq: Optional[float] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive multi-panel analysis figure.
    
    Combines multiple visualizations into one figure for
    complete overview of simulation results.
    """
    setup_figure_style()
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    sorted_results = sorted(results, key=lambda r: r.frequency)
    escaped_results = [r for r in sorted_results if r.escaped]
    
    # 1. Frequency sweep binary (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    freqs = [r.frequency for r in sorted_results]
    escaped = [1 if r.escaped else 0 for r in sorted_results]
    colors = [COLORS['escape'] if e else COLORS['no_escape'] for e in escaped]
    ax1.scatter(freqs, escaped, c=colors, s=60, edgecolor='white')
    if threshold_freq:
        ax1.axvline(x=threshold_freq, color=COLORS['threshold'], linestyle='--')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Escaped')
    ax1.set_title('Escape Threshold')
    
    # 2. Escape times (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if escaped_results:
        esc_freqs = [r.frequency for r in escaped_results]
        esc_times = [r.escape_time for r in escaped_results]
        ax2.scatter(esc_freqs, esc_times, c=COLORS['store'], s=60, edgecolor='white')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Escape Time')
    ax2.set_title('Escape Time vs Frequency')
    
    # 3. Max store reached (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    max_stores = [r.get_max_store() for r in sorted_results]
    ax3.scatter(freqs, max_stores, c=colors, s=60, edgecolor='white')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Max Store')
    ax3.set_title('Maximum Store Reached')
    
    # 4-5. Example time series below threshold (middle left)
    below_thresh = [r for r in sorted_results if not r.escaped]
    if below_thresh:
        example_below = below_thresh[len(below_thresh)//2]
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(example_below.times, example_below.positions, color=COLORS['position'])
        ax4.set_title(f'Below Threshold (f={example_below.frequency:.3f})')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Position')
    
    # 6. Example time series above threshold (middle middle)
    if escaped_results:
        example_above = escaped_results[0]
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(example_above.times, example_above.positions, color=COLORS['escape'])
        ax5.set_title(f'Above Threshold (f={example_above.frequency:.3f})')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Position')
    
    # 7. Store comparison (middle right)
    ax6 = fig.add_subplot(gs[1, 2])
    if below_thresh:
        ax6.plot(example_below.times, example_below.stores, 
                color=COLORS['no_escape'], label=f'f={example_below.frequency:.3f}')
    if escaped_results:
        ax6.plot(example_above.times, example_above.stores,
                color=COLORS['escape'], label=f'f={example_above.frequency:.3f}')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Store')
    ax6.set_title('Store Evolution Comparison')
    ax6.legend()
    
    # 8-9. Phase portraits (bottom)
    if below_thresh:
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(example_below.positions, example_below.velocities, 
                color=COLORS['no_escape'], alpha=0.7)
        ax7.set_xlabel('Position')
        ax7.set_ylabel('Velocity')
        ax7.set_title('Phase Portrait (Below Threshold)')
    
    if escaped_results:
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(example_above.positions, example_above.velocities,
                color=COLORS['escape'], alpha=0.7)
        ax8.set_xlabel('Position')
        ax8.set_ylabel('Velocity')
        ax8.set_title('Phase Portrait (Above Threshold)')
    
    # 10. Max amplitude reached
    ax9 = fig.add_subplot(gs[2, 2])
    max_amps = [r.get_max_amplitude() for r in sorted_results]
    ax9.scatter(freqs, max_amps, c=colors, s=60, edgecolor='white')
    ax9.set_xlabel('Frequency')
    ax9.set_ylabel('Max Amplitude')
    ax9.set_title('Maximum Amplitude Reached')
    
    plt.suptitle('Comprehensive Photoelectric Model Analysis', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comprehensive analysis to {save_path}")
    
    if show:
        plt.show()
    
    return fig

