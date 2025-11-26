"""
Analysis Module - Tools for Analyzing Simulation Results

This module provides tools for:
- Finding the frequency threshold
- Analyzing escape time vs frequency relationships
- Comparing amplitude dependence
- Statistical analysis of results
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .simulation import SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class ThresholdAnalysis:
    """Results from threshold frequency analysis."""
    
    threshold_frequency: float
    threshold_uncertainty: float
    method: str
    
    # Additional data
    frequencies_below: np.ndarray
    frequencies_above: np.ndarray
    
    # Quality metrics
    confidence: float  # 0-1 confidence in the result
    n_data_points: int
    
    def __repr__(self) -> str:
        return (
            f"ThresholdAnalysis(f_th={self.threshold_frequency:.4f} ± "
            f"{self.threshold_uncertainty:.4f}, method={self.method}, "
            f"confidence={self.confidence:.2f})"
        )


def find_threshold_frequency(
    results: List[SimulationResult],
    method: str = 'first_escape'
) -> ThresholdAnalysis:
    """
    Find the threshold frequency from a frequency sweep.
    
    The threshold is the frequency below which no escape occurs
    and above which escape does occur (photoelectric effect analog).
    
    Args:
        results: List of simulation results from frequency sweep
        method: Detection method - 'first_escape', 'interpolate', or 'fit'
        
    Returns:
        ThresholdAnalysis with threshold frequency and metadata
    """
    logger.info(f"Finding threshold frequency using method: {method}")
    
    # Sort by frequency
    sorted_results = sorted(results, key=lambda r: r.frequency)
    
    # Separate escaped and non-escaped
    escaped = [r for r in sorted_results if r.escaped]
    not_escaped = [r for r in sorted_results if not r.escaped]
    
    if len(escaped) == 0:
        logger.warning("No escapes detected in any simulation")
        return ThresholdAnalysis(
            threshold_frequency=float('inf'),
            threshold_uncertainty=float('inf'),
            method=method,
            frequencies_below=np.array([r.frequency for r in not_escaped]),
            frequencies_above=np.array([]),
            confidence=0.0,
            n_data_points=len(results)
        )
    
    if len(not_escaped) == 0:
        logger.warning("All simulations resulted in escape")
        return ThresholdAnalysis(
            threshold_frequency=0.0,
            threshold_uncertainty=sorted_results[0].frequency,
            method=method,
            frequencies_below=np.array([]),
            frequencies_above=np.array([r.frequency for r in escaped]),
            confidence=0.5,
            n_data_points=len(results)
        )
    
    # Find threshold based on method
    if method == 'first_escape':
        # Threshold is between highest non-escape and lowest escape
        max_no_escape = max(r.frequency for r in not_escaped)
        min_escape = min(r.frequency for r in escaped)
        
        threshold = (max_no_escape + min_escape) / 2.0
        uncertainty = (min_escape - max_no_escape) / 2.0
        
        # Check for clean separation
        if max_no_escape >= min_escape:
            logger.warning(
                "Non-clean threshold: some frequencies both escape and don't"
            )
            confidence = 0.5
        else:
            confidence = 1.0 - uncertainty / threshold if threshold > 0 else 0.5
    
    elif method == 'interpolate':
        # Use escape times to interpolate
        # Shorter escape time = more above threshold
        threshold, uncertainty, confidence = _interpolate_threshold(sorted_results)
    
    else:
        logger.warning(f"Unknown method {method}, falling back to first_escape")
        return find_threshold_frequency(results, method='first_escape')
    
    logger.info(
        f"Threshold found: f_th = {threshold:.4f} ± {uncertainty:.4f}"
    )
    
    return ThresholdAnalysis(
        threshold_frequency=threshold,
        threshold_uncertainty=uncertainty,
        method=method,
        frequencies_below=np.array([r.frequency for r in not_escaped]),
        frequencies_above=np.array([r.frequency for r in escaped]),
        confidence=confidence,
        n_data_points=len(results)
    )


def _interpolate_threshold(results: List[SimulationResult]) -> Tuple[float, float, float]:
    """
    Interpolate threshold using escape times.
    
    Near the threshold, escape time should be long.
    Far above threshold, escape time should be short.
    """
    # Get escaped results with finite escape times
    escaped = [(r.frequency, r.escape_time) for r in results 
               if r.escaped and r.escape_time is not None]
    
    if len(escaped) < 2:
        # Fall back to simple method
        escaped_freqs = [r.frequency for r in results if r.escaped]
        not_escaped_freqs = [r.frequency for r in results if not r.escaped]
        
        if escaped_freqs and not_escaped_freqs:
            threshold = (max(not_escaped_freqs) + min(escaped_freqs)) / 2
            uncertainty = abs(min(escaped_freqs) - max(not_escaped_freqs)) / 2
            return threshold, uncertainty, 0.5
        else:
            return 0.0, float('inf'), 0.0
    
    # Sort by frequency
    escaped.sort(key=lambda x: x[0])
    freqs = np.array([e[0] for e in escaped])
    times = np.array([e[1] for e in escaped])
    
    # Escape time typically decreases as frequency increases above threshold
    # Find where escape time is longest (closest to threshold)
    max_time_idx = np.argmax(times)
    
    # Non-escaped frequencies
    not_escaped_freqs = [r.frequency for r in results if not r.escaped]
    
    if not_escaped_freqs:
        max_not_escaped = max(not_escaped_freqs)
        # Threshold is between max_not_escaped and frequency of longest escape
        threshold = (max_not_escaped + freqs[max_time_idx]) / 2
        uncertainty = abs(freqs[max_time_idx] - max_not_escaped) / 2
    else:
        threshold = freqs[max_time_idx]
        uncertainty = freqs[1] - freqs[0] if len(freqs) > 1 else 0.1
    
    confidence = 0.7  # Moderate confidence for interpolation
    
    return threshold, uncertainty, confidence


def analyze_escape_times(results: List[SimulationResult]) -> Dict:
    """
    Analyze how escape time varies with frequency.
    
    This is important for the photoelectric model:
    - Above threshold, escape should be relatively quick
    - Near threshold, escape takes longer
    - Escape time should not strongly depend on amplitude
    
    Args:
        results: List of simulation results
        
    Returns:
        Dictionary with analysis results
    """
    escaped = [r for r in results if r.escaped and r.escape_time is not None]
    
    if len(escaped) == 0:
        logger.warning("No escapes to analyze")
        return {'error': 'No escapes detected'}
    
    freqs = np.array([r.frequency for r in escaped])
    times = np.array([r.escape_time for r in escaped])
    
    # Sort by frequency
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    times = times[sort_idx]
    
    # Compute statistics
    analysis = {
        'n_escaped': len(escaped),
        'frequencies': freqs,
        'escape_times': times,
        'min_escape_time': np.min(times),
        'max_escape_time': np.max(times),
        'mean_escape_time': np.mean(times),
        'freq_at_min_time': freqs[np.argmin(times)],
        'freq_at_max_time': freqs[np.argmax(times)]
    }
    
    # Try to fit escape time vs frequency
    if len(escaped) >= 3:
        try:
            # Fit: escape_time ~ 1 / (f - f_th)^n
            # Or simpler: escape_time = a / f + b
            coeffs = np.polyfit(freqs, times, deg=2)
            analysis['fit_coeffs'] = coeffs
            analysis['fit_type'] = 'quadratic'
        except Exception as e:
            logger.warning(f"Could not fit escape time data: {e}")
    
    logger.info(
        f"Escape time analysis: {len(escaped)} escapes, "
        f"time range [{analysis['min_escape_time']:.2f}, "
        f"{analysis['max_escape_time']:.2f}]"
    )
    
    return analysis


def analyze_amplitude_independence(
    results_by_amplitude: Dict[float, List[SimulationResult]]
) -> Dict:
    """
    Analyze how threshold depends on amplitude.
    
    For the photoelectric effect, threshold should be nearly independent
    of amplitude - this is a key prediction to test.
    
    Args:
        results_by_amplitude: Dictionary mapping amplitude to results list
        
    Returns:
        Analysis of amplitude dependence
    """
    thresholds = []
    amplitudes = []
    
    for amp, results in results_by_amplitude.items():
        thresh = find_threshold_frequency(results)
        if thresh.confidence > 0.3:  # Only include reliable thresholds
            thresholds.append(thresh.threshold_frequency)
            amplitudes.append(amp)
    
    if len(thresholds) < 2:
        return {
            'error': 'Not enough reliable thresholds',
            'n_amplitudes': len(results_by_amplitude)
        }
    
    thresholds = np.array(thresholds)
    amplitudes = np.array(amplitudes)
    
    # Calculate variation
    threshold_variation = np.std(thresholds) / np.mean(thresholds)
    amplitude_range = amplitudes.max() / amplitudes.min()
    
    # Fit threshold vs amplitude
    try:
        # log-log fit to check for power law
        log_amp = np.log(amplitudes)
        log_thresh = np.log(thresholds)
        slope, intercept = np.polyfit(log_amp, log_thresh, 1)
        
        # slope ~ 0 means threshold independent of amplitude (good!)
        # slope ~ -0.5 or similar would indicate amplitude dependence
    except:
        slope = None
        intercept = None
    
    analysis = {
        'amplitudes': amplitudes,
        'thresholds': thresholds,
        'threshold_mean': np.mean(thresholds),
        'threshold_std': np.std(thresholds),
        'threshold_variation': threshold_variation,
        'amplitude_range': amplitude_range,
        'power_law_slope': slope,
        'is_amplitude_independent': threshold_variation < 0.1  # <10% variation
    }
    
    if analysis['is_amplitude_independent']:
        logger.info("✓ Threshold appears AMPLITUDE INDEPENDENT (photoelectric-like)")
    else:
        logger.info(f"✗ Threshold varies with amplitude (variation: {threshold_variation:.2%})")
    
    return analysis


def compute_energy_balance(result: SimulationResult) -> Dict:
    """
    Compute energy balance during simulation.
    
    Tracks:
    - Kinetic energy: ½v²
    - Potential energy: ½ω₀²x²
    - Internal store: s (not energy, but tracks internal state)
    
    This helps understand the energy flow in the model.
    """
    kinetic = 0.5 * result.velocities ** 2
    # Note: We need omega0 from the model, assume 1.0 for now
    potential = 0.5 * result.positions ** 2
    mechanical = kinetic + potential
    
    return {
        'times': result.times,
        'kinetic': kinetic,
        'potential': potential,
        'mechanical': mechanical,
        'store': result.stores,
        'final_kinetic': kinetic[-1] if len(kinetic) > 0 else 0,
        'max_kinetic': np.max(kinetic) if len(kinetic) > 0 else 0,
        'energy_gained': mechanical[-1] - mechanical[0] if len(mechanical) > 1 else 0
    }


@dataclass
class PhotoelectricComparison:
    """
    Comparison of model behavior with photoelectric effect characteristics.
    """
    
    # Core photoelectric properties
    has_frequency_threshold: bool
    threshold_frequency: float
    threshold_uncertainty: float
    
    # Amplitude independence
    is_amplitude_independent: bool
    amplitude_variation: float
    
    # Escape time behavior
    escape_time_decreases_with_freq: bool
    
    # Overall score (0-1)
    photoelectric_score: float
    
    # Detailed notes
    notes: List[str]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "PHOTOELECTRIC EFFECT MODEL COMPARISON",
            "=" * 60,
            "",
            f"Frequency Threshold: {'✓' if self.has_frequency_threshold else '✗'}",
            f"  f_th = {self.threshold_frequency:.4f} ± {self.threshold_uncertainty:.4f}",
            "",
            f"Amplitude Independence: {'✓' if self.is_amplitude_independent else '✗'}",
            f"  Threshold variation: {self.amplitude_variation:.2%}",
            "",
            f"Escape Time Behavior: {'✓' if self.escape_time_decreases_with_freq else '✗'}",
            "",
            f"Overall Photoelectric Score: {self.photoelectric_score:.0%}",
            "",
            "Notes:",
        ]
        for note in self.notes:
            lines.append(f"  - {note}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def compare_to_photoelectric_effect(
    frequency_sweep_results: List[SimulationResult],
    amplitude_sweep_results: Optional[Dict[float, List[SimulationResult]]] = None
) -> PhotoelectricComparison:
    """
    Compare model behavior to photoelectric effect characteristics.
    
    Checks:
    1. Is there a frequency threshold?
    2. Is the threshold independent of amplitude?
    3. Does escape time decrease with frequency above threshold?
    
    Args:
        frequency_sweep_results: Results from frequency sweep
        amplitude_sweep_results: Optional results from multiple amplitude sweeps
        
    Returns:
        PhotoelectricComparison with detailed analysis
    """
    notes = []
    score = 0.0
    
    # 1. Check for frequency threshold
    threshold = find_threshold_frequency(frequency_sweep_results)
    has_threshold = (
        threshold.threshold_frequency > 0 and 
        threshold.threshold_frequency < float('inf') and
        threshold.confidence > 0.5
    )
    
    if has_threshold:
        score += 0.4
        notes.append(f"Clear frequency threshold detected at f={threshold.threshold_frequency:.3f}")
    else:
        notes.append("No clear frequency threshold detected")
    
    # 2. Check amplitude independence
    is_amp_independent = False
    amp_variation = float('inf')
    
    if amplitude_sweep_results is not None:
        amp_analysis = analyze_amplitude_independence(amplitude_sweep_results)
        if 'is_amplitude_independent' in amp_analysis:
            is_amp_independent = amp_analysis['is_amplitude_independent']
            amp_variation = amp_analysis.get('threshold_variation', float('inf'))
            
            if is_amp_independent:
                score += 0.3
                notes.append("Threshold is amplitude-independent (key photoelectric property)")
            else:
                notes.append(f"Threshold varies with amplitude by {amp_variation:.1%}")
    else:
        notes.append("Amplitude independence not tested")
    
    # 3. Check escape time behavior
    escape_analysis = analyze_escape_times(frequency_sweep_results)
    
    escape_time_correct = False
    if 'frequencies' in escape_analysis and len(escape_analysis['frequencies']) > 2:
        freqs = escape_analysis['frequencies']
        times = escape_analysis['escape_times']
        
        # Generally escape time should decrease as frequency increases
        # Check correlation
        if len(freqs) > 2:
            correlation = np.corrcoef(freqs, times)[0, 1]
            escape_time_correct = correlation < -0.3  # Negative correlation
            
            if escape_time_correct:
                score += 0.3
                notes.append("Escape time decreases with frequency (correct behavior)")
            else:
                notes.append(f"Escape time correlation with frequency: {correlation:.2f}")
    
    return PhotoelectricComparison(
        has_frequency_threshold=has_threshold,
        threshold_frequency=threshold.threshold_frequency,
        threshold_uncertainty=threshold.threshold_uncertainty,
        is_amplitude_independent=is_amp_independent,
        amplitude_variation=amp_variation,
        escape_time_decreases_with_freq=escape_time_correct,
        photoelectric_score=score,
        notes=notes
    )

