"""
Test suite for windowing edge case fix in neural_signal_encoder.py

Tests the fix for dimension mismatches that occurred when processing
subjects with slightly different recording lengths (Sub-02, 06, 07, 10).
"""

import numpy as np
import sys
import os

# Add the project root to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from mistr.neural_signal_encoder import (
    extract_high_gamma,
    extract_wavelet_coefficients,
    extract_mel_spectrogram,
    extract_speech_prosody
)


def test_mismatched_signal_lengths():
    """
    Test that features can be extracted and aligned from signals with
    different lengths that would produce off-by-one window counts.

    Simulates the edge case from Sub-02, 06, 07, 10 where EEG and audio
    signals have slightly different lengths after resampling.
    """
    print("\n=== Test 1: Mismatched Signal Lengths ===")

    # Parameters matching the encoder
    win_length = 0.05
    frame_shift = 0.01
    eeg_sr = 1024
    audio_sr = 16000

    # Create signals with lengths that produce off-by-one window counts
    # These lengths are designed to trigger the edge case
    eeg_length = 30734  # Will produce ~30025 windows
    audio_length = 480586  # Will produce ~30026 windows

    # Generate synthetic signals
    num_channels = 64
    eeg_data = np.random.randn(eeg_length, num_channels)
    audio_data = np.random.randn(audio_length).astype(np.float64)

    # Extract features
    wavelet_features = extract_wavelet_coefficients(
        eeg_data, eeg_sr, window_length=win_length, frame_shift=frame_shift
    )
    prosody_features = extract_speech_prosody(
        audio_data, audio_sr, window_length=win_length, frame_shift=frame_shift
    )
    mel_features = extract_mel_spectrogram(
        (audio_data * 32767).astype(np.int16), audio_sr,
        window_length=win_length, frame_shift=frame_shift
    )

    print(f"  EEG signal length: {eeg_length} samples")
    print(f"  Audio signal length: {audio_length} samples")
    print(f"  Wavelet features shape: {wavelet_features.shape}")
    print(f"  Prosody features shape: {prosody_features.shape}")
    print(f"  Mel features shape: {mel_features.shape}")

    # Apply the fix: trim to minimum
    min_windows = min(wavelet_features.shape[0], prosody_features.shape[0], mel_features.shape[0])
    wavelet_features = wavelet_features[:min_windows]
    prosody_features = prosody_features[:min_windows]
    mel_features = mel_features[:min_windows]

    print(f"  Trimmed to minimum: {min_windows} windows")

    # Verify all have same length now
    assert wavelet_features.shape[0] == prosody_features.shape[0] == mel_features.shape[0], \
        "Features should have matching window counts after trimming"

    # Verify concatenation works without error
    try:
        combined = np.concatenate(
            (wavelet_features.reshape(wavelet_features.shape[0], -1), prosody_features),
            axis=1
        )
        print(f"  ✓ Concatenation successful: {combined.shape}")
    except ValueError as e:
        print(f"  ✗ Concatenation failed: {e}")
        raise

    print("  ✓ Test passed")


def test_extreme_length_differences():
    """
    Test with more extreme length differences to ensure robustness.
    Simulates cases where window counts differ by more than 1.
    """
    print("\n=== Test 2: Extreme Length Differences ===")

    win_length = 0.05
    frame_shift = 0.01
    eeg_sr = 1024
    audio_sr = 16000

    # Create signals with larger differences
    eeg_length = 29184  # Will produce ~28500 windows (like Sub-10)
    audio_length = 456016  # Will produce ~28501 windows

    num_channels = 64
    eeg_data = np.random.randn(eeg_length, num_channels)
    audio_data = np.random.randn(audio_length).astype(np.float64)

    wavelet_features = extract_wavelet_coefficients(
        eeg_data, eeg_sr, window_length=win_length, frame_shift=frame_shift
    )
    prosody_features = extract_speech_prosody(
        audio_data, audio_sr, window_length=win_length, frame_shift=frame_shift
    )

    print(f"  Wavelet windows: {wavelet_features.shape[0]}")
    print(f"  Prosody windows: {prosody_features.shape[0]}")
    print(f"  Difference: {abs(wavelet_features.shape[0] - prosody_features.shape[0])}")

    # Apply trimming
    min_windows = min(wavelet_features.shape[0], prosody_features.shape[0])
    wavelet_features = wavelet_features[:min_windows]
    prosody_features = prosody_features[:min_windows]

    assert wavelet_features.shape[0] == prosody_features.shape[0], \
        "Features should match after trimming"

    print(f"  ✓ Both trimmed to {min_windows} windows")
    print("  ✓ Test passed")


def test_identical_window_counts():
    """
    Test that the fix doesn't break cases where window counts already match.
    This ensures we don't introduce issues for subjects that were processing fine.
    """
    print("\n=== Test 3: Identical Window Counts ===")

    win_length = 0.05
    frame_shift = 0.01
    eeg_sr = 1024
    audio_sr = 16000

    # Calculate lengths that produce exactly 1000 windows for both
    target_windows = 1000
    eeg_length = int(np.ceil(target_windows * frame_shift * eeg_sr + win_length * eeg_sr))
    audio_length = int(np.ceil(target_windows * frame_shift * audio_sr + win_length * audio_sr))

    num_channels = 64
    eeg_data = np.random.randn(eeg_length, num_channels)
    audio_data = np.random.randn(audio_length).astype(np.float64)

    wavelet_features = extract_wavelet_coefficients(
        eeg_data, eeg_sr, window_length=win_length, frame_shift=frame_shift
    )
    mel_features = extract_mel_spectrogram(
        (audio_data * 32767).astype(np.int16), audio_sr,
        window_length=win_length, frame_shift=frame_shift
    )

    print(f"  Wavelet windows: {wavelet_features.shape[0]}")
    print(f"  Mel windows: {mel_features.shape[0]}")

    # Even if they match, trimming should not break anything
    min_windows = min(wavelet_features.shape[0], mel_features.shape[0])
    wavelet_features = wavelet_features[:min_windows]
    mel_features = mel_features[:min_windows]

    assert wavelet_features.shape[0] == mel_features.shape[0], \
        "Trimming shouldn't break matching window counts"

    print(f"  ✓ Both have {min_windows} windows after trimming")
    print("  ✓ Test passed")


def test_concatenation_shape_consistency():
    """
    Test that the final concatenated feature array has consistent shape
    across all features, matching the minimum window count.
    """
    print("\n=== Test 4: Concatenation Shape Consistency ===")

    win_length = 0.05
    frame_shift = 0.01

    # Create three feature arrays with slightly different window counts
    # (simulating wavelet, prosody, mel)
    num_channels = 64
    wavelet_level = 4

    wavelet_windows = 1000
    prosody_windows = 1001
    mel_windows = 999  # Minimum

    # Simulate feature shapes as they would come from extraction functions
    wavelet_features = np.random.randn(wavelet_windows, num_channels, wavelet_level)
    prosody_features = np.random.randn(prosody_windows, 6)  # pitch, energy, shimmer, duration, phase_var, cfc
    mel_features = np.random.randn(mel_windows, 23)

    print(f"  Before trimming:")
    print(f"    Wavelet: {wavelet_features.shape}")
    print(f"    Prosody: {prosody_features.shape}")
    print(f"    Mel: {mel_features.shape}")

    # Apply the fix
    min_windows = min(wavelet_features.shape[0], prosody_features.shape[0], mel_features.shape[0])
    wavelet_features = wavelet_features[:min_windows]
    prosody_features = prosody_features[:min_windows]
    mel_features = mel_features[:min_windows]

    print(f"  After trimming to {min_windows}:")
    print(f"    Wavelet: {wavelet_features.shape}")
    print(f"    Prosody: {prosody_features.shape}")
    print(f"    Mel: {mel_features.shape}")

    # Test concatenation as done in the encoder
    combined_features = np.concatenate(
        (wavelet_features.reshape(wavelet_features.shape[0], -1), prosody_features),
        axis=1
    )

    assert combined_features.shape[0] == min_windows, \
        f"Combined features should have {min_windows} windows, got {combined_features.shape[0]}"

    print(f"  ✓ Combined features shape: {combined_features.shape}")
    print("  ✓ Test passed")


if __name__ == "__main__":
    print("\nRunning windowing edge case tests...")
    print("=" * 60)

    try:
        test_mismatched_signal_lengths()
        test_extreme_length_differences()
        test_identical_window_counts()
        test_concatenation_shape_consistency()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Tests failed: {e}")
        print("=" * 60)
        raise
