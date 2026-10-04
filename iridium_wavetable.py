#!/usr/bin/env python3

"""
Waldorf Iridium WAV -> Wavetable preparation tool
=================================================

Prepares arbitrary WAV files for import into Waldorf Iridium.

Default:
    period      = 2048 samples
    max waves   = 430
    max samples = 880640

Features:
    - WAV input
    - mono conversion
    - DC removal
    - silence trimming
    - automatic loud/dynamic section selection
    - spectral activity analysis
    - clipping penalty
    - boundary alignment
    - per-wave cyclic phase optimization
    - optional per-wave seam crossfade
    - global normalization
    - 16-bit PCM output
    - CSV report
    - command-line options

Dependencies:
    pip install numpy soundfile

Usage:

    python iridium_wavetable.py

    python iridium_wavetable.py --input <dir> --output <dir>

    python iridium_wavetable.py --period 2048

    python iridium_wavetable.py --period 1024

    python iridium_wavetable.py --mode dynamic

    python iridium_wavetable.py --mode loudest

    python iridium_wavetable.py --seam 32

    python iridium_wavetable.py --period 2048 --mode auto --seam 16
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf


# ============================================================
# IRIDIUM LIMITS
# ============================================================

IRIDIUM_LIMITS = {
    4096: 217,
    2048: 430,
    1024: 846,
    512: 1645,
    256: 2000,
    128: 2000,
    64: 2000,
}


# ============================================================
# DEFAULT SETTINGS
# ============================================================

DEFAULT_PERIOD = 2048
DEFAULT_MODE = "auto"

# Number of samples used to evaluate spectral activity.
ANALYSIS_FRAME = 2048

# Silence threshold relative to file peak.
SILENCE_DB = -55.0

# Default seam smoothing length.
DEFAULT_SEAM = 16

# Maximum number of candidate regions to evaluate.
# Larger = more precise, slower.
MAX_CANDIDATES = 5000


# ============================================================
# BASIC AUDIO UTILITIES
# ============================================================

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0

    return float(np.sqrt(np.mean(x * x) + 1e-15))


def peak(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0

    return float(np.max(np.abs(x)))


def normalize(x: np.ndarray, target_peak: float = 0.98) -> np.ndarray:
    p = peak(x)

    if p < 1e-12:
        return x

    return x * (target_peak / p)


def remove_dc(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x

    return x - np.mean(x)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float64)

    return np.mean(audio, axis=1, dtype=np.float64)


# ============================================================
# SILENCE TRIMMING
# ============================================================

def trim_silence(audio: np.ndarray, silence_db: float) -> tuple[np.ndarray, int, int]:
    """
    Remove leading/trailing low-level audio.

    Returns:
        trimmed_audio
        start_sample
        end_sample
    """

    if len(audio) == 0:
        return audio, 0, 0

    absolute = np.abs(audio)

    file_peak = np.max(absolute)

    if file_peak < 1e-12:
        return audio, 0, len(audio)

    threshold = file_peak * db_to_linear(silence_db)

    active = absolute >= threshold

    if not np.any(active):
        return audio, 0, len(audio)

    start = int(np.argmax(active))
    end = int(len(active) - np.argmax(active[::-1]))

    return audio[start:end], start, end


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

def spectral_features(frame: np.ndarray) -> tuple[float, float]:
    """
    Calculate two useful descriptors:

        spectral_flux
        spectral_centroid

    Values are normalized and used only for comparing regions.
    """

    if len(frame) < 4:
        return 0.0, 0.0

    window = np.hanning(len(frame))

    x = frame * window

    spectrum = np.abs(np.fft.rfft(x))

    if len(spectrum) < 2:
        return 0.0, 0.0

    spectrum = spectrum[1:]

    total = np.sum(spectrum) + 1e-15

    frequencies = np.arange(1, len(spectrum) + 1)

    centroid = float(
        np.sum(frequencies * spectrum) / total
    )

    # Normalize centroid to 0..1
    centroid /= max(len(spectrum), 1)

    # Normalize spectral energy
    spectrum_norm = spectrum / total

    # Spectral flux against a flat reference.
    flat = np.ones_like(spectrum_norm) / len(spectrum_norm)

    flux = float(
        np.sqrt(np.mean((spectrum_norm - flat) ** 2))
    )

    return flux, centroid


def calculate_frame_features(
    audio: np.ndarray,
    period: int
) -> dict[str, np.ndarray]:
    """
    Divide audio into period-sized frames and calculate
    features for every frame.
    """

    frame_count = len(audio) // period

    if frame_count == 0:
        return {
            "energy": np.array([]),
            "peak": np.array([]),
            "flux": np.array([]),
            "centroid": np.array([]),
            "clip": np.array([]),
        }

    usable = audio[:frame_count * period]

    frames = usable.reshape(frame_count, period)

    energy = np.sqrt(
        np.mean(frames * frames, axis=1)
    )

    frame_peak = np.max(
        np.abs(frames),
        axis=1
    )

    clip = np.mean(
        np.abs(frames) >= 0.995,
        axis=1
    )

    flux = np.zeros(frame_count)
    centroid = np.zeros(frame_count)

    for i, frame in enumerate(frames):
        f, c = spectral_features(frame)
        flux[i] = f
        centroid[i] = c

    return {
        "energy": energy,
        "peak": frame_peak,
        "flux": flux,
        "centroid": centroid,
        "clip": clip,
    }


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def robust_normalize(x: np.ndarray) -> np.ndarray:
    """
    Robust 0..1 normalization.

    Uses percentiles rather than min/max to prevent one
    pathological frame from dominating the selection.
    """

    if len(x) == 0:
        return x

    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)

    if hi - lo < 1e-12:
        return np.ones_like(x) * 0.5

    y = (x - lo) / (hi - lo)

    return np.clip(y, 0.0, 1.0)


# ============================================================
# REGION SCORING
# ============================================================

def score_regions(
    features: dict[str, np.ndarray],
    region_frames: int,
    mode: str,
) -> np.ndarray:
    """
    Score every possible contiguous region.

    The score is calculated from:

        energy
        spectral activity
        spectral movement
        clipping

    Different modes use different weights.
    """

    energy = robust_normalize(features["energy"])
    flux = robust_normalize(features["flux"])
    centroid = robust_normalize(features["centroid"])
    clip = np.clip(features["clip"] * 10.0, 0.0, 1.0)

    # Spectral movement from frame to frame.
    centroid_delta = np.zeros_like(centroid)

    if len(centroid) > 1:
        centroid_delta[1:] = np.abs(
            np.diff(centroid)
        )

    centroid_delta = robust_normalize(centroid_delta)

    if mode == "loudest":

        weights = {
            "energy": 0.80,
            "flux": 0.05,
            "movement": 0.05,
            "centroid": 0.10,
        }

    elif mode == "dynamic":

        weights = {
            "energy": 0.35,
            "flux": 0.30,
            "movement": 0.30,
            "centroid": 0.05,
        }

    elif mode == "balanced":

        weights = {
            "energy": 0.55,
            "flux": 0.20,
            "movement": 0.20,
            "centroid": 0.05,
        }

    else:
        # AUTO
        weights = {
            "energy": 0.55,
            "flux": 0.20,
            "movement": 0.20,
            "centroid": 0.05,
        }

    frame_score = (
        weights["energy"] * energy
        + weights["flux"] * flux
        + weights["movement"] * centroid_delta
        + weights["centroid"] * centroid
        - 0.75 * clip
    )

    n = len(frame_score)

    if n < region_frames:
        return np.array([])

    # Rolling average using cumulative sum.
    cumulative = np.concatenate(
        ([0.0], np.cumsum(frame_score))
    )

    scores = (
        cumulative[region_frames:]
        - cumulative[:-region_frames]
    ) / region_frames

    return scores


def choose_best_region(
    audio: np.ndarray,
    period: int,
    max_waves: int,
    mode: str,
) -> tuple[int, int, float]:
    """
    Return:

        start sample
        end sample
        score
    """

    features = calculate_frame_features(
        audio,
        period
    )

    region_frames = min(
        max_waves,
        len(features["energy"])
    )

    if region_frames == 0:
        return 0, 0, 0.0

    scores = score_regions(
        features,
        region_frames,
        mode
    )

    if len(scores) == 0:
        return 0, region_frames * period, 0.0

    # If there are a huge number of candidate regions,
    # downsample the search while retaining good precision.
    step = 1

    if len(scores) > MAX_CANDIDATES:
        step = math.ceil(
            len(scores) / MAX_CANDIDATES
        )

    candidate_scores = scores[::step]

    candidate_index = int(
        np.argmax(candidate_scores)
    )

    frame_index = candidate_index * step

    start = frame_index * period
    end = start + region_frames * period

    return start, end, float(scores[frame_index])


# ============================================================
# WAVE SEAM OPTIMIZATION
# ============================================================

def best_circular_shift(wave: np.ndarray, search: int = 128) -> np.ndarray:
    """
    Find a nearby circular shift which minimizes the difference
    between the beginning and end of the waveform.

    We deliberately keep the search local. Huge shifts would
    destroy the temporal relationship between neighboring waves.
    """

    n = len(wave)

    if n < 16:
        return wave

    search = min(search, n // 4)

    best_shift = 0
    best_cost = float("inf")

    compare = min(32, n // 16)

    if compare < 2:
        return wave

    for shift in range(-search, search + 1):

        shifted = np.roll(wave, shift)

        a = shifted[:compare]
        b = shifted[-compare:]

        cost = float(
            np.mean((a - b) ** 2)
        )

        if cost < best_cost:
            best_cost = cost
            best_shift = shift

    return np.roll(wave, best_shift)


def seam_crossfade(
    wave: np.ndarray,
    seam: int
) -> np.ndarray:
    """
    Make the beginning and end of one wave meet more smoothly.

    Only the local seam is modified.
    """

    if seam <= 0:
        return wave

    n = len(wave)

    if seam * 2 >= n:
        return wave

    result = wave.copy()

    fade = np.linspace(
        0.0,
        1.0,
        seam,
        endpoint=False
    )

    start = result[:seam].copy()
    end = result[-seam:].copy()

    # Blend both sides toward their average.
    blend = (
        end * (1.0 - fade)
        + start * fade
    )

    result[:seam] = blend
    result[-seam:] = blend

    return result


def optimize_waves(
    audio: np.ndarray,
    period: int,
    seam: int,
    phase_search: int,
) -> np.ndarray:
    """
    Process each wavetable wave independently.
    """

    wave_count = len(audio) // period

    if wave_count == 0:
        return audio

    usable = audio[:wave_count * period]

    waves = usable.reshape(wave_count, period).copy()

    output = np.empty_like(waves)

    for i in range(wave_count):

        wave = waves[i]

        # Remove DC individually.
        wave = remove_dc(wave)

        # Find a better cyclic boundary.
        wave = best_circular_shift(
            wave,
            search=phase_search
        )

        # Optional seam smoothing.
        wave = seam_crossfade(
            wave,
            seam
        )

        output[i] = wave

    return output.reshape(-1)


# ============================================================
# PERIOD / LIMIT HELPERS
# ============================================================

def get_max_waves(period: int) -> int:
    """
    Get the known Iridium limit.

    For values between documented presets we use the conservative
    2000-wave maximum only if the resulting size is reasonable.
    """

    if period in IRIDIUM_LIMITS:
        return IRIDIUM_LIMITS[period]

    if not 64 <= period <= 4096:
        raise ValueError(
            "Iridium period must be between 64 and 4096."
        )

    # Conservative fallback.
    #
    # We do not invent a larger table limit for undocumented
    # periods. 2000 is the global waveform-count ceiling commonly
    # reported for current Iridium/Quantum firmware.
    return min(
        2000,
        4096 // period * 2000
    )


# ============================================================
# FILE PROCESSING
# ============================================================

def process_file(
    input_path: Path,
    output_dir: Path,
    period: int,
    mode: str,
    seam: int,
    phase_search: int,
) -> dict:

    result = {
        "file": input_path.name,
        "status": "OK",
        "sample_rate": "",
        "original_samples": "",
        "trimmed_samples": "",
        "output_samples": "",
        "waves": "",
        "period": period,
        "selected_start": "",
        "selected_end": "",
        "selection_score": "",
        "mode": mode,
        "note": "",
    }

    try:
        audio, sample_rate = sf.read(
            input_path,
            always_2d=False,
            dtype="float64"
        )
    except Exception as e:

        result["status"] = "ERROR"
        result["note"] = f"Read error: {e}"

        return result

    result["sample_rate"] = sample_rate
    result["original_samples"] = len(audio)

    # --------------------------------------------------------
    # Mono
    # --------------------------------------------------------

    audio = to_mono(audio)

    # --------------------------------------------------------
    # DC removal
    # --------------------------------------------------------

    audio = remove_dc(audio)

    # --------------------------------------------------------
    # Silence trimming
    # --------------------------------------------------------

    trimmed, trim_start, trim_end = trim_silence(
        audio,
        SILENCE_DB
    )

    result["trimmed_samples"] = len(trimmed)

    if len(trimmed) < period:

        # Repeat short files.
        repetitions = math.ceil(
            period / max(len(trimmed), 1)
        )

        if len(trimmed) == 0:

            result["status"] = "ERROR"
            result["note"] = "Empty audio."

            return result

        trimmed = np.tile(
            trimmed,
            repetitions
        )[:period]

        result["note"] = (
            "Input shorter than one period; repeated to one wave."
        )

    max_waves = get_max_waves(period)

    max_samples = max_waves * period

    # --------------------------------------------------------
    # Region selection
    # --------------------------------------------------------

    if len(trimmed) <= max_samples:

        selected = trimmed[
            :(len(trimmed) // period) * period
        ]

        start = 0
        end = len(selected)
        score = 0.0

    else:

        start, end, score = choose_best_region(
            trimmed,
            period,
            max_waves,
            mode
        )

        selected = trimmed[start:end]

        result["selected_start"] = (
            start + trim_start
        )

        result["selected_end"] = (
            end + trim_start
        )

        result["selection_score"] = score

    # --------------------------------------------------------
    # Safety: complete waves only
    # --------------------------------------------------------

    usable = (
        len(selected) // period
    ) * period

    selected = selected[:usable]

    # --------------------------------------------------------
    # Wave-by-wave optimization
    # --------------------------------------------------------

    selected = optimize_waves(
        selected,
        period,
        seam,
        phase_search
    )

    # --------------------------------------------------------
    # Final DC removal
    # --------------------------------------------------------

    selected = remove_dc(selected)

    # --------------------------------------------------------
    # Global normalization
    # --------------------------------------------------------

    selected = normalize(
        selected,
        target_peak=0.98
    )

    wave_count = len(selected) // period

    result["output_samples"] = len(selected)
    result["waves"] = wave_count

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_path = output_dir / input_path.name

    try:

        sf.write(
            output_path,
            selected.astype(np.float32),
            sample_rate,
            subtype="PCM_16",
            format="WAV"
        )

    except Exception as e:

        result["status"] = "ERROR"
        result["note"] = f"Write error: {e}"

        return result

    return result


# ============================================================
# REPORT
# ============================================================

def write_report(
    path: Path,
    results: list[dict]
):

    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare WAV files for Waldorf Iridium wavetable import."
        )
    )

    parser.add_argument(
        "--period",
        type=int,
        default=DEFAULT_PERIOD,
        choices=[
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ],
        help=(
            "Samples per wavetable wave. "
            "Default: 2048."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "auto",
            "loudest",
            "balanced",
            "dynamic",
        ],
        default=DEFAULT_MODE,
        help=(
            "How to choose the best region. "
            "Default: auto."
        )
    )

    parser.add_argument(
        "--seam",
        type=int,
        default=DEFAULT_SEAM,
        help=(
            "Samples used for per-wave seam smoothing. "
            "Use 0 to disable. Default: 16."
        )
    )

    parser.add_argument(
        "--phase-search",
        type=int,
        default=128,
        help=(
            "Maximum circular phase shift searched for "
            "each wave. Default: 128."
        )
    )

    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable silence trimming."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path.cwd(),
        help=(
            "Input directory. Default: current directory."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Default: ./iridium_wavetables/"
        )
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    input_dir = args.input.resolve()

    if args.output is None:

        output_dir = (
            input_dir / "iridium_wavetables"
        )

    else:

        output_dir = args.output.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Global silence setting
    # --------------------------------------------------------

    global SILENCE_DB

    if args.no_trim:
        SILENCE_DB = -120.0

    max_waves = get_max_waves(
        args.period
    )

    max_samples = (
        args.period * max_waves
    )

    max_seconds_44k = (
        max_samples / 44100.0
    )

    print()
    print("=" * 70)
    print("WALDORF IRIDIUM WAVETABLE PREPARER")
    print("=" * 70)

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print(f"Period          : {args.period} samples")
    print(f"Maximum waves   : {max_waves}")
    print(f"Maximum samples : {max_samples:,}")
    print(
        f"Equivalent time : {max_seconds_44k:.2f} sec @ 44.1 kHz"
    )
    print()
    print(f"Selection mode  : {args.mode}")
    print(f"Seam smoothing  : {args.seam} samples")
    print(f"Phase search    : ±{args.phase_search} samples")
    print(f"Silence trim    : {not args.no_trim}")

    wav_files = sorted(
        p
        for p in input_dir.rglob("*.wav")
        if p.is_file()
        and output_dir not in p.parents
    )

    if not wav_files:

        print()
        print("No WAV files found.")
        return

    print()
    print(
        f"Found {len(wav_files)} WAV file(s)."
    )
    print("=" * 70)

    results = []

    for i, wav_file in enumerate(wav_files, 1):

        # Mirror the input subdirectory structure.
        out_sub = (
            output_dir
            / wav_file.relative_to(input_dir).parent
        )

        out_sub.mkdir(
            parents=True,
            exist_ok=True
        )

        print()
        print(
            f"[{i}/{len(wav_files)}] "
            f"{wav_file.relative_to(input_dir)}"
        )

        result = process_file(
            wav_file,
            out_sub,
            args.period,
            args.mode,
            args.seam,
            args.phase_search
        )

        results.append(result)

        if result["status"] == "OK":

            print(
                f"  OK: "
                f"{result['waves']} waves / "
                f"{result['output_samples']:,} samples"
            )

            if result["selected_start"] != "":

                print(
                    f"  Selected samples: "
                    f"{result['selected_start']} - "
                    f"{result['selected_end']}"
                )

        else:

            print(
                f"  ERROR: {result['note']}"
            )

    # --------------------------------------------------------
    # CSV report
    # --------------------------------------------------------

    report_path = (
        output_dir / "iridium_report.csv"
    )

    write_report(
        report_path,
        results
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    success = sum(
        r["status"] == "OK"
        for r in results
    )

    errors = len(results) - success

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Successful: {success}")
    print(f"Errors    : {errors}")
    print()
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")
    print()


if __name__ == "__main__":
    main()