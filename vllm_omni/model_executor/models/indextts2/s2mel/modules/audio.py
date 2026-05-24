# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch


def dynamic_range_compression_torch(x, compression=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * compression)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None) -> np.ndarray:
    """Create mel filterbank matrix (replaces librosa.filters.mel)."""
    if fmax is None:
        fmax = float(sr) / 2.0
    fmin_mel = 2595.0 * np.log10(1.0 + fmin / 700.0)
    fmax_mel = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    weights = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for i in range(n_mels):
        lower = freqs[i]
        center_f = freqs[i + 1]
        upper = freqs[i + 2]
        for j, f in enumerate(fft_freqs):
            if lower <= f <= center_f and center_f != lower:
                weights[i, j] = (f - lower) / (center_f - lower)
            elif center_f < f <= upper and upper != center_f:
                weights[i, j] = (upper - f) / (upper - center_f)
    enorm = 2.0 / (freqs[2 : n_mels + 2] - freqs[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{sampling_rate}_{fmax}_{y.device}" not in mel_basis:
        mel = _mel_filterbank(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(sampling_rate) + "_" + str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(sampling_rate) + "_" + str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec
