"""GPU-accelerated spectral extraction (PyTorch).

Replaces spectral.py for the GPU path.  The public entry point
``spectral_gpu`` has the same signature and return type (numpy float64)
as the original ``spectral``, so the rest of the codebase needs only a
one-line switch.

Design
------
* Filter coefficients are computed on CPU (they are tiny).
* filtfilt is reimplemented in pure PyTorch to keep data on the GPU the
  whole time, matching scipy's "odd" reflection padding and causal FIR
  forward/backward pass.
* After filtering, the Hilbert transform is applied to ALL frequency bands
  in one batched FFT call – the main GPU speedup.
* The Morlet wavelet path uses batched 1-D convolution via conv1d.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FIR filter design helpers (unchanged from spectral.py – CPU only)
# ---------------------------------------------------------------------------

def fir_order(fs, sizevec, flow, cycle=3):
    filtorder = cycle * (fs // flow)
    if sizevec < 3 * filtorder:
        filtorder = (sizevec - 1) // 3
    return int(filtorder)


def _n_odd_fcn(f, o, w, l):
    b0 = 0
    m  = np.array(range(int(l + 1)))
    k  = m[1:len(m)]
    b  = np.zeros(k.shape)
    for s in range(0, len(f), 2):
        m_  = (o[s + 1] - o[s]) / (f[s + 1] - f[s])
        b1  = o[s] - m_ * f[s]
        b0 += (b1 * (f[s + 1] - f[s]) + m_ / 2 * (
               f[s + 1] ** 2 - f[s] ** 2)) * abs(np.square(w[round((s + 1) / 2)]))
        b  += (m_ / (4 * np.pi ** 2) * (
               np.cos(2 * np.pi * k * f[s + 1]) -
               np.cos(2 * np.pi * k * f[s])) / (k * k)) * abs(
               np.square(w[round((s + 1) / 2)]))
        b  += (f[s + 1] * (m_ * f[s + 1] + b1) * np.sinc(2 * k * f[s + 1]) -
               f[s]     * (m_ * f[s]     + b1) * np.sinc(2 * k * f[s])) * abs(
               np.square(w[round((s + 1) / 2)]))
    b   = np.insert(b, 0, b0)
    a   = np.square(w[0]) * 4 * b
    a[0] /= 2
    aud = np.flipud(a[1:]) / 2
    a2  = np.insert(aud, len(aud), a[0])
    return np.concatenate((a2, a[1:] / 2))


def _n_even_fcn(f, o, w, l):
    k = np.array(range(0, int(l) + 1)) + 0.5
    b = np.zeros(k.shape)
    for s in range(0, len(f), 2):
        m_ = (o[s + 1] - o[s]) / (f[s + 1] - f[s])
        b1 = o[s] - m_ * f[s]
        b += (m_ / (4 * np.pi ** 2) * (
              np.cos(2 * np.pi * k * f[s + 1]) -
              np.cos(2 * np.pi * k * f[s])) / (k * k)) * abs(
              np.square(w[round((s + 1) / 2)]))
        b += (f[s + 1] * (m_ * f[s + 1] + b1) * np.sinc(2 * k * f[s + 1]) -
              f[s]     * (m_ * f[s]     + b1) * np.sinc(2 * k * f[s])) * abs(
              np.square(w[round((s + 1) / 2)]))
    a = np.square(w[0]) * 4 * b
    return 0.5 * np.concatenate((np.flipud(a), a))


def _firls(n, f, o):
    w  = np.ones(round(len(f) / 2))
    n += 1
    f /= 2
    lo = (n - 1) / 2
    return _n_odd_fcn(f, o, w, lo) if bool(n % 2) else _n_even_fcn(f, o, w, lo)


def fir1(n, wn):
    nbands = len(wn) + 1
    ff  = np.array((0, wn[0], wn[0], wn[1], wn[1], 1))
    f0  = np.mean(ff[2:4])
    lo  = n + 1
    mags = np.array(range(nbands)).reshape(1, -1) % 2
    aa   = np.ravel(np.tile(mags, (2, 1)), order='F')
    h    = _firls(lo - 1, ff, aa)
    wind = np.hamming(lo)
    b    = h * wind
    c    = np.exp(-1j * 2 * np.pi * (f0 / 2) * np.array(range(lo)))
    b   /= abs(c @ b)
    return b, 1


# ---------------------------------------------------------------------------
# GPU filtfilt – zero-phase FIR filter
# ---------------------------------------------------------------------------

def _lfilter_fir_gpu(b_t, x_flat):
    """Causal FIR filter on GPU: y[n] = Σ_k b[k] * x[n-k].

    Parameters
    ----------
    b_t    : (M,) float tensor – FIR coefficients
    x_flat : (batch, L) float tensor

    Returns
    -------
    y : (batch, L) float tensor
    """
    M      = b_t.shape[0]
    batch  = x_flat.shape[0]
    # Pad left by M-1 zeros for causal output
    x_in   = F.pad(x_flat.unsqueeze(1), (M - 1, 0))   # (batch, 1, L+M-1)
    # conv1d does cross-correlation; flip b to get convolution semantics
    kernel = b_t.flip(0).reshape(1, 1, M)
    out    = F.conv1d(x_in, kernel, padding=0)          # (batch, 1, L)
    return out[:, 0, :]


def _filtfilt_gpu(x, b_np, padlen, device, dtype):
    """Zero-phase FIR filter matching scipy.signal.filtfilt (odd extension).

    Parameters
    ----------
    x      : (..., n_times) tensor on *device*
    b_np   : numpy array of FIR coefficients
    padlen : int – number of reflection samples (scipy default: 3*(M-1))
    device, dtype : torch device / dtype

    Returns
    -------
    filtered : same shape and device as x
    """
    n_times   = x.shape[-1]
    orig_shape = x.shape
    x_flat     = x.reshape(-1, n_times)          # (batch, n_times)

    # Odd reflection extension (matches scipy's "odd" mode)
    left  = 2.0 * x_flat[:, :1]  - x_flat[:, 1:padlen + 1].flip(-1)
    right = 2.0 * x_flat[:, -1:] - x_flat[:, -padlen - 1:-1].flip(-1)
    x_pad = torch.cat([left, x_flat, right], dim=-1)   # (batch, n+2p)

    b_t   = torch.tensor(b_np, dtype=dtype, device=device)
    n_pad = x_pad.shape[-1]

    # Forward causal pass
    y = _lfilter_fir_gpu(b_t, x_pad)

    # Backward causal pass (flip → filter → flip)
    y = _lfilter_fir_gpu(b_t, y.flip(-1)).flip(-1)

    # Strip padding
    result = y[:, padlen:padlen + n_times]
    return result.reshape(orig_shape)


# ---------------------------------------------------------------------------
# GPU Hilbert transform (batched FFT)
# ---------------------------------------------------------------------------

def _hilbert_gpu(x):
    """Analytic signal via Hilbert transform – fully on GPU.

    Parameters
    ----------
    x : (..., n_times) real tensor

    Returns
    -------
    xa : (..., n_times) complex tensor
    """
    n   = x.shape[-1]
    Xf  = torch.fft.fft(x, n=n, dim=-1)
    # Build one-sided doubling weights
    h   = torch.zeros(n, dtype=x.dtype, device=x.device)
    if n % 2 == 0:
        h[0] = 1.0; h[n // 2] = 1.0; h[1:n // 2] = 2.0
    else:
        h[0] = 1.0; h[1:(n + 1) // 2] = 2.0
    # Broadcast h over all leading dims
    for _ in range(x.dim() - 1):
        h = h.unsqueeze(0)
    return torch.fft.ifft(Xf * h, n=n, dim=-1)


# ---------------------------------------------------------------------------
# GPU Morlet wavelet
# ---------------------------------------------------------------------------

def _morlet_gpu(x_t, sf, f_center, width, device, dtype):
    """Complex Morlet decomposition for one center frequency.

    Parameters
    ----------
    x_t      : (n_epochs, n_times) tensor
    sf, f_center, width : scalars
    device, dtype

    Returns
    -------
    xout : (n_epochs, n_times) complex tensor
    """
    dt   = 1.0 / sf
    sf_w = f_center / width
    st   = 1.0 / (2.0 * math.pi * sf_w)

    t_np = np.arange(-width * st / 2.0, width * st / 2.0, dt)
    a    = 1.0 / math.sqrt(st * math.sqrt(math.pi))
    m_np = a * np.exp(-(t_np ** 2) / (2.0 * st ** 2)) * np.exp(
           1j * 2.0 * math.pi * f_center * t_np)

    M       = len(m_np)
    m_real  = torch.tensor(m_np.real, dtype=dtype, device=device).flip(0).reshape(1, 1, M)
    m_imag  = torch.tensor(m_np.imag, dtype=dtype, device=device).flip(0).reshape(1, 1, M)

    n_times = x_t.shape[-1]
    x_in    = F.pad(x_t.unsqueeze(1), (M - 1, M - 1))   # (batch, 1, n+2(M-1))

    r_out   = F.conv1d(x_in, m_real, padding=0)[:, 0, :]  # (batch, n+M-1)
    i_out   = F.conv1d(x_in, m_imag, padding=0)[:, 0, :]

    # Trim to original length (same slice as np.convolve "full" → centre)
    start   = int(math.ceil(M / 2)) - 1
    r_out   = r_out[:, start:start + n_times]
    i_out   = i_out[:, start:start + n_times]
    return torch.complex(r_out, i_out)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def spectral_gpu(x, sf, f, stype, dcomplex, cycle, width, device, dtype):
    """GPU replacement for torchpac.spectral.spectral.

    Parameters
    ----------
    x       : (n_epochs, n_times) numpy array
    sf      : float – sampling frequency
    f       : (n_freqs, 2) numpy array – [f_low, f_high] per band
    stype   : 'pha' | 'amp' | None
    dcomplex: 'hilbert' | 'wavelet'
    cycle   : int – number of FIR cycles
    width   : int – Morlet width
    device  : torch.device
    dtype   : torch.dtype (float32 recommended)

    Returns
    -------
    out : (n_freqs, n_epochs, n_times) numpy float64 array
    """
    n_freqs, n_epochs, n_times = f.shape[0], x.shape[0], x.shape[1]
    x_t = torch.tensor(x, dtype=dtype, device=device)   # (n_epochs, n_times)

    if dcomplex == 'hilbert':
        # ---- compute FIR coefficients on CPU (fast, tiny arrays) ----
        b_list, pad_list = [], []
        for k in range(n_freqs):
            fo = fir_order(sf, n_times, f[k, 0], cycle=cycle)
            b, _ = fir1(fo, f[k, :] / (sf / 2.0))
            b_list.append(b)
            pad_list.append(fo)   # scipy uses filtorder as default padlen

        # ---- filter each band (GPU) ----
        xf = torch.zeros(n_freqs, n_epochs, n_times, dtype=dtype, device=device)
        for k in range(n_freqs):
            xf[k] = _filtfilt_gpu(x_t, b_list[k], pad_list[k], device, dtype)

        # ---- Hilbert on all bands at once (big GPU win) ----
        if stype is not None:
            xd = _hilbert_gpu(xf)   # (n_freqs, n_epochs, n_times) complex
        else:
            xd = xf

    elif dcomplex == 'wavelet':
        fc   = f.mean(1)   # centre frequencies
        # complex dtype for accumulation
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        xd = torch.zeros(n_freqs, n_epochs, n_times, dtype=cdtype, device=device)
        for k_idx, fc_k in enumerate(fc):
            xd[k_idx] = _morlet_gpu(x_t, sf, float(fc_k), width, device, dtype)

    else:
        raise ValueError("dcomplex must be 'hilbert' or 'wavelet'.")

    # ---- extract phase / amplitude ----
    if stype == 'pha':
        return xd.angle().cpu().numpy().astype(np.float64)
    elif stype == 'amp':
        return xd.abs().cpu().numpy().astype(np.float64)
    else:
        return xd.cpu().numpy().astype(np.complex128)
