"""GPU-accelerated PAC methods (PyTorch).

All public functions follow the same numpy interface as meth_pac.py:
they accept numpy arrays, run computation on the specified device, and
return numpy float64 arrays.  The device/dtype are captured at creation
time via the factory function ``make_pac_gpu_fcns``.
"""
import math
import numpy as np
import torch

from torchpac.gcmi_gpu import copnorm_gpu, nd_mi_gg_gpu


# ---------------------------------------------------------------------------
# Internal tensor-level helpers
# ---------------------------------------------------------------------------

def _to(arr, device, dtype):
    return torch.tensor(arr, dtype=dtype, device=device)


def _out(t):
    return t.cpu().numpy().astype(np.float64)


def _mvl(pha_t, amp_t):
    """MVL on tensors: uses real/imag split to stay in float32."""
    n_times = pha_t.shape[-1]
    re = torch.einsum('i...j,k...j->ik...', amp_t, torch.cos(pha_t))
    im = torch.einsum('i...j,k...j->ik...', amp_t, torch.sin(pha_t))
    return torch.sqrt(re ** 2 + im ** 2) / n_times


def _kl_hr(pha_t, amp_t, n_bins):
    """Bin amplitude by phase using searchsorted (replaces np.digitize)."""
    device = pha_t.device
    edges = torch.linspace(-math.pi, math.pi, n_bins + 1,
                           dtype=pha_t.dtype, device=device)
    # searchsorted returns insertion index; subtract 1 to get bin index
    phad = torch.searchsorted(edges.contiguous(),
                              pha_t.contiguous()) - 1
    phad = phad.clamp(0, n_bins - 1)

    abin = []
    for i in range(n_bins):
        idx = (phad == i).to(amp_t.dtype)
        m = idx.sum().clamp(min=1.0)
        abin.append(torch.einsum('i...j,k...j->ik...', amp_t, idx) / m)
    return torch.stack(abin, dim=0)   # (n_bins, n_amp, n_pha, ...)


def _modulation_index(pha_t, amp_t, n_bins):
    p_j = _kl_hr(pha_t, amp_t, n_bins)
    p_j = p_j / p_j.sum(dim=0, keepdim=True)
    log_p = torch.log(p_j)
    log_p = torch.where(torch.isfinite(log_p),
                        log_p,
                        torch.full_like(log_p, -1e38))
    pac = 1.0 + (p_j * log_p).sum(dim=0) / math.log(n_bins)
    pac[~torch.isfinite(pac)] = 0.0
    return pac


def _heights_ratio(pha_t, amp_t, n_bins):
    p_j = _kl_hr(pha_t, amp_t, n_bins)
    p_j = p_j / p_j.sum(dim=0, keepdim=True)
    h_max = p_j.max(dim=0).values
    h_min = p_j.min(dim=0).values
    return (h_max - h_min) / h_max.clamp(min=1e-12)


def _norm_direct_pac(pha_t, amp_t, p_thresh):
    n_times = amp_t.shape[-1]
    mu = amp_t.mean(dim=-1, keepdim=True)
    # ddof=1 sample std
    sd = amp_t.std(dim=-1, correction=1, keepdim=True)
    amp_n = (amp_t - mu) / sd.clamp(min=1e-12)

    re = torch.einsum('i...j,k...j->ik...', amp_n, torch.cos(pha_t))
    im = torch.einsum('i...j,k...j->ik...', amp_n, torch.sin(pha_t))
    pac = torch.sqrt(re ** 2 + im ** 2)

    if p_thresh == 1.0 or p_thresh is None:
        return pac / n_times

    s = pac ** 2
    pac = pac / n_times
    xlim = n_times * torch.special.erfinv(
        torch.tensor(1.0 - p_thresh, dtype=pha_t.dtype, device=pha_t.device)
    ) ** 2
    pac[s <= 2 * xlim] = 0.0
    return pac


def _plv(pha_t, pha_amp_t):
    """Phase Locking Value: |Σ exp(i*(pha - pha_amp))| / n."""
    n_times = pha_t.shape[-1]
    cp, sp = torch.cos(pha_t), torch.sin(pha_t)
    ca, sa = torch.cos(pha_amp_t), torch.sin(pha_amp_t)
    # exp(-i*pha_amp) * exp(i*pha):  i→pha_amp axis, k→pha axis
    re = (torch.einsum('i...j,k...j->ik...', ca, cp) +
          torch.einsum('i...j,k...j->ik...', sa, sp))
    im = (torch.einsum('i...j,k...j->ik...', ca, sp) -
          torch.einsum('i...j,k...j->ik...', sa, cp))
    return torch.sqrt(re ** 2 + im ** 2) / n_times


def _gauss_cop_pac(pha_t, amp_t):
    """Vectorised gcPAC: all (amp, pha) pairs in one batched Cholesky.

    pha_t : (n_pha, n_epochs, 2, n_times)   – already copnormed
    amp_t : (n_amp, n_epochs, 1, n_times)   – already copnormed
    returns (n_amp, n_pha, n_epochs)
    """
    n_pha = pha_t.shape[0]
    n_amp = amp_t.shape[0]
    # expand to (n_amp, n_pha, n_epochs, mvaxis, n_times)
    pha_exp = pha_t.unsqueeze(0).expand(n_amp, -1, -1, -1, -1)
    amp_exp = amp_t.unsqueeze(1).expand(-1, n_pha, -1, -1, -1)
    return nd_mi_gg_gpu(pha_exp, amp_exp)   # (n_amp, n_pha, n_epochs)


# ---------------------------------------------------------------------------
# Public numpy-in / numpy-out wrappers
# ---------------------------------------------------------------------------

def mean_vector_length_gpu(pha, amp, device, dtype):
    pha_t = _to(pha, device, dtype)
    amp_t = _to(amp, device, dtype)
    return _out(_mvl(pha_t, amp_t))


def modulation_index_gpu(pha, amp, n_bins, device, dtype):
    pha_t = _to(pha, device, dtype)
    amp_t = _to(amp, device, dtype)
    return _out(_modulation_index(pha_t, amp_t, n_bins))


def heights_ratio_gpu(pha, amp, n_bins, device, dtype):
    pha_t = _to(pha, device, dtype)
    amp_t = _to(amp, device, dtype)
    return _out(_heights_ratio(pha_t, amp_t, n_bins))


def norm_direct_pac_gpu(pha, amp, p, device, dtype):
    pha_t = _to(pha, device, dtype)
    amp_t = _to(amp, device, dtype)
    return _out(_norm_direct_pac(pha_t, amp_t, p))


def phase_locking_value_gpu(pha, pha_amp, device, dtype):
    pha_t     = _to(pha,     device, dtype)
    pha_amp_t = _to(pha_amp, device, dtype)
    return _out(_plv(pha_t, pha_amp_t))


def gauss_cop_pac_gpu(pha, amp, device, dtype):
    """gcPAC – inputs must already be copnormed (same contract as CPU ver)."""
    pha_t = _to(pha, device, dtype)
    amp_t = _to(amp, device, dtype)
    return _out(_gauss_cop_pac(pha_t, amp_t))


# ---------------------------------------------------------------------------
# Factory: return bound callables matching the CPU API
# ---------------------------------------------------------------------------

def make_pac_gpu_fcns(device, dtype):
    """Return a dict of GPU PAC functions keyed by method index (1-6).

    Each function has signature ``f(pha, amp) -> np.ndarray``.
    """
    from functools import partial
    return {
        1: partial(mean_vector_length_gpu,   device=device, dtype=dtype),
        2: partial(modulation_index_gpu,     device=device, dtype=dtype,
                   n_bins=18),
        3: partial(heights_ratio_gpu,        device=device, dtype=dtype,
                   n_bins=18),
        4: partial(norm_direct_pac_gpu,      device=device, dtype=dtype,
                   p=0.05),
        5: partial(phase_locking_value_gpu,  device=device, dtype=dtype),
        6: partial(gauss_cop_pac_gpu,        device=device, dtype=dtype),
    }
