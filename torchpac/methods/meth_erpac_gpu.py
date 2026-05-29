"""GPU-accelerated ERPAC methods (PyTorch).

Two methods are provided:
  * ``erpac_gpu``    – circular angular-linear correlation (Voytek 2013)
  * ``ergcpac_gpu``  – Gaussian-Copula MI-based ERPAC (Ince 2017)

Both accept/return numpy arrays.  p-values for the circular method are
computed on CPU via scipy (one small call, negligible overhead).
"""
import numpy as np
import torch
from scipy.stats import chi2

from torchpac.gcmi_gpu import copnorm_gpu, nd_mi_gg_gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to(arr, device, dtype):
    return torch.tensor(arr, dtype=dtype, device=device)


def _pearson_gpu(x, y, st='i...j,k...j->ik...'):
    """Batched Pearson correlation – replicates meth_erpac.pearson exactly."""
    n    = x.shape[-1]
    mu_x = x.mean(-1, keepdim=True)
    mu_y = y.mean(-1, keepdim=True)
    # ddof = n-1  →  denominator = 1  →  std = sqrt(Σ(x-μ)²)
    s_x  = ((x - mu_x) ** 2).sum(-1, keepdim=True).sqrt()
    s_y  = ((y - mu_y) ** 2).sum(-1, keepdim=True).sqrt()
    cov   = torch.einsum(st, x,   y)
    mu_xy = torch.einsum(st, mu_x, mu_y)
    cov  -= n * mu_xy
    cov  /= (torch.einsum(st, s_x, s_y) + 1e-12)
    return cov


# ---------------------------------------------------------------------------
# Circular ERPAC (Voytek 2013)
# ---------------------------------------------------------------------------

def erpac_gpu(pha, amp, device, dtype):
    """GPU circular ERPAC.

    Parameters
    ----------
    pha : (n_pha, ..., n_epochs)  numpy array (trial axis last)
    amp : (n_amp, ..., n_epochs)  numpy array

    Returns
    -------
    rho  : (n_amp, n_pha, ...) numpy float64
    pval : (n_amp, n_pha, ...) numpy float64  – computed via scipy chi2
    """
    pha_t = _to(pha, device, dtype)
    amp_t = _to(amp, device, dtype)

    n    = pha_t.shape[-1]
    sa   = torch.sin(pha_t)
    ca   = torch.cos(pha_t)

    rxs  = _pearson_gpu(amp_t, sa)
    rxc  = _pearson_gpu(amp_t, ca)
    rcs  = _pearson_gpu(sa,  ca, st='i...j,k...j->i...')
    rcs  = rcs.unsqueeze(0)

    rho  = torch.sqrt(
        (rxc ** 2 + rxs ** 2 - 2.0 * rxc * rxs * rcs) /
        (1.0 - rcs ** 2 + 1e-12)
    )

    rho_np = rho.cpu().numpy().astype(np.float64)
    pval   = 1.0 - chi2.cdf(n * rho_np ** 2, 2)
    return rho_np, pval


# ---------------------------------------------------------------------------
# Gaussian-Copula ERPAC (Ince 2017)
# ---------------------------------------------------------------------------

def _ergcpac_core_gpu(pha_t, amp_t):
    """Fully vectorised: compute MI for all (amp, pha, time) at once.

    pha_t : (n_pha, n_times, 2, n_epochs)   – copnormed
    amp_t : (n_amp, n_times, 1, n_epochs)   – copnormed

    Returns
    -------
    out : (n_amp, n_pha, n_times) tensor
    """
    n_pha, n_times = pha_t.shape[0], pha_t.shape[1]
    n_amp          = amp_t.shape[0]

    # Expand to (n_amp, n_pha, n_times, mvaxis, n_epochs)
    pha_exp = pha_t.unsqueeze(0).expand(n_amp, -1, -1, -1, -1)
    amp_exp = amp_t.unsqueeze(1).expand(-1, n_pha, -1, -1, -1)

    # nd_mi_gg_gpu handles arbitrary batch dims; here batch = (n_amp, n_pha, n_times)
    return nd_mi_gg_gpu(pha_exp, amp_exp)   # (n_amp, n_pha, n_times)


def ergcpac_gpu(pha, amp, smooth=None, device='cpu', dtype=torch.float32):
    """GPU Gaussian-Copula ERPAC.

    Parameters
    ----------
    pha    : (n_pha, n_times, 2, n_epochs)  numpy array – already copnormed
    amp    : (n_amp, n_times, 1, n_epochs)  numpy array – already copnormed
    smooth : int | None – temporal smoothing half-width
    device, dtype

    Returns
    -------
    out : (n_amp, n_pha, n_times) numpy float64
    """
    pha_t = _to(pha, device, dtype)
    amp_t = _to(amp, device, dtype)

    n_pha, n_times = pha_t.shape[0], pha_t.shape[1]
    n_amp          = amp_t.shape[0]

    if isinstance(smooth, int):
        out = torch.zeros(n_amp, n_pha, n_times, dtype=dtype, device=device)
        cnt = torch.zeros(n_times, dtype=dtype, device=device)

        for t in range(smooth, n_times - smooth):
            sl   = slice(t - smooth, t + smooth + 1)
            # extract window: (n_pha/amp, window, mvaxis, n_epochs)
            ph_w = pha_t[:, sl, :, :]
            am_w = amp_t[:, sl, :, :]
            # reshape window into epoch axis for MI
            # ph_w: (n_pha, W, 2, n_epochs) → (n_pha, 1, 2, W*n_epochs)
            W, n_ep = ph_w.shape[1], ph_w.shape[3]
            ph_w = ph_w.reshape(n_pha, 1, 2, W * n_ep)
            am_w = am_w.reshape(n_amp, 1, 1, W * n_ep)
            # squeeze time dim so nd_mi_gg_gpu sees (n_amp, n_pha, mvaxis, trials)
            ph_exp = ph_w.unsqueeze(0).expand(n_amp, -1, -1, -1, -1).squeeze(2)
            am_exp = am_w.unsqueeze(1).expand(-1, n_pha, -1, -1, -1).squeeze(2)
            out[:, :, t] = nd_mi_gg_gpu(ph_exp, am_exp)
            cnt[t] = 1.0

        # zero-out uncomputed edges
        out[:, :, cnt == 0] = 0.0
    else:
        out = _ergcpac_core_gpu(pha_t, amp_t)

    return out.cpu().numpy().astype(np.float64)


def _ergcpac_perm_gpu(pha, amp, smooth=None, n_perm=200,
                       device='cpu', dtype=torch.float32):
    """Permutation distribution for GPU gcERPAC.

    Shuffles the trial (epoch) axis of phase for each permutation.

    Returns
    -------
    surrogates : (n_perm, n_amp, n_pha, n_times) numpy float64
    """
    pha_t = _to(pha, device, dtype)
    n_ep  = pha_t.shape[-1]
    out   = []
    for _ in range(n_perm):
        tr_    = torch.randperm(n_ep, device=device)
        pha_s  = pha_t[..., tr_]
        out.append(ergcpac_gpu(pha_s.cpu().numpy(), amp,
                               smooth=smooth, device=device, dtype=dtype))
    return np.stack(out, axis=0)
