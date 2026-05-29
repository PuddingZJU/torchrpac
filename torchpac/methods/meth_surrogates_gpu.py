"""GPU-accelerated surrogate generation (PyTorch).

Mirrors meth_surrogates.py but keeps tensors on-device for the full loop.
The three shuffling strategies use torch.randperm / torch.roll so no
CPU round-trips are needed between permutations.
"""
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shuffling strategies (operate on tensors)
# ---------------------------------------------------------------------------

def _swap_pha_amp_gpu(pha_t, amp_t, seed, device):
    """Permute trial axis of phase (axis 1)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    n_trials = pha_t.shape[1]
    tr_ = torch.randperm(n_trials, generator=g, device=device)
    return pha_t[:, tr_, ...], amp_t


def _swap_blocks_gpu(pha_t, amp_t, seed, device):
    """Cut amplitude at a random time point and swap the two blocks."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    n_times = amp_t.shape[-1]
    cut_at  = int(torch.randint(1, n_times, (1,), generator=g).item())
    amp_new = torch.cat([amp_t[..., cut_at:], amp_t[..., :cut_at]], dim=-1)
    return pha_t, amp_new


def _time_lag_gpu(pha_t, amp_t, seed, device):
    """Circular-shift the phase by a random lag."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    shift = int(torch.randint(0, pha_t.shape[-1], (1,), generator=g).item())
    return torch.roll(pha_t, shift, dims=-1), amp_t


_SURRO_FCN = {
    1: _swap_pha_amp_gpu,
    2: _swap_blocks_gpu,
    3: _time_lag_gpu,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_surrogates_gpu(pha, amp, ids, fcn_gpu, n_perm,
                           random_state, device, dtype):
    """Compute surrogate PAC values on the GPU.

    Parameters
    ----------
    pha, amp     : numpy arrays – phase (n_pha, n_epochs, n_times) and
                   amplitude (n_amp, n_epochs, n_times)
    ids          : int – surrogate method {1, 2, 3}
    fcn_gpu      : callable(pha_np, amp_np) -> np.ndarray – GPU PAC function
    n_perm       : int – number of permutations
    random_state : int – base seed (each perm uses random_state + k)
    device       : torch.device
    dtype        : torch.dtype

    Returns
    -------
    surrogates : numpy float64 array of shape (n_perm, n_amp, n_pha, n_epochs)
                 or None if ids == 0.
    """
    if ids == 0:
        return None

    shuffle_fn = _SURRO_FCN[ids]

    pha_t = torch.tensor(pha, dtype=dtype, device=device)
    amp_t = torch.tensor(amp, dtype=dtype, device=device)

    surros = []
    for k in range(n_perm):
        pha_s, amp_s = shuffle_fn(pha_t, amp_t, random_state + k, device)
        # fcn_gpu accepts numpy, so convert briefly – copies are small vs compute
        pac_k = fcn_gpu(pha_s.cpu().numpy(), amp_s.cpu().numpy())
        surros.append(pac_k)

    return np.array(surros)   # (n_perm, n_amp, n_pha, n_epochs)


# ---------------------------------------------------------------------------
# Normalisation (identical logic to CPU version, numpy only)
# ---------------------------------------------------------------------------

def normalize_gpu(idn, pac, surro):
    """Inplace normalisation of PAC by surrogate distribution."""
    s_mean = np.mean(surro, axis=0)
    s_std  = np.std(surro,  axis=0)
    if idn == 1:
        pac -= s_mean
    elif idn == 2:
        pac /= s_mean
    elif idn == 3:
        pac -= s_mean
        pac /= s_mean
    elif idn == 4:
        pac -= s_mean
        pac /= s_std
