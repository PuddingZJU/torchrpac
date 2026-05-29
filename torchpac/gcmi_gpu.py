"""GPU-accelerated Gaussian Copula Mutual Information (PyTorch)."""
import math
import torch
from torchpac.config import CONFIG


def ctransform_gpu(x):
    """Empirical CDF (copula transform) along last axis.

    Parameters
    ----------
    x : torch.Tensor, shape (..., n_trials)

    Returns
    -------
    xr : torch.Tensor, shape (..., n_trials)
        Ranks scaled to open interval (0, 1).
    """
    xi = torch.argsort(x, dim=-1)
    xr = torch.argsort(xi, dim=-1).to(x.dtype)
    xr = xr + 1.0
    xr = xr / float(x.shape[-1] + 1)
    return xr


def _ndtri_gpu(p):
    """Inverse normal CDF: sqrt(2) * erfinv(2p - 1).

    Equivalent to scipy.special.ndtri but runs on the tensor's device.
    """
    return math.sqrt(2.0) * torch.special.erfinv(2.0 * p - 1.0)


def copnorm_gpu(x):
    """Gaussian copula normalisation along last axis.

    Parameters
    ----------
    x : torch.Tensor, shape (..., n_trials)

    Returns
    -------
    cx : torch.Tensor
        Standard-normal samples with the same empirical CDF as x.
    """
    return _ndtri_gpu(ctransform_gpu(x))


def nd_mi_gg_gpu(x, y):
    """Multi-dimensional Gaussian Copula MI in bits (GPU).

    Parameters
    ----------
    x : torch.Tensor, shape (..., x_mvaxis, traxis)
    y : torch.Tensor, shape (..., y_mvaxis, traxis)

    Returns
    -------
    mi : torch.Tensor, shape (...)
        Mutual information in bits.
    """
    ntrl = x.shape[-1]
    nvarx = x.shape[-2]
    nvary = y.shape[-2]
    nvarxy = nvarx + nvary

    # joint variable
    xy = torch.cat([x, y], dim=-2)
    if CONFIG['MI_DEMEAN']:
        xy = xy - xy.mean(dim=-1, keepdim=True)

    # covariance matrix
    cxy = torch.einsum('...ij,...kj->...ik', xy, xy) / float(ntrl - 1)
    cx = cxy[..., :nvarx, :nvarx]
    cy = cxy[..., nvarx:, nvarx:]

    # Cholesky – batched over all leading dims
    chcxy = torch.linalg.cholesky(cxy)
    chcx = torch.linalg.cholesky(cx)
    chcy = torch.linalg.cholesky(cy)

    # entropy via log-diagonal of Cholesky (normalisation cancels in MI)
    hx  = torch.log(torch.einsum('...ii->...i', chcx)).sum(-1)
    hy  = torch.log(torch.einsum('...ii->...i', chcy)).sum(-1)
    hxy = torch.log(torch.einsum('...ii->...i', chcxy)).sum(-1)

    ln2 = math.log(2.0)
    if CONFIG['MI_BIASCORRECT']:
        vec = torch.arange(1, nvarxy + 1, dtype=x.dtype, device=x.device)
        psiterms = torch.special.digamma((ntrl - vec) / 2.0) / 2.0
        dterm = (ln2 - math.log(ntrl - 1.0)) / 2.0
        hx  = hx  - nvarx  * dterm - psiterms[:nvarx].sum()
        hy  = hy  - nvary  * dterm - psiterms[:nvary].sum()
        hxy = hxy - nvarxy * dterm - psiterms[:nvarxy].sum()

    return (hx + hy - hxy) / ln2
