"""Switch and utility functions for PAC methods."""
import numpy as np
from functools import partial


def get_pac_fcn(idp, n_bins, p, implementation="tensor", full=False,
                device=None, dtype=None):
    """Get the function for computing Phase-Amplitude coupling.

    Parameters
    ----------
    idp            : int  – PAC method index (1-6)
    n_bins         : int  – number of phase bins (MI / HR methods)
    p              : float – statistical threshold (ndPAC)
    implementation : {'tensor', 'numba', 'gpu'}
    full           : bool – if True return the full dict instead of one fcn
    device         : torch.device | None – required when implementation='gpu'
    dtype          : torch.dtype  | None – required when implementation='gpu'

    Returns
    -------
    fcn or dict of fcns with signature f(pha, amp) -> np.ndarray
    """
    n_bins = np.int64(n_bins)
    p      = np.float64(p)

    assert implementation in ['tensor', 'numba', 'gpu'], (
        "implementation must be 'tensor', 'numba', or 'gpu'.")

    if implementation == 'gpu':
        from torchpac.methods.meth_pac_gpu import (
            mean_vector_length_gpu, modulation_index_gpu, heights_ratio_gpu,
            norm_direct_pac_gpu, phase_locking_value_gpu, gauss_cop_pac_gpu)
        METH = {
            1: partial(mean_vector_length_gpu,  device=device, dtype=dtype),
            2: partial(modulation_index_gpu,    device=device, dtype=dtype,
                       n_bins=int(n_bins)),
            3: partial(heights_ratio_gpu,       device=device, dtype=dtype,
                       n_bins=int(n_bins)),
            4: partial(norm_direct_pac_gpu,     device=device, dtype=dtype,
                       p=float(p)),
            5: partial(phase_locking_value_gpu, device=device, dtype=dtype),
            6: partial(gauss_cop_pac_gpu,       device=device, dtype=dtype),
        }
    elif implementation == 'tensor':
        from torchpac.methods.meth_pac import (
            mean_vector_length, modulation_index, heights_ratio,
            norm_direct_pac, phase_locking_value, gauss_cop_pac)
        METH = {
            1: partial(mean_vector_length),
            2: partial(modulation_index, n_bins=n_bins),
            3: partial(heights_ratio, n_bins=n_bins),
            4: partial(norm_direct_pac, p=p),
            5: partial(phase_locking_value),
            6: partial(gauss_cop_pac)}
    else:  # numba
        from torchpac.methods.meth_pac_nb import (
            mean_vector_length_nb, modulation_index_nb, heights_ratio_nb,
            norm_direct_pac_nb, phase_locking_value_nb)
        from torchpac.methods.meth_pac import gauss_cop_pac
        METH = {
            1: partial(mean_vector_length_nb),
            2: partial(modulation_index_nb, n_bins=n_bins),
            3: partial(heights_ratio_nb, n_bins=n_bins),
            4: partial(norm_direct_pac_nb, p=p),
            5: partial(phase_locking_value_nb),
            6: partial(gauss_cop_pac)}

    return METH if full else METH[idp]


def pacstr(idpac):
    """Return correspond methods string."""
    # Pac methods :
    if idpac[0] == 1:
        method = 'Mean Vector Length (MVL, Canolty et al. 2006)'
    elif idpac[0] == 2:
        method = 'Modulation Index (MI, Tort et al. 2010)'
    elif idpac[0] == 3:
        method = 'Heights ratio (HR, Lakatos et al. 2005)'
    elif idpac[0] == 4:
        method = 'Normalized Direct Pac (ndPac, Ozkurt et al. 2012)'
    elif idpac[0] == 5:
        method = 'Phase-Locking Value (PLV, Penny et al. 2008)'
    elif idpac[0] == 6:
        method = 'Gaussian Copula PAC (gcPac)'
    else:
        raise ValueError("No corresponding pac method.")

    # Surrogate method :
    if idpac[1] == 0:
        suro = 'No surrogates'
    elif idpac[1] == 1:
        suro = 'Permute phase across trials (Tort et al. 2010)'
    elif idpac[1] == 2:
        suro = 'Swap amplitude time blocks (Bahramisharif et al. 2013)'
    elif idpac[1] == 3:
        suro = 'Time lag (Canolty et al. 2006)'
    else:
        raise ValueError("No corresponding surrogate method.")

    # Normalization methods :
    if idpac[2] == 0:
        norm = 'No normalization'
    elif idpac[2] == 1:
        norm = 'Substract the mean of surrogates'
    elif idpac[2] == 2:
        norm = 'Divide by the mean of surrogates'
    elif idpac[2] == 3:
        norm = 'Substract then divide by the mean of surrogates'
    elif idpac[2] == 4:
        norm = "Substract the mean and divide by the deviation of the " + \
               "surrogates"
    else:
        raise ValueError("No corresponding normalization method.")

    return method, suro, norm
