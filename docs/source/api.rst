.. _api:

API
===

.. contents::
   :local:
   :depth: 2

.. _fcncfc:

Compute phase-amplitude coupling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:mod:`torchpac`:

.. currentmodule:: torchpac

.. autosummary::
   :toctree: generated/

   Pac
   EventRelatedPac
   PreferredPhase

Utility functions
~~~~~~~~~~~~~~~~~

:py:mod:`torchpac.utils`:

.. currentmodule:: torchpac.utils

.. autosummary::
   :toctree: generated/

   PSD
   ITC
   BinAmplitude
   PeakLockedTF
   pac_vec
   pac_trivec

Generate synthetic signals
~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:mod:`torchpac.signals`:

.. currentmodule:: torchpac.signals

.. autosummary::
   :toctree: generated/

   pac_signals_wavelet
   pac_signals_tort

Statistics
~~~~~~~~~~

:py:mod:`torchpac.stats`:

.. currentmodule:: torchpac.stats

.. autosummary::
   :toctree: generated/

   test_stationarity


Individual methods
~~~~~~~~~~~~~~~~~~

:py:mod:`torchpac.methods`:

PAC methods
+++++++++++

If you don't want to use the :class:`torchpac.Pac` class, you can also manually import the method of your choice
and use it on phase / amplitude to compute PAC. Note that some functions have both a tensor or Numba-based implementation.

Tensor-based implementation
***************************

.. currentmodule:: torchpac.methods

.. autosummary::
   :toctree: generated/

   mean_vector_length
   modulation_index
   heights_ratio
   norm_direct_pac
   phase_locking_value
   gauss_cop_pac

Numba-based implementation
**************************

.. autosummary::
   :toctree: generated/

   mean_vector_length_nb
   modulation_index_nb
   heights_ratio_nb
   norm_direct_pac_nb
   phase_locking_value_nb


Event Related PAC methods
+++++++++++++++++++++++++

.. currentmodule:: torchpac.methods

.. autosummary::
   :toctree: generated/

   erpac
   ergcpac

Preferred phase
+++++++++++++++

.. currentmodule:: torchpac.methods

.. autosummary::
   :toctree: generated/

   preferred_phase

Surrogates methods
++++++++++++++++++

.. currentmodule:: torchpac.methods

.. autosummary::
   :toctree: generated/

   swap_pha_amp
   swap_blocks
   time_lag

Normalization
+++++++++++++

.. currentmodule:: torchpac.methods

.. autosummary::
   :toctree: generated/

   normalize
