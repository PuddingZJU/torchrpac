.. _Installation:

Installation
============

Requirements
++++++++++++

Torchpac relies on three packages :

* `NumPy <https://www.numpy.org/>`_
* `SciPy <https://www.scipy.org/>`_
* `Joblib <https://joblib.readthedocs.io/en/latest/>`_

Then if you want to be able to plot your results you'll need to install
`Matplotlib <https://matplotlib.org/>`_.

Some additional packages might also be required, in particular :

* `MNE Python <https://mne.tools/stable/index.html>`_ for running some examples, in particular some statistical functions are needed
* `Numba <http://numba.pydata.org/>`_, a Python compiler to speed up some functions

Standard installation
+++++++++++++++++++++

Torchpac can be installed using pip. In a terminal, run the following command :

.. code-block:: shell

    pip install torchpac

And if you want want to update to the latest version :

.. code-block:: shell

    pip install -U torchpac

Install the most up-to-date version
+++++++++++++++++++++++++++++++++++

The latest version is hosted on `github <https://github.com/EtienneCmb/torchpac>`_.
This is always going to be the most up-to-date version, with the latest features and fixes.
If you want to install this version, open a terminal and run the following commands :

.. code-block:: shell

    git clone https://github.com/EtienneCmb/torchpac.git
    cd torchpac/
    python setup.py develop

Finally, if you want to update your version you can use :

.. code-block:: shell

    git pull
