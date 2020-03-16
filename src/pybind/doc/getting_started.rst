
Getting Started
###############


Compiling Kaldi Pybind
======================

First, you have to install Kaldi. You can find
detailed information for Kaldi installation
from
`http://kaldi-asr.org/doc/install.html <http://kaldi-asr.org/doc/install.html>`_.

.. Note::

    Kaldi Pybind is still under active development and has not
    yet been merged into the master branch. You should checkout
    the ``pybind11`` branch before compilation.

.. Note::

    We support **ONLY** Python3. If you are still using Python2,
    please upgrade to Python3. Python3.5 is known to work.

The following is a quick start:

.. code-block:: bash

    git clone https://github.com/kaldi-asr/kaldi.git
    cd kaldi
    git checkout pybind11
    cd tools
    extras/check_dependencies.sh
    make -j4
    cd ../src
    ./configure --shared
    make -j4
    cd pybind
    pip install pybind11
    make
    make test

After a successful compilation, you have to modify the environment
variable ``PYTHONPATH``:

.. code-block:: bash

  export KALDI_ROOT=/path/to/your/kaldi
  export PYTHONPATH=$KALDI_ROOT/src/pybind:$PYTHONPATH

.. HINT::

  There is no ``make install``. Once compiled, you are ready to
  use Kaldi Pybind.
