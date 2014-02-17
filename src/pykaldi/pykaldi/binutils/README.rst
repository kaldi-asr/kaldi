Running the example Pykaldi scripts
===================================

Summary
-------
The `<pykaldi-latgen-faster-decoder.py>`_
demonstrates how to use the class ``PyGmmLatgenWrapper``,
which takes audio on the input and outputs the decoded lattice.

Dependencies
------------
* For inspecting the saved lattices you need `dot` binary 
  from `Graphviz <http://www.graphviz.org/Download..php>`_ library.
* For running the live demo you need ``pyaudio`` package.

Running the example scripts
---------------------------
* Run the decoding with ``PyGmmLatgenWrapper`` on small test set
  with pretrained Language (LM) and Acoustic (AM) models.
  The data as well as the models are downloaded from our server.
  The logic is written in `<pykaldi-latgen-faster-decoder.py>`_.

.. code-block:: bash

    make pykaldi-latgen-faster

* Run the decoding with Kaldi binary utils wrapped in `<run_gmm-latgen-faster.sh>`_.
  The same data, AM a LM are used as for ``make pykaldi-latgen-faster``.
  We use this script as reference.

.. code-block:: bash

    make gmm-latgen-faster

* The stupid simple live demo should decode speech from your microphone.
  It uses the pretrained AM and LM and wraps `<live-demo.py>`_. 
  The ``pyaudio`` package is used for capturing the sound from your microphone.
  We were able to use it under `Ubuntu 12.10`, but we guarantee nothing on your system.

.. code-block:: bash

    make live
