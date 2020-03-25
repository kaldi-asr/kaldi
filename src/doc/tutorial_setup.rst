
Kaldi tutorial: Getting started (15 minutes)
=============================================


The first step is to download and install Kaldi. Be aware that the code and scripts in the "trunk" (which is always up to date) is easier to install and is generally better. If you use the "trunk" code you can also try to use the most recent scripts, which are in the directory ``"egs/rm/s5"``. But be aware that if you do that some aspects of the tutorial may be out of date.

Assuming Git is installed, to get the latest code you can type::

    git clone https://github.com/kaldi-asr/kaldi.git

Then cd to kaldi. Look at the ``INSTALL`` file and follow the instructions (it points you to two subdirectories). Look carefully at the output of the installation scripts, as they try to guide you on what to do. Some installation errors are non-fatal, and the installation scripts will tell you so (i.e. there are some things it installs which are nice to have but are not really needed). The "best-case" scenario is that you do::

    cd kaldi/tools/; make; cd ../src;  ./configure; make

and everything will just work; however, if this does not happen there are fallback plans (e.g. you may have to install some package on your machine, or run ``install_atlas.sh`` in tools/, or run some steps in ``tools/INSTALL`` manually, or provide options to the configure script in ``src/``). If there are problems, there may be some information in `The build process (how Kaldi is compiled) <pages/api-undefined.md#build_setup>`_ that will help you; otherwise, feel free to contact the maintainers (\ `Other Kaldi-related resources (and how to get help) <pages/api-undefined.md#other>`_\ ) and we will be happy to help.

