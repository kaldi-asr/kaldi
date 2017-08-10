#!/bin/bash

# this script checks if TF is installed to be used with python
#                    and if TF related binaries in kaldi is ready to use
. path.sh

if which lattice-lmrescore-tf-rnnlm 2>&1>/dev/null; then
  echo TensorFlow relate binaries found. This is good.
else
  echo TF related binaries not compiled.
  echo You would need to run tools/extras/install_tensorflow_cc.sh first
  echo and then do \"make\" under both src/tfrnnlm and src/tfrnnlmbin
  exit 1
fi

echo

if python steps/tfrnnlm/check_py.py 2>/dev/null; then
  echo TensorFlow ready to use on the python side. This is good.
else
  echo TensorFlow not found on the python side.
  echo Please run tools/extras/install_tensorflow_py.sh to install it
  echo If you already have TensorFlow installed somewhere else, you would need
  echo to add it to your PATH
  exit 1
fi
