 #!/bin/bash

# The script automatically choose default settings of miniconda for installation
# Miniconda will be installed in the HOME directory. ($HOME/miniconda3).
# Also don't make miniconda's python as default.

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b

$HOME/miniconda3/bin/python -m pip install --user tqdm
$HOME/miniconda3/bin/python -m pip install --user scikit-learn
$HOME/miniconda3/bin/python -m pip install --user librosa
$HOME/miniconda3/bin/python -m pip install --user h5py
