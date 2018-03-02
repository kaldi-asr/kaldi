 #!/bin/bash

# Choose default settings for installation
# Miniconda should be installed in the HOME directory. This is the default setting.
# Please don't change it.
# Also don't make miniconda's python as default.

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

$HOME/miniconda3/bin/python -m pip install --user tqdm
$HOME/miniconda3/bin/python -m pip install --user scikit-learn
$HOME/miniconda3/bin/python -m pip install --user librosa
$HOME/miniconda3/bin/python -m pip install --user h5py
