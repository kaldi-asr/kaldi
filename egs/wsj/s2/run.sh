#!/bin/bash



######################################################################
# Prepare the data, train the baseline GMM-based systems
bash run.A.gmm.sh || exit 1


######################################################################
# Train the pure hybdir systems (GMM replaced by ANN)
bash run.B.hybrid.sh || exit 1


######################################################################
# Train the tandem systems (ANN as feature extraction for GMM)
bash run.C.tandem.sh || exit 1


######################################################################
# Train deep hybrid system (ANN pretrained by RBMs and Xentropy opt.)
bash run.D.deep.sh || exit 1

