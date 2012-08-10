#!/bin/bash



######################################################################
# Prepare the data, train the baseline GMM-based systems
bash run.A.gmm.sh


######################################################################
# Train the pure hybdir systems (GMM replaced by ANN)
bash run.B.hybrid.sh


######################################################################
# Train the tandem systems (ANN as feature extraction for GMM)
bash run.C.tandem.sh


######################################################################
# Train deep hybrid system (ANN pretrained by RBMs and Xentropy opt.)
bash run.D.deep.sh

