 
 This is a Kaldi recipe for speaker verification using the VoxCeleb1 and
 VoxCeleb2 corpora.  See http://www.robots.ox.ac.uk/~vgg/data/voxceleb/ and 
 http://www.robots.ox.ac.uk/~vgg/data/voxceleb2/ for additional details and
 information on how to obtain them.

 Note: This recipe requires ffmpeg to be installed and its location included
 in $PATH

 The subdirectories "v1" and so on are different speaker recognition
 recipes. The recipe in v1 demonstrates a standard approach using a
 full-covariance GMM-UBM, iVectors, and a PLDA backend.  The example 
 in v2 demonstrates DNN speaker embeddings with a PLDA backend.
