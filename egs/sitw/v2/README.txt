 This recipe replaces i-vectors used in the v1 recipe with embeddings extracted
 from a deep neural network.  In the scripts, we refer to these embeddings as
 "x-vectors."  The recipe in local/nnet3/xvector/tuning/run_xvector_1a.sh is
 closesly based on the following paper:

 @inproceedings{snyder2018xvector,
 title={X-vectors: Robust DNN Embeddings for Speaker Recognition},
 author={Snyder, D. and Garcia-Romero, D. and Sell, G. and Povey, D. and Khudanpur, S.},
 booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 year={2018},
 organization={IEEE},
 url={http://www.danielpovey.com/files/2018_icassp_xvectors.pdf}
 }

 The recipe uses the following datasets:

 Evaluation
     
     Speakers in the Wild    http://www.speech.sri.com/projects/sitw

 System Development
     
     VoxCeleb 1              http://www.robots.ox.ac.uk/~vgg/data/voxceleb
     VoxCeleb 2              http://www.robots.ox.ac.uk/~vgg/data/voxceleb2
     MUSAN                   http://www.openslr.org/17
     RIR_NOISES              http://www.openslr.org/28
