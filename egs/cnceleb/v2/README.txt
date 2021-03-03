 This recipe replaces ivectors used in the v1 recipe with embeddings extracted
 from a deep neural network.  In the scripts, we refer to these embeddings as
 "xvectors".  The recipe is closely based on the following paper:
 http://www.danielpovey.com/files/2018_icassp_xvectors.pdf but uses a wideband
 rather than narrowband MFCC config.

 In addition to the CN-Celeb datasets used for training and evaluation (see
 ../README.txt), we also use the following datasets for augmentation.

     MUSAN               http://www.openslr.org/17
     RIR_NOISES          http://www.openslr.org/28
