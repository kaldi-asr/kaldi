This recipe is using a publicly available subset of Resource Management data, 
distributed by CMU.

To run the recipe the data should be downloaded first, for which ./getdata.sh
command can be used. Then ./run.sh script can be executed to automatically perform
all steps or the commands can be started manually by copy/pasting them. 

The script and data layout are based on egs/rm/s3 recipe, with several exceptions:

- because this recipe uses pre-extracted feature vectors no conversion from .sph
to .wav format and consequent feature extraction is needed. The features are just
converted from CMU Sphinx feature files to Kaldi Tables.

- only one test set is available instead of several (e.g. mar87, oct87 and so on)
as in the original recipe

- no speaker-dependent processing

- only the steps up to tri2a stage are implemented

- on the plus side it requires less disk space (about 220MB)
