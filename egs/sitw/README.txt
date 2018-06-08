 
 This directory (sitw) contains example scripts for the Speakers in the
 Wild (SITW) Speaker Recognition Challenge.  The SITW corpus is required,
 and can be obtained by following the directions at the url
 http://www.speech.sri.com/projects/sitw/

 Additional data sources (e.g., VoxCeleb and MUSAN) are required to train
 the systems in the subdirectories.  See the corresponding README.txt files
 in the subdirectories for more details. 

 Note: This recipe requires ffmpeg to be installed and its location included
 in $PATH.

 The subdirectories "v1" and so on are different speaker recognition
 recipes. The recipe in v1 is a traditional i-vector system while the v2
 recipe uses DNN embeddings called x-vectors.
