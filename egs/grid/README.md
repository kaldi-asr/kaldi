Kaldi-based audio-visual speech recognition
================================

A baseline system for audio-visual speech recognition using the Kaldi speech recognition toolkit is provided.

The scripts contain the early integration approach that is presented in:

H. Meutzner, N. Ma, R. Nickel, C. Schymura, D. Kolossa, "Improving Audio-Visual Speech Recognition using Deep Neural Networks with Dynamic Stream Reliability Estimates", ICASSP, New Orleans, USA, March 2017.

Future releases will also contain the late-integration approach using dynamic stream weights.


Data description
--------------------------

The experiments are based on the audio data of the CHiME-2 challenge [1] and the video data of the GRID audio-visual speech corpus [2,3].

The audio data has to be manually obtained from the official CHiME-2 track 1 website [3].

The video features have been precomputed using the video files of the GRID corpus and will be automatically obtained from
http://doi.org/10.5281/zenodo.260211
when running the scripts.

The video features contain the 63-dimensional DCT coefficients of the landmark points extracted using the Viola-Jones algorithm. The features have been end-pointed and interpolated using a differential digital analyser in order to match the length of the utterances when using a frame length of 25ms and a frame shift of 10ms, which is the default configuration of Kaldi's feature extraction scripts.

[1] http://spandh.dcs.shef.ac.uk/chime_challenge/chime2013/chime2_task1.html

[2] http://spandh.dcs.shef.ac.uk/gridcorpus

[3] Martin Cooke, Jon Barker, and Stuart Cunningham and Xu Shao, "An audio-visual corpus for speech perception and automatic speech recognition", The Journal of the Acoustical Society of America 120, 2421 (2006); http://doi.org/10.1121/1.2229005


License and Citation
--------------------------

The scripts are released under the Apache 2.0 license. The video features are released under the Creative Commons Attribution-NonCommercial 4.0 license.

When using these scripts for your research, please cite the following paper

	@inproceedings{meutzner2017,
	  author = {Hendrik Meutzner, Ning Ma, Robert Nickel, Christopher Schymura, Dorothea Kolossa},
	  title = {{Improving Audio-Visual Speech Recognition using Deep Neural Networks with Dynamic Stream Reliability Estimates}},
	  booktitle = {{IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},      
	  year = {2017}
	}
