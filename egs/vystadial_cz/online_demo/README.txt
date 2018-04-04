Running the example Pykaldi scripts
===================================

Summary
-------
The demo presents three new Kaldi features on pretrained Czech AMs:
* Online Lattice Recogniser. The best results were obtained using MFCC, LDA+MLLT and bMMI.
* Python wrapper which interfaces the OnlineLatticeRecogniser to Python.
* Training scripts which can be used with standard Kaldi tools or with the new OnlineLatticeRecogniser.

The pykaldi-latgen-faster-decoder.py
demonstrates how to use the class PyOnlineLatgenRecogniser,
which takes audio on the input and outputs the decoded lattice.
There are also the OnlineLatgenRecogniser C++ and Kaldi standard gmm-latgen-faster demos.
All three demos produce the same results.

TODO: Publish English AM and add English demo

In March 2014, the PyOnlineLatticeRecogniser recogniser was evaluated on domain of SDS Alex. 
See graphs evaluating OnlineLatticeRecogniser performance at 
http://nbviewer.ipython.org/github/oplatek/pykaldi-eval/blob/master/Pykaldi-evaluation.ipynb.

An example posterior word lattice output for one Czech utterance can be seen at 
http://oplatek.blogspot.it/2014/02/ipython-demo-pykaldi-decoders-on-short.html

Dependencies
------------
* Build (make) and test (make test) the code under  kaldi/src, kaldi/src/pykaldi and kaldi/src/onl-rec
* For inspecting the saved lattices you need dot binary 
  from Graphviz <http://www.graphviz.org/Download..php library.
* For running the live demo you need pyaudio package.

Running the example scripts
---------------------------


    make online-latgen-recogniser

* Run the test src/onl-rec/onl-rec-latgen-recogniser-test for OnlineLatgenRecogniser
  which shows C++ example of how to use the recogniser.
  The same data, AM a LM are used as for make pyonline-latgen-recogniser.
  The pretrained Language (LM) and Acoustic (AM) models are used.
  The data as well as the models are downloaded from our server.


    make pyonline-latgen-recogniser

* Run the decoding with PyOnlineFasterRecogniser. 
  Example Python script pykaldi-online-latgen-recogniser.py shows 
  PyOnlineFasterRecogniser decoding  on small test set.
  The same pretrained Language (LM) and Acoustic (AM) models.


    make gmm-latgen-faster

* Run the decoding with Kaldi gmm-latgen-faster executable wrapped in `<run_gmm-latgen-faster.sh>`_.
  This is the reference executable for 
  The same data, AM a LM are used as for make pyonline-latgen-recogniser.
  We use this script as reference.


    make live

* The simple live demo should decode speech from your microphone.
  It uses the pretrained AM and LM and wraps `<live-demo.py>`_. 
  The pyaudio package is used for capturing the sound from your microphone.
  We were able to use it under `Ubuntu 12.10` and Python 2.7, but we guarantee nothing on your system.


Notes
-----
 The scripts for Czech and English support acoustic models obtained using MFCC, LDA+MLLT/delta+delta-delta feature transformations and acoustic models trained generatively or by MPE or bMMI training.

The new functionality is separated to different directories:
 * kaldi/src/onl-rec stores C++ code for OnlineLatticeRecogniser.
 * kaldi/scr/pykaldi stores Python wrapper PyOnlineLatticeRecogniser.
 * kaldi/egs/vystadial/s5 stores training scripts.
 * kaldi/egs/vystadial/online_demo shows Kaldi standard decoder, OnlineLatticeRecogniser and PyOnlineLatticeRecogniser, which produce the exact same lattices using the same setup.

The OnlineLatticeRecogniser is used in Alex dialogue system (https://github.com/UFAL-DSG/alex).
