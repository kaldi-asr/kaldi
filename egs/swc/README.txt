
		About The Sheffield Wargame Corpus (SWC)




                                --

			About The Recipes


                                s5b)

The scripts under directory "s5b" build systems using SWC data only, this 
includes training, development and evaluation sets in different setups.
The scripts also prepare a shared infrastructure for user-defined dataset.

Note that the recipe WILL NOT download any of the SWC data for you, because
it requires a signed ethical agreement before the SWC data could be
downloaded.

One language model (LM) built from web data and Warhammer 40k blog data is
provided especially for SWC task. You could download it from the following
website:
	
	http://mini-vm20.dcs.shef.ac.uk/swc/LM.html

This recipe follows the framework in the AMI recipe "s5b" and builds 
systems using

        * IHM (Individual Headset Microphone)
        * SDM (Single Distant Microphone)
        * MDM (Multiple Distant Microphone)

The first channel of the 8 channel circular array on the table (*TBL1-01*)
is used in this recipe as SDM baseline.

The same array (*TBL1-*) is suggested to run with BeamformIt to build a
comparable MDM baseline.

To reproduce the results in this recipe, the following (non-standard)
software is expected to be installed for beamforming:

        * BeamformIt (for MDM scenario, installed with Kaldi tools)



			      ---

			      Results

The scoring results for baseline systems reported in [2] are based on an
earlier version of the recipe ("s5") which could be downloaded either from
the SWC website:

        http://mini-vm20.dcs.shef.ac.uk/swc/Kaldi_recipe.html

or from Yulan's git repository (the master branch):

        https://github.com/wings619/kaldi.git

It is worth mentioning that all the results and analysis on the SWC website 
were acquired with the "s5" version. 

The scoring results with for the baseline systems based on the current "s5b" 
version recipe are detailed in the file "s5b/RESULTS.txt". 



                                ---

			      Citations


[1] SWC1: 	"The Sheffield Wargame Corpus", Charles Fox, Yulan Liu, Erich
Zwyssig and Thomas Hain, Interspeech 2013, Lyon, France, 25-29 Aug 2013. 

[2] SWC2, SWC3:	"The Sheffield Wargame Corpus - Day Two and Day Three", Yulan 
Liu, Charles Fox, Madina Hasan and Thomas Hain, Interspeech 2016, San Francisco, 
USA, 8-12 Sep 2016.

[3] BeamformIt: "Acoustic beamforming for speaker diarization of meetings",
Xavier Anguera, Chuck Wooters and Javier Hernando, IEEE Transactions on Audio,
Speech and Language Processing, September 2007, volume 15, number 7,
pp.2011-2023.



                               ----

			      Contacts

For any problems, bugs in this recipe or SWC data, please contact:

	* Yulan Liu <acp12yl@sheffield.ac.uk>
	* Thomas Hain <t.hain@sheffield.ac.uk>  http://mini.dcs.shef.ac.uk



                               -----

			  Acknowledgements

Thanks the bloggers who agreed to allow free use of their blog text to
build a LM tailered for desktop game Warhammer 40k. The blog data and the
LM are strictly limited for research purposes only. For more information
about these contributers, please feel free to visit their blogs:

        http://castigatorschaos.blogspot.co.uk
        http://atomicspud40k.blogspot.co.uk
        http://cadiascreed40k.blogspot.co.uk
        http://40kaddict.blogspot.co.uk


                              ------

Last update: 22 Feb 2017 by Yulan Liu.

