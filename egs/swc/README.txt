
		About The Sheffield Wargame Corpus (SWC)




                                --

			About This Recipe
				s5)

The scripts under this directory build systems using SWC data only, this
includes training, development and evaluation sets in different setup.
The scripts also prepares a shared infrastructure for user-defined dataset.

Note that the recipe WILL NOT download any of SWC data for you, because
it requires to sign an ethical agreement before downloading could be
processed.

One language model (LM) built from web data and Warhammer 40k blog data is 
provided especially for SWC task. It will be downloaded from Sheffield 
University website within this recipe. 

This recipe follows the framework in AMI recipe and builds systems using

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

The scoring folders for baseline systems reported in [1] have been copied 
to file "s5/RESULTS.txt".

For more information about SWC, please visit the following website:
	
	http://mini-vm20.dcs.shef.ac.uk/swc/SWC-home.html

For any questions, please send email to the swc-user google group:

        swc-users@googlegroups.com
        https://groups.google.com/forum/#!forum/swc-users


                                ---

			      Citation


[1] SWC1: 	"The Sheffield Wargame Corpus", Charles Fox, Yulan Liu, Erich
Zwyssig and Thomas Hain, Interspeech 2013, Lyon, France, 25-29 Aug 2013. 

[2] SWC2, SWC3:	"The Sheffield Wargame Corpus - Day Two And Day Three", Yulan
Liu, Charles Fox, Madina Hasan and Thomas Hain, Interspeech 2016, San
Fransisco, USA, 8-12 Sep 2016.

[3] BeamformIt: "Acoustic beamforming for speaker diarization of meetings",
Xavier Anguera, Chuck Wooters and Javier Hernando, IEEE Transactions on Audio,
Speech and Language Processing, September 2007, volume 15, number 7,
pp.2011-2023.


                               ----

			      Contact

For any problems, bugs in this recipe or SWC data, please contact:

	* Yulan Liu <acp12yl@sheffield.ac.uk>
	* Thomas Hain <t.hain@sheffield.ac.uk>  http://mini.dcs.shef.ac.uk



                               -----

			  Acknowledgement

Thanks the bloggers who agreed to allow free use of their blog text to
build a LM tailered for desktop game Warhammer 40k. The blog data and the
LM are strictly limited for research purposes only. For more information
about these contributers, please feel free to visit their blogs:

        http://castigatorschaos.blogspot.co.uk
        http://atomicspud40k.blogspot.co.uk
        http://cadiascreed40k.blogspot.co.uk
        http://40kaddict.blogspot.co.uk


                              ------

