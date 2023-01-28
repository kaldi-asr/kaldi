A. Alexander Thornton (apt2141)

B. May 8, 2022

C. Project Title: Speaker Diarization: Deep Speech Embeddings for Time Delay Neural Networks (TDNN)

D. Project Summary:

Abstract—The fundamental problem of Speaker Diarization
can be simplified as ”who spoke when”. At its essence, Speaker
Diarization can be reduced to the traditional Speaker Identifica-
tion problem, but expanded to N interleaving speakers through
time. This work improves upon the existing Speaker Diarization
project in Kaldi, which was incomplete and unfinished prior to
my efforts.
Index Terms—speaker identification, diarization, time delay
neural networks, time series learning

E. All tools are included with the code here. Build Kaldi, and you can just run the run.sh

F. Only use run.sh, the stage is set to 7 for decoding

G. Run the code with this simple command:

	./run.sh

All environment variables are defined inside

Sample output will appear at the bottom, with the test accuracy

H. The data used was built off a MANIFEXT file downloaded here:

https://groups.inf.ed.ac.uk/ami/download/temp/amiBuild-1372-Thu-Apr-28-2022.manifest.txt

It's important to know that those files change daily, and are constantly changing, so this one might already be gone


	
