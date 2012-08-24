
This directory hosts example binaries that implement online decoding.

The online decoding code depends on libportaudio and it is assumed that you 
already compiled and installed portaudio in the ../../tools/ folder with the help of
the install_portaudio.sh script that is provided there. Unfortunately, portaudio
is more OS dependent then the other Kaldi code and you may face some issues, 
depending on your OS.
Please also refer to the "Install instructions for PortAudio" section in tools/INSTALL.

!!CAUTION!! IF YOU ARE TRYING TO RUN THESE BINARIES ON MAC OS, YOU MAY
NEED TO TELL THE BINARIES WHERE THEY CAN FIND LIBPORTAUDIO, SINCE IT IS
DYNAMICALLY LINKED. TO AVOID THIS, YOU CAN SIMPLY COPY libportaudio.dylib
TO /usr/local/lib. YOU CAN FIND THE DYLIB FILES IN tools/portaudio/install/lib
AFTER HAVING RUN THE INSTALLATION SCRIPT FOR PORTAUDIO IN THE tools FOLDER.

Here is a short overview on the provided binaries.

#########################

online-gmm-decode-faster: 

Demonstrates one possible implementation for online decoding. The binary
records audio from your sound card, computes the features on-the-fly, performs 
decoding and finally displays the recognition output on stdout. Recognition 
output is displayed at a very low latency (and even before the end of an utterance
is reached) by doing a partial trace-back whenever possible.

A very simple heuristic to detect the end of an utterance is employed:
whenever x frames of silence (any consecutive sequence of "silence" phones) 
are detected at the beginning of  the trace-back, we display the full trace-back 
and re-set the language model context. For easy readability, we insert two line
breaks at the end of an utterance. The value of x (50 frames initially) is being
lowered automatically whenever the current utterances becomes too long (longer than
max-utt-length frames). The end pointing behavior can therefore be influenced via 
the optional parameter max-utt-length (the lower the value, the shorter the average 
utterance length) and by defining what phones constitute silence (sil, noise, 
laughter, etc.)

Decoding has to happen "fast enough" (real time factor, RTF < 1) so that the decoder
can keep up with the live recorded audio. If the decoder is too slow (because of a 
slow machine, a too high beam, too big models), you will observe buffer overflow 
error messages (provided that you have set the "--verbose=1" option), which means 
that audio samples got lost. 

To avoid dropped audio samples, an (imperfect) heuristic is used that tries to 
keep the RTF between the two values rt-min and rt-max by adjusting the decoding 
beam in the following manner: The effective beam width is updated every 
"update-interval"-th frame. The beam is scaled up or down by a fraction of its
current size given by "beam-update" times a factor which depends on how much the
current decoding time is off from the set target. The fraction by which the beam
is updated however cannot be more than "max-beam-update" and the beam can never
become wider than the value configured with the "--beam" option.

!!CAUTION!! AS MENTIONED, THE HEURISTIC DOES NOT ALWAYS WORK. IT IS THEREFORE 
IMPORTANT TO USE A REASONABLE MAX BEAM AND A REASONABLE NUMBER FOR MAX ACTIVE 
STATE. USE THE VERBOSE OPTION TO LOOK FOR BUFFER OVERFLOW MESSAGES!!

Another option that can influence RTF and also latency is "batch-size". 
Feature computation and decoding happens in batches of batch-size frames. 
The default value is 27 frames. The higher the value, the higher the latency, 
but smaller values may increase RTF.

#########################

online-wav-gmm-decode-faster:

Simulates online decoding on wav files. This is useful to measure word error rate
and to tune parameters, such as batch size etc.

#########################

online-net-client & online-server-gmm-decode-faster:

Enables online decoding in a client-server fashion, that is, doing audio recording
and feature computation on the client and sending the raw features to the recognition
server to do the decoding. The recognition server sends the recognition result back
to the client for display. UDP is used for communication. This is very useful when
one wants to run big models that require lots of CPU power to keep the RTF low. 
We chose UDP, because we want to decode in real time and retransmissions would lead
to delay and losing samples at PortAudio level. No guarantees are made about
re-ordered packages but we haven't faced any such issues in our testing.
