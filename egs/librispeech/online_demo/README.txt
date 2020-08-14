This directory contains a small demo, which gives an example of how the online
decoding binaries can be used. These binaries are optional so if you want to use
them please type "make ext" in KALDI_ROOT/src. Pre-built acoustic models are 
utilized, which were trained on VoxForge's data(using egs/voxforge/s5). The 
decoding graphs are constructed using a language model built on a public domain 
book's text. There are two test modes: simulated online decoding, where the audio
comes from pre-recorded audio snippets, and live decoding mode, where the user can 
try how his speech is recognized with the afore-mentioned language and acoustic
models. The microphone input is implemented using PortAudio library.
The audio(for the "simulated input" mode) and the models are automatically
downloaded the first time when the script is started.

You can start the demo in "simulated" mode by just calling:
./run.sh

To try using your own voice(assuming you have a working microphone) call:
./run.sh --test-mode live

By default a discriminatively trained acoustic models is used. To try a
maximum likelihood trained one(usually has lower accuracy) add
"--ac-model-type tri2a" when running the above commands.
