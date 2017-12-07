import online2_nnet3_latgen_i2x_wrapper as m
import numpy

wavefile = numpy.random.randint(low=-2**15, high=2**15, size=16000*10, dtype=numpy.int16)
dfactory = m.DecoderFactory(b"../online2bin/resourcedir/")
decoder = dfactory.StartDecodingSession()
decoder.FeedBytestring(wavefile.tostring())
decoder.FeedBytestring(b"")
rr = m.RecognitionResult()
decoder.GetResultAndFinalize(rr.this)
print(str(rr.transcript, 'utf8'))

import wave
f = open(b"/home/christoph/data/transcription/to_transcribe2/16khz/e9603e8f814dbe98d233b04af040ad3154f689c34128026a7b93bbd8924b8623.webm.wav", 'rb')
w = wave.open(f)
www = w.readframes(w.getnframes())
decoder = dfactory.StartDecodingSession()
decoder.FeedBytestring(www)
decoder.FeedBytestring(b"")
rr = m.RecognitionResult()
decoder.GetResultAndFinalize(rr.this)
print(str(rr.transcript, 'utf8'))
