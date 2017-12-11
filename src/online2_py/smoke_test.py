#!/usr/bin/env python3

import online2_nnet3_latgen_i2x_wrapper as m

import numpy
import random
import timeit
import wave


def printRecoResult(rr):
    print("[ num frames = '{:5d}', average per-frame likelihood = {:02.3f} ], {}".format(
        rr.num_frames, rr.mean_frame_likelihood, str(rr.transcript, 'utf8')))

wavefile = numpy.random.randint(low=-2**15, high=2**15, size=16000*10, dtype=numpy.int16)
dfactory = m.DecoderFactory(b"../online2bin/resourcedir/")
decoder = dfactory.StartDecodingSession()
decoder.FeedBytestring(wavefile.tostring())
decoder.Finalize()
rr = decoder.GetResult()
printRecoResult(rr)

f = open(b"/home/christoph/data/transcription/to_transcribe2/16khz/e9603e8f814dbe98d233b04af040ad3154f689c34128026a7b93bbd8924b8623.webm.wav", 'rb')
w = wave.open(f)
w_samples = w.readframes(w.getnframes())

def run_decoding(partial, low=1, high=200):
    decoder = dfactory.StartDecodingSession()
    i = 0
    while i < len(w_samples):
        chunk_length = random.randint(low, high)
        decoder.FeedBytestring(w_samples[i:i+chunk_length])
        if partial:
            result = decoder.GetResult()
            printRecoResult(result)
        i += chunk_length
    decoder.Finalize()
    return decoder

decoder = run_decoding(partial=True, low=100, high=16000)
decoder.Finalize()
rr = decoder.GetResult()
decoder.InvalidateAndFree()
printRecoResult(rr)

NUM = 3
elapsed = timeit.repeat("run_decoding(partial=False)", setup="from __main__ import run_decoding", number=1, repeat=NUM)
length = w.getnframes() / w.getframerate()
print("Elapsed: {}, Sound length: {}, best-case RTF: {}".format(elapsed, length, min(elapsed) / length))
