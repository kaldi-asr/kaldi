#!/usr/bin/env python3

import concurrent.futures
import wave

import online2_nnet3_latgen_i2x_wrapper as m

dfactory = m.DecoderFactory(b"../online2bin/resourcedir/")

wave_list = []
with open(b"../online2bin/resourcedir/wav.scp") as wavscp_fid:
    for line in wavscp_fid.readlines():
        wave_list.append(line.split()[1])


def printRecoResult(rr):
    print("[ num frames = '{:5d}', average per-frame likelihood = {:02.3f} ], {}".format(
        rr.num_frames, rr.mean_frame_likelihood, str(rr.transcript, 'utf8')))


def recognize(wavfile):
    with wave.open(wavfile, 'rb') as wave_fid:
        data = wave_fid.readframes(wave_fid.getnframes())
        decoder = dfactory.StartDecodingSession()
        decoder.FeedBytestring(data)
        decoder.Finalize()
        rr = decoder.GetResult()
        return rr.transcript


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for x in executor.map(recognize, wave_list):
            print(x.decode('utf-8'))


if __name__ == '__main__':
    main()
