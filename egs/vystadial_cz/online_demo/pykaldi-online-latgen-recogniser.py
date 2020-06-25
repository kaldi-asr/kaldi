#!/usr/bin/env python
# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License. #
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from kaldi.utils import load_wav, wst2dict, lattice_to_nbest
from kaldi.decoders import PyOnlineLatgenRecogniser
import sys
import fst
import time

# DEBUG = True
DEBUG = False


def write_decoded(f, wav_name, word_ids, wst):
    assert(len(word_ids) > 0)
    best_weight, best_path = word_ids[0]
    if wst is not None:
        decoded = [wst[w] for w in best_path]
    else:
        decoded = [str(w) for w in best_path]
    line = u' '.join([wav_name] + decoded + ['\n'])
    if DEBUG:
        print('%s best path %s' % (wav_name, decoded.encode('UTF-8')))
        for i, s in enumerate(word_ids):
            if i > 0:
                break
            print('best path %d: %s' % (i, str(s)))
    f.write(line.encode('UTF-8'))


# @profile
def decode(d, pcm):
    frame_len = (2 * audio_batch_size)  # 16-bit audio so 1 sample = 2 chars
    i, decoded_frames, max_end = 0, 0, len(pcm)
    start = time.time()
    while i * frame_len < len(pcm):
        i, begin, end = i + 1, i * frame_len, min(max_end, (i + 1) * frame_len)
        audio_chunk = pcm[begin:end]
        d.frame_in(audio_chunk)
        dec_t = d.decode(max_frames=10)
        while dec_t > 0:
            decoded_frames += dec_t
            dec_t = d.decode(max_frames=10)
    print("forward decode: %s secs" % str(time.time() - start))
    start = time.time()
    d.prune_final()
    lik, lat = d.get_lattice()
    print("backward decode: %s secs" % str(time.time() - start))
    d.reset(keep_buffer_data=False)
    return (lat, lik, decoded_frames)


def decode_wrap(argv, audio_batch_size, wav_paths,
        file_output, wst_path=None):
    wst = wst2dict(wst_path)
    d = PyOnlineLatgenRecogniser()
    d.setup(argv)
    for wav_name, wav_path in wav_paths:
        sw, sr = 2, 16000  # 16-bit audio so 1 sample_width = 2 chars
        pcm = load_wav(wav_path, def_sample_width=sw, def_sample_rate=sr)
        print('%s has %f sec' % (wav_name, (float(len(pcm)) / sw) / sr))
        lat, lik, decoded_frames = decode(d, pcm)
        lat.isyms = lat.osyms = fst.read_symbols_text(wst_path)
        if DEBUG:
            with open('pykaldi_%s.svg' % wav_name, 'w') as f:
                f.write(lat._repr_svg_())
            lat.write('%s_pykaldi.fst' % wav_name)

        print("Log-likelihood per frame for utterance %s is %f over %d frames" % (
            wav_name, int(lik / decoded_frames), decoded_frames))
        word_ids = lattice_to_nbest(lat, n=10)
        write_decoded(file_output, wav_name, word_ids, wst)


if __name__ == '__main__':
    audio_scp, audio_batch_size = sys.argv[1], int(sys.argv[2])
    dec_hypo, wst_path = sys.argv[3], sys.argv[4]
    argv = sys.argv[5:]
    print('Python args: %s' % str(sys.argv), file=sys.stderr)

    # open audio_scp, decode and write to dec_hypo file
    with open(audio_scp, 'rb') as r:
        with open(dec_hypo, 'wb') as w:
            lines = r.readlines()
            scp = [tuple(line.strip().split(' ', 1)) for line in lines]
            decode_wrap(argv, audio_batch_size, scp, w, wst_path)
