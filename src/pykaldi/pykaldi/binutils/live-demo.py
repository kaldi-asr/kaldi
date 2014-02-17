#!/usr/bin/env python
# encoding: utf-8
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

import pyaudio
from pykaldi.decoders import PyGmmLatgenWrapper
from pykaldi.utils import wst2dict, lattice_to_nbest
import sys
import time
import select
import tty
import termios


def setup_pyaudio(samples_per_frame):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(pyaudio.paInt32),
                    channels=1,
                    rate=16000,
                    input=True,
                    output=True,
                    frames_per_buffer=samples_per_frame)
    return (p, stream)


def teardown_pyaudio(p, stream):
    stream.stop_stream()
    stream.close()
    p.terminate()
    p, stream = None, None


def user_control(pause):
    '''Simply stupid sollution how to control state of recognizer.
    Three boolean states which should be
    set up by are returned by the function.'''

    utt_end, dialog_end = False, False
    old_settings = termios.tcgetattr(sys.stdin)
    # raise NotImplementedError('TODO not working reading characters from terminal')
    try:
        tty.setcbreak(sys.stdin.fileno())
        # if is data on input
        while (select.select([sys.stdin], [], [], 1) == ([sys.stdin], [], [])):
            c = sys.stdin.read(1)
            print 'character %s' % c
            if c == 'u':
                utt_end = True
            elif c == 'p':
                pause = not pause
            elif c == 'c':
                dialog_end = True
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    print """
Utterance end %d : press 'u'
The recognition is paused %d: press 'p'
For terminating the program press 'c'\n\n""" % (utt_end, pause)

    return (utt_end, dialog_end, pause)


def decode_loop(d, audio_batch_size, wst, stream):
    utt_frames, new_frames, pause = 0, 0, False
    while True:
        utt_end, dialog_end, pause = user_control(pause)
        if pause:
            d.reset(keep_buffer_data=False)
            time.sleep(0.1)
            continue
        new_frames = d.decode(max_frames=10)
        if utt_end or dialog_end:
            start = time.time()
            d.prune_final()
            prob, lat = d.get_lattice()
            nbest = lattice_to_nbest(lat, n=1)
            best_prob, best_path = nbest[0]
            decoded = [wst[w] for w in best_path]
            print "%s secs, frames: %d, prob: %f, %s " % (
                str(time.time() - start), utt_frames, prob, decoded.encode('UTF-8'))
            utt_frames = 0
        if dialog_end:
            d.reset(keep_buffer_data=False)
            break
        if new_frames == 0:
            frame = stream.read(2 * audio_batch_size)  # 16bit audio 16/8=2
            d.frame_in(frame)
        else:
            utt_frames += new_frames

if __name__ == '__main__':
    audio_batch_size, wst_path = int(sys.argv[1]), sys.argv[2]
    argv = sys.argv[3:]
    print >> sys.stderr, 'Python args: %s' % str(sys.argv)

    print """ Press space for pause
    Pres 'Enter' to see output at the end of utterance
    Prec 'Esc' for terminating the program"""
    wst = wst2dict(wst_path)
    d = PyGmmLatgenWrapper()
    d.setup(argv)

    p, stream = setup_pyaudio(audio_batch_size)
    decode_loop(d, audio_batch_size, wst, stream)
    teardown_pyaudio(p, stream)
