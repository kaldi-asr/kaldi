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
from __future__ import print_function

import pyaudio
from kaldi.decoders import PyOnlineLatgenRecogniser
from kaldi.utils import wst2dict, lattice_to_nbest
import sys
import time
import select
import tty
import termios
import wave

CHANNELS, RATE, FORMAT = 1, 16000, pyaudio.paInt16


class LiveDemo(object):

    def __init__(self, audio_batch_size, wst, dec_args):
        self.batch_size = audio_batch_size
        self.wst = wst
        self.args = dec_args
        self.d = PyOnlineLatgenRecogniser()
        self.pin, self.stream = None, None
        self.frames = []
        self.utt_frames, self.new_frames = 0, 0
        self.utt_end, self.dialog_end = False, False

    def setup(self):
        self.d.reset()
        self.d.setup(argv)
        self.pin = pyaudio.PyAudio()
        self.stream = self.pin.open(format=FORMAT, channels=CHANNELS,
                                    rate=RATE, input=True, frames_per_buffer=self.batch_size,
                                    stream_callback=self.get_audio_callback())
        self.utt_frames, self.new_frames = 0, 0
        self.utt_end, self.dialog_end = False, False
        self.frames = []

    def tear_down(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.pin is not None:
            self.pin.terminate()
        p, stream = None, None
        self.frames = []

    def get_audio_callback(self):
        def frame_in(in_data, frame_count, time_info, status):
            self.d.frame_in(in_data)
            self.frames.append(in_data)
            return in_data, pyaudio.paContinue
        return frame_in

    def _user_control(self):
        '''Simply stupid sollution how to control state of recogniser.'''

        self.utt_end, self.dialog_end = False, False
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            # if is data on input
            while (select.select([sys.stdin], [], [], 1) == ([sys.stdin], [], [])):
                c = sys.stdin.read(1)
                if c == 'u':
                    print('\nMarked end of utterance\n')
                    self.utt_end = True
                elif c == 'c':
                    self.dialog_end = True
                    print('\nMarked end of dialogue\n')
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("""Chunks: %d ; Utterance %d ; end %d : press 'u'\nFor terminating press 'c'\n\n""" % (len(self.frames), self.utt_frames, self.utt_end))

    def run(self):
        while True:
            time.sleep(0.1)
            self._user_control()
            new_frames = self.d.decode(max_frames=10)
            while new_frames > 0:
                self.utt_frames += new_frames
                new_frames = self.d.decode(max_frames=10)
            if self.utt_end or self.dialog_end:
                start = time.time()
                self.d.prune_final()
                prob, lat = self.d.get_lattice()
                # lat.write('live-demo-recorded.fst')
                nbest = lattice_to_nbest(lat, n=10)
                if nbest:
                    best_prob, best_path = nbest[0]
                    decoded = ' '.join([wst[w] for w in best_path])
                else:
                    decoded = 'Empty hypothesis'
                print("%s secs, frames: %d, prob: %f, %s " % (
                    str(time.time() - start), self.utt_frames, prob, decoded))
                self.utt_frames = 0
                self.d.reset(keep_buffer_data=False)
            if self.dialog_end:
                self.save_wav()
                break

    def save_wav(self):
        wf = wave.open('live-demo-record.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setframerate(RATE)
        wf.setsampwidth(self.pin.get_sample_size(FORMAT))
        wf.writeframes(b''.join(self.frames))
        wf.close()


if __name__ == '__main__':
    audio_batch_size, wst_path = int(sys.argv[1]), sys.argv[2]
    argv = sys.argv[3:]
    print('Python args: %s' % str(sys.argv), file=sys.stderr)

    wst = wst2dict(wst_path)
    demo = LiveDemo(audio_batch_size, wst, argv)
    demo.setup()
    demo.run()
