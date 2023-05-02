#!/usr/bin/python3
#
# This script gets speech segments from whole recordings using webrtcvad
# Modified from: https://github.com/wiseman/py-webrtcvad/blob/master/example.py
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import collections, sys, os, argparse, contextlib
import wave
import webrtcvad

def get_args():
	parser = argparse.ArgumentParser(description="Obtain speech segments for all wav files in a dir."
        " Writes the output to the stdout." 
		"Usage: apply_webrtcvad.py [options...] <data-dir>"
		"E.g.: apply_webrtcvad.py --aggressiveness 2 --reco2channels data/reco2channels data",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--mode", type=int, dest = "mode", default=1,
		help="Integer in {0,1,2,3} specifying the VAD aggressiveness. 0 is the least aggressive"
        " about filtering out non-speech, 3 is the most aggressive.")

	parser.add_argument("--reco2channels", type=str, dest="reco2ch_file",
		help="In multi-channel setting, specifying this would avoid computing VAD for each channel"
        " separately. Only first channel will be used to compute VAD and all channels will share.")

	parser.add_argument("data_dir", help="Data directory containing wav.scp")

	args = parser.parse_args()

	return args

def check_args(args):
    if (args.mode not in [0,1,2,3]):
        raise Exception("Aggressiveness mode must be in {0,1,2,3}")
    if (not os.path.exists(os.path.join(args.data_dir,'wav.scp'))):
        raise Exception("No wav.scp file exists")
    return

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_segments(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: List of (start_time,end_time) tuples.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    segments = []
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                start_time = voiced_frames[0].timestamp
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                end_time = frame.timestamp + frame.duration
                triggered = False
                ring_buffer.clear()
                voiced_frames = []
                # Write to segments list
                segments.append((start_time, end_time))
    # If we have any leftover voiced audio when we run out of input,
    # add it to segments list.
    if voiced_frames:
        end_time = voiced_frames[-1].timestamp
        segments.append((start_time, end_time))
    return segments


def get_reco2channels(reco2ch_file):
    """
    Given a file containing reco id and channel ids for the recording, return
    the corresponding dictionary.
    """
    reco2channels = {}
    with open(reco2ch_file, 'r') as f:
        for line in f.readlines():
            reco, channels = line.strip.split(maxsplit=1)
            channels = channels.split()
            reco2channels[reco] = channels
    return reco2channels

def get_wav_list(data_dir, reco2channels=None):
    """
    Return a dictionary of uttid with wav paths. Optionally takes reco2channels and,
    if provided, the uttid is actually the recoid.
    """
    if reco2channels is not None:
        keep_wavs = {reco2channels[reco][0]:reco for reco in reco2channels.keys()}
    wav_list = {}
    with open(os.path.join(data_dir,'wav.scp'),'r') as f:
        for line in f.readlines():
            utt, wav = line.strip().split()
            if reco2channels is not None:
                if utt in keep_wavs:
                    wav_list[keep_wavs[utt]] = wav
            else:
                wav_list[utt] = wav
    return wav_list

def get_speech_segments(uttid, wav, vad):
    """
    Compute and print the segments for the given uttid. It is in the format:
    <segment-id> <utt-id> <start-time> <end-time>
    """
    audio, sample_rate = read_wave(wav)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_segments(sample_rate, 30, 300, vad, frames)
    for segment in segments:
        start = float("{:.2f}".format(segment[0]))
        end = float("{:.2f}".format(segment[1]))
        segment_id = '{}_{}_{}'.format(uttid,'{:.0f}'.format(100*start).zfill(6), '{:.0f}'.format(100*end).zfill(6))
        print ("{} {} {} {}".format(segment_id, uttid, start, end))
    return

def main():
    # First we read and check the arguments
    args = get_args()
    check_args(args)
    
    if (args.reco2ch_file is not None):
        reco2channels = get_reco2channels(args.reco2ch_file)
        wav_list = get_wav_list(args.data_dir, reco2channels)
    else:
        wav_list = get_wav_list(args.data_dir)

    vad = webrtcvad.Vad(args.mode)
    for utt in wav_list.keys():
        get_speech_segments(utt, wav_list[utt], vad)


if __name__ == '__main__':
    main()