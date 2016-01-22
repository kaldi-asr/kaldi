#!/nfs/disk/perm/linux-x86_64/bin/python2
###!/usr/bin/python
import argparse
import random
import bisect
import re
import logging
import wave
import math
import struct
import sys
import pyximport; pyximport.install()
from util import *

def dirichlet(params):
    samples = [random.gammavariate(x, 1) if x > 0 else 0. for x in params]
    samples = [x / sum(samples) for x in samples]
    for x in xrange(1, len(samples)):
        samples[x] += samples[x - 1]
    return bisect.bisect_left(samples, random.random())

#def energy(wav_filename):
#    return float(filter(lambda x: 'amp' in x,
#                 filter(lambda x: 'RMS' in x, 
#                 subprocess.check_output(['sox', wav_filename, '-n', 'stat'], stderr=subprocess.STDOUT).split('\n')))[0].split()[-1])

def wave_mat(wav_filename):
    f = wave.open(wav_filename, 'r')
    n = f.getnframes()
    ret = f.readframes(n)
    f.close()
    return list(struct.unpack('%dh' % n, ret))

"""
def energy(mat):
    return float(sum([x * x for x in mat])) / len(mat)
"""

def num_samples(mat):
    return len(mat)

def scp(scp_filename):
    with open(scp_filename) as f:
        for l in f:
            yield tuple(l.strip().split())

def wave_header(sample_array, sample_rate):
  byte_count = (len(sample_array)) * 2 # short
  # write the header
  hdr = struct.pack('<ccccIccccccccIHHIIHH',
    'R', 'I', 'F', 'F',
    byte_count + 0x2c - 8,  # header size
    'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
    0x10,  # size of 'fmt ' header
    1,  # format 1
    1,  # channels
    sample_rate,  # samples / second
    sample_rate * 2,  # bytes / second
    2,  # block alignment
    16)  # bits / sample
  hdr += struct.pack('<ccccI',
    'd', 'a', 't', 'a', byte_count)
  return hdr


def output(tag, mat):
    sys.stdout.write(tag + ' ')
    sys.stdout.write(wave_header(mat, 16000))
    sys.stdout.write(struct.pack('%dh' % len(mat), *mat))

def output_wave_file(dir, tag, mat):
    with open('%s/%s.wav' % (dir,tag), 'w') as f:
        f.write(wave_header(mat, 16000))
        f.write(struct.pack('%dh' % len(mat), *mat))

"""
def mix(mat, type, scale):
    pos, noise = noises[type]
    ret = []
    l = len(noise)
    for i in xrange(len(mat)):
        x = mat[i]
        d = int(x + scale * noise[pos])
        #if d > 32767 or d < -32768:
        #    logging.debug('overflow occurred!')
        d = max(min(d, 32767), -32768)
        ret.append(d)
        pos += 1
        if pos == l:
            pos = 0
    noises[type] = (pos, noise)
    return ret
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-level', type=float, help='')
    parser.add_argument('--noise-src', type=str, help='')
    parser.add_argument('--noise-prior', type=str, help='')
    parser.add_argument('--seed', type=int, help='')
    parser.add_argument('--sigma0', type=float, help='')
    parser.add_argument('--wav-src', type=str, help='')
    parser.add_argument('--verbose', type=bool, help='')
    parser.add_argument('--wavdir', type=str, help='')
    args = parser.parse_args()
    random.seed(args.seed)
    params = [float(x) for x in args.noise_prior.split(',')]
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    global noises
    noise_energies = [0.]
    noises = [(0, [])]
    for tag, wav in scp(args.noise_src):
        logging.debug('noise wav: %s', wav)
        mat = wave_mat(wav)
        e = energy(mat)
        logging.debug('noise energy: %f', e)
        noise_energies.append(e)
        noises.append((0, mat))
        
    for tag, wav in scp(args.wav_src):
        logging.debug('wav: %s', wav)
        noise_level = random.gauss(args.noise_level, args.sigma0)
        logging.debug('noise level: %f', noise_level)
        mat = wave_mat(wav)
        signal = energy(mat)
        logging.debug('signal energy: %f', signal)
        noise = signal / (10 ** (noise_level / 10.))
        logging.debug('noise energy: %f', noise)
        type = dirichlet(params)
        logging.debug('selected type: %d', type)
        if type == 0:
            if args.wavdir != 'NULL':
               output_wave_file(args.wavdir, tag, mat)
            output(tag, mat)
        else:
            p,n = noises[type]
            if p+len(mat) > len(n):
                noise_energies[type] = energy(n[p::]+n[0:len(n)-p:])
            else:
                noise_energies[type] = energy(n[p:p+len(mat):])
            scale = math.sqrt(noise / noise_energies[type])
            logging.debug('noise scale: %f', scale)
            pos, result = mix(mat, n, p, scale)
            noises[type] = (pos, n)
            if args.wavdir != 'NULL':
                output_wave_file(args.wavdir, tag, result)
            else:
                output(tag, result)
            #output(tag, mix(mat, type, scale))

if __name__ == '__main__':
    main()



