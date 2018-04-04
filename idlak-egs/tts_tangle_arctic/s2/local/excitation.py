#!/usr/bin/python
# -*- coding: utf-8 -*-

import cmath, math, random
import numpy as np
import os, sys, copy, timeit, array
try:
  import pylab
except:
  pass

class RESD:
    
    
    # Set up constants
    def __init__(self, fftlen, srate, fshift, extra_opts):
        self.len = int(fftlen)
        self.hlen = (self.len) / 2
        self.srate = int(srate)
        self.fshift = float(fshift)
        self.f0min = 70.
        self.uvf0 = .5 / self.fshift
        self.rnd_val = 0
        self.frame_shift = int(self.fshift * self.srate)
        
        if hasattr(extra_opts, 'no_mix'):
            self.no_mix = extra_opts.no_mix
        else:
            self.no_mix = False
        
        if hasattr(extra_opts, 'verbose'):
            self.verbose = extra_opts.verbose
        else:
            self.verbose = False

        if hasattr(extra_opts, 'midbands'):
            self.midbands = extra_opts.midbands
        else:
            self.midbands = None

    ########################################################
    
    # f0 is fundamental frequency in Hz, while fs is sampling frequency
    # N2 is glottal opening duration, N1 is "duty" of the cycle 
    def rosenberg_pulse(self, N1, N2, pulselength, randomize=False):
        
        #T = 1/f0 # period
        #pulselength = math.floor(T*fs)
        N2 = int(math.floor(pulselength*N2))
        N1 = int(math.floor(N1*N2))
        gn = np.zeros(self.len + 1)
        offset = self.len/2 - N1
        # Opening phase
        for n in range(0, N1):
            gn[n + offset] = 0.5 * (1-math.cos(np.pi*n / N1))
        # Closing phase
        for n in range(N1, N2):
            gn[n + offset] = math.cos(np.pi*(n-N1)/(N2-N1)/2)
        #print gn
        if randomize:
            self.rnd_val += (random.random() - 0.5) * np.pi * 0.01
        else:
            self.rnd_val = 0
        #return gn
        gn = np.diff(gn)
        # Normalise in the FFT domain
        gn = np.fft.fftshift(gn)
        pulse_fft = np.fft.rfft(gn)
        for i, c in enumerate(pulse_fft):
            #n = abs(c)
            #if n > 0:
            #    pulse_fft[i] = c / n
            #else:
            #    pulse_fft[i] = 1.0 + 0j
            if i != 0 and i != len(pulse_fft) -1:
                pulse_fft[i] = cmath.rect(1.0, cmath.polar(c)[1] + self.rnd_val)
        gn = np.fft.irfft(pulse_fft)
        gn = np.fft.ifftshift(gn)
        return gn
    
    #def rms(self, v):
        
    # Mixed excitation
    def mixed_excitation(self, _pulse, _noise, bndaps, pitch_period=0.0, voiced=True):

        # Calculatation is done in the frequency domain
        pulse = _pulse
        noise = _noise

        # Mix the noise and the pulse
        if not self.no_mix:
            noise_fft = np.fft.fft(noise)
            pulse_fft = np.fft.fft(pulse)

            if self.midbands:
                # if the middle of the bands are specified then use linear interpolation

                noiseweights = np.array(map(lambda b: np.power(10., b / 20.), bndaps))
                noiseweights = noiseweights.clip(0., 1.)

                # make sure it starts and ends at the right point
                mb = copy.copy(self.midbands)
                if mb[0] != 0: 
                    mb = [0] + mb
                    noiseweights = np.hstack(([0.], noiseweights))
                if mb[-1] != self.len/2: 
                    mb = mb + [self.len/2]
                    noiseweights = np.hstack((noiseweights, [1.]))

                noiseWeights = np.interp(range(self.len/2), mb, noiseweights)
                pulseWeights = np.sqrt(1 - noiseWeights**2).clip(0.,1.)
                noiseWeights = np.array(np.hstack((noiseWeights,noiseWeights[::-1])), dtype=complex)
                pulseWeights = np.array(np.hstack((pulseWeights,pulseWeights[::-1])), dtype=complex)

                noise_fft *= noiseWeights
                pulse_fft *= pulseWeights
            else:

                for band_idx, band_val in enumerate(bndaps):
                    noise_weight = np.power(10., band_val / 20.)
                    noise_weight = min(noise_weight, 1.)

                    #if pulse[self.hlen] == 0.0:
                    #    noise_weight = 1.0

                    pulse_weight = np.sqrt(max(0.0, 1.0 - noise_weight**2))

                    # Weight each band seperately
                    for k in range(self.bands[band_idx][0], self.bands[band_idx][1]):
                        noise_fft[k] *= noise_weight
                        pulse_fft[k] *= pulse_weight

                        if not (k == 0 or k == self.len/2):
                            noise_fft[-k] *= noise_weight
                            pulse_fft[-k] *= pulse_weight

            noise = np.real(np.fft.ifft(noise_fft))
            pulse = np.real(np.fft.ifft(pulse_fft))
            excitation = np.real(np.fft.ifft(noise_fft + pulse_fft))
        else:
            if voiced:
                noise = np.zeros(noise.size)
            excitation = pulse + noise

        # Mix the excitation
        #excitation = pulse + noise

        # Normalise energy
        excitation /= np.sqrt(pitch_period / sum(excitation ** 2)) 
        
        
        return excitation, pulse, noise 
    
    ###################################################################


    


    
    
    ###################################################################
    
    
    # Apply the vocoder
    def vocode(self, f0, bands, verbose = False, opts=None):
        
        verbose = self.verbose or verbose
        
        self.opts_ = opts
        # Make sure the inputs are numpy arrays
        self.f0s = np.array(f0)
        self.bndaps = np.array(bands)
        
        # orders
        self.O_bndap = self.bndaps.shape[1]
        
        self.calculate_bands(self.O_bndap)
        self.nrframes = max(f0.size, bands.shape[0])
        
        # reinitialise the memory of the filters and the random number seed
        np.random.seed(0)
        self.iteration = 0
        self.current_coefs = None
                
        # Raw wav out
        raw = np.zeros(self.nrframes * self.frame_shift)
        # Coefficients for OLA
        self.hanning = np.zeros(self.nrframes * self.frame_shift)
        

        time = 0.
        time_idx = 0
        excitation_frame_idx = 0
        mcep_frame_idx = 0
        pitch_period_prv = int(float(self.srate) / self.uvf0)
        
       
        self.lstsample = 0
        while excitation_frame_idx < self.nrframes and self.lstsample <  raw.size:
            if verbose:
                print "Building excitation, Iteration: {0:> 4d}    {1:> 4d} / {2}".format(self.iteration, excitation_frame_idx, self.nrframes)
                
            # Get the f0 for the frame (NOT LOG F0)
            if excitation_frame_idx > self.f0s.size: 
                frame_f0 = 0.0
            else:
                frame_f0 = self.f0s[excitation_frame_idx]

            bndap = self.bndaps[excitation_frame_idx,:]

            # Find the pitch period (in samples) from the F0 & then the pulse's magnitude
            if frame_f0 > 0.0:
                frame_f0 = max(self.f0min, frame_f0) # ensure the pitch period isn't too short^W long (BP)
                pitch_period = float(self.srate) / frame_f0
                voiced = True
            else:
                frame_f0 = self.uvf0
                pitch_period = float(self.srate) / self.uvf0
                voiced = False
                bndap = np.zeros(self.O_bndap)

            pitch_period_int = int(pitch_period)
            pulse_magnitude = np.sqrt(pitch_period_int)

            # Create Excitation
            pulse = np.zeros(self.len)
            #pulse = self.rosenberg_pulse(0.7, 0.7, pitch_period_int, False)#True)
            #if len(pulse) < self.len:
            #    pulse_pre = np.zeros((self.len - len(pulse)) / 2)
            #    pulse_post = np.zeros(self.len - len(pulse) - len(pulse_pre))
            #    pulse = np.concatenate((pulse_pre, pulse, pulse_post))
            pulse[self.hlen] = pulse_magnitude
            pulse *= np.sqrt(pitch_period / sum(pulse ** 2))

            #pylab.figure(); pylab.plot(pulse); pylab.show()
            #sys.stderr.write("len: %d, hlen: %d, vals: %f,%f\n" % (len(pulse), self.hlen, pulse[0], pulse[self.hlen]))
            #

            # Noise only needs to covers the middle of window
            noise = np.zeros(self.len)
            start_point =  self.hlen - pitch_period_prv
            #if (start_point < 0): print self.hlen, pitch_period_prv
            flen =  pitch_period_prv + pitch_period_int
            # To ensure the energy of noise is the same as pulse
            noise_norm_fact = np.sqrt(pitch_period / flen)
            noise[start_point: start_point + flen] = np.random.normal(0., 1.0 * noise_norm_fact, flen)
            #sys.stderr.write("nrgs: %f <> %f\n" % ( sum(pulse ** 2), sum(noise ** 2)))

            # Pitch synch hanning window
            hanning_window = self.pitch_sync_hann(pitch_period_prv, pitch_period_int)


            # Mix to get excitation (and get the new pulse and noise)
            excitation, pulse, noise = self.mixed_excitation(pulse, noise, bndap, pitch_period, voiced)
            #if (voiced):
            #    pylab.figure(); pylab.plot(excitation); pylab.show()
            self.overlap_and_add(raw, excitation, hanning_window, time_idx)
            #pylab.figure(); pylab.plot(raw[:self.len]); pylab.show()

            # Update times, frame index, etc
            pitch_period_prv = pitch_period_int
            time += 1. / frame_f0
            time_idx = int(self.srate * time)

            while time_idx > (excitation_frame_idx + 1) * self.frame_shift:
                excitation_frame_idx += 1

            if excitation_frame_idx >= self.nrframes:
                excitation_frame_idx = self.nrframes
                break

            self.iteration += 1

        excitation = np.divide(raw, np.where(self.hanning > 0, self.hanning, 1))
        #return excitation
        t_start = self.len/2 - self.frame_shift 
        return excitation[t_start:]


    ###################################################################

    

    # Builds a pitch synchronous hanning filter
    def pitch_sync_hann(self, left_period, right_period, hlen = None):

        if hlen is None:
            hlen = self.hlen

        left = np.hanning(left_period*4)[:left_period*2]
        right = np.hanning(right_period*4)[-right_period*2:]
        
        leftzeros = []
        rightzeros = []
        if left.size < hlen:
            leftzeros = np.zeros(hlen - left.size)
        if right.size < (self.len - hlen):
            rightzeros = np.zeros((self.len - hlen) - right.size)
        hann = np.hstack([leftzeros, left[-hlen:], right[:self.len - hlen], rightzeros])

        ## Plot window
        #pylab.figure(); pylab.plot(hann); pylab.show(); exit()

        return hann


    # Overlap and add function
    def overlap_and_add(self, raw, excitation, hanning, start_time):
        excitation = excitation * hanning

        hlen = self.hlen
        win_center = start_time #int((i + 0.5) * period)
        win_lb = win_center - hlen
        win_ub = win_lb + hlen * 2
        slc_exc = slice(max(0, -win_lb), min(excitation.size, hlen + raw.size - win_ub))
        #slc_exc = slice(0, min(excitation.size, raw.size - start_time))
        len_exc_slc = slc_exc.stop - slc_exc.start
        slc_raw = slice(start_time, min(start_time + len_exc_slc, raw.size))
        raw[slc_raw] = raw[slc_raw] + excitation[slc_exc]
        self.hanning[slc_raw] = self.hanning[slc_raw] + hanning[slc_exc]
        self.lstsample = slc_raw.stop
        #print slc_raw.start, slc_raw.stop
        #print slc_exc, slc_raw


    # Get bands
    def calculate_bands(self, nr_bands):
        niquest = self.srate / 2

        if nr_bands == 5:
            self.bands = [[0,                   self.len/16],
                          [self.len/16,         self.len/8],
                          [self.len/8,          self.len/4],
                          [self.len/4,          (self.len * 3) / 8],
                          [(self.len * 3) / 8,  self.len/2 + 1]]
        else:
            self.bands = []
            for b in range(nr_bands):
                startf = 1960 / (26.81 / (b + 0.53) - 1)
                startf = round (startf / 100 ) * 100   
                bs = startf * self.len / self.srate


                endf = 1960 / (26.81 / (b + 1 + 0.53) - 1)
                endf = round (endf / 100 ) * 100
                if endf < niquest and b < nr_bands-1:
                    be = endf * self.len / self.srate
                else:
                    be = self.len/2 + 1

                self.bands.append((int(bs), int(be)))

        if not self.midbands:
            self.midbands = []
            for b in self.bands:
                self.midbands.append((b[0] + (b[1] - b[0])/2))





def load_file(filename, dim):
    if filename[-5:] == "float":
        dd = np.fromfile(filename,'f4')
        if dim == 1:
            return np.reshape(dd, (-1,))
        return np.reshape(dd, (-1, dim))
    else:
        ll = [map(float, l.strip().split()) for l in open(filename).readlines()]
        if len(ll[0]) != dim:
            print "Dim mismatch: %d <> %d" % (dim, len(ll[0]))
            sys.exit(-2)
        if dim == 1:
            return np.reshape(ll, (-1, ))
        return np.reshape(ll, (-1, dim))
    
    
def main():
    from optparse import OptionParser
    usage="usage: %prog [options] <f0file> [<bndapfile> [<output>]]\n" \
        "e.g. python to_wav test.raw "
    parser = OptionParser(usage=usage)
    parser.add_option('-s','--srate', default = 48000,
                      help = 'sample rate')
    parser.add_option('-f','--fftlen', default = 4096,
                      help = 'fft length')
    parser.add_option('-F','--fshift', default = 0.005,
                      help = 'frame shift')
    
    parser.add_option('-b','--bndaporder', default = 25,
                      help = 'bndap order')
    parser.add_option('-n','--no-mix', default = False, action="store_true",
                      help = 'Disable mixed excitation')
    parser.add_option('-G','--gain', default = 1.0,
                      help = 'Change output gain')
    parser.add_option('-B', '--midbands', default = None, 
                      help = 'comma separated mid points of each band')
                      # kaldi bands = 8,15,22,30,38,47,58,69,81,95,110,127,147,169,194,224,259,301,353,416,498,606,757,980,1344

    opts, args = parser.parse_args()
    if len(args) < 1:
        parser.error("missing arguments, at least one is required")

    if not opts.no_mix and len(args) < 2:
        parser.error("missing arguments, bndap are required if mixing is active")

    if len(args) > 3 or (opts.no_mix and len(args) > 2):
        parser.error("Too many arguments")

    outfile = "-"
    bndapfile = None
    f0file = args[0]
    if not opts.no_mix:
        bndapfile = args[1]
        if len(args) == 3:
            outfile = args[2]
    else:
        if len(args) == 2:
            outfile = args[1]
    
    if outfile == "-":
        out = sys.stdout
    else:
        out = open(outfile, "w")
    
    f0s = load_file(f0file, 1)
    if bndapfile:
        bndaps = load_file(bndapfile, int(opts.bndaporder))
    else:
        bndaps = np.array([[0.] * int(opts.bndaporder)] * len(f0s))

    #f lf0:
    #   f0s = np.exp(f0s)

    if opts.midbands:
        opts.midbands = [int(b) for b in opts.midbands.split(',')]
        if not opts.bndaporder == len(opts.midbands):
            print "bandap order does not match number of mid band points"
            sys.exit(-1)

    residual = RESD(opts.fftlen, int(opts.srate), float(opts.fshift), opts)
    excitation = residual.vocode(f0s, bndaps)
    sys.stderr.write("Excitation: duration %f\n" % (len(excitation) / float(opts.srate)))
    
    out.write(array.array('f', excitation * float(opts.gain)).tostring())
    


if __name__ == '__main__':
    main()

