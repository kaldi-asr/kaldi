import scipy.io.wavfile, scipy.signal, numpy, math
import argparse

# Change these as you see fit :-)
#inputimpulse='/home/potard/cereproc/trunk/apps/basictts/Impulse Response Files/Medium Plate - Mono Verb.wav'
#inputwav = '/share/audio/en/ga/jrm/48000/jrm_a0001_001.wav' #'/share/audio/en/ga/jrm/original.48/jrm_a0001_001.wav'
#outputwav='/tmp/test.wav'

# For spectrum based one:
# 
#mixvalue = 0.7
from scipy.fftpack import (fftn, ifftn)
from numpy.fft import rfftn, irfftn
from numpy import (array, asarray)
import numpy as np
try:
  import pylab
except:
  pass


def _check_valid_mode_shapes(shape1, shape2):
  for d1, d2 in zip(shape1, shape2):
    if not d1 >= d2:
      raise ValueError(
        "in1 should have at least as many items as in2 in "
        "every dimension for 'valid' mode.")

def _centered(arr, newsize):
  # Return the center newsize portion of the array.
  newsize = asarray(newsize)
  currsize = array(arr.shape)
  startind = (currsize - newsize) // 2
  endind = startind + newsize
  myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
  return arr[tuple(myslice)]

def _next_regular(target):
  """
  Find the next regular number greater than or equal to target.
  Regular numbers are composites of the prime factors 2, 3, and 5.
  Also known as 5-smooth numbers or Hamming numbers, these are the optimal
  size for inputs to FFTPACK.
  Target must be a positive integer.
  """
  if target <= 6:
    return target
    
  # Quickly check if it's already a power of 2
  if not (target & (target-1)):
    return target

  match = float('inf')  # Anything found will be smaller
  p5 = 1
  while p5 < target:
    p35 = p5
    while p35 < target:
      # Ceiling integer division, avoiding conversion to float
      # (quotient = ceil(target / p35))
      quotient = -(-target // p35)
      
      # Quickly find next power of 2 >= quotient
      try:
        p2 = 2**((quotient - 1).bit_length())
      except AttributeError:
        # Fallback for Python <2.7
        p2 = 2**(len(bin(quotient - 1)) - 2)

      N = p2 * p35
      if N == target:
        return N
      elif N < match:
        match = N
      p35 *= 3
      if p35 == target:
        return p35
    if p35 < match:
      match = p35
    p5 *= 5
    if p5 == target:
      return p5
  if p5 < match:
    match = p5
  return match

def _unfold_fft(in1, fshape):
  assert(in1.shape == array(fshape) / 2 + 1)
  #print in1[::-1]
  ret = np.concatenate((in1, np.conj(in1[::-1][1:])))#, axis=1)
  print len(ret), len(in1)
  return ret

def customfftconvolve(in1, in2, mode="full", types=('','')):
  """ Pretty much the same as original fftconvolve, but supports
      having operands as fft already 
  """

  in1 = asarray(in1)
  in2 = asarray(in2)

  if in1.ndim == in2.ndim == 0:  # scalar inputs
    return in1 * in2
  elif not in1.ndim == in2.ndim:
    raise ValueError("in1 and in2 should have the same dimensionality")
  elif in1.size == 0 or in2.size == 0:  # empty arrays
    return array([])

  s1 = array(in1.shape)
  s2 = array(in2.shape)
  complex_result = False
  #complex_result = (np.issubdtype(in1.dtype, np.complex) or
  #                  np.issubdtype(in2.dtype, np.complex))
  shape = s1 + s2 - 1
  
  if mode == "valid":
    _check_valid_mode_shapes(s1, s2)

  # Speed up FFT by padding to optimal size for FFTPACK
  fshape = [_next_regular(int(d)) for d in shape]
  fslice = tuple([slice(0, int(sz)) for sz in shape])

  if not complex_result:
    if types[0] == 'fft':
      fin1 = in1#_unfold_fft(in1, fshape)
    else:
      fin1 = rfftn(in1, fshape)

    if types[1] == 'fft':
      fin2 = in2#_unfold_fft(in2, fshape)
    else:
      fin2 = rfftn(in2, fshape)
    ret = irfftn(fin1 * fin2, fshape)[fslice].copy()
  else:
    if types[0] == 'fft':
      fin1 = _unfold_fft(in1, fshape)
    else:
      fin1 = fftn(in1, fshape)
    if types[1] == 'fft':
      fin2 = _unfold_fft(in2, fshape)
    else:
      fin2 = fftn(in2, fshape)
    ret = ifftn(fin1 * fin2)[fslice].copy()

  if mode == "full":
    return ret
  elif mode == "same":
    return _centered(ret, s1)
  elif mode == "valid":
    return _centered(ret, s1 - s2 + 1)
  else:
    raise ValueError("Acceptable mode flags are 'valid',"
                     " 'same', or 'full'.")

def siltrim(impulse, maxv):
  lowb = 0
  upb = len(impulse) - 1
  while abs(impulse[lowb]) < maxv:
    lowb += 1
  while abs(impulse[upb]) < maxv:
    upb -= 1
  #print "Trimming %d samples out of %d" %(len(impulse) - 1 - upb + lowb, len(impulse))
  return impulse[lowb:upb]

# Extract a single frame of the original signal
def make_frames(wav, period, length=1024):
  wav_size = len(wav)
  nframes = int(math.ceil(float(wav_size) / period))
  hsz = 2 * period
  off = (length - hsz) / 2
  hpad = np.zeros(off)
  hann = np.concatenate((hpad, numpy.hanning(hsz), hpad))

  for i in range(nframes):
    start_time = i * period
    win_center = start_time + length / 2  #start_time #int((i + 0.5) * period)
    win_lb = win_center - length / 2
    win_ub = win_lb + length
    data = np.concatenate(([0.] * -win_lb, wav[max(0, win_lb): min(wav_size, win_ub)], [0.] * (win_ub - wav_size)))
    #data = [0.] * -win_lb
    #data += wav[max(0, win_lb): min(wav_size, win_ub)]
    #data += [0.] * (win_ub - wav_size)
    #data = numpy.array(data, 'd')
    #start_time = win_lb
    slc_win = slice(max(0, -win_lb), min(data.size, data.size + wav_size - win_ub))
    #print wav_size - win_ub
    len_slc = slc_win.stop - slc_win.start
    slc_raw = slice(start_time, min(start_time + len_slc, wav_size))
    yield (data, hann, slc_win, slc_raw)
  

def spectrum_at_time(spectrum_int, t):
  return spectrum_int(t)

from scipy.interpolate import interp1d

def apply_convolve(input_wav, input_impulse, output_wav, opts):
  if opts.spectrum:
    rate=int(opts.rate)
    # Assume input is raw file
    wav = numpy.fromstring(open(input_wav, 'rb').read(), dtype=np.float32)
    # Consider the impulse file as a raw sequence of complex of size fftlen * nperiods
    spectrums = numpy.fromstring(open(input_impulse, 'rb').read(), dtype=np.complex64)
    spectrum_sequence = spectrums.reshape((-1, opts.fftlen/2 + 1))
    #pylab.figure(); pylab.plot(np.log(np.abs(spectrum_sequence[0]))); pylab.show()
    # We consider that all the number were described polar coordinates complex,
    # so restore them to their normal form
    for n in range(spectrum_sequence.shape[0]):
      for k in range(len(spectrum_sequence[n])):
        c = spectrum_sequence[n][k]
        nc = c.real * np.exp(1j*c.imag)
        spectrum_sequence[n][k] = nc
    
    nx = spectrum_sequence.shape[0]
    x = np.linspace(0, nx * float( opts.period) / rate, nx, endpoint=False)
    # BP: it appears the slinear interpolation sounds better, but it is 10x slower
    spectrum_int = interp1d(x, spectrum_sequence, axis=0)#, kind='slinear')
    nframes = int(math.ceil(float(len(wav) / opts.period)))
    #print nframes, spectrum_sequence.shape[0]
    #assert(nframes == spectrum_sequence.shape[0])
    # Now cut into frames:
    wav_size = len(wav)
    raw = np.zeros(wav_size)
    norm = np.zeros(wav_size) + 1e-8
    #print wav_size, nframes
    for i, (f, hann, sw, sr) in enumerate(make_frames(wav, opts.interperiod, opts.fftlen/2)):
      #spectrum = spectrum_sequence[i]
      time = i * float(opts.interperiod) / rate
      try:
        spectrum = spectrum_at_time(spectrum_int, time)
      except:
        spectrum = oldspectrum
      outf = customfftconvolve(f, spectrum, types=('','fft'))[opts.fftlen/4:]
      # OLA
      #print i, time, wav_size, sr, sw#, len(outf), len(hann)
      raw[sr] = raw[sr] + numpy.multiply(outf[sw], hann[sw])
      norm[sr] = norm[sr] + hann[sw]
      oldspectrum = spectrum
    open(output_wav + ".hann", 'w').write(np.array(norm, dtype=numpy.float32).tostring())
    open(output_wav + ".raw", 'w').write(np.array(raw, dtype=numpy.float32).tostring())
    newwav = numpy.divide(raw * opts.mixvalue, norm)
    
    #if max(abs(newwav)) > 32000.0:
    #  newwav = newwav / max(abs(newwav)) * 32000.0
    #print max(newwav)
  else:
    rate, wav = scipy.io.wavfile.read(input_wav)
    irate, impulse = scipy.io.wavfile.read(input_impulse)
    # low quality silence trimming
    impulse = siltrim(impulse, 3.0)
    # Convert to float
    impulsefloat = scipy.array(impulse, dtype=float)
    wavfloat = scipy.array(wav, dtype=float)
  
    # Perform convolution, truncate end
    if opts.fast:
      outwav = (scipy.signal.fftconvolve(wavfloat, impulsefloat) / sum(impulsefloat))[:len(wav)]
    else:
      outwav = (numpy.convolve(wavfloat, impulsefloat) / sum(impulsefloat))[:len(wav)]
    # Normalisation of output
    outwav = outwav / max(abs(outwav)) * max(abs(wavfloat))
    # Mixup with original signal
    newwav = opts.mixvalue * wavfloat + (1.0 - opts.mixvalue) * outwav
  # Write output file
  scipy.io.wavfile.write(output_wav, rate, scipy.array(newwav, dtype=scipy.int16))


def main():
  # Create a parser
  parser = argparse.ArgumentParser(description='Convolve a signal with a filter')
  
  # Add the program specific options
  #parser.add_argument("--encode", default=True, action=ConfigureAction, help="Encode the input residual")
  parser.add_argument("-f", "--fast", default=False, action="store_true", help="Perform convolution with FFT")
  parser.add_argument("-m", "--mixvalue", default=0.7, type=float, help="Coefficient of original file in mixing, or gain if spectrum")
  parser.add_argument("-s", "--spectrum", action="store_true", help="Treat input impulse as sequence of spectrogram")
  parser.add_argument("-l", "--fftlen", default=1024, type=int, help="Number of spectrum coefficients when using spectrum option")
  parser.add_argument("-r", "--rate", default=48000, type=int, help="Sampling rate")
  parser.add_argument("-p", "--period", default=240, type=int, help="Sampling step when using spectrum option")
  parser.add_argument("-i", "--interperiod", default=80, type=int, help="Sampling step when using spectrum option")
  parser.add_argument("input_wav", help="input audio file")
  parser.add_argument("input_impulse", help="input filter file for convolve")
  parser.add_argument("output_wav", help="Output wav file")
  
  # Processing args
  args = parser.parse_args()

  apply_convolve(args.input_wav, args.input_impulse, args.output_wav, args)

if __name__ == "__main__":
    main()
