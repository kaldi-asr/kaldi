#!/usr/bin/env python
# Copyright 2014  Johns Hopkins University (Authors: Vijayaditya Peddinti).  Apache 2.0.

# corrupts the wave files supplied via input pipe with the specified
# room-impulse response (RIR) and additive noise distortions (specified by corresponding files)

import wave, struct, sys, scipy.signal as signal, numpy as np, argparse, scipy.io.wavfile, warnings, subprocess

def wave_load_from_command(wav_command, temp_file='temp.wav'):
  try:
    subprocess.check_call(wav_command + ' > ' + temp_file, shell = True)
    return wave_load(temp_file)
  except subprocess.CalledProcessError:
    return None

def wave_load_from_command_secure(wav_command):
  sub_commands = wav_command.split('|')
  subprocess_list = []
  input = None
  for command in sub_commands:
    subprocess_list.append(subprocess.Popen(command.split(), stdin = input, stdout = subprocess.PIPE))
    input = subprocess_list[-1].stdout
  return wave_load(subprocess_list[-1].stdout)
   
def wave_load(file):
  if hasattr(file,'read'):
    warnings.warn(' Assuming that the input is int stream.')
    wav = wave.open(file)
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()
    if sampwidth == 2:
      dtype = np.int16
    elif sampwidth == 4:
      dtype = np.int32
    elif sampwidth == 8:
      dtype = np.int64
    frames = wav.readframes(nframes * nchannels)
    out = struct.unpack_from("%dh" % nframes * nchannels, frames)
    out = np.array(out, dtype = dtype)
    out = np.reshape(out, [nchannels, -1], order = 'F')
    out = out.transpose()
  else:
    [framerate, out] =  scipy.io.wavfile.read(file)
    if len(out.shape) == 1:
      out = out.reshape([out.shape[0], 1])
  if issubclass(out.dtype.type, np.integer):
    max_val = float(np.iinfo(out.dtype).max)
    out = out / max_val
  return (framerate, out)

def wav_write(file_handle, fs, data):
  if str(data.dtype) in set(['float64', 'float32']):
    #rms_val = np.sqrt(np.mean(data * data))
    #data = (0.25 * data / rms_val ) * (2 ** 15)
    data = (0.99 * data / np.max(np.abs(data))) * (2 ** 15)
    data = data.astype('int16', copy = False)
  elif str(data.dtype) == 'int16':
    pass
  else:
    raise Exception('Not implemented for '+str(data.dtype))
  scipy.io.wavfile.write(file_handle, fs, data)
 
def corrupt(x, h, n, snr):
    # x : signal, single channel signal
    # h : room impulse response, can be multi-channel
    # n : noise signal, can be multi-channel (same as h)
    # snr : snr of the noise added signal

    # compute direct reverberation of the RIR
    fs = x[0]
    if h is not None:
      x = x[1][:, 0] # make input single channel
      assert(h[0] == fs)
      h = h[1] # copy the samples from (sampling_rate, samples) tuple
      channel_one = h[:,0]
      max_h = max(channel_one)
      delay_impulse = [i for i, j in enumerate(channel_one) if j == max_h][0]
      before_impulse = np.floor(fs * 0.001)
      after_impulse = np.floor(fs * 0.05)
      direct_rir = channel_one[max(0, delay_impulse - before_impulse):min(len(channel_one), delay_impulse + after_impulse)]
      direct_rir = np.array(direct_rir)
      direct_signal = signal.fftconvolve(x, direct_rir)

      # compute the reverberant signal
      y = np.zeros([x.shape[0] + h.shape[0] - 1, h.shape[1]])
      for channel in xrange(h.shape[1]):
        y[:, channel] = signal.fftconvolve(x, h[:,channel])
    else:
      y = x
      direct_signal = x[:,1].reshape([x.shape[0], 1])
      delay_impulse = 0
    # compute the scaled noise    
    if n is not None:
      fs_n = n[0]
      n = n[1]
      sys.stderr.write('Noise signal : '+str(n.shape) + '\n')
      n_y = np.zeros(y.shape) 
      assert(fs_n == fs) # sampling rate of noise and signal is same
      assert(n.shape[1] == y.shape[1]) # both the reverberant signal and noise signal have the same number of channels
      # repeat the source noise data "n" to match the length of the reverberant signal
      num_reps = int(np.floor(n_y.shape[0] / n.shape[0]))
      dest_array_index = 0
      for i in xrange(0, num_reps):
        np.copyto(n_y[n.shape[0] * i : n.shape[0] * (i+1), :], n)
        dest_array_index = n.shape[0] * (i+1)
      # fill the remaining portion of destination with the initial samples of n
      np.copyto(n_y[dest_array_index:, :], n[0 : n_y.shape[0] - dest_array_index, :])
      # normalize noise data according to the prefixed SNR value
      n_ref = n_y[:, 0]
      n_power = float(np.mean(n_ref**2))
      x_power = float(np.mean(direct_signal**2))
      M_snr = np.multiply(1/n_power, x_power)
      M_snr = np.sqrt((10**(-snr/10))*M_snr)
      n_scaled = np.dot(n_y, np.diagflat(M_snr))
      y = y + n_scaled
      # scipy.io.savemat('debug.mat',{'n_scaled':n_scaled, 'y':y, 'x':x, 'h':h})
    return y[delay_impulse:(delay_impulse + x.shape[0]), :]

if __name__ == "__main__":
  usage = """ Python script to corrupt the input wav stream with
  the specified room impulse response and noise source."""
  sys.stderr.write(str(" ".join(sys.argv)))
  main_parser = argparse.ArgumentParser(usage)
  main_parser.add_argument('--temp-file-name', type=str, default='temp.wav', help='file name of temp file to be used')
  main_parser.add_argument('input_file', type=str, help='file with list of wave files and corresponding corruption parameters')
  main_params = main_parser.parse_args() 
  temp_file = main_params.temp_file_name
  wav_param_list = map( lambda x: x.strip(), open(main_params.input_file))
  
  for line in wav_param_list:
    try:
      parser = argparse.ArgumentParser()
      parser.add_argument('--rir-file', type=str, help='file with the room impulse response')
      parser.add_argument('--noise-file', type=str, help='file with additive noise')
      parser.add_argument('--snr-db', type=float, default=20, help='desired SNR(dB) of the output')
      parser.add_argument('--multi-channel', type=str, default='False', help='is output multi-channel')
      parser.add_argument('input_file', type=str, help='input-file')
      parser.add_argument('output_file', type=str, help='output-file')

      parts = line.split('|') 
      wav_command = "|".join(parts[:-1])
      params = parser.parse_args(parts[-1].split())
      if params.multi_channel.lower() == 'true':
        params.multi_channel = True
        raise Exception("Cannot generate multi-channel outputs")
      else:
        params.multi_channel = False
      sys.stderr.write(line)
      # read the wav input from the stdin
      x = wave_load_from_command(wav_command, temp_file)
      if x is None:
        sys.stderr.write('There was error trying to run the command\n'+wav_command)
        continue
        
      sys.stderr.write('Input signal : '+str(x[1].shape) + '\n')
      fs = x[0]
      if x[1].shape[1] > 1:
        raise Exception('Input wave file cannot be multi-channel')
      # read the impulse response if available from the file
      if params.rir_file is not None:
        h = wave_load(params.rir_file)
        if not params.multi_channel:
          sys.stderr.write('Impulse response : '+str(h[1].shape) + '\n')
          channel1 = h[1][:, 0]
          h = (h[0], channel1.reshape([channel1.shape[0],1])) # just select the first channel
      else:
        h = None

      # read the noise if available from the file
      if params.noise_file is not None:
        n = wave_load(params.noise_file)
        if not params.multi_channel:
          channel1 = n[1][:, 0]
          n = (n[0], channel1.reshape([channel1.shape[0], 1]))
      else:
        n = None 

      y = corrupt(x, h, n, params.snr_db)
      wav_write(params.output_file, fs, y)
      sys.stderr.write('Output signal : '+str(y.shape) + '\n')
      if hasattr(params.output_file, 'write'):
        params.output_file.flush()
    except struct.error:
      warnings.warn("Could not reverberate signal {0}")
      continue
