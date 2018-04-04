import cmath, random, math

import numpy as np

# f0 is fundamental frequency in Hz, while fs is sampling frequency
# N2 is glottal opening duration, N1 is "duty" of the cycle 
def rosenberg_pulse(N1, N2, pulselength, fftlen=1024, randomize=False, differentiate=True, normalize=False):

    N2 = int(math.floor(pulselength*N2))
    N1 = int(math.floor(N1*N2))
    if differentiate:
        gn = np.zeros(fftlen + 1)
    else:
        gn = np.zeros(fftlen)
    offset = fftlen/2 - N1
    # Opening phase
    for n in range(0, N1):
        gn[n + offset] = 0.5 * (1-math.cos(np.pi*n / N1))
    # Closing phase
    for n in range(N1, N2):
        gn[n + offset] = math.cos(np.pi*(n-N1)/(N2-N1)/2)
    if randomize:
        rnd_val += (random.random() - 0.5) * np.pi * 0.01
    else:
        rnd_val = 0
    if differentiate:
        gn = np.diff(gn)
    # Normalise in the FFT domain
    if normalize:
        gn = np.fft.fftshift(gn)
        pulse_fft = np.fft.rfft(gn)
        for i, c in enumerate(pulse_fft):
            if i != 0 and i != len(pulse_fft) -1:
                pulse_fft[i] = cmath.rect(1.0, cmath.polar(c)[1] + rnd_val)
        gn = np.fft.irfft(pulse_fft)
        gn = np.fft.ifftshift(gn)
    return gn

def excitation(f0s, srate, frame_shift):
    time = 0.
    time_idx = 0
    excitation_frame_idx = 0
    nrframes = len(f0s)
    f0min = 50.0
    raw_excitation = []
        
    while excitation_frame_idx < nrframes:
            
        # Get the f0 for the frame (NOT LOG F0)
        # We *could* try to interpolate, but we have to be careful with unvoiced
        # regions marked with 0.0
        if excitation_frame_idx > nrframes:
            frame_f0 = 0.0
        else:
            frame_f0 = f0s[excitation_frame_idx]

        if frame_f0 > 0.0:
            frame_f0 = max(f0min, frame_f0) # ensure the pitch period isn't too long
            pitch_period = srate / frame_f0
            voiced = True
        else:
            frame_f0 = 1.0 / frame_shift
            pitch_period = srate * frame_shift
            voiced = False

        pitch_period_int = int(pitch_period)
        pulse_magnitude = np.sqrt(pitch_period_int)
        if voiced:
            noise_factor = 0
        else:
            noise_factor = 1

        # Create Excitation
        pulse = rosenberg_pulse(0.6, 0.5, pitch_period_int, pitch_period_int)
        pulse *= pulse_magnitude / np.sqrt(sum(pulse ** 2))

        noise = np.random.normal(0., 1.0, pitch_period_int)

        mixed = (1.0 - noise_factor) * pulse + noise_factor * noise
        

        raw_excitation += list(mixed)
        time += 1. / frame_f0
        #time_idx = int(srate * time)

        while time > (excitation_frame_idx + 1) * frame_shift:
            excitation_frame_idx += 1

        if excitation_frame_idx >= nrframes:
            excitation_frame_idx = nrframes
            break
    return raw_excitation

def main():
    srate = 48000
    f0s = [100.0, 110.0, 115.0, 0.0, 0.0, 50.0, 55.0, 60.0, 55.0, 50.0]
    raw = excitation(f0s, srate, 0.010)
    open("/tmp/excitation.raw", 'w').write(np.array(raw, dtype=np.float32).tostring())
    
if __name__ == "__main__":
    main()
