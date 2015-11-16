warning('off','all');

[x,fs]=wavread('test.wav');
x=2^15 * x; % kaldi uses 16 bit integers directly
param.samp_freq=fs;
param.window_type='hamming';
param.frame_shift_ms=10;
param.frame_length_ms=30;
param.round_to_power_of_two='true';
param.snip_edges='false';
param.output_type='real_and_imaginary';
param.output_layout='block';
param.cut_dc='false';
param.cut_nyquist='false';
param.nsamples=length(x);
param
fp=param.frame_length_ms*0.001; % frame period in seconds

Sx1=stft(x,param);
writehtk('test.wav.stft_htk.1',Sx1.',fp,9);
xhat1=istft(Sx1,param);
fprintf('NMSE between istft(stft(x)) and x = %f .\n',norm(x-xhat1)/norm(x));

param.output_type='real_and_imaginary';
param.output_layout='interleaved';
param.cut_dc='false';
param.cut_nyquist='false';
param

Sx2=stft(x,param);
writehtk('test.wav.stft_htk.2',Sx2.',fp,9);
xhat2=istft(Sx2,param);
fprintf('NMSE between istft(stft(x)) and x = %f .\n',norm(x-xhat2)/norm(x));

param.round_to_power_of_two='false';
param.output_type='real_and_imaginary';
param.output_layout='block';
param.cut_dc='false';
param.cut_nyquist='false';
param

Sx3=stft(x,param);
writehtk('test.wav.stft_htk.3',Sx3.',fp,9);
xhat3=istft(Sx3,param);
fprintf('NMSE between istft(stft(x)) and x = %f .\n',norm(x-xhat3)/norm(x));

param.round_to_power_of_two='true';
param.output_type='real_and_imaginary';
param.output_layout='block';
param.cut_dc='false';
param.cut_nyquist='true';
param

Sx4=stft(x,param);
writehtk('test.wav.stft_htk.4',Sx4.',fp,9);
xhat4=istft(Sx4,param);
fprintf('NMSE between istft(stft(x)) and x = %f .\n',norm(x-xhat4)/norm(x));

param.snip_edges='true';
param.output_type='real_and_imaginary';
param.output_layout='block';
param.cut_dc='false';
param.cut_nyquist='false';
param

Sx5=stft(x,param);
writehtk('test.wav.stft_htk.5',Sx5.',fp,9);
xhat5=istft(Sx5,param);
fprintf('NMSE between istft(stft(x)) and x = %f .\n',norm(x-xhat5)/norm(x));
