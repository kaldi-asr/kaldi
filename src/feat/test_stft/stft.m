function X=stft(x,param)

% stft 
%
% X=stft(x,param)
%
% Inputs:
% x: vector of signal of length nsample
% param: kaldi-like options
%
% Output:
% X: nbins x nframes matrix containing the STFT coefficients 
% note that the FFT vector is in the columns of the matrix
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 Hakan Erdogan
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Errors and warnings %%%
if nargin<1, error('Not enough input arguments.'); end

if ~exist('param', 'var')
    param = struct;
end

paramdefaults={ ...
% frame extraction options
    'samp_freq',             'ns',      '16000';...% sampling rate
    'frame_shift_ms',        'ns',      '10';...% frame shift in ms
    'frame_length_ms',       'ns',      '30';...% frame length in ms
    'window_type',           'string',  'hamming';...% hamming, hanning, povey, rectangular
    'round_to_power_of_two', 'string',  'true';... % fft size power of two or not?
    'snip_edges',            'string',  'perfect';... % true, false, or perfect. snip edges or not
% stft options
    'output_type',           'string',  'complex' ;... % complex, amplitude_and_phase, amplitude, phase
    'output_layout',         'string',  'block' ;...% block or interleaved
    'cut_dc',                'string',  'false' ;...% true or false
    'cut_nyquist',           'string',  'false' ;...% true or false
};

[nopt,nf]=size(paramdefaults); % nf is always 3 and not used

% initialize parameters as given above 
% if not provided in param struct already
for i=1:nopt
    if (~isfield(param,paramdefaults{i,1})),
        if (strcmp(paramdefaults{i,2},'string')==1),
            eval(sprintf('param.%s = ''%s'';',paramdefaults{i,1},paramdefaults{i,3}));
        else
            eval(sprintf('param.%s = %s;',paramdefaults{i,1},paramdefaults{i,3}));
        end
    end
end

frame_shift=param.frame_shift_ms * 0.001 * param.samp_freq;
frame_length=param.frame_length_ms * 0.001 * param.samp_freq;
x=x(:);
nsamples=length(x);

% determine number of frames
if (strcmp(param.snip_edges,'true'))
    if (nsamples < frame_length)
        num_frames = 0;
    else
        num_frames = 1 + floor((nsamples - frame_length)/frame_shift);
    end
elseif (strcmp(param.snip_edges,'false'))
    num_frames = round(nsamples / frame_shift);
elseif (strcmp(param.snip_edges,'perfect'))
    num_frames = ceil( 1 + (nsamples+frame_length)/frame_shift );
end

if (num_frames == 0)
    return;
end

% determine frequency bins
if (strcmp(param.round_to_power_of_two,'true'))
    num_fft = 2^(ceil(log2(frame_length)));
else
    num_fft = frame_length;
end

num_bins = num_fft/2 + 1;

% 
% if (strcmp(param.cut_dc,'true'))
%     num_bins = num_bins - 1;
% end
% 
% if (strcmp(param.cut_nyquist,'true'))
%     num_bins = num_bins - 1;
% end

SX=zeros(num_bins, num_frames);

if (strcmp(param.window_type,'hamming'))
    wfunc = hamming(frame_length);
elseif (strcmp(param.window_type,'hanning'))
    wfunc = hanning(frame_length);
elseif (strcmp(param.window_type,'rectangular'))
    wfunc = ones(1,frame_length);
elseif (strcmp(param.window_type,'povey'))
    wfunc = (0.5 - 0.5*cos(2*pi*[0:(frame_length-1)] ./ (frame_length-1))).^0.85;
end
wfunc=wfunc(:);

if (strcmp(param.snip_edges,'true'))
    for fr=1:num_frames
        ind_vec = (fr-1)*frame_shift+(1:frame_length); % one-based index
        xfft=fft(x(ind_vec).*wfunc,num_fft);
        SX(:,fr)=xfft(1:num_bins);
    end
elseif (strcmp(param.snip_edges,'false'))
    for fr=1:num_frames
        ind_vec = (fr-0.5)*frame_shift-frame_length/2+(0:frame_length-1); % 0-based index
        ind_vec(ind_vec < 0) = -ind_vec(ind_vec < 0); % 0-based index
        ind_vec(ind_vec > nsamples-1) = nsamples - 2 - (ind_vec(ind_vec > nsamples - 1) - nsamples) ; % 0-based index
        xfft=fft(x(1+ind_vec).*wfunc,num_fft);
        SX(:,fr)=xfft(1:num_bins);
    end
elseif (strcmp(param.snip_edges,'perfect'))
        for fr=1:num_frames
        ind_vec = fr*frame_shift-frame_length+(0:frame_length-1); % 0-based index
        ind_vec(ind_vec < 0) = -ind_vec(ind_vec < 0); % 0-based index
        ind_vec(ind_vec > nsamples-1) = nsamples - 2 - (ind_vec(ind_vec > nsamples - 1) - nsamples) ; % 0-based index
        xfft=fft(x(1+ind_vec).*wfunc,num_fft);
        SX(:,fr)=xfft(1:num_bins);
    end
end

if (strcmp(param.cut_dc,'true'))
    SX=SX(2:end,:);
end

if (strcmp(param.cut_nyquist,'true'))
    SX=SX(1:end-1,:);
end

% num_bins may be changed now due to cut_dc and/or cut_nyquist
[num_bins, num_frames]=size(SX);

if (strcmp(param.output_type,'amplitude_and_phase'))
    X=[abs(SX);angle(SX)];
elseif (strcmp(param.output_type,'real_and_imaginary'))
    X=[real(SX);imag(SX)];
elseif (strcmp(param.output_type,'complex'))
    X=SX;
elseif (strcmp(param.output_type,'amplitude'))
    X=[abs(SX)];
elseif (strcmp(param.output_type,'phase'))
    X=[angle(SX)];
end

if (strcmp(param.output_type,'amplitude_and_phase') || strcmp(param.output_type,'real_and_imaginary')) 
    if (strcmp(param.output_layout,'interleaved'))
        reorder_index = [(1:num_bins)' num_bins+(1:num_bins)']';
        reorder_index = reorder_index(:);
        X=X(reorder_index,:);
    end
end

return;
