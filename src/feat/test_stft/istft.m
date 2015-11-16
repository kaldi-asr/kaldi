function x=istft(X,param)

% stft 
%
% x=istft(X,param)
%
% Inputs:
% X: STFT matrix
% param: kaldi-like options
%
% Output:
% x: nsamples long vector containing the time-domain signal 
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
    'snip_edges',            'string',  'perfect';... % true, false or perfect. snip edges or not
% istft options
    'output_type',           'string',  'complex' ;... % amplitude_and_phase, amplitude, phase
    'output_layout',         'string',  'block' ;...% block or interleaved
    'cut_dc',                'string',  'false' ;...% true or false
    'cut_nyquist',           'string',  'false' ;...% true or false
    'nsamples',              'ns',      '-1';...% <=0 means self-calculate
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

[num_bins, num_frames]=size(X);

if (strcmp(param.output_type,'amplitude') || strcmp(param.output_type,'phase'))
    fprintf('Cannot invert just amplitude or phase.\n');
    x=[];
    return;
end
    
    
% undo interleaving if needed
if (strcmp(param.output_type,'amplitude_and_phase') || strcmp(param.output_type,'real_and_imaginary')) 
    if (strcmp(param.output_layout,'interleaved'))
        reorder_index = [1:2:num_bins, 2:2:num_bins];
        X=X(reorder_index,:);
    end
end

nb2=num_bins/2;

% form complex stft
if (strcmp(param.output_type,'amplitude_and_phase'))
    SX=X(1:nb2,:).*exp(sqrt(-1)*X(nb2+(1:nb2),:));
elseif (strcmp(param.output_type,'real_and_imaginary'))
    SX=X(1:nb2,:)+sqrt(-1)*X(nb2+(1:nb2),:);
elseif (strcmp(param.output_type,'complex'))
    SX=X;
end

if (strcmp(param.cut_dc,'true'))
    SX=[zeros(1,num_frames);SX];
end

if (strcmp(param.cut_nyquist,'true'))
    SX=[SX;zeros(1,num_frames)];
end

SS=conj(SX(end-1:-1:2,:));
SX=[SX;SS];

[num_fft, num_frames]=size(SX);

if (num_fft < frame_length)
    fprintf('num_fft=%d must be greater than or equal to frame_length=%d\n',num_fft,frame_length);
    return;
end

% determine number of samples
if (param.nsamples <= 0)
    if (strcmp(param.snip_edges,'true'))
        nsamples = (num_frames-1) * frame_shift;
    elseif (strcmp(param.snip_edges,'false'))
        nsamples = num_frames * frame_shift;
    elseif (strcmp(param.snip_edges,'perfect'))
        nsamples = (num_frames-1) * frame_shift;
    end
else
    nsamples=param.nsamples;
end

x=zeros(nsamples,1);

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

scale_factor=frame_shift/sum(wfunc);

if (strcmp(param.snip_edges,'true'))
    for fr=1:num_frames
        ind_vec = (fr-1)*frame_shift+(1:num_fft); % one-based index
        x(ind_vec)=x(ind_vec)+real(ifft(SX(:,fr)));
    end
elseif (strcmp(param.snip_edges,'false'))
    for fr=1:num_frames
        ind_vec = (fr-0.5)*frame_shift-frame_length/2+(1:num_fft); % 1-based index
        xfft=real(ifft(SX(:,fr)));
        valid_ones=(ind_vec > 0) & (ind_vec <= nsamples);
        x(ind_vec(valid_ones))=x(ind_vec(valid_ones))+xfft(valid_ones);
    end
elseif (strcmp(param.snip_edges,'perfect'))
    for fr=1:num_frames
        ind_vec = fr*frame_shift-frame_length+(1:num_fft); % 1-based index
        xfft=real(ifft(SX(:,fr)));
        valid_ones=(ind_vec > 0) & (ind_vec <= nsamples);
        x(ind_vec(valid_ones))=x(ind_vec(valid_ones))+xfft(valid_ones);
    end
end

x=scale_factor*x;

return;
