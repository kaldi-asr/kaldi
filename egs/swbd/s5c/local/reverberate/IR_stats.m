function [rt,drr,cte,cfs,edt] = IR_stats(x, fs, varargin)
%function [rt,drr,cte,cfs,edt] = IR_stats(filename, varargin)
%function [rt,drr,cte,cfs,edt] = IR_stats(x, fs, varargin)
% Calculate RT, DRR, Cte, and EDT for impulse response file
% 
%   RT = IR_STATS(FILENAME) returns the reverberation time (to -60 dB)
%   using a method based on ISO 3382-1:2009. The function uses reverse
%   cumulative trapezoidal integration to estimate the decay curve, and a
%   linear least-square fit to estimate the slope between 0 dB and -60 dB.
%   Estimates are taken in octave bands and the overall figure is an
%   average of the 500 Hz and 1 kHz bands.
% 
%   FILENAME should be the full path to an audio file or the name of an
%   audio file on the Matlab search path. The file can be of any format
%   supported by the AUDIOREAD function, and have any number of channels;
%   estimates (and plots) will be returned for each channel.
% 
%   The function returns a 1xN vector of RTs, where N is the number of
%   channels in the audio file.
% 
%   The function determines the direct sound as the peak of the squared
%   impulse response.
% 
%   [RT,DRR] = IR_STATS(FILENAME) returns the direct-to-reverberant-ratio
%   DRR for the impulse; DRR is the same size as RT. This is calculated
%   in the following way:
%   
%   DRR = 10 * log10( X(T0-C:T0+C)^2 / X(T0+C+1:end)^2 )
% 
%   where X is the approximated integral of the impulse, T0 is the time of
%   the direct impulse, and C=2.5ms [1].
% 
%   [RT,DRR,CTE] = IR_STATS(FILENAME) returns the early-to-late index CTE
%   for the impulse; CTE is the same size as RT. This is calculated in
%   the following way:
%   
%   CTE = 10 * log10( X(T0-C:T0+TE)^2 / X(T0+TE+1:end)^2 )
% 
%   where TE is 50 ms.
% 
%   [RT,DRR,CTE,CFS] = IR_STATS(FILENAME) returns the octave-band centre
%   frequencies CFS used in the calculation of RT.
% 
%   [RT,DRR,CTE,CFS,EDT] = IR_STATS(FILENAME) returns the early decay
%   time EDT, which is the same size as RT. The slope of the decay curve
%   is determined from the fit between 0 and -10 dB. The decay time is
%   calculated from the slope as the time required for a 60 dB decay.
% 
%   ... = IR_STATS(...,'PARAMETER',VALUE) allows numerous
%   parameters to be specified. These parameters are:
% 
%       'graph'      : {false} | true
%           Controls whether decay curves are plotted. Specifically, graphs
%           are plotted of the impulse response, decay curves, and linear
%           least-square fit for each octave band and channel of the audio
%           file. If the EDT output is specified, the EDT fit will also be
%           plotted.
%       'te'         : {0.05} | scalar
%           Specifies the early time limit (in seconds).
%       'spec'       : {'mean'} | 'full'
%           Determines the nature of RT and EDT outputs. With spec='mean'
%           (default) the reported RT and EDT are the mean of the 500 Hz
%           and 1 kHz bands. With spec='full', the function returns the
%           RT and EDT as calculated for each octave band returned in
%           CFS; RT and EDT have size [M N] where M=length(CFS).
%       'y_fit'      : {[0 60]} | two-element vector
%           Specifies the decibel range over which the decay curve should
%           be evaluated. For example, 'y_fit' may be [-5 -25] or [-5 -35]
%           corresponding to the RT20 and RT30 respectively.
%       'correction' : {0.0025} | scalar
%           Specifies the correction parameter C (in seconds) given above
%           for DRR and CTE calculations. Values of up to 10 ms have been
%           suggested in the literature.
% 
%   Octave-band filters are calculated according to ANSI S1.1-1986 and IEC
%   standards. Note that the OCTDSGN function recommends centre frequencies
%   fc in the range fs/200 < fc < fs/5.
% 
%   The author would like to thank Feifei Xiong for his input on the
%   correction parameter.
% 
%   References
% 
%   [1] Zahorik, P., 2002: 'Direct-to-reverberant energy ratio
%       sensitivity', The Journal of the Acoustical Society of America, 
%       112, 2110-2117.
% 
%   See also AUDIOREAD, OCTDSGN.

% =========================================================================
% Last changed:     $Date: 2015-06-19 12:05:24 +0100 (Fri, 19 Jun 2015) $
% Last committed:   $Revision: 387 $
% Last changed by:  $Author: ch0022 $
% =========================================================================

    %% validate inputs and set options

    % check file exists
    %assert(exist(filename,'file')==2,['IR_stats: ' filename ' does not exist'])

    % set defaults
    options = struct(...
        'graph',false,...
        'te',0.05,...
        'spec','mean',...
        'y_fit',[0 -60],...
        'correction',0.0025);

    % read parameter/value inputs
    if nargin>1 % if parameters are specified
        % read the acceptable names
        optionNames = fieldnames(options);
        % count arguments
        nArgs = length(varargin);
        if round(nArgs/2)~=nArgs/2
           error('IR_STATS needs propertyName/propertyValue pairs')
        end
        % overwrite defults
        for pair = reshape(varargin,2,[]) % pair is {propName;propValue}
           IX = strcmpi(pair{1},optionNames); % find match parameter names
           if any(IX)
              % do the overwrite
              options.(optionNames{IX}) = pair{2};
           else
              error('%s is not a recognized parameter name',pair{1})
           end
        end
    end
    
    %% read in audio file

    % read in impulse
    %[x,fs] = audioread(filename);
    assert(fs>=5000,'Sampling frequency is too low. FS must be at least 5000 Hz.')

    % set te in samples
    te = round(options.te*fs);

    % Check sanity of te
    assert(te<length(x),'The specified early time limit te is longer than the duration of the impulse!')

    % get number of channels
    numchans = size(x,2);
    
    %% set up octave-band filters
    
    % octave-band center frequencies
    cfs = [31.25 62.5 125 250 500 1000 2000 4000 8000 16000];

    % octave-band filter order
    N = 3;

    % limit centre frequencies so filter coefficients are stable
    cfs = cfs(cfs>fs/200 & cfs<fs/5);
    cfs = cfs(:);

    % calculate filter coefficients
    a = zeros(length(cfs),(2*N)+1);
    b = zeros(length(cfs),(2*N)+1);
    for f = 1:length(cfs)
        [b(f,:),a(f,:)] = octdsgn(cfs(f),fs,N);
    end
    
    %% perform calculations

    % empty matrices to fill
    z = zeros([length(cfs) size(x)]);
    rt_temp = zeros([length(cfs) numchans]);
    edt = zeros([length(cfs) numchans]);
    t0 = zeros(1,numchans);
    drr = zeros(1,numchans);
    cte = zeros(1,numchans);

    correction = round(options.correction*fs);

    % filter and integrate
    for n = 1:numchans
        t0(n) = find(x(:,n).^2==max(x(:,n).^2)); % find direct impulse
        if options.graph
            scrsz = get(0,'ScreenSize');
            figpos = [((n-1)/numchans)*scrsz(3) scrsz(4) scrsz(3)/2 scrsz(4)];
            figure('Name',['Channel ' num2str(n)],'OuterPosition',figpos);
        end
        for f = 1:length(cfs)
            y = filter(b(f,:),a(f,:),x(:,n)); % octave-band filter
            temp = cumtrapz(y(end:-1:1).^2); % decay curve
            z(f,:,n) = temp(end:-1:1);
            [rt_temp(f,n),E_rt,fit_rt] = calc_decay(z(f,t0:end,n),options.y_fit,60,fs); % estimate RT
            [edt(f,n),E_edt,fit_edt] = calc_decay(z(f,t0:end,n),[0,-10],60,fs); % estimate EDT
            if options.graph % plot
                % time axes for different vectors
                ty = ((0:length(y)-1)-t0(n))./fs;
                tE_rt = (0:length(E_rt)-1)./fs;
                tE_edt = (0:length(E_edt)-1)./fs;
                % plot
                subplot(length(cfs),2,(2*f)-1)
                plot(ty,y,'k') % octave-band impulse
                if f==1
                    title({'Impulse response'; ''; [num2str(cfs(f)) ' Hz octave band']})
                else
                    title([num2str(cfs(f)) ' Hz octave band'])
                end
                if f==length(cfs)
                    xlabel('Time [s]')
                else
                    set(gca,'xticklabel',[]);
                end
                ylabel('Amplitude')
                set(gca,'position',[1 1 1 1.05].*get(gca,'position'),'xlim',[min(ty) max(ty)]);
                subplot(length(cfs),2,2*f)
                % energy decay and linear least-square fit
                if nargout==5
                    % plot EDT fit if EDT wanted
                    plot(tE_rt,E_rt,'-k',tE_rt,fit_rt,'--r',tE_edt,fit_edt,':b')
                else
                    plot(tE_rt,E_rt,'-k',tE_rt,fit_rt,'--r')
                end
                % title for top row
                if f==1
                    title({'Decay curve'; ''; [num2str(cfs(f)) ' Hz octave band']})
                else
                    title([num2str(cfs(f)) ' Hz octave band'])
                end
                % x label for bottom row
                if f==length(cfs)
                    xlabel('Time [s]')
                else
                    set(gca,'xticklabel',[]);
                end
                ylabel('Energy [dB]')
                set(gca,'position',[1 1 1 1.05].*get(gca,'position'),'ylim',[-70 0],'xlim',[0 max(tE_rt)]);
                % choose legend according to EDT request
                fitstr = num2str(abs(diff(options.y_fit)));
                if nargout==5
                    legend('Energy decay curve',['Linear fit (RT' fitstr ')'],'Linear fit (EDT)','location','northeast')
                else
                    legend('Energy decay curve',['Linear fit (RT' fitstr ')'],'location','northeast')
                end
            end
        end
        % DRR
        if nargout>=2
            drr(n) = 10.*log(...
                trapz(x(max(1,t0(n)-correction):t0(n)+correction,n).^2)/...
                trapz(x(t0(n)+correction+1:end,n).^2)...
                );
        end
        % Cte
        if nargout>=3
            if t0(n)+te+1>size(x,1)
                warning(['Early time limit (te) out of range in channel ' num2str(n) '. Try lowering te.'])
                cte(n) = NaN;
            else
                cte(n) = 10.*log(...
                    trapz(x(max(1,t0(n)-correction):t0(n)+te).^2)/...
                    trapz(x(t0(n)+te+1:end,n).^2)...
                    );
            end
        end
    end
    
    %% write output

    switch lower(options.spec)
        case 'full'
            rt = rt_temp;
        case 'mean'
            rt = mean(rt_temp(cfs==500 | cfs==1000,:)); % overall RT
            edt = mean(edt(cfs==500 | cfs==1000,:)); % overall EDT
        otherwise
            error('Unknown ''spec'': must be ''full'' or ''mean''.')
    end


function [t,E,fit] = calc_decay(z,y_fit,y_dec,fs)
% CALC_DECAY calculate decay time from decay curve
% Returns the time for a specified decay y_dec calculated
% from the fit over the range y_fit. The input is the
% integral of the impulse sample at fs Hz. The function also
% returns the energy decay curve in dB and the corresponding
% fit.

    E = 10.*log10(z); % put into dB
    E = E-max(E); % normalise to max 0
    E = E(1:find(isinf(E),1,'first')-1); % remove trailing infinite values
    IX = find(E<=max(y_fit),1,'first'):find(E<=min(y_fit),1,'first'); % find yfit x-range
    if isempty(IX)
        error('Impulse response has insufficient dynamic range to evaluate to %i dB',min(y_fit))
    end

    % calculate fit over yfit
    x = reshape(IX,1,length(IX));
    y = reshape(E(IX),1,length(IX));
    p = polyfit(x,y,1);
    fit = polyval(p,1:2*length(E)); % actual fit
    fit2 = fit-max(fit); % fit anchored to 0dB

    diff_y = abs(diff(y_fit)); % dB range diff
    t = (y_dec/diff_y)*find(fit2<=-diff_y,1,'first')/fs; % estimate decay time

    fit = fit(1:length(E));


function [B,A] = octdsgn(Fc,Fs,N); 
% OCTDSGN  Design of an octave filter.
%    [B,A] = OCTDSGN(Fc,Fs,N) designs a digital octave filter with 
%    center frequency Fc for sampling frequency Fs. 
%    The filter are designed according to the Order-N specification 
%    of the ANSI S1.1-1986 standard. Default value for N is 3. 
%    Warning: for meaningful design results, center values used
%    should preferably be in range Fs/200 < Fc < Fs/5.
%    Usage of the filter: Y = FILTER(B,A,X). 
%
%    Requires the Signal Processing Toolbox. 
%
%    See also OCTSPEC, OCT3DSGN, OCT3SPEC.

% Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
%         couvreur@thor.fpms.ac.be
% Last modification: Aug. 22, 1997, 9:00pm.

% References: 
%    [1] ANSI S1.1-1986 (ASA 65-1986): Specifications for
%        Octave-Band and Fractional-Octave-Band Analog and
%        Digital Filters, 1993.

if (nargin > 3) | (nargin < 2)
  error('Invalide number of arguments.');
end
if (nargin == 2)
  N = 3; 
end
if (Fc > 0.70*(Fs/2))
  error('Design not possible. Check frequencies.');
end

% Design Butterworth 2Nth-order octave filter 
% Note: BUTTER is based on a bilinear transformation, as suggested in [1]. 
%W1 = Fc/(Fs/2)*sqrt(1/2);
%W2 = Fc/(Fs/2)*sqrt(2); 
pi = 3.14159265358979;
beta = pi/2/N/sin(pi/2/N); 
alpha = (1+sqrt(1+8*beta^2))/4/beta;
W1 = Fc/(Fs/2)*sqrt(1/2)/alpha; 
W2 = Fc/(Fs/2)*sqrt(2)*alpha;
[B,A] = butter(N,[W1,W2]); 


