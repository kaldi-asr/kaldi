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


