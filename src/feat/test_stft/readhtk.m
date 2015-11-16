function [d,fp,dt,tc,t]=readhtk(file)
%READHTK  read an HTK parameter file [D,FP,DT,TC,T]=(FILE)
%
% Input:
%    FILE = name of HTX file
% Outputs:
%       D = data: column vector for waveforms, one row per frame for other types
%      FP = frame period in seconds
%      DT = data type (also includes Voicebox code for generating data)
%             0  WAVEFORM     Acoustic waveform
%             1  LPC          Linear prediction coefficients
%             2  LPREFC       LPC Reflection coefficients:  -lpcar2rf([1 LPC]);LPREFC(1)=[];
%             3  LPCEPSTRA    LPC Cepstral coefficients
%             4  LPDELCEP     LPC cepstral+delta coefficients (obsolete)
%             5  IREFC        LPC Reflection coefficients (16 bit fixed point)
%             6  MFCC         Mel frequency cepstral coefficients
%             7  FBANK        Log Fliter bank energies
%             8  MELSPEC      linear Mel-scaled spectrum
%             9  USER         User defined features
%            10  DISCRETE     Vector quantised codebook
%            11  PLP          Perceptual Linear prediction
%            12  ANON
%      TC = full type code = DT plus (optionally) one or more of the following modifiers
%               64  _E  Includes energy terms
%              128  _N  Suppress absolute energy
%              256  _D  Include delta coefs
%              512  _A  Include acceleration coefs
%             1024  _C  Compressed
%             2048  _Z  Zero mean static coefs
%             4096  _K  CRC checksum (not implemented yet)
%             8192  _0  Include 0'th cepstral coef
%            16384  _V  Attach VQ index
%            32768  _T  Attach delta-delta-delta index
%       T = text version of type code e.g. LPC_C_K

%   Thanks to Dan Ellis (ee.columbia.edu) for sorting out decompression.
%   Thanks to Stuart Anderson (whispersys.com) for making it work on 64 bit machines.

%      Copyright (C) Mike Brookes 2005
%      Version: $Id: readhtk.m 713 2011-10-16 14:45:43Z dmb $
%
%   VOICEBOX is a MATLAB toolbox for speech processing.
%   Home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This program is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation; either version 2 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You can obtain a copy of the GNU General Public License from
%   http://www.gnu.org/copyleft/gpl.html or by writing to
%   Free Software Foundation, Inc.,675 Mass Ave, Cambridge, MA 02139, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid=fopen(file,'r','b');
if fid < 0
    error(sprintf('Cannot read from file %s',file));
end
nf=fread(fid,1,'int32');             % number of frames
fp=fread(fid,1,'int32')*1.E-7;       % frame interval (converted to seconds)
by=fread(fid,1,'int16');            % bytes per frame
tc=fread(fid,1,'int16');            % type code (see comments above for interpretation)
tc=tc+65536*(tc<0);
cc='ENDACZK0VT';                    % list of suffix codes
nhb=length(cc);                     % number of suffix codes
ndt=6;                              % number of bits for base type
hb=floor(tc*pow2(-(ndt+nhb):-ndt));
hd=hb(nhb+1:-1:2)-2*hb(nhb:-1:1);   % extract bits from type code
dt=tc-pow2(hb(end),ndt);            % low six bits of tc represent data type

% hd(7)=1 CRC check
% hd(5)=1 compressed data
if (dt==5)  % hack to fix error in IREFC files which are sometimes stored as compressed LPREFC
    fseek(fid,0,'eof');
    flen=ftell(fid);        % find length of file
    fseek(fid,12,'bof');
    if flen>14+by*nf        % if file is too long (including possible CRCC) then assume compression constants exist
        dt=2;               % change type to LPREFC
        hd(5)=1;            % set compressed flag
        nf=nf+4;            % frame count doesn't include compression constants in this case
    end
end

if any(dt==[0,5,10])        % 16 bit data for waveforms, IREFC and DISCRETE
    d=fread(fid,[by/2,nf],'int16').';
    if ( dt == 5),
        d=d/32767;                    % scale IREFC
    end
else
    if hd(5)                            % compressed data - first read scales
        nf = nf - 4;                    % frame count includes compression constants
        ncol = by / 2;
        scales = fread(fid, ncol, 'float');
        biases = fread(fid, ncol, 'float');
        d = ((fread(fid,[ncol, nf], 'int16')+repmat(biases,1,nf)).*repmat(1./scales,1,nf)).';
    else                              % uncompressed data
        d=fread(fid,[by/4,nf],'float').';
    end
end;
fclose(fid);
if nargout > 4
    ns=sum(hd);                 % number of suffixes
    kinds={'WAVEFORM' 'LPC' 'LPREFC' 'LPCEPSTRA' 'LPDELCEP' 'IREFC' 'MFCC' 'FBANK' 'MELSPEC' 'USER' 'DISCRETE' 'PLP' 'ANON' '???'};
    t=[kinds{min(dt+1,length(kinds))} reshape(['_'*ones(1,ns);cc(hd>0)],1,2*ns)];
end
