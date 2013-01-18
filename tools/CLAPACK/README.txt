The files in this directory were downloaded from
http://www.netlib.org/clapack/{clapack,cblas,f2c}.h
In f2c.h, the following lines were modified:

typedef /*long*/ int integer;
typedef unsigned /*long*/ int uinteger;

The "long"'s were commented out.  This seems to be necessary to get OpenBLAS to
link properly; it seems to be compiling Fortran integers as 32 bits, even when
compiling for a 64 bit architecture.  If we compiled with INTERFACE64=1, this
caused much harder problems, by changing the whole of CBLAS to have a 64 bit
interface without giving us a usable header that we could #include that would
correspond to that.
