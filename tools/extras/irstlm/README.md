# irstlm
IRSTLM Toolkit 


CONTENT:

- src: source code
- scripts: supporting scripts
- doc: documentation (in Latex) and in pdf (to be generated)
- bin: binaries (to be generated) and scripts
- lib: libraries (to be generated)
- readme: this file


DOCUMENTATION

A User Manual is available under https://sourceforge.net/projects/irstlm
The data for the examples described in the User Manual are available under http://sourceforge.net/projects/irstlm/files/irstlm/sampledata/


HOW TO INSTALL WITH AUTOMAKE


Step 0: 

$> sh regenerate-makefiles.sh [--force]

Set parameter force to the value "--force" if you want to recreate all links to the autotools


Step 1: 

$> ./configure [--prefix=/path/where/to/install] ...

Run "configure --help" to get more details on the compilation options

If your g++ compiler does not support '-std=c++0x', please add parameter '--disable-cxx0'. To check whether g++ complier does support '-std=c++0x', please run the following command:

$> echo | g++ -E -x c++ -std=c++0x -dM - >& /dev/null ; echo $?

If it returns 0, g++ complier does support '-std=c++0x'; otherwise, it does not, and hence please use '--disable-cxx0'

To enable/disable assert for debugging purpose, please add parameter '--enable-assert' (default) or '--disable-assert'

To modify debugging level, please add parameter '--with-tracelevel=<val>' (default is 0)


Step 2: 

$> make


Step 3: 

$> make install

These steps will generate the irstlm library and commands, respectively, under the specified path where to install.


HOW TO INSTALL WITH CMAKE


Step 0: 

$> cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/path/where/to/install"

Note: If your g++ compiler does not support '-std=c++0x', please add parameter '-DCXX0:BOOL=OFF'. To check whether g++ complier does support '-std=c++0x', please run the following command:

$> echo | g++ -E -x c++ -std=c++0x -dM - >& /dev/null ; echo $?

If it returns 0, g++ complier does support '-std=c++0x'; otherwise, it does not, and hence please use '-DCXX0:BOOL=OFF'

To enable/disable assert for debugging purpose, please add parameter '-DASSERT:BOOL=ON' (default) or '-DASSERT:BOOL=OFF'

To modify debugging level, please add parameter '-DTRACE_LEVEL=<val>' (default is 0)


Step 2: 

$> make


Step 3: 

$> make install


HOW TO CONTRIBUTE

If you wish to contribute to the Open Source IRSTLM toolkit just tell us! 

Marcello Federico
FBK, Trento, ITALY
email: federico AT fbk DOT eu

