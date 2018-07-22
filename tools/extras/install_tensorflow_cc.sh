#!/bin/bash

set -e

#export JAVA_HOME=/LOCATION_ON_YOUR_MACHINE/java/jdk1.8.0_121
PATH=$PWD/bazel/output:$PATH
export HOME=$PWD/tensorflow_build/
mkdir -p $HOME

java=`which java`
if [ -z "$JAVA_HOME" ]; then
  java=$JAVA_HOME/bin/java
fi

java_version=`$java -version 2>&1 | head -n 1 | awk '{print $3}' | sed s=\"==g`
good_version=`echo 1.8 $java_version | awk '{if($1<$2)print 1; else print 0}'`
if [ $good_version -eq 0 ]; then
  echo You have jdk version = $java_version, which is older than 1.8
  echo You need to download a later than 1.8 JDK version at
  echo http://www.oracle.com/technetwork/pt/java/javase/downloads/jdk8-downloads-2133151.html
  echo and set your JAVA_HOME to point to where it is installed
  exit 1
else
  echo You have jdk version = $java_version, which is newer than 1.8. We will continue the installation
fi


[ ! -f bazel.zip ] && wget https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-dist.zip -O bazel.zip
mkdir -p bazel
cd bazel
unzip ../bazel.zip
./compile.sh
cd ../

# now bazel is built
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout r1.4
./configure

tensorflow/contrib/makefile/download_dependencies.sh 
bazel build -c opt //tensorflow:libtensorflow.so
bazel build -c opt //tensorflow:libtensorflow_cc.so

echo Building tensorflow completed. You will need to go to kaldi/src/ and do
echo \"make\" under tfrnnlm/ and tfrnnlmbin/ to generate the binaries
exit
# the following would utilize the highest optimization but might not work in a
# grid where each machine might have different configurations
bazel build --config=opt //tensorflow:libtensorflow.so
bazel build --config=opt //tensorflow:libtensorflow_cc.so
