#!/bin/bash

set  -e

if [ ! -f cuedrnnlm.tar.gz ]; then
  wget http://mi.eng.cam.ac.uk/projects/cued-rnnlm/cuedrnnlm.tar.gz
fi
tar -zxvf cuedrnnlm.tar.gz

cd cuedrnnlm

cat > patch <<EOF
--- fileops.h   2016-06-01 09:48:49.000000000 -0400
+++ fileops.h2  2016-08-20 20:16:04.925567534 -0400
@@ -70,7 +70,7 @@
             // getvalidchar (fptr, c);
             if (c == '\n')
             {
-                if (cnt==0 && word[0] != '<')
+                if (cnt==0 && strcmp(word, "<s>") != 0)
                 {
                     linevec.push_back("<s>");
                     cnt ++;
@@ -90,7 +90,7 @@
             else if ((c == ' ' || c=='\t') && index > 0) // space in the middle of line
             {
                 word[index] = 0;
-                if (cnt==0 && word[0] != '<')
+                if (cnt==0 && strcmp(word, "<s>") != 0)
                 {
                     linevec.push_back("<s>");
                     cnt ++;
@@ -105,7 +105,7 @@
                 index ++;
             }
         }
-        if (cnt>0 && word[0] != '<')
+        if (cnt > 0 && strcmp(word, "</s>") != 0)
         {
             linevec.push_back("</s>");
             cnt ++;
EOF

patch -p0 < patch

if [ ! -f /usr/local/cuda/bin/nvcc ]; then
  echo This needs to be done on a machine with GPUs!
  exit 1
fi

echo nvcc found. Will start building cued-rnnlm

export PATH=$PATH:/usr/local/cuda/bin/
./build.sh

[ -f rnnlm ] && rm rnnlm
ln -s rnnlm.cued rnnlm

exit

# below is not needed anymore
wget http://mi.eng.cam.ac.uk/projects/cued-rnnlm/src.evaloncpu.tar.gz
tar -zxvf src.evaloncpu.tar.gz
cd src.evaloncpu
./build.sh
cd ../
ln -s src.evaloncpu/rnnlm.eval

echo cued-rnnlm succesfully installed
