#!/usr/bin/env python3

import glob
import os
import sys

for line in glob.glob("{}/*.wav".format(sys.argv[1])):
    fname = os.path.basename(line.strip())
    ext = os.path.splitext(line.strip())[1]
    if ext != ".wav":
        print (ext, file=sys.stderr)
        raise SystemExit(1)

    print ("--noise-id {} --noise-type point-source "
           "--bg-fg-type foreground {}".format(fname, line.strip()))
