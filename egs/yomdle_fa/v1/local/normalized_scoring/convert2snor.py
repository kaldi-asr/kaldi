#!/usr/bin/env python3

import sys
import io

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf8")

for line in sys.stdin:
    line = line.strip()
    line_vect = line.split()
    utt_id = line_vect[0]
    utt = ' '.join(line_vect[1:])
    sys.stdout.write(utt + " (" + utt_id + ")\n")
